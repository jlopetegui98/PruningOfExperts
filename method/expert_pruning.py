from tqdm import tqdm
from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

from model import PrunableMixtralSparseMoeBlockWrapper, PrunableDeepseekMoEWrapper


def progressive_pruning_mixtral(model, calib_loader, r=6):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=r)
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate()
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = r
    model.config.num_local_experts = r

    return model, (global_loss_history, )


def progressive_pruning_deepseek(model, calib_loader, r=16):
    moe_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx == 0:
            continue

        if hasattr(layer.mlp, 'experts') and hasattr(layer.mlp, 'gate'):
            moe_layers.append(layer_idx)
            print(f"Found MoE layer in decoder layer {layer_idx}")

    if not moe_layers:
        raise ValueError("No MoE layers found in the model")
    print(f"Found {len(moe_layers)} MoE layers in the model")

    global_loss_history = {}

    for layer_idx in moe_layers:
        print(f"\nProcessing layer {layer_idx}")
        layer = model.model.layers[layer_idx]

        original_mlp = layer.mlp
        try:
            if not isinstance(layer.mlp, PrunableDeepseekMoEWrapper):
                layer.mlp = PrunableDeepseekMoEWrapper(original_mlp, r=r)
                print(f"Wrapped layer {layer_idx} with prunable wrapper")

                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                batch_size = 1
                seq_len = 8
                hidden_dim = layer.mlp.model.gate.gating_dim

                test_input = torch.randn(batch_size, seq_len, hidden_dim,
                                        device=device, dtype=dtype)

                _ = layer.mlp(test_input)
                print(f"Layer {layer_idx} wrapper tested successfully!")
        except Exception as e:
            print(f"Error wrapping layer {layer_idx}: {e}")
            layer.mlp = original_mlp
            continue

        layer.mlp.cache_space = CacheDataset()

        print(f"Collecting Z activations for layer {layer_idx}...")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(calib_loader):
                success = collect_activations(model, layer, batch, collect_z=True, collect_x=False)
                if not success or batch_idx >= 1:
                    break

        torch.cuda.empty_cache()

        print(f"Collecting X activations for layer {layer_idx}...")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(calib_loader):
                success = collect_activations(model, layer, batch, collect_z=False, collect_x=True)
                if not success or batch_idx >= 1:
                    break

        torch.cuda.empty_cache()

        layer.mlp.cache_X = False
        layer.mlp.cache_Z = False

        try:
            print(f"Enumerating expert combinations for layer {layer_idx}...")
            loss_history = layer.mlp.enumerate()
            global_loss_history[layer_idx] = loss_history

            print(f"Pruning layer {layer_idx}...")
            layer.mlp.prune()
            print(f"Layer {layer_idx} pruned successfully")

            if layer_idx % 4 == 0:
                temp_path = f"deepseek-moe-16b-pruned-checkpoint-layer-{layer_idx}"
                try:
                    print(f"Saving checkpoint after layer {layer_idx}...")
                    model.save_pretrained(temp_path)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

        except Exception as e:
            print(f"Error during enumeration/pruning of layer {layer_idx}: {e}")
            pruned_experts = torch.nn.ModuleList([
                layer.mlp.model.experts[i] for i in range(min(r, len(layer.mlp.model.experts)))
            ])

            layer.mlp.model.experts = pruned_experts

            gate_weights = layer.mlp.model.gate.weight.data
            gate_weights_pruned = gate_weights[:r, :]
            layer.mlp.model.gate.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)

            if hasattr(layer.mlp.model.gate, 'n_routed_experts'):
                layer.mlp.model.gate.n_routed_experts = r

            layer.mlp = layer.mlp.model
            continue

        for layer_idx_reset in range(layer_idx+1, max(moe_layers)+1):
            if layer_idx_reset in moe_layers:
                reset_layer = model.model.layers[layer_idx_reset]
                if isinstance(reset_layer.mlp, PrunableDeepseekMoEWrapper):
                    reset_layer.mlp.experts_to_drop = None
                    reset_layer.mlp.experts_to_keep = None
                    if hasattr(reset_layer.mlp, 'cache_space'):
                        reset_layer.mlp.cache_space = CacheDataset()

    if hasattr(model.config, 'n_routed_experts'):
        model.config.n_routed_experts = r
        print(f"Updated model config n_routed_experts to {r}")

    print("\nPruning complete for all layers!")

    print("\nSaving pruned model...")
    try:
        model.save_pretrained("deepseek-moe-16b-pruned")
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving final model: {e}")

    return model, (global_loss_history,)

def collect_activations(model, layer, batch, collect_z=True, collect_x=False):
    try:
        batch = {k: v.to(model.device) for k, v in batch.items()}

        if collect_z:
            layer.mlp.cache_Z = True
            layer.mlp.cache_X = False
        elif collect_x:
            layer.mlp.cache_Z = False
            layer.mlp.cache_X = True
        else:
            layer.mlp.cache_Z = False
            layer.mlp.cache_X = False

        num_experts = len(layer.mlp.model.experts)
        original_top_k = getattr(layer.mlp, 'top_k', 6)
        layer.mlp.top_k = min(original_top_k, num_experts)

        model_inputs = model.prepare_inputs_for_generation(**batch)
        outputs = model(**model_inputs)

        layer.mlp.top_k = original_top_k

        del model_inputs
        del outputs

        return True
    except Exception as e:
        print(f"Error collecting activations: {e}")
        if 'model_inputs' in locals():
            del model_inputs
        if 'outputs' in locals():
            del outputs
        return False
