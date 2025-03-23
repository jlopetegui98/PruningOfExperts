import torch
import torch.nn.functional as F
import itertools
from data import CacheDataset
from datasets import load_dataset, Dataset
from transformers.testing_utils import CaptureLogger
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Linear4bit
from bitsandbytes.functional import dequantize_4bit, quantize_4bit
from bitsandbytes.functional import QuantState
import random
import math
from tqdm import tqdm


class PrunableMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r

        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.model.gate(hidden_states)

        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            
            current_hidden_states = expert_layer(
                current_state)

        
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            print(
                f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @torch.no_grad()
    def enumerate(self):
        """
        Method to compute the approximation loss for each possible final combination of experts
        w.r.t to the original model.
        """
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        with torch.inference_mode():
            for dropped in itertools.combinations(range(self.model.num_experts), self.model.num_experts - self.r):
                self.experts_to_drop = dropped
                loss = 0

                for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                    hidden_states = hidden_states.to(
                        device=self.model.gate.weight.data.device, non_blocking=True)
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64, device=self.model.gate.weight.data.device, non_blocking=True)

                    final_hidden_states_e, _ = self.forward(
                        hidden_states.unsqueeze(0))
                    loss += torch.norm(final_hidden_states -
                                       final_hidden_states_e.squeeze(0).to(torch.float64)).item()
                loss_history[dropped] = loss

        self.experts_to_drop = min(loss_history, key=loss_history.get)
        return loss_history

    @torch.no_grad()
    def prune(self):
        """
        Main method in the pruning pipeline. It will modify the MoE corresponding layer
        given the best experts configuration found. The main difference with 
        respect to the original implementation is that in this case we allow quantized
        versions of Mixtral (4-bit quantization).
        """
        
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop)
        )

        print(f"Original gate shape: {self.model.gate.weight.shape}")

        # Debugging info for quantization state
        print(f"Gate weight type: {type(self.model.gate.weight)}")

        quantized_flag = False
        # DEQUANTIZE the weights (convert 4-bit back to float)
        # we must consider all possible escenarios of the Params4bit dequantization pipeline
        # to avoid bugs
        if isinstance(self.model.gate.weight, bnb.nn.Params4bit):
            quantized_flag = True
            print("Detected 4-bit quantization, dequantizing...")
            # Check if quant_state exists and is properly defined
            if hasattr(self.model.gate, 'quant_state') and self.model.gate.quant_state is not None:
                print("Using gate.quant_state for dequantization")
                gate_weights_fp32 = bnb.functional.dequantize_4bit(
                    self.model.gate.weight.data, self.model.gate.quant_state
                )
            elif hasattr(self.model.gate.weight, 'quant_state') and self.model.gate.weight.quant_state is not None:
                print("Using weight.quant_state for dequantization")
                gate_weights_fp32 = bnb.functional.dequantize_4bit(
                    self.model.gate.weight.data, self.model.gate.weight.quant_state
                )
            else:
                # Create an output tensor with the right shape
                print("No quant_state found, creating output tensor manually")
                out_shape = (self.model.gate.out_features, self.model.gate.in_features)
                out_tensor = torch.empty(out_shape,
                                       dtype=torch.float32,
                                       device=self.model.gate.weight.data.device)

                # Try accessing absmax from the weight itself if available
                if hasattr(self.model.gate.weight, 'absmax'):
                    print("Using weight.absmax for dequantization")
                    absmax = self.model.gate.weight.absmax
                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=None,
                        absmax=absmax,
                        out=out_tensor,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type
                    )
                elif hasattr(self.model.gate, 'absmax'):
                    print("Using gate.absmax for dequantization")
                    absmax = self.model.gate.absmax
                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=None,
                        absmax=absmax,
                        out=out_tensor,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type
                    )
                else:
                    # Try to create a new quant_state with default values
                    print("Creating a new QuantState with default values")
                    # Create a dummy absmax
                    dummy_absmax = torch.ones((self.model.gate.weight.data.shape[0], 1),
                                             device=self.model.gate.weight.data.device)

                    quant_state = QuantState(
                        absmax=dummy_absmax,
                        shape=out_shape,
                        dtype=torch.float32,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type,
                    )

                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=quant_state
                    )
        else:
            print("No quantization detected, using weights as is")
            gate_weights_fp32 = self.model.gate.weight.data  # Not quantized, use as is

        print(f"Dequantized gate shape: {gate_weights_fp32.shape}")

        # PRUNE the dequantized weights
        gate_weights_pruned = gate_weights_fp32[experts_to_reserve, :]

        print(f"Pruned gate shape: {gate_weights_pruned.shape}")

        # RE-QUANTIZE the pruned weights back to 4-bit
        if quantized_flag:
            try:
                gate_weights_4bit, new_quant_state = bnb.functional.quantize_4bit(
                    gate_weights_pruned,
                    blocksize=self.model.gate.weight.blocksize,
                    compress_statistics=self.model.gate.weight.compress_statistics,
                    quant_type=self.model.gate.weight.quant_type,
                    quant_storage=self.model.gate.weight.quant_storage
                )

                print("Successfully re-quantized the weights")
            except Exception as e:
                print(f"Error during re-quantization: {e}")
                print("Falling back to using pruned weights directly")
                # Fallback: use the pruned weights directly without re-quantizing
                gate_weights_4bit = gate_weights_pruned
                new_quant_state = None

            # CREATE a new gate layer with the updated quantized weights
            try:
                gate_new = bnb.nn.Linear4bit(
                    input_features=self.model.gate.in_features,
                    output_features=self.r,
                    bias=False,
                    device="cpu",
                )

                for param in gate_new.parameters():
                    param.requires_grad = False

                # Convert the new gate weight depending on whether re-quantization worked
                if new_quant_state is not None:
                    gate_new.weight = bnb.nn.Params4bit(
                        gate_weights_4bit,
                        requires_grad=False,
                        quant_state=new_quant_state,
                        blocksize=self.model.gate.weight.blocksize,
                        compress_statistics=self.model.gate.weight.compress_statistics,
                        quant_type=self.model.gate.weight.quant_type,
                        quant_storage=self.model.gate.weight.quant_storage
                    )
                else:
                    # If re-quantization failed, use the pruned weights directly
                    gate_new.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)

                print(f"New pruned gate shape: {gate_new.weight.shape}")
            except Exception as e:
                print(f"Error creating new gate layer: {e}")
                # Try a more direct approach
                print("Attempting to modify gate layer directly")
                self.model.gate.out_features = self.r
                self.model.gate.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)
        
        else:
            gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=self.r, bias=False, device='cpu', dtype=torch.bfloat16)
            gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]

        # UPDATE the model with the new pruned gate if creation was successful
        if 'gate_new' in locals():
            self.model.gate = gate_new

        # PRUNE the expert list
        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve]
        )
        
        self.model.num_experts = self.r

        print(f"Successfully pruned the model to {self.r} experts")


class PrunableDeepseekMoEWrapper(torch.nn.Module):
    def __init__(self, model, r=None):
        super().__init__()
        self.model = model
        self.r = r

        self.num_experts = len(self.model.experts)
        self.n_routed_experts = getattr(self.model.gate, 'n_routed_experts', self.num_experts)
        self.top_k = getattr(self.model.gate, 'top_k', 6)

        self.experts_to_drop = None
        self.experts_to_keep = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

        self.config = getattr(self.model, 'config', None)
        self.training = self.model.training

        print(f"Initialized wrapper with {self.num_experts} experts and top_k={self.top_k}")

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape

        if self.cache_X:
            self.cache_space.append(X=hidden_states.detach())

        if hidden_states.dim() == 2:
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.view(1, -1, hidden_dim)

        with torch.no_grad():
            try:
                topk_idx, topk_weight, _ = self.model.gate(hidden_states)

                if hasattr(self.model, 'experts'):
                    actual_num_experts = len(self.model.experts)
                    flat_topk_idx = topk_idx.view(-1)
                    invalid_indices = flat_topk_idx >= actual_num_experts
                    if invalid_indices.any():
                        flat_topk_idx[invalid_indices] = 0
                        topk_idx = flat_topk_idx.view_as(topk_idx)
            except ValueError:
                hidden_dim = hidden_states.shape[-1]
                router_logits = F.linear(
                    hidden_states.view(-1, hidden_dim),
                    self.model.gate.weight,
                    None
                )

                if self.experts_to_drop is not None:
                    for e in self.experts_to_drop:
                        if e < router_logits.shape[1]:
                            router_logits[:, e] = -float('inf')

                scores = router_logits.softmax(dim=-1)
                k = min(self.top_k, scores.shape[1])
                topk_weight, topk_idx = torch.topk(scores, k=k, dim=-1)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.experts_to_drop is not None:
            for expert_idx in self.experts_to_drop:
                if expert_idx >= len(self.model.experts):
                    continue

                mask = (flat_topk_idx == expert_idx)
                if mask.any():
                    if self.experts_to_keep is None:
                        self.experts_to_keep = [i for i in range(min(self.num_experts, len(self.model.experts)))
                                              if i not in self.experts_to_drop]

                    if self.experts_to_keep:
                        flat_topk_idx[mask] = self.experts_to_keep[0]

        if self.training:
            hidden_states = hidden_states.repeat_interleave(topk_idx.shape[1], dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.model.experts):
                if self.experts_to_drop is not None and i in self.experts_to_drop:
                    continue
                mask = (flat_topk_idx == i)
                if mask.any():
                    y[mask] = expert(hidden_states[mask])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            if hasattr(self.model, 'moe_infer'):
                y = self.model.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1))
                y = y.view(*orig_shape)
            else:
                outputs = []
                for i, expert in enumerate(self.model.experts):
                    if self.experts_to_drop is not None and i in self.experts_to_drop:
                        continue
                    mask = (flat_topk_idx == i)
                    if mask.any():
                        expert_output = expert(hidden_states[mask])
                        for idx, output in zip(torch.where(mask)[0], expert_output):
                            if len(outputs) <= idx:
                                outputs.extend([None] * (idx - len(outputs) + 1))
                            outputs[idx] = output

                assert all(o is not None for o in outputs), "Some tokens have no expert outputs"
                y = torch.stack(outputs).view(*orig_shape)

        if hasattr(self.model, 'shared_experts') and self.model.shared_experts is not None:
            if self.config is not None and getattr(self.config, 'n_shared_experts', None) is not None:
                y = y + self.model.shared_experts(identity)

        if self.cache_Z:
            self.cache_space.append(Z=y.detach())

        return y

    @torch.no_grad()
    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        num_experts = len(self.model.experts)
        keep_count = min(self.r, num_experts)
        drop_count = max(0, num_experts - keep_count)

        if drop_count == 0:
            print(f"Already at target expert count ({num_experts}), no dropping needed")
            self.experts_to_drop = tuple()
            self.experts_to_keep = list(range(num_experts))
            return loss_history

        max_combinations = 100

        if math.comb(num_experts, drop_count) <= max_combinations:
            max_combinations = math.comb(num_experts, drop_count)
            print(f"Enumerating all {max_combinations} possible combinations")
        else:
            print(f"Total possible combinations: {math.comb(num_experts, drop_count)}, evaluating {max_combinations}")

        if hasattr(self, 'cache_space') and hasattr(self.cache_space, 'Xs') and len(self.cache_space.Xs) > 0:
            try:
                with torch.inference_mode():
                    hidden_states = self.cache_space.Xs[0].to(self.model.gate.weight.device)

                    router_logits = F.linear(
                        hidden_states.view(-1, hidden_states.shape[-1]),
                        self.model.gate.weight,
                        None
                    )

                    k = min(self.top_k, router_logits.shape[1])
                    top_k_indices = torch.topk(router_logits, k=k, dim=-1).indices
                    expert_counts = torch.zeros(num_experts, device=top_k_indices.device)

                    for i in range(num_experts):
                        expert_counts[i] = (top_k_indices == i).sum().item()

                    sorted_experts = torch.argsort(expert_counts, descending=True)

                    print(f"Expert importance (top {min(10, num_experts)}): {expert_counts[sorted_experts[:min(10, num_experts)]]}")

                    if drop_count > 0:
                        print(f"Least important experts: {sorted_experts[-min(drop_count, len(sorted_experts)):]}")

                        default_drop = tuple(sorted_experts[-min(drop_count, len(sorted_experts)):].cpu().tolist())
                        combinations_to_try = [default_drop]

                        sampling_weights = 1.0 / (expert_counts + 1.0)

                        for _ in range(max_combinations - 1):
                            drop_candidates = torch.multinomial(
                                sampling_weights,
                                min(drop_count, num_experts - 1),
                                replacement=False
                            ).cpu().tolist()

                            drop_tuple = tuple(sorted(drop_candidates))
                            if drop_tuple not in combinations_to_try:
                                combinations_to_try.append(drop_tuple)

                        print(f"Evaluating {len(combinations_to_try)} combinations")
                    else:
                        print("No experts to drop, keeping all experts")
                        self.experts_to_drop = tuple()
                        self.experts_to_keep = list(range(num_experts))
                        return loss_history

            except Exception as e:
                print(f"Error in importance-based sampling: {e}")
                if drop_count > 0:
                    combinations_to_try = []
                    for _ in range(max_combinations):
                        dropped = tuple(sorted(random.sample(range(num_experts), min(drop_count, num_experts-1))))
                        if dropped not in combinations_to_try:
                            combinations_to_try.append(dropped)
                else:
                    self.experts_to_drop = tuple()
                    self.experts_to_keep = list(range(num_experts))
                    return loss_history
        else:
            if drop_count > 0:
                print("No cached activations available, using random sampling")
                combinations_to_try = []
                for _ in range(max_combinations):
                    dropped = tuple(sorted(random.sample(range(num_experts), min(drop_count, num_experts-1))))
                    if dropped not in combinations_to_try:
                        combinations_to_try.append(dropped)
            else:
                print("No experts to drop and no cached activations, keeping all experts")
                self.experts_to_drop = tuple()
                self.experts_to_keep = list(range(num_experts))
                return loss_history

        with torch.inference_mode():
            for dropped in tqdm(combinations_to_try, desc="Evaluating expert combinations"):
                self.experts_to_drop = dropped
                self.experts_to_keep = [i for i in range(num_experts) if i not in dropped]
                loss = 0

                if hasattr(self.cache_space, 'Xs') and hasattr(self.cache_space, 'Zs'):
                    for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                        hidden_states = hidden_states.to(
                            device=self.model.gate.weight.device, non_blocking=True)
                        final_hidden_states = final_hidden_states.to(
                            dtype=torch.float32, device=self.model.gate.weight.device, non_blocking=True)

                        try:
                            final_hidden_states_e = self.forward(hidden_states)
                            loss += torch.norm(final_hidden_states - final_hidden_states_e.to(torch.float32)).item()
                        except Exception as e:
                            print(f"Error with experts {dropped}: {e}")
                            loss = float('inf')
                            break

                    loss_history[dropped] = loss

                    if len(loss_history) % 5 == 0:
                        current_best = min(loss_history.values()) if loss_history else float('inf')
                        print(f"Evaluated {len(loss_history)}/{len(combinations_to_try)} combinations. Current best loss: {current_best}")

        if loss_history:
            best_drop_combo = min(loss_history, key=loss_history.get)
            best_loss = loss_history[best_drop_combo]
            self.experts_to_drop = best_drop_combo
            self.experts_to_keep = [i for i in range(num_experts) if i not in best_drop_combo]

            print(f"Best combination found - experts to DROP: {best_drop_combo}")
            print(f"These {len(best_drop_combo)} experts will be dropped, resulting in loss: {best_loss}")
            print(f"Keeping {len(self.experts_to_keep)} experts: {self.experts_to_keep}")
        else:
            print("All combinations failed, using default expert dropping")
            drop_count = max(0, num_experts - self.r)
            if drop_count > 0:
                self.experts_to_drop = tuple(range(num_experts - drop_count, num_experts))
                self.experts_to_keep = [i for i in range(num_experts - drop_count)]
            else:
                self.experts_to_drop = tuple()
                self.experts_to_keep = list(range(num_experts))
            print(f"Dropping experts: {self.experts_to_drop}")
            print(f"Keeping experts: {self.experts_to_keep}")

        return loss_history

    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.num_experts - self.r

        if self.experts_to_keep is None:
            self.experts_to_keep = [i for i in range(self.num_experts) if i not in self.experts_to_drop]

        print(f"Starting pruning: dropping {len(self.experts_to_drop)} experts, keeping {len(self.experts_to_keep)}")

        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        print(f"Original gate shape: {self.model.gate.weight.shape}")
        print(f"Gate weight type: {type(self.model.gate.weight)}")

        if isinstance(self.model.gate.weight, bnb.nn.Params4bit):
            print("Detected 4-bit quantization, dequantizing...")

            if hasattr(self.model.gate, 'quant_state') and self.model.gate.quant_state is not None:
                print("Using gate.quant_state for dequantization")
                gate_weights_fp32 = bnb.functional.dequantize_4bit(
                    self.model.gate.weight.data, self.model.gate.quant_state
                )
            elif hasattr(self.model.gate.weight, 'quant_state') and self.model.gate.weight.quant_state is not None:
                print("Using weight.quant_state for dequantization")
                gate_weights_fp32 = bnb.functional.dequantize_4bit(
                    self.model.gate.weight.data, self.model.gate.weight.quant_state
                )
            else:
                print("No quant_state found, creating output tensor manually")
                out_shape = (self.model.gate.out_features, self.model.gate.in_features)
                out_tensor = torch.empty(out_shape,
                                       dtype=torch.float32,
                                       device=self.model.gate.weight.data.device)

                if hasattr(self.model.gate.weight, 'absmax'):
                    print("Using weight.absmax for dequantization")
                    absmax = self.model.gate.weight.absmax
                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=None,
                        absmax=absmax,
                        out=out_tensor,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type
                    )
                elif hasattr(self.model.gate, 'absmax'):
                    print("Using gate.absmax for dequantization")
                    absmax = self.model.gate.absmax
                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=None,
                        absmax=absmax,
                        out=out_tensor,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type
                    )
                else:
                    dummy_absmax = torch.ones((self.model.gate.weight.data.shape[0], 1),
                                             device=self.model.gate.weight.data.device)

                    quant_state = QuantState(
                        absmax=dummy_absmax,
                        shape=out_shape,
                        dtype=torch.float32,
                        blocksize=self.model.gate.weight.blocksize,
                        quant_type=self.model.gate.weight.quant_type
                    )

                    gate_weights_fp32 = bnb.functional.dequantize_4bit(
                        self.model.gate.weight.data,
                        quant_state=quant_state
                    )
        else:
            print("No quantization detected, using weights as is")
            gate_weights_fp32 = self.model.gate.weight.data

        print(f"Dequantized gate shape: {gate_weights_fp32.shape}")

        gate_weights_pruned = gate_weights_fp32[self.experts_to_keep, :]

        print(f"Pruned gate shape: {gate_weights_pruned.shape}")

        if isinstance(self.model.gate.weight, bnb.nn.Params4bit):
            try:
                gate_weights_4bit, new_quant_state = bnb.functional.quantize_4bit(
                    gate_weights_pruned,
                    blocksize=self.model.gate.weight.blocksize,
                    compress_statistics=self.model.gate.weight.compress_statistics,
                    quant_type=self.model.gate.weight.quant_type,
                    quant_storage=self.model.gate.weight.quant_storage
                )
                print("Successfully re-quantized the weights")
            except Exception as e:
                print(f"Error during re-quantization: {e}")
                print("Falling back to using pruned weights directly")
                gate_weights_4bit = gate_weights_pruned
                new_quant_state = None
        else:
            gate_weights_4bit = gate_weights_pruned
            new_quant_state = None

        try:
            if isinstance(self.model.gate.weight, bnb.nn.Params4bit):
                gate_new = bnb.nn.Linear4bit(
                    input_features=self.model.gate.gating_dim,
                    output_features=self.r,
                    bias=False,
                    device="cpu",
                )

                for param in gate_new.parameters():
                    param.requires_grad = False

                if new_quant_state is not None:
                    gate_new.weight = bnb.nn.Params4bit(
                        gate_weights_4bit,
                        requires_grad=False,
                        quant_state=new_quant_state,
                        blocksize=self.model.gate.weight.blocksize,
                        compress_statistics=self.model.gate.weight.compress_statistics,
                        quant_type=self.model.gate.weight.quant_type,
                        quant_storage=self.model.gate.weight.quant_storage
                    )
                else:
                    gate_new.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)

                print(f"New pruned gate shape: {gate_new.weight.shape}")
            else:
                gate_new = torch.nn.Linear(
                    in_features=self.model.gate.gating_dim,
                    out_features=self.r,
                    bias=False,
                )
                gate_new.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)

                print(f"New pruned gate shape: {gate_new.weight.shape}")
        except Exception as e:
            print(f"Error creating new gate layer: {e}")
            print("Attempting to modify gate layer directly")
            self.model.gate.n_routed_experts = self.r
            self.model.gate.weight = torch.nn.Parameter(gate_weights_pruned, requires_grad=False)

        if 'gate_new' in locals():
            gate_new.top_k = self.model.gate.top_k
            gate_new.n_routed_experts = self.r
            gate_new.gating_dim = self.model.gate.gating_dim
            gate_new.scoring_func = getattr(self.model.gate, 'scoring_func', 'softmax')
            gate_new.norm_topk_prob = getattr(self.model.gate, 'norm_topk_prob', False)
            gate_new.aux_loss_alpha = getattr(self.model.gate, 'alpha', 0.001)
            gate_new.seq_aux = getattr(self.model.gate, 'seq_aux', True)
            gate_new.training = self.model.gate.training

            self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in self.experts_to_keep]
        )

        print(f"Successfully pruned the model from {self.num_experts} to {len(self.model.experts)} experts")

        return self.model
