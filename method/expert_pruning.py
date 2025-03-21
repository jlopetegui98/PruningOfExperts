from tqdm import tqdm
from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

from model import PrunableMixtralSparseMoeBlockWrapper



def progressive_pruning(model, calib_loader, r=6):
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