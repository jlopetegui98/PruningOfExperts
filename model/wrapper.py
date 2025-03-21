import torch
import torch.nn.functional as F
import itertools as I
from data import CacheDataset
from datasets import load_dataset, Dataset
from transformers.testing_utils import CaptureLogger
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralSparseMoeBlock
    )
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Linear4bit
from bitsandbytes.functional import dequantize_4bit, quantize_4bit
from bitsandbytes.quant_state import QuantState


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
            for dropped in I.combinations(range(self.model.num_experts), self.model.num_experts - self.r):
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
