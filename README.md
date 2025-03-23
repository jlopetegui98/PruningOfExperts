# PruningOfExperts
This repository contains the implementation for the final project of the LLMs class at MVA.

The code of this repo is mainly based on the following repository: [Expert_Sparsity](https://github.com/Lucky-Lance/Expert_Sparsity/tree/main) which contains the code from the paper [Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2402.14800).

**Implementation details:**

The main goal of our work was to extend the previous implementation for the efficient expert pruning from the paper mentioned above under severe memory constraints. Particularly, we modified the base implementation to allow the pruning over a 4-bit quantized version of the Mixtral8x7B model (model used in the original work) and the DeepSeek MoE 16B Base model. There are significant changes in the architecture and representation of the model after 4-bit quantization, including changes in the shape of layers' weights, that had to be considered in the current implementation.

**Pruned models:**

We used as calibration data the c4 dataset, one of the two considered in the paper mentioned before. We got two pruned versions of the [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model, the first one with 6 experts per layer and a second one with 4, as well as one pruned version of the [deepseek-moe-16b-base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base) model (keeping 16 out of 64 experts per layer):

- [Mixtral8x7B-4bit-pruned-1](https://huggingface.co/JavierLopetegui/Mixtral8x7B-4bit-pruned): 6 experts - ~18GB in V-RAM
- [Mixtral8x7B-4bit-pruned-2](https://huggingface.co/JavierLopetegui/Mixtral8x7B-4bit-pruned_4_experts): 4 experts - ~12GB in V-RAM 
- [deepseek-moe-16b-pruned](https://huggingface.co/olijacklu/deepseek-moe-16b-pruned): 16 experts - ~4GB in V-RAM 
