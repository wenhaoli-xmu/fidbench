# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
from ..modifier import Modifier

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class CompressionConfig(dict):
    def __init__(
        self,
        compress_method=None,
        attention_number=12,
        quantize_bit=0,
        group_num=0,
        group_size = 0,
        rank=0.0,
        rankv=0.0,
        prefill_rank = 0.0,
        prefill_rankv = 0.0,
        loop=0,
        top_k=0.0,
        left=0.0,
        stage=1,
        device_num=0,
        batch_num=1,
        start_saving=0,
        locality_saving=0,
        token_preserving=False,
        streaming=False,
        streaming_gap=0,
        stream_grouping = False,
        iter=0,
        # h2o setings
        heavy_size=0,
        recent_size=0,
    ):
        self.compress_method = compress_method
        self.quantize_bit = quantize_bit
        self.group_num = group_num
        self.group_size = group_size
        self.rank = rank
        self.rankv = rankv
        self.ranv = rankv
        self.prefill_rank = prefill_rank
        self.prefill_rankv = prefill_rankv
        self.loop = loop
        self.device_num = device_num
        self.attention_number = attention_number
        self.top_k = top_k
        self.left = left
        self.batch_num = batch_num
        self.stage = stage
        self.start_saving = start_saving
        self.locality_saving = locality_saving
        self.token_preserving = token_preserving
        self.iter = iter
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.streaming = streaming
        self.streaming_gap = streaming_gap
        self.stream_grouping = stream_grouping


    def create_attention_config(self, config):
        attention_config = []
        for i in range(self.attention_number):
            attention_config.append(config)
        return attention_config

    def copy_for_all_attention(self):
        self.compress_method = self.create_attention_config(self.compress_method)
        self.quantize_bit = self.create_attention_config(self.quantize_bit)
        self.group_num = self.create_attention_config(self.group_num)
        self.rank = self.create_attention_config(self.rank)
        self.prefill_rank = self.create_attention_config(self.prefill_rank)
        self.loop = self.create_attention_config(self.loop)
        self.top_k = self.create_attention_config(self.top_k)
        self.device_num = self.create_attention_config(self.device_num)
        self.left = self.create_attention_config(self.left)
        self.stage = self.create_attention_config(self.stage)
        self.rankv = self.create_attention_config(self.rankv)
        self.prefill_rankv = self.create_attention_config(self.prefill_rankv)
        self.start_saving = self.create_attention_config(self.start_saving)
        self.locality_saving = self.create_attention_config(self.locality_saving)
        self.token_preserving = self.create_attention_config(self.token_preserving)
        self.iter = self.create_attention_config(self.iter)
        self.heavy_size = self.create_attention_config(self.heavy_size)
        self.recent_size = self.create_attention_config(self.recent_size)
        self.streaming = self.create_attention_config(self.streaming)
        self.streaming_gap = self.create_attention_config(self.streaming_gap)
        self.group_size = self.create_attention_config(self.group_size)
        self.stream_grouping = self.create_attention_config(self.stream_grouping)

    def compress_ratio(
        self,
        compress_method,
        seqlen,
        model_dim,
        rank=0,
        rankv=0,
        quantize_bit=0,
        top_k=0,
        left=0.0,
        stage=1,
        batch_num=1,
    ):
        if compress_method == None:
            return 1.0
        elif compress_method == "Picache":
            if seqlen > rank and seqlen > rankv:
                return (
                    2
                    * seqlen
                    * batch_num
                    * model_dim
                    / (
                        ((model_dim + seqlen * batch_num) * (rank + rankv))
                        * quantize_bit
                        / 16
                    )
                )
            elif seqlen <= rank:
                return (
                    (
                        2
                        * seqlen
                        * batch_num
                        * model_dim
                        / (
                            (model_dim + seqlen * batch_num) * rankv
                            + seqlen * batch_num * model_dim
                        )
                    )
                    * 16
                    / quantize_bit
                )

            elif seqlen <= rankv:
                return (
                    (
                        2
                        * seqlen
                        * batch_num
                        * model_dim
                        / (
                            (model_dim + seqlen * batch_num) * rank
                            + seqlen * batch_num * model_dim
                        )
                    )
                    * 16
                    / quantize_bit
                )
        elif compress_method == "poweriteration":
            return (
                seqlen
                * batch_num
                * model_dim
                / ((model_dim + seqlen * batch_num) * rank)
            )
        elif compress_method == "stagept":
            return (
                seqlen
                * batch_num
                * model_dim
                / (model_dim * rank + seqlen * batch_num * (rank / stage))
            )
        elif (
            compress_method == "uniformquantization"
            or compress_method == "groupquantization"
            or compress_method == "sortquantization"
        ):
            return 16 / quantize_bit
        elif compress_method == "pruning":
            return 1 / top_k
        elif (
            compress_method == "densesparseuniformquantization"
            or compress_method == "densesparsesortquantization"
        ):
            return 1 / (quantize_bit / 16 + left)
        elif compress_method == "pt+outlier":
            return (
                seqlen
                * batch_num
                * model_dim
                * 16
                / quantize_bit
                / ((model_dim + seqlen * batch_num) * rank)
            )

    def calculate_compress_ratio_list(self, seqlen, model_dim):
        self.compress_ratio_list = []
        for i, compress_method in enumerate(self.compress_method):
            if compress_method == None:
                self.compress_ratio_list.append(
                    self.compress_ratio(compress_method, seqlen, model_dim)
                )
            elif compress_method == "Picache":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        rankv=self.rankv[i],
                        quantize_bit=self.quantize_bit[i],
                        batch_num=self.batch_num,
                        left=self.left[i],
                    )
                )
            elif compress_method == "poweriteration":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        batch_num=self.batch_num,
                    )
                )
            elif compress_method == "stagept":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        batch_num=self.batch_num,
                        stage=self.stage[i],
                    )
                )
            elif (
                compress_method == "uniformquantization"
                or compress_method == "groupquantization"
                or compress_method == "sortquantization"
            ):
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=0,
                        quantize_bit=self.quantize_bit[i],
                    )
                )
            elif compress_method == "pruning":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        top_k=self.top_k[i],
                    )
                )
            elif compress_method == "densesparseuniformquantization":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        quantize_bit=self.quantize_bit[i],
                        left=self.left[i],
                    )
                )
            elif compress_method == "densesparsesortquantization":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        quantize_bit=self.quantize_bit[i],
                        left=self.left[i],
                    )
                )
            elif compress_method == "pt+outlier":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        quantize_bit=self.quantize_bit[i],
                        batch_num=self.batch_num,
                        left=self.left[i],
                    )
                )

    def calculate_compress_ratio_total(self):
        return sum(self.compress_ratio_list) / len(self.compress_ratio_list)

    def __str__(self):
        return f"compress_method:{self.compress_method},\nquantize_bit:{self.quantize_bit},\nrank:{self.rank},\nloop:{self.loop},\ndevice_num:{self.device_num},\ncompressratio:{self.compress_ratio_list},\ncompressratio_total:{self.calculate_compress_ratio_total()}"


def fake_groupwise_token_asymmetric_quantization( ####
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_channel_asymmetric_quantization_new(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size

    fixed_input = input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
    
    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_poweriteration_group(input: torch.Tensor, loop, rank, device, p_base, q_base):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    dtype = input.dtype
    batch, dim1, dim2, dim3 = input.shape


    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch,dim1,dim3, rank).to(input.device)]
        q_base = [torch.rand(batch,dim1,dim2, rank).to(input.device)]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 2, 3) @ q_base[0]
    input = q_base[0] @ torch.transpose(p_base[0], 2, 3)
    input = input.view(batch, dim1, dim2, dim3)

    input = input.type(dtype)

    return input

def fake_groupwise_channel_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size
    fixed_length = int(group_num * group_size)
    fixed_input = input[:,:fixed_length,:]
    residual_input = input[:,fixed_length:,:]
    fixed_input = fixed_input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / cluster_num
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head * sep_dim)
    concat_input = torch.cat((dequantized_input,residual_input),dim=1)
    dequantized_input = concat_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_token_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / cluster_num
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input


def gearslkivi_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearlkivi_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)
    

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    
    return quantized_output



def gearslkivi_tokenQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1): ####
    input = input.float()
    cloned_input = input.clone()
    output = gears_tokenQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_channelQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1): ####
    input = input.float()
    cloned_input = input.clone()
    output = gears_channelQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearlkivi_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output

     
def gears_channelQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output
def gears_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output
def tokenwise_gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = cloned_input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,

                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr

def gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr
def gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr
def tokenwise_gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = cloned_input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
 
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr


def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    prefill=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape
    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    
    if compress_config.compress_method[layer_idx] == "KCVT":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )

    if compress_config.compress_method[layer_idx] == "KIVI_V2":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_value[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )

    if compress_config.compress_method[layer_idx] == "GEAR":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEAR-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEARL":

        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rank_used,
            compress_config.loop[layer_idx],

            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx],
 
        )
    if compress_config.compress_method[layer_idx] == "GEARL-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            rank_used,
            compress_config.loop[layer_idx],
            
            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            rankv_used,
            compress_config.loop[layer_idx],
            
        )

    return previous_key, previous_value


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError(
            "Make sure to implement `get_max_length` in a subclass."
        )

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __setitem__(
        self, layer_idx: int, key_value_states: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Support for backwards-compatible `past_key_value` assignment, e.g. `past_key_value[0] = (key_states,
        value_states)` to update the cache for the first layer.
        """
        key_states, value_states = key_value_states
        self.key_cache[layer_idx], self.value_cache[layer_idx] = (
            key_states,
            value_states,
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :,
                :,
                -self.window_length + self.num_sink_tokens + key_states.shape[-2] :,
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, cos[: self.window_length], sin[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(
                    keys_to_keep, rerotation_cos, rerotation_sin
                )
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat(
                [sink_keys, keys_to_keep, key_states], dim=-2
            )

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :,
                :,
                -self.window_length + self.num_sink_tokens + value_states.shape[-2] :,
            ]
            self.value_cache[layer_idx] = torch.cat(
                [sink_values, values_to_keep, value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape,
        dtype=dtype,
        device=device,
        past_key_values_length=past_key_values_length,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int] = None, compress_config=None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.prefill = True
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.compress_config = compress_config
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()
        if compress_config is not None:
            self.rank = compress_config.rank[self.layer_idx]
            self.rankv = compress_config.rankv[self.layer_idx]
            self.dveice_num = compress_config.device_num[self.layer_idx]
            if (
                compress_config.compress_method[self.layer_idx] == "poweriteration"
                or compress_config.compress_method[self.layer_idx] == "stagept"
                or compress_config.compress_method[self.layer_idx] == "pt+outlier"
                or compress_config.compress_method[self.layer_idx] == "Picache"
            ):
                # self.k_cache = PiCache((1, 12, 1023, 64), 100, 4, 0, 200)
                # self.v_cache = PiCache((1, 12, 1023, 64), 100, 4, 0, 200)
                # TODO 1023 change to inputsize
                if self.hidden_size > self.rank:
                    self.pbase1 = [
                        torch.rand(self.hidden_size, self.rank).to(self.dveice_num)
                    ]
                    # max input size is 1023
                    self.qbase1 = [
                        torch.rand(config.max_position_embeddings - 1, self.rank).to(
                            self.dveice_num
                        )
                    ]
                else:
                    self.pbase1, self.qbase1 = None, None
                if self.hidden_size > self.rankv:
                    self.pbase2 = [
                        torch.rand(self.hidden_size, self.rankv).to(self.dveice_num)
                    ]
                    self.qbase2 = [
                        torch.rand(config.max_position_embeddings - 1, self.rankv).to(
                            self.dveice_num
                        )
                    ]
                else:
                    self.pbase2, self.qbase2 = None, None

            else:
                self.pbase1, self.qbase1, self.pbase2, self.qbase2 = (
                    None,
                    None,
                    None,
                    None,
                )
                if compress_config.compress_method[self.layer_idx] == "H2O":
                    self.h2ocache = H2OCache(100)
        else:
            self.pbase1, self.qbase1, self.pbase2, self.qbase2 = (
                None,
                None,
                None,
                None,
            )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if (
            self.compress_config is not None
            and self.compress_config.compress_method[self.layer_idx] == "H2O"
        ):
            key_states, value_states, query_states = self.h2ocache.selection(
                attn_weights, key_states, value_states, query_states
            )
            kv_tuple = (key_states, value_states)
            past_key_value = past_key_value.__setitem__(self.layer_idx, kv_tuple)
        # print("query shape",query_states.shape)
        # print("key shape",key_states.shape)
        # print("value shape",value_states.shape)
        # torch.save(key_states,"key_states" +str(self.layer_idx)+".pt")
        # torch.save(value_states,"value_states" +str(self.layer_idx)+".pt")

        if (
            self.compress_config is not None
            and self.compress_config.compress_method[self.layer_idx] == "H2O"
        ):
            if q_seq_len > 1:
                batch, q_seq_len, sep_dim = query_states.shape
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
                print("attn weights shape", attn_weights.shape)
                print("head dim", self.head_dim)
                attn_weights = attn_weights / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len > 1:
            self.prefill = True
        if past_key_value is not None:
            # TODO : add compress_config and compress functions
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            if self.compress_config is not None:
                if self.compress_config.streaming[self.layer_idx] is True:
                    if len(past_key_value) >= self.layer_idx + 1:
                        past_key, past_value = past_key_value[self.layer_idx]
                    else:
                        past_key, past_value = key_states, value_states
                    bsz, num_heads, seq_len, head_dim = past_key.shape
                    if (
                        self.prefill is True
                        or seq_len % self.compress_config.streaming_gap[self.layer_idx]
                        == 0
                    ):
                        if self.compress_config.stream_grouping[self.layer_idx] == True:
                            bsz, num_heads, seq_len, head_dim = past_key.shape
                            if self.prefill is True:
                                residual_length = seq_len % self.compress_config.streaming_gap[
                                    self.layer_idx
                                ]
                                past_key_compress = past_key[:,:,0:seq_len - residual_length:,:]
                                past_value_compress = past_value[:,:,0:seq_len - residual_length:,:]
                                if residual_length == 0:
                                    past_key_full = None
                                    past_value_full = None
                                else:
                                    past_key_full = past_key[:,:, -residual_length:,:]
                                    past_value_full = past_value[:,:, -residual_length:,:]
                            else:
                                residual_length = self.compress_config.streaming_gap[self.layer_idx]
                                past_key_compress = past_key[:,:, -residual_length:,:]
                                past_value_compress = past_value[:,:, -residual_length:,:]
                                past_key_full = past_key[:,:, 0:-residual_length:,:]
                                past_value_full = past_value[:,:, 0:-residual_length:,:]
                            (past_key_compress, past_value_compress) = compress_insert_function(
                                past_key_compress,
                                past_value_compress,
                                self.compress_config,
                                self.layer_idx,
                                pbase1=self.pbase1,
                                qbase1=self.qbase1,
                                pbase2=self.pbase2,
                                qbase2=self.qbase2,
                                prefill = self.prefill,
                            )
                            if past_key_full is not None:
                                if self.prefill is True:
                                    past_key = torch.cat([past_key_compress, past_key_full], dim=2)
                                    past_value = torch.cat([past_value_compress, past_value_full], dim=2)
                                else:
                                    past_key = torch.cat([past_key_full, past_key_compress], dim=2)
                                    past_value = torch.cat([past_value_full, past_value_compress], dim=2)
                        else:
                            if self.compress_config.compress_method[self.layer_idx] == "KIVI":
                                bsz, num_heads, seq_len, head_dim = past_key.shape
                                fixed_length = seq_len // self.compress_config.group_size[
                                    self.layer_idx
                                ] * self.compress_config.group_size[self.layer_idx]
                                residual_key = past_key[:, :, fixed_length:, :]
                                residual_value = past_value[:, :, fixed_length:, :]
                                past_key, past_value = past_key[:,:,:fixed_length,:], past_value[:,:,:fixed_length,:]
                            # not streaming compress is compress every geneartion
                            (
                                past_key,
                                past_value,
                            ) = compress_insert_function(
                                past_key,
                                past_value,
                                self.compress_config,
                                self.layer_idx,
                                pbase1=self.pbase1,
                                qbase1=self.qbase1,
                                pbase2=self.pbase2,
                                qbase2=self.qbase2,
                            )
                            if self.compress_config.compress_method[self.layer_idx] == "KIVI":
                                past_key = torch.cat([past_key, residual_key], dim=2)
                                past_value = torch.cat([past_value, residual_value], dim=2)
                        if self.prefill is False:
                            past_key_value.__setitem__(
                                self.layer_idx, (past_key, past_value)
                            )
                        self.prefill = False

            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
 
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, compress_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        # here use sdpa attn
        config._attn_implementation = "sdpa"
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, compress_config=compress_config
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, compress_config=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx, compress_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SimulatedGearLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, compress_config=None):
        super().__init__(config)
        self.model = LlamaModel(config, compress_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class Gear(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)

        compress_config = CompressionConfig(
            compress_method="GEAR",
            rank=self.conf['rank'],
            rankv=self.conf['rankv'],
            prefill_rank = self.conf['prefillrank'],
            prefill_rankv = self.conf['prefillrankv'],
            
            loop=self.conf['loop'],
            quantize_bit=self.conf['quantize_bit'],
            group_num=self.conf['group_num'],
            group_size = self.conf['group_size'],
            top_k=self.conf['top_kprun'],
            left=self.conf['left'],
            attention_number=self.conf['attention_number'],
            device_num=self.conf['gpu'],
            batch_num=1,

            streaming=self.conf['streaming'],
            streaming_gap=self.conf['streaming_gap'],
            stream_grouping=self.conf['stream_grouping'])
        
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)

        model_name_or_path = model.config._name_or_path
        del model

        model = SimulatedGearLlamaForCausalLM.from_pretrained(
            model_name_or_path, 
            compress_config,
            device_map='auto')
        
        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []

    
    def reset(self):
        pass
    

    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        output = self.model(input_ids=p_ids)
        kv_cache = output.past_key_values

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        for tok, label in zip(
                torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                torch.chunk(g_ids[:, 1:], turns, dim=-1)):

            output = self.model(input_ids=tok, past_key_values=kv_cache)
            logits, kv_cache = output.logits, output.past_key_values

            label = label.ravel().item()
            next_1 = logits.argmax(dim=-1).ravel().item()
            next_5 = logits.topk(k=5, dim=-1).indices.ravel().tolist()

            acc1 += next_1 == label
            acc5 += label in next_5

        acc1 /= turns
        acc5 /= turns

        return acc1, acc5


    @torch.no_grad()
    def compute_ppl(self, input_ids):
        assert input_ids.shape[0] == 1, 'only support batch size 1'
        assert input_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        output = self.model(input_ids=input_ids)
        logits = output.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

        ppl = torch.exp(loss).item()

        return ppl