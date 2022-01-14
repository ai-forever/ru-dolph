# -*- coding: utf-8 -*-
import torch


def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """
    Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


def get_attention_mask(masks, bs, l_text_tokens, image_tokens_per_dim, r_text_tokens, device):
    total_seq_length = l_text_tokens + image_tokens_per_dim*image_tokens_per_dim + r_text_tokens
    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))
    for i, index in enumerate(torch.argmin(masks, 1)):
        if index:
            attention_mask[i, 0, l_text_tokens + image_tokens_per_dim * image_tokens_per_dim + index:, :] = 0
    return attention_mask


def get_t2t_attention_mask(bs, l_text_tokens, image_tokens_per_dim, r_text_tokens, device):
    total_seq_length = l_text_tokens + image_tokens_per_dim*image_tokens_per_dim + r_text_tokens
    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))
    attention_mask[:, :, l_text_tokens:, :] = 0
    return attention_mask
