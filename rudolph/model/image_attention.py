# -*- coding: utf-8 -*-
import torch


def _init_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask=False):
    attn_size = l_text_tokens + image_tokens_per_dim**2 + r_text_tokens
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool if is_bool_mask else torch.float32))
    mask[-r_text_tokens:, :l_text_tokens] = 0
    return mask


def get_row_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask=False):
    mask = _init_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask)
    step = image_tokens_per_dim + 1
    image_end = l_text_tokens+image_tokens_per_dim**2
    for col in range(l_text_tokens, image_end):
        mask[col + step:image_end, col] = False if is_bool_mask else 0.0
    return mask


def get_col_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask=False):
    mask = _init_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask)
    step = image_tokens_per_dim - 1
    image_end = l_text_tokens+image_tokens_per_dim**2
    for col in range(l_text_tokens, image_end):
        for i in range(1, image_end, step+1):
            a = max(col + i, l_text_tokens)
            b = min(col + i + step, image_end)
            mask[a:b, col] = False if is_bool_mask else 0.0
    return mask


def get_conv_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, kernel=11, is_bool_mask=False):
    mask = _init_mask(l_text_tokens, image_tokens_per_dim, r_text_tokens, is_bool_mask)
    shift = kernel // 2
    image_end = l_text_tokens+image_tokens_per_dim**2
    for pos in range(l_text_tokens, image_end):
        mask[pos+1:image_end, pos] = False if is_bool_mask else 0.0
        img = torch.zeros(image_tokens_per_dim, image_tokens_per_dim)
        pixel_id = pos - l_text_tokens
        row = pixel_id // image_tokens_per_dim
        col = pixel_id % image_tokens_per_dim
        for r in range(-shift, shift+1):
            for c in range(-shift, shift+1):
                c_abs = max(min(c + col, image_tokens_per_dim - 1), 0)
                r_abs = max(min(r + row, image_tokens_per_dim - 1), 0)
                img[r_abs, c_abs] = 0.2
                cell_id = r_abs * image_tokens_per_dim + c_abs
                if l_text_tokens + cell_id > pos:
                    mask[l_text_tokens + cell_id, pos] = True if is_bool_mask else 1.0
        img[row, col] = 1.0
    return mask
