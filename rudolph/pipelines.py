# -*- coding: utf-8 -*-
import os
from glob import glob
from os.path import join
from datetime import datetime

import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm.auto import tqdm
from einops import rearrange

from . import utils
from .model.utils import get_i2t_attention_mask, get_t2t_attention_mask


def generate_codebooks(
        text,
        tokenizer,
        model,
        top_k, top_p, images_num,
        image_prompts=None,
        temperature=1.0, bs=8,
        seed=None, use_cache=True,
):
    # TODO docstring
    if seed is not None:
        utils.seed_everything(seed)
    else:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
        utils.seed_everything(seed)

    vocab_size = model.get_param('vocab_size')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    total_seq_length = l_text_seq_length + image_seq_length + r_text_seq_length
    device = model.get_param('device')

    text = text.lower().strip()
    encoded = tokenizer.encode_text(text, text_seq_length=r_text_seq_length)
    codebooks = []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = encoded.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            has_cache = False
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(l_text_seq_length, l_text_seq_length + image_seq_length)):
                idx -= l_text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    logits, has_cache = model(out, attention_mask,
                                              has_cache=has_cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
            codebooks.append(out[:, -image_seq_length:])

    return torch.cat(codebooks)


def generate_captions(
        pil_img,
        tokenizer,
        model,
        vae,
        template='',
        top_k=32, top_p=0.6, captions_num=128,
        temperature=1.0, bs=64,
        seed=None, use_cache=True,
):
    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
    utils.seed_everything(seed)

    vocab_size = model.get_param('vocab_size')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    device = model.get_param('device')

    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(image_tokens_per_dim * 8,
                            scale=(1., 1.),
                            ratio=(1., 1.)),
        T.ToTensor()
    ])
    img = image_transform(pil_img)

    template = template.lower().strip()
    template_encoded = tokenizer.encode_text(template, text_seq_length=r_text_seq_length)
    template_size = (template_encoded != 0).sum() - 1  # eos
    template_encoded = template_encoded[:template_size]

    generated_tokens = []
    for chunk in more_itertools.chunked(range(captions_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            masks = torch.ones(chunk_bs, r_text_seq_length, dtype=torch.int32)
            attention_mask = get_i2t_attention_mask(masks, chunk_bs, l_text_seq_length, image_tokens_per_dim,
                                                    r_text_seq_length, device)

            images = img.unsqueeze(0).repeat((chunk_bs, 1, 1, 1)).to(device)
            image_input_ids = vae.get_codebook_indices(images)

            out = torch.cat((
                torch.zeros((chunk_bs, l_text_seq_length), dtype=torch.int64).to(device),
                image_input_ids,
                template_encoded.repeat(chunk_bs, 1).to(device),
            ), dim=1)

            has_cache = False
            for _ in tqdm(range(
                    l_text_seq_length + image_seq_length + template_size,
                    l_text_seq_length + image_seq_length + r_text_seq_length
            )):
                logits, has_cache = model(out, attention_mask,
                                          has_cache=has_cache, use_cache=use_cache, return_loss=False)

                logits = logits[:, -1, :vocab_size]
                logits /= temperature
                filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                sample = torch.multinomial(probs, 1)
                indexes = torch.where(sample >= vocab_size - l_text_seq_length)
                sample[indexes] = 3
                out = torch.cat((out, sample), dim=-1)

            generated_tokens.append(out[:, -r_text_seq_length:])

    generated_tokens = torch.cat(generated_tokens)

    texts = set()
    for tokens in generated_tokens:
        end = torch.where(tokens == 3)[0].shape[0] or tokens.shape[0]
        text = tokenizer.decode_text(tokens[:end]).strip()
        if text:
            texts.add(text)

    return list(texts)


def show(pil_images, nrow=4, size=14, save_dir=None, show=True):
    """
    :param pil_images: list of images in PIL
    :param nrow: number of rows
    :param size: size of the images
    :param save_dir: dir for separately saving of images, example: save_dir='./pics'
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        count = len(glob(join(save_dir, 'img_*.png')))
        for i, pil_image in enumerate(pil_images):
            pil_image.save(join(save_dir, f'img_{count+i}.png'))

    pil_images = [pil_image.convert('RGB') for pil_image in pil_images]
    imgs = torchvision.utils.make_grid(utils.pil_list_to_torch_tensors(pil_images), nrow=nrow)
    if not isinstance(imgs, list):
        imgs = [imgs.cpu()]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        if save_dir is not None:
            count = len(glob(join(save_dir, 'group_*.png')))
            img.save(join(save_dir, f'group_{count+i}.png'))
        if show:
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if show:
        fix.show()
        plt.show()


def self_reranking_by_text(
        text,
        codebooks,
        tokenizer,
        model,
        bs=64,
):
    vocab_size = model.get_param('vocab_size')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    device = model.get_param('device')

    text = text.lower().strip()
    encoded = tokenizer.encode_text(text, text_seq_length=r_text_seq_length)
    mask = torch.zeros(r_text_seq_length, dtype=torch.int64)
    mask[encoded != 0] = 1
    ppl_text, ppl_image = [], []
    for chunk in more_itertools.chunked(codebooks, bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = get_i2t_attention_mask(
                mask.unsqueeze(0).repeat(chunk_bs, 1).to(device),
                chunk_bs, l_text_seq_length, image_tokens_per_dim, r_text_seq_length, device
            )
            input_ids = torch.cat((
                torch.zeros((chunk_bs, l_text_seq_length), dtype=torch.int64).to(device),
                torch.stack(chunk),
                encoded.unsqueeze(0).repeat(chunk_bs, 1).to(device),
            ), dim=1)

            logits, _ = model(input_ids, attention_mask, has_cache=False, use_cache=False, return_loss=False)
            logits = rearrange(logits, 'b n c -> b c n')
            image_logits = logits[:, vocab_size:,
                                  l_text_seq_length:l_text_seq_length + image_seq_length - 1].contiguous().float()
            r_text_logits = logits[:, :vocab_size, -r_text_seq_length:-1].contiguous().float()

            input_ids = input_ids.contiguous().long()

            ppl_image.append(
                ce_to_ppl(F.cross_entropy(
                    image_logits,
                    input_ids[:, l_text_seq_length + 1:l_text_seq_length + image_seq_length],
                    reduction='none',
                ))
            )
            ppl_text.append(
                ce_to_ppl(F.cross_entropy(
                    r_text_logits,
                    input_ids[:, -(r_text_seq_length - 1):],
                    ignore_index=0,
                    reduction='none',
                ))
            )
    return torch.cat(ppl_text), torch.cat(ppl_image)


def self_reranking_by_image(
        texts,
        pil_img,
        tokenizer,
        model,
        vae,
        bs=64,
        seed=42,
):
    if seed is not None:
        utils.seed_everything(seed)

    vocab_size = model.get_param('vocab_size')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    device = model.get_param('device')

    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(image_tokens_per_dim * 8,
                            scale=(1., 1.),
                            ratio=(1., 1.)),
        T.ToTensor()
    ])

    img = image_transform(pil_img)

    ppl_text, ppl_image = [], []
    for chunk in more_itertools.chunked(texts, bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            chunk_encoded, masks = [], []
            for text in chunk:
                text = text.lower().strip()
                encoded = tokenizer.encode_text(text, text_seq_length=r_text_seq_length)
                mask = torch.zeros(r_text_seq_length, dtype=torch.int64)
                mask[encoded != 0] = 1
                chunk_encoded.append(encoded)
                masks.append(mask)

            chunk_encoded = torch.stack(chunk_encoded)
            masks = torch.stack(masks)

            attention_mask = get_i2t_attention_mask(
                masks.to(device),
                chunk_bs, l_text_seq_length, image_tokens_per_dim, r_text_seq_length, device
            )

            images = img.unsqueeze(0).repeat((chunk_bs, 1, 1, 1)).to(device)
            image_input_ids = vae.get_codebook_indices(images)

            input_ids = torch.cat((
                chunk_encoded.to(device),
                image_input_ids,
                chunk_encoded.to(device),
            ), dim=1)

            logits, _ = model(input_ids, attention_mask, has_cache=False, use_cache=False, return_loss=False)
            logits = rearrange(logits, 'b n c -> b c n')

            image_logits = logits[:, vocab_size:,
                                  l_text_seq_length:l_text_seq_length + image_seq_length - 1].contiguous().float()
            l_text_logits = logits[:, :vocab_size, :l_text_seq_length - 1].contiguous().float()
            input_ids = input_ids.contiguous().long()

            ppl_image.append(
                ce_to_ppl(F.cross_entropy(
                    image_logits,
                    input_ids[:, l_text_seq_length + 1:l_text_seq_length + image_seq_length],
                    reduction='none',
                ))
            )

            ppl_text.append(
                ce_to_ppl(F.cross_entropy(
                    l_text_logits,
                    input_ids[:, 1:l_text_seq_length],
                    ignore_index=0,
                    reduction='none',
                ))
            )

    ppl_text = torch.cat(ppl_text)
    ppl_image = torch.cat(ppl_image)
    return ppl_text, ppl_image


def zs_clf(pil_img, classes, model, tokenizer, vae, template=''):
    """
    classes - list of strings
    template - prefix template
    """
    vocab_size = model.get_param('vocab_size')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    image_seq_length = model.get_param('image_seq_length')
    device = model.get_param('device')

    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(image_tokens_per_dim * 8,
                            scale=(1., 1.),
                            ratio=(1., 1.)),
        T.ToTensor()
    ])

    template = template.lower().strip()
    template_encoded = tokenizer.encode_text(template, text_seq_length=r_text_seq_length)
    template_size = (template_encoded != 0).sum() - 2  # bos, eos
    template_encoded = template_encoded[:template_size + 1]

    encoded, masks = [], []
    for _class in classes:
        _class = _class.lower().strip()
        class_encoded = tokenizer.encode_text(f' {_class}', text_seq_length=r_text_seq_length)
        if template_size:
            class_encoded = torch.cat([template_encoded, class_encoded[1:-template_size]])
        encoded.append(class_encoded)

        mask = torch.zeros(r_text_seq_length, dtype=torch.int64)
        mask[class_encoded != 0] = 1
        masks.append(mask)

    encoded = torch.stack(encoded, 0)
    masks = torch.stack(masks, 0)

    bs = len(classes)

    with torch.no_grad():
        attention_mask = get_i2t_attention_mask(masks, bs, l_text_seq_length, image_tokens_per_dim, r_text_seq_length,
                                                device)
        img = image_transform(pil_img)
        images = img.unsqueeze(0).to(device)
        image_input_ids = vae.get_codebook_indices(images)

        input_ids = torch.cat((
            torch.zeros(l_text_seq_length, dtype=torch.int64).repeat(bs, 1).to(device),
            image_input_ids.repeat(bs, 1),
            encoded.to(device),
        ), dim=1)

        logits, _ = model(input_ids, attention_mask, has_cache=False, use_cache=False, return_loss=False)
        logits = rearrange(logits, 'b n c -> b c n')

        image_logits = logits[:, vocab_size:,
                              l_text_seq_length:l_text_seq_length + image_seq_length - 1].contiguous()
        r_text_logits = logits[:, :vocab_size, -r_text_seq_length:-1].contiguous()

        ppl_text = ce_to_ppl(F.cross_entropy(
            r_text_logits[:, :, template_size:],
            input_ids[:, -(r_text_seq_length - 1 - template_size):],
            ignore_index=0,
            reduction='none',
        ))

        ppl_image = ce_to_ppl(F.cross_entropy(
            image_logits,
            input_ids[:, l_text_seq_length + 1:l_text_seq_length + image_seq_length],
            reduction='none',
        ))

        pred = ppl_text.argmin().item()

    return {
        'label': pred,
        'class': classes[pred],
        'ppl_text': ppl_text.cpu().numpy(),
        'ppl_image': ppl_image.cpu().numpy(),
    }


def generate_texts(
        tokenizer,
        model,
        template='',
        top_k=32, top_p=0.8, texts_num=128,
        temperature=1.0, bs=64,
        seed=None, use_cache=True,
):
    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
    utils.seed_everything(seed)

    vocab_size = model.get_param('vocab_size')
    image_tokens_per_dim = model.get_param('image_tokens_per_dim')
    l_text_seq_length = model.get_param('l_text_seq_length')
    r_text_seq_length = model.get_param('r_text_seq_length')
    device = model.get_param('device')

    template = template.lower().strip()
    template_encoded = tokenizer.encode_text(template, text_seq_length=l_text_seq_length)
    template_size = (template_encoded != 0).sum() - 1  # eos
    template_encoded = template_encoded[:template_size]

    generated_tokens = []
    for chunk in more_itertools.chunked(range(texts_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = get_t2t_attention_mask(chunk_bs, l_text_seq_length, image_tokens_per_dim,
                                                    r_text_seq_length, device)
            out = template_encoded.repeat(chunk_bs, 1).to(device)
            has_cache = False
            for _ in tqdm(range(template_size, l_text_seq_length)):
                logits, has_cache = model(out, attention_mask,
                                          has_cache=has_cache, use_cache=use_cache, return_loss=False)

                logits = logits[:, -1, :vocab_size]
                logits /= temperature
                filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                sample = torch.multinomial(probs, 1)
                indexes = torch.where(sample > vocab_size - l_text_seq_length)
                sample[indexes] = 3
                out = torch.cat((out, sample), dim=-1)
            generated_tokens.append(out[:, :l_text_seq_length])

    generated_tokens = torch.cat(generated_tokens)

    texts = set()
    for tokens in generated_tokens:
        end = torch.where(tokens == 3)[0].shape[0] or tokens.shape[0]
        text = tokenizer.decode_text(tokens[:end]).strip()
        if text:
            texts.add(text)
    texts = list(texts)

    ppl_text = []
    for chunk in more_itertools.chunked(texts, bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            chunk_encoded = []
            for text in chunk:
                text = text.lower().strip()
                encoded = tokenizer.encode_text(text, text_seq_length=l_text_seq_length)
                chunk_encoded.append(encoded)

            chunk_encoded = torch.stack(chunk_encoded)
            attention_mask = get_t2t_attention_mask(
                chunk_bs, l_text_seq_length, image_tokens_per_dim, r_text_seq_length, device
            )
            input_ids = chunk_encoded.to(device)

            logits, _ = model(input_ids, attention_mask, has_cache=False, use_cache=False, return_loss=False)
            logits = rearrange(logits, 'b n c -> b c n')

            l_text_logits = logits[:, :vocab_size, :l_text_seq_length - 1].contiguous().float()
            input_ids = input_ids.contiguous().long()

            ppl_text.append(
                ce_to_ppl(F.cross_entropy(
                    l_text_logits,
                    input_ids[:, 1:l_text_seq_length],
                    ignore_index=0,
                    reduction='none',
                ))
            )

    ppl_text = torch.cat(ppl_text)

    result = []
    for idx in ppl_text.argsort():
        idx = idx.item()
        result.append({
            'text': texts[idx],
            'ppl': round(ppl_text[idx].item(), 2),
        })

    return result


def ce_to_ppl(ce):
    indexes = torch.where(ce)
    ce[indexes] = torch.exp(ce[indexes])
    ppl = ce.sum(1) / torch.unique(indexes[0], return_counts=True)[1]
    return ppl
