# -*- coding: utf-8 -*-
import os
from glob import glob
from os.path import join
from datetime import datetime
from collections import Counter

import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import youtokentome as yttm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm.auto import tqdm
from copy import deepcopy
from einops import rearrange

from . import utils


DEFAULT_SPC_TOKENS = {
    '<LT_UNK>': 16384,
    '<RT_UNK>': 16385,
    '<LT_T2I>': 16386,
    '<LT_I2T>': 16387,
    '<LT_T2T>': 16388,
    '<RT_I2T>': 16389,
}


class ruDolphApi:

    spc_id = -1

    def __init__(self, model, tokenizer, vae, spc_tokens=None, quite=False, *, bs=24, q=0.5, txt_top_k=64,
                 img_top_k=768, txt_top_p=0.8, img_top_p=0.99, txt_temperature=0.9, img_temperature=1.0):
        self.spc_tokens = spc_tokens or deepcopy(DEFAULT_SPC_TOKENS)
        self.model = model
        self.tokenizer = tokenizer
        self.vae = vae
        ###
        self.vocab_size = self.model.get_param('vocab_size')
        self.l_text_seq_length = self.model.get_param('l_text_seq_length')
        self.r_text_seq_length = self.model.get_param('r_text_seq_length')
        self.image_seq_length = self.model.get_param('image_seq_length')
        self.image_tokens_per_dim = self.model.get_param('image_tokens_per_dim')
        self.text_special_tokens = self.model.get_param('text_special_tokens')
        self.image_special_tokens = self.model.get_param('image_special_tokens')
        self.total_seq_length = self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
        self.text_vocab_size = self.vocab_size - self.l_text_seq_length - self.text_special_tokens
        self.image_size = self.image_tokens_per_dim * 8
        self.device = self.model.get_param('device')
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        ###
        self.bs = bs
        self.q = q
        self.txt_top_k = txt_top_k
        self.txt_top_p = txt_top_p
        self.txt_temperature = txt_temperature
        self.img_top_p = img_top_p
        self.img_top_k = img_top_k
        self.img_temperature = img_temperature
        self.ignore_ids = [
            self.tokenizer.eos_id, self.tokenizer.bos_id, self.tokenizer.unk_id, self.tokenizer.pad_id,
            self.spc_id, *list(self.spc_tokens.values())
        ]
        self.quite = quite

    def dream(self, text, images_num, codebooks_num, image_prompts=None, template=None,
              top_k=768, top_p=0.99, temperature=1.0,
              bs=None, seed=None, use_cache=True, special_token='<LT_T2I>', ppl_txt_w=0.95):
        template = template or '{text}'
        codebooks = self.generate_codebooks(
            text=template.format(text=text), images_num=codebooks_num, top_k=top_k, top_p=top_p, bs=bs, seed=seed,
            temperature=temperature, use_cache=use_cache, special_token=special_token, image_prompts=image_prompts
        )
        ppl_txt, ppl_img, scores = self.self_reranking_by_text(text, codebooks, bs=bs, seed=seed, ppl_txt_w=ppl_txt_w)
        result = []
        with torch.no_grad():
            indexes = scores.argsort()[:images_num]
            images = self.vae.decode(codebooks[indexes])
            pil_images = utils.torch_tensors_to_pil_list(images)
            for idx in indexes:
                result.append({
                    'ppl_txt': ppl_txt[idx],
                    'ppl_img': ppl_img[idx],
                    'score': scores[idx],
                })
        return pil_images, result

    def image_captioning(self, pil_img, r_template='на картинке', early_stop=64, captions_num=5, seed=None, bs=None,
                         generations_num=48, top_k=None, top_p=None, temperature=None, ppl_txt_w=0.05,
                         l_special_token='<LT_T2I>', r_special_token='<RT_I2T>'):
        texts, counts = self.generate_captions(
            pil_img, r_template=r_template, early_stop=early_stop, captions_num=generations_num,
            bs=bs, top_k=top_k, top_p=top_p, temperature=temperature, seed=seed, r_special_token=r_special_token,
        )
        ppl_txt, ppl_img, scores = self.self_reranking_by_image(
            texts, pil_img,
            bs=bs, seed=seed, l_special_token=l_special_token, ppl_txt_w=ppl_txt_w
        )
        scores = scores / counts
        result = []
        for idx in scores.argsort()[:captions_num]:
            result.append({
                'text': texts[idx],
                'score': scores[idx],
                'count': counts[idx],
                'ppl_txt': ppl_txt[idx],
                'ppl_img': ppl_img[idx],
            })
        return result

    def generate_codebooks(self, text, images_num, image_prompts=None, top_k=None, top_p=None,
                           temperature=None, bs=None, seed=None, use_cache=True, special_token='<LT_T2I>'):
        torch.cuda.empty_cache()
        bs = bs or self.bs
        top_k = top_k or self.img_top_k
        top_p = top_p or self.img_top_p
        temperature = temperature or self.img_temperature
        self.seed_everything(seed)
        text = text.lower().strip()
        encoded = self.encode_text(text, text_seq_length=self.l_text_seq_length)
        encoded[torch.where(encoded == self.spc_id)] = self.spc_tokens[special_token]

        codebooks = []
        for chunk in more_itertools.chunked(range(images_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                out = encoded.unsqueeze(0).repeat(chunk_bs, 1).to(self.device)
                cache = None
                if image_prompts is not None:
                    prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                    prompts = prompts.repeat(chunk_bs, 1)
                iter_range = range(self.l_text_seq_length, self.l_text_seq_length + self.image_seq_length)
                if not self.quite:
                    iter_range = tqdm(iter_range)
                for idx in iter_range:
                    idx -= self.l_text_seq_length
                    if image_prompts is not None and idx in prompts_idx:
                        out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                    else:
                        logits, cache = self.model(out, attention_mask,
                                                   cache=cache, use_cache=use_cache, return_loss=False)
                        logits = logits[:, -1, self.vocab_size:]
                        if self.image_special_tokens:
                            logits = logits[:, :-self.image_special_tokens]
                        logits /= temperature
                        filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        out = torch.cat((out, sample), dim=-1)
                codebooks.append(out[:, -self.image_seq_length:])

        return torch.cat(codebooks)

    def self_reranking_by_text(self, text, codebooks, bs=None, seed=None, l_template='',
                               l_special_token='<LT_I2T>', r_special_token='<RT_I2T>', ppl_txt_w=0.95):

        torch.cuda.empty_cache()
        bs = bs or self.bs
        self.seed_everything(seed)
        text = text.lower().strip()
        r_encoded = self.encode_text(text, text_seq_length=self.r_text_seq_length)
        r_encoded[torch.where(r_encoded == self.spc_id)] = self.spc_tokens[r_special_token]

        l_encoded = self.encode_text(l_template, text_seq_length=self.l_text_seq_length)
        l_encoded[torch.where(l_encoded == self.spc_id)] = self.spc_tokens[l_special_token]

        ppl_txt, ppl_img = [], []
        for chunk in more_itertools.chunked(codebooks, bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                input_ids = torch.cat((
                    l_encoded.unsqueeze(0).repeat(chunk_bs, 1).to(self.device),
                    torch.stack(chunk),
                    r_encoded.unsqueeze(0).repeat(chunk_bs, 1).to(self.device),
                ), dim=1)

                logits, _ = self.model(input_ids, attention_mask, cache=None, use_cache=False, return_loss=False)
                logits = rearrange(logits, 'b n c -> b c n')
                image_logits = logits[:, self.vocab_size:,
                                      self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length - 1]
                if self.image_special_tokens:
                    image_logits = image_logits[:, :-self.image_special_tokens]
                image_logits = image_logits.contiguous().float()
                r_text_logits = logits[:, :self.vocab_size - self.l_text_seq_length,
                                       -self.r_text_seq_length:-1].contiguous().float()
                input_ids = input_ids.contiguous().long()

                ppl_img.append(
                    self.ce_to_ppl(F.cross_entropy(
                        image_logits,
                        input_ids[:, self.l_text_seq_length + 1:self.l_text_seq_length + self.image_seq_length],
                        reduction='none',
                    ))
                )
                ppl_txt.append(
                    self.ce_to_ppl(F.cross_entropy(
                        r_text_logits,
                        input_ids[:, -(self.r_text_seq_length - 1):],
                        ignore_index=0,
                        reduction='none',
                    ))
                )

        ppl_txt = torch.cat(ppl_txt)
        ppl_img = torch.cat(ppl_img)
        scores = (ppl_txt - ppl_txt.min()) / (ppl_txt.quantile(q=self.q) - ppl_txt.min()) * ppl_txt_w + \
                 (ppl_img - ppl_img.min()) / (ppl_img.quantile(q=self.q) - ppl_img.min()) * (1 - ppl_txt_w)

        return ppl_txt.cpu().numpy(), ppl_img.cpu().numpy(), scores.cpu().numpy()

    def zs_clf(self, pil_img, classes, bs=10, r_template=None, l_template=None, seed=None,
               r_special_token='<RT_I2T>', l_special_token='<LT_I2T>'):
        self.seed_everything(seed)
        r_template, l_template = r_template or '{}', l_template or ''
        r_encoded, l_encoded = [], []
        for _class in classes:
            r_text = r_template.format(_class).lower().strip()
            r_encoded.append(self.encode_text(r_text, text_seq_length=self.r_text_seq_length))
            l_text = l_template.format(_class).lower().strip()
            l_encoded.append(self.encode_text(l_text, text_seq_length=self.l_text_seq_length))
        r_encoded = torch.stack(r_encoded, 0)
        r_encoded[torch.where(r_encoded == self.spc_id)] = self.spc_tokens[r_special_token]
        l_encoded = torch.stack(l_encoded, 0)
        l_encoded[torch.where(l_encoded == self.spc_id)] = self.spc_tokens[l_special_token]

        with torch.no_grad():
            img = self.image_transform(pil_img)
            images = img.unsqueeze(0).to(self.device)
            image_input_ids = self.vae.get_codebook_indices(images, disable_gumbel_softmax=True)
            cache = None

            ppl_txt, ppl_img = [], []  # noqa
            for indexes in more_itertools.chunked(range(len(classes)), bs):
                chunk_r_encoded = r_encoded[indexes]
                chunk_l_encoded = l_encoded[indexes]
                chunk_bs = chunk_r_encoded.shape[0]
                attention_mask = self.get_attention_mask(chunk_bs)

                input_ids = torch.cat((
                    chunk_l_encoded.to(self.device),
                    image_input_ids.repeat(chunk_bs, 1),
                    chunk_r_encoded.to(self.device),
                ), dim=1)

                if cache is not None:
                    cache = list(map(list, cache.values()))
                    for i, e in enumerate(cache):
                        for j, c in enumerate(e):
                            t = cache[i][j]
                            t = t[..., :self.l_text_seq_length + self.image_seq_length, :]
                            cache[i][j] = t
                    cache = dict(zip(range(len(cache)), cache))

                logits, cache = self.model(input_ids, attention_mask, cache=cache, use_cache=True, return_loss=False)
                logits = rearrange(logits, 'b n c -> b c n')

                r_text_logits = logits[:, :self.vocab_size-self.l_text_seq_length,
                                       -self.r_text_seq_length:-1].contiguous()

                chunk_ppl_txt = self.ce_to_ppl(F.cross_entropy(
                    r_text_logits[:, :, :],
                    input_ids[:, -(self.r_text_seq_length - 1):],
                    ignore_index=0,
                    reduction='none',
                ))
                ppl_txt.append(chunk_ppl_txt)

            ppl_txt = torch.cat(ppl_txt)
            ppl_txt = ppl_txt / ppl_txt.norm(dim=0, keepdim=True)
            scores = ppl_txt.softmax(0)
            pred = scores.argmin().item()

        return {
            'label': pred,
            'class': classes[pred],
            'scores': scores.cpu().numpy(),
        }

    def generate_texts(
        self, template='',
        top_k=None, top_p=None, texts_num=48,
        early_stop=None,
        temperature=None, bs=None, seed=None, use_cache=True, special_token='<LT_T2T>',
        allowed_token_ids=None,
    ):
        torch.cuda.empty_cache()
        bs = bs or self.bs
        top_k = top_k or self.txt_top_k
        top_p = top_p or self.txt_top_p
        temperature = temperature or self.txt_temperature
        self.seed_everything(seed)

        early_stop = early_stop or self.l_text_seq_length

        template = template.lower().strip()
        template_encoded = self.encode_text(template, text_seq_length=self.l_text_seq_length)
        template_size = (template_encoded != 0).sum() - 1  # eos
        if not self.quite:
            print('--> template_size:', template_size.item())
        template_encoded = template_encoded[:template_size]
        template_encoded[torch.where(template_encoded == self.spc_id)] = self.spc_tokens[special_token]

        generated_tokens = []
        for chunk in more_itertools.chunked(range(texts_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                out = template_encoded.repeat(chunk_bs, 1).to(self.device)
                cache = None
                iter_range = range(template_size, min(early_stop, self.l_text_seq_length))
                if not self.quite:
                    iter_range = tqdm(iter_range)
                for _ in iter_range:
                    logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, :self.vocab_size]
                    if allowed_token_ids:
                        logits = logits[:, allowed_token_ids]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    if allowed_token_ids:
                        sample = torch.tensor(allowed_token_ids).to(self.device)[sample]
                    indexes = torch.where(sample > self.text_vocab_size)
                    sample[indexes] = self.tokenizer.eos_id
                    out = torch.cat((out, sample), dim=-1)

                generated_tokens.append(out[:, :self.l_text_seq_length])

        generated_tokens = torch.cat(generated_tokens)

        texts = Counter()
        for tokens in generated_tokens:
            end = torch.where(tokens == self.tokenizer.eos_id)[0]
            if end.shape[0]:
                end = min(end)
            else:
                end = tokens.shape[0]

            text = self.decode_text(tokens[:end]).strip()
            if text:
                texts.update([text])

        texts = list(texts.items())

        ppl_txt = []
        for chunk in more_itertools.chunked(texts, bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                chunk_encoded = []
                for text, _ in chunk:
                    text = text.lower().strip()
                    encoded = self.encode_text(text, text_seq_length=self.l_text_seq_length)
                    chunk_encoded.append(encoded)

                chunk_encoded = torch.stack(chunk_encoded)
                chunk_encoded[torch.where(chunk_encoded == self.spc_id)] = self.spc_tokens[special_token]

                attention_mask = self.get_attention_mask(chunk_bs)
                input_ids = chunk_encoded.to(self.device)

                logits, _ = self.model(input_ids, attention_mask, cache=None, use_cache=False, return_loss=False)
                logits = rearrange(logits, 'b n c -> b c n')

                l_text_logits = logits[:, :self.vocab_size - self.l_text_seq_length,
                                       :self.l_text_seq_length - 1].contiguous().float()
                input_ids = input_ids.contiguous().long()

                ppl_txt.append(
                    self.ce_to_ppl(F.cross_entropy(
                        l_text_logits,
                        input_ids[:, 1:self.l_text_seq_length],
                        ignore_index=0,
                        reduction='none',
                    ))
                )

        ppl_txt = torch.cat(ppl_txt)

        result = []
        for idx in ppl_txt.argsort():
            idx = idx.item()
            text, count = texts[idx]
            result.append({
                'text': text,
                'ppl_txt': round(ppl_txt[idx].item(), 2),
                'count': count,
            })

        return result

    def generate_captions(
        self, pil_img, early_stop=None, top_k=None, top_p=None, captions_num=48,
            temperature=None, bs=None, seed=None, use_cache=True,
            l_template='', r_template='', l_special_token='<LT_I2T>', r_special_token='<RT_I2T>',
    ):
        torch.cuda.empty_cache()
        bs = bs or self.bs
        top_k = top_k or self.txt_top_k
        top_p = top_p or self.txt_top_p
        temperature = temperature or self.txt_temperature
        self.seed_everything(seed)

        early_stop = early_stop or self.r_text_seq_length

        img = self.image_transform(pil_img)

        r_encoded = self.encode_text(r_template.lower().strip(), text_seq_length=self.r_text_seq_length)
        r_encoded[torch.where(r_encoded == self.spc_id)] = self.spc_tokens[r_special_token]
        template_size = (r_encoded != 0).sum() - 1  # eos
        if not self.quite:
            print('--> template_size:', template_size.item())
        r_encoded = r_encoded[:template_size]

        l_encoded = self.encode_text(l_template, text_seq_length=self.l_text_seq_length)
        l_encoded[torch.where(l_encoded == self.spc_id)] = self.spc_tokens[l_special_token]

        generated_tokens = []
        for chunk in more_itertools.chunked(range(captions_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                images = img.unsqueeze(0).repeat((chunk_bs, 1, 1, 1)).to(self.device)
                image_input_ids = self.vae.get_codebook_indices(images, disable_gumbel_softmax=True)

                out = torch.cat((
                    l_encoded.repeat(chunk_bs, 1).to(self.device),
                    image_input_ids,
                    r_encoded.repeat(chunk_bs, 1).to(self.device),
                ), dim=1)

                cache = None
                iter_range = range(
                    self.l_text_seq_length + self.image_seq_length + template_size,
                    min(self.l_text_seq_length + self.image_seq_length + early_stop, self.total_seq_length)
                )
                if not self.quite:
                    iter_range = tqdm(iter_range)
                for _ in iter_range:
                    logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, :self.vocab_size - self.l_text_seq_length]

                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    indexes = torch.where(sample >= self.text_vocab_size)
                    sample[indexes] = self.tokenizer.eos_id
                    out = torch.cat((out, sample), dim=-1)

                generated_tokens.append(out[:, self.l_text_seq_length + self.image_seq_length:])

        generated_tokens = torch.cat(generated_tokens)

        c = Counter()
        for tokens in generated_tokens:
            end = torch.where(tokens == self.tokenizer.eos_id)[0]
            if end.shape[0]:
                end = min(end)
            else:
                end = tokens.shape[0]
            text = self.decode_text(tokens[:end]).strip()
            if text:
                c.update([text])

        texts, counts = [], []
        for text, count in c.items():
            texts.append(text)
            counts.append(count)
        return texts, np.array(counts)

    def self_reranking_by_image(
            self,
            texts,
            pil_img,
            bs=8,
            seed=42,
            l_special_token='<LT_T2I>',
            ppl_txt_w=0.05,
    ):
        torch.cuda.empty_cache()
        bs = bs or self.bs
        self.seed_everything(seed)

        img = self.image_transform(pil_img)

        ppl_txt, ppl_img = [], []
        for chunk in more_itertools.chunked(texts, bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                chunk_encoded = []
                for text in chunk:
                    text = text.lower().strip()
                    encoded = self.encode_text(text, text_seq_length=self.l_text_seq_length)
                    encoded[torch.where(encoded == self.spc_id)] = self.spc_tokens[l_special_token]
                    chunk_encoded.append(encoded)

                chunk_encoded = torch.stack(chunk_encoded)
                attention_mask = self.get_attention_mask(chunk_bs)

                images = img.unsqueeze(0).repeat((chunk_bs, 1, 1, 1)).to(self.device)
                image_input_ids = self.vae.get_codebook_indices(images, disable_gumbel_softmax=True)

                input_ids = torch.cat((
                    chunk_encoded.to(self.device),
                    image_input_ids,
                ), dim=1)

                logits, _ = self.model(input_ids, attention_mask, cache=None, use_cache=False, return_loss=False)
                logits = rearrange(logits, 'b n c -> b c n')

                image_logits = logits[:, self.vocab_size:,
                                      self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length - 1]
                if self.image_special_tokens:
                    image_logits = image_logits[:, :-self.image_special_tokens]
                image_logits = image_logits.contiguous().float()

                l_text_logits = logits[:, :self.vocab_size - self.l_text_seq_length, :self.l_text_seq_length - 1]
                l_text_logits = l_text_logits.contiguous().float()

                input_ids = input_ids.contiguous().long()

                ppl_img.append(
                    self.ce_to_ppl(F.cross_entropy(
                        image_logits,
                        input_ids[:, self.l_text_seq_length + 1:self.l_text_seq_length + self.image_seq_length],
                        reduction='none',
                    ))
                )

                ppl_txt.append(
                    self.ce_to_ppl(F.cross_entropy(
                        l_text_logits,
                        input_ids[:, 1:self.l_text_seq_length],
                        ignore_index=0,
                        reduction='none',
                    ))
                )

        ppl_txt = torch.cat(ppl_txt)
        ppl_img = torch.cat(ppl_img)
        scores = (ppl_txt - ppl_txt.min()) / (ppl_txt.quantile(q=self.q) - ppl_txt.min()) * ppl_txt_w + \
                 (ppl_img - ppl_img.min()) / (ppl_img.quantile(q=self.q) - ppl_img.min()) * (1 - ppl_txt_w)

        return ppl_txt.cpu().numpy(), ppl_img.cpu().numpy(), scores.cpu().numpy()

    def generate_codebooks_cfg(self, text, images_num, template=None, image_prompts=None, top_k=None, top_p=None,
                               late_softmax=False, temperature=None, bs=None, seed=None, use_cache=True, weight_cfg=0.2,
                               special_token='<LT_UNK>'):
        """ by @neverix """
        torch.cuda.empty_cache()
        bs = bs or self.bs
        top_k = top_k or self.img_top_k
        top_p = top_p or self.img_top_p
        temperature = temperature or self.img_temperature
        self.seed_everything(seed)
        template = template or ''

        text = text.lower().strip()
        encoded = self.encode_text(text, text_seq_length=self.l_text_seq_length)
        encoded[torch.where(encoded == self.spc_id)] = self.spc_tokens['<LT_T2I>']

        encoded_CFG = self.encode_text(template.format(text=text), text_seq_length=self.l_text_seq_length)
        encoded_CFG[torch.where(encoded_CFG == self.spc_id)] = self.spc_tokens[special_token]

        encoded = torch.stack([encoded, encoded_CFG])
        weights = [1-weight_cfg, weight_cfg]
        bs //= len(weights)
        codebooks = []
        for chunk in more_itertools.chunked(range(images_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                out = encoded.repeat(chunk_bs, 1).to(self.device)
                caches = [None for _ in range(len(weights))]
                if image_prompts is not None:
                    prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                    prompts = prompts.repeat(chunk_bs, 1)
                iter_range = range(self.l_text_seq_length, self.l_text_seq_length + self.image_seq_length)
                if not self.quite:
                    iter_range = tqdm(iter_range)
                for idx in iter_range:
                    idx -= self.l_text_seq_length
                    if image_prompts is not None and idx in prompts_idx:
                        out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                    else:
                        true_probs = []
                        for d, weight in enumerate(weights):
                            logits, cache_ = self.model(out[d::len(weights)], attention_mask, use_cache=use_cache,
                                                        cache=caches[d], return_loss=False)
                            caches[d] = cache_
                            logits = logits[:, -1, self.vocab_size:]
                            if self.image_special_tokens:
                                logits = logits[:, :-self.image_special_tokens]
                            logits /= temperature
                            probs = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                            if not late_softmax:
                                probs = torch.nn.functional.softmax(probs, dim=-1)
                            true_probs.append(probs * weight)
                        probs = torch.stack(true_probs, dim=0)
                        probs = probs.sum(dim=0)
                        if late_softmax:
                            probs = torch.nn.functional.softmax(probs, dim=-1)

                        sample = torch.multinomial(probs, 1)
                        out = torch.stack(
                            [torch.cat((out[i::len(weights)], sample), dim=-1) for i in range(len(weights))]
                        )
                        out = out.transpose(0, 1).reshape(-1, *out.shape[2:])

                codebooks.append(
                    out[::len(weights), self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length:]
                )

        return torch.cat(codebooks)

    @staticmethod
    def show(pil_images, nrow=4, size=11, save_dir=None, show=True):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            count = len(glob(join(save_dir, 'img_*.png')))
            for i, pil_image in enumerate(pil_images):
                pil_image.save(join(save_dir, f'img_{count + i}.png'))

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
                img.save(join(save_dir, f'group_{count + i}.png'))
            if show:
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if show:
            fix.show()
            plt.show()

    def get_attention_mask(self, bs):
        return torch.tril(torch.ones((bs, 1, self.total_seq_length, self.total_seq_length), device=self.device))

    @staticmethod
    def seed_everything(seed):
        # TODO docstring
        if seed is not None:
            utils.seed_everything(seed)
        elif seed is not False:
            seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
            utils.seed_everything(seed)

    @staticmethod
    def ce_to_ppl(ce):
        indexes = torch.where(ce)
        ce[indexes] = torch.exp(ce[indexes])
        ppl = ce.sum(1) / torch.unique(indexes[0], return_counts=True)[1]
        return ppl

    def encode_text(self, text, text_seq_length):
        tokens = self.tokenizer.tokenizer.encode([text], output_type=yttm.OutputType.ID)[0]
        bos = [self.tokenizer.bos_id]
        if self.text_special_tokens:
            bos.append(self.spc_id)
        tokens = bos + tokens + [self.tokenizer.eos_id]
        return self.tokenizer.prepare_tokens(tokens, text_seq_length)

    def decode_text(self, encoded):
        return self.tokenizer.tokenizer.decode(encoded.cpu().numpy().tolist(), ignore_ids=self.ignore_ids)[0]
