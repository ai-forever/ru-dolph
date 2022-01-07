# -*- coding: utf-8 -*-
import os

import torch
from huggingface_hub import hf_hub_url, cached_download

from .model import ruDolphModel
from .fp16 import FP16Module


__all__ = ['ruDolphModel', 'FP16Module', 'get_rudolph_model']


MODELS = {
    '350M': dict(
        description='Russian Diffusion On Language Picture Hyper-modality (RuDOLPH 🦌🎄☃️) 350M is a fast and light '
                    'text-image-text transformer (350M GPT-3) designed for a quick and easy fine-tuning setup '
                    'for the solution of various tasks: from generating images by text description and '
                    'image classification to visual question answering and more. \n'
                    'This model demonstrates the power of Hyper-modality Transformers.',
        model_params=dict(
            num_layers=24,
            hidden_size=1024,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            l_text_seq_length=64,
            image_tokens_per_dim=16,
            r_text_seq_length=64,
            kernel_size=7,
            last_kernel_size=9,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 64,
            image_vocab_size=8192,
        ),
        repo_id='sberbank-ai/RuDOLPH-350M',
        filename='pytorch_model.bin',
        authors='SberAI, SberDevices',
        full_description='',  # TODO
    ),
}


def get_rudolph_model(name, pretrained=True, fp16=False, device='cpu', cache_dir='/tmp/rudolph', **model_kwargs):
    # TODO docstring
    assert name in MODELS

    if fp16 and device == 'cpu':
        print('Warning! Using both fp16 and cpu doesnt support. You can use cuda device or turn off fp16.')

    config = MODELS[name].copy()
    config['model_params'].update(model_kwargs)
    model = ruDolphModel(device=device, **config['model_params'])
    if pretrained:
        cache_dir = os.path.join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
        checkpoint = torch.load(os.path.join(cache_dir, config['filename']), map_location='cpu')
        model.load_state_dict(checkpoint)
    if fp16:
        model = FP16Module(model)
    model.eval()
    model = model.to(device)
    if config['description'] and pretrained:
        print(config['description'])
    return model
