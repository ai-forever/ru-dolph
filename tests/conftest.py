# -*- coding: utf-8 -*-
from os.path import abspath, dirname

import pytest

from rudolph.model.model import ruDolphModel

TEST_ROOT = dirname(abspath(__file__))


@pytest.fixture(scope='module')
def toy_model():
    model_params = dict(
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        embedding_dropout_prob=0.1,
        output_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        image_tokens_per_dim=16,
        l_text_seq_length=64,
        r_text_seq_length=64,
        kernel_size=7,
        last_kernel_size=9,
        cogview_sandwich_layernorm=True,
        cogview_pb_relax=True,
        vocab_size=16384+64,
        image_vocab_size=8192,
    )
    model = ruDolphModel(device='cpu', **model_params)
    yield model
