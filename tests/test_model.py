# -*- coding: utf-8 -*-
import torch


def test_forward(toy_model):
    bs = 4
    device = toy_model.get_param('device')
    l_text_seq_length = toy_model.get_param('l_text_seq_length')
    r_text_seq_length = toy_model.get_param('r_text_seq_length')
    image_tokens_per_dim = toy_model.get_param('image_tokens_per_dim')
    total_seq_length = l_text_seq_length + image_tokens_per_dim*image_tokens_per_dim + r_text_seq_length
    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))

    with torch.no_grad():
        l_text_input_ids = torch.tensor([
            [*range(1000, 1000 + l_text_seq_length - 11), 2, *[0]*10] for _ in range(bs)
        ]).long()
        #
        image_input_ids = torch.tensor([
            [*range(5000, 5000 + image_tokens_per_dim**2)] for _ in range(bs)
        ]).long()
        #
        r_text_input_ids = torch.tensor([
            [*range(2000, 2000 + r_text_seq_length - 11), 2, *[0]*10] for _ in range(bs)
        ]).long()
        #
        input_ids = torch.cat((l_text_input_ids, image_input_ids, r_text_input_ids), dim=1)
        loss, loss_values, hidden_states = toy_model.forward(input_ids, attention_mask, return_loss=True,
                                                             return_hidden_states=True)
        assert type(loss.data.detach().item()) == float
        assert type(loss_values) == dict
        assert len(hidden_states) == 24
