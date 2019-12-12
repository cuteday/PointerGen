import torch
import torch.nn as nn

rand_unif_init_mag=0.02
trunc_norm_init_std = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_lstm_wt(lstm):
    """
        LSTM weight 使用一致分布初始化
        Bias 填0
    """
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    """
        Linear 全部正态初始化
    """
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)

def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss

#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=True):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.tensor(order).to(device)
    sorted_ = sequence_emb.index_select(batch_dim, order)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.tensor(order).to(device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states

def lstm_encoder(sequence, lstm ,seq_lens=None, batch_first = True):
    """ 
        functional LSTM encoder (sequence is [b, t]/[b, t, d], lstm should be rolled lstm)
        简单的LSTM包装 自动长度降序重排, Pad&Pack
    """
    batch_size = sequence.size(0)
    if not batch_first:
        sequence = sequence.transpose(0, 1)

    if seq_lens:
        # print(batch_size)
        # print(seq_lens)
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        try:
            sequence = reorder_sequence(sequence, sort_ind, batch_first)
        except:
            print(sequence.size())
            print(sort_ind)
            assert False

    if seq_lens:
        packed_seq = nn.utils.rnn.pack_padded_sequence(sequence, seq_lens, batch_first = batch_first)
        packed_out, final_states = lstm(packed_seq)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = batch_first)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        try:
            lstm_out = reorder_sequence(lstm_out, reorder_ind, batch_first)
        except:
            print(lstm_out.size())
            print(reorder_ind)
            assert False
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(sequence)

    return lstm_out, final_states