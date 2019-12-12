import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.utils import *
from models.neural import Attention, ReduceState
 
class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size + 5, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        init_lstm_wt(self.lstm)
        init_wt_normal(self.embedding.weight)   # 采用正态初始化embedding 不使用预训练

    #seq_lens should be in descending order
    def forward(self, inputs, seq_lens):
        """
            encoder outputs: LSTM hidden state
            encoder feature: linear transformed hidden 
        """
        embedded = self.embedding(inputs)
 
        # packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        # output, hidden = self.lstm(packed)
        # # output, (h_n, c_n)

        # encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        # encoder_outputs = encoder_outputs.contiguous()

        encoder_outputs, hidden = lstm_encoder(embedded, self.lstm, seq_lens, True)
        
        encoder_feature = encoder_outputs.view(-1, 2 * self.config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.attention_network = Attention(config)
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

        init_wt_normal(self.embedding.weight)  # embedding 使用正态初始化...
        init_lstm_wt(self.lstm)
        #init_linear_wt(self.out1)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        """
            y_t: token idx
            s_t: prev decoder state: [h_t; c_t]->reduce 
            c_t: prev context vector
            enc_feature: encoder hidden transformed
            enc_outputs: encoder hidden     
        """
        config = self.config
        if not self.training and step == 0:     # 测试时用于predict hypothesis
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)  # 输入的idx
        # 将s_t, c_t, y_emb 变换为emb大小的输入
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))   # coverage vec concats 
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        # 此context vector 是根据当前decoder state计算的
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist    # 分布在src vocab上
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(nn.Module):
    def __init__(self, config, is_eval=False):
        super(Model, self).__init__()
        encoder = LSTMEncoder(config)
        decoder = Decoder(config)
        reduce_state = ReduceState(config)

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        # if model_file_path is not None:
        #     state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        #     self.load_state_dict(state['model'])
            # self.encoder.load_state_dict(state['encoder_state_dict'])
            # self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            # self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self):
        pass
