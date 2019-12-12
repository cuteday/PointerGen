import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import init_linear_wt, init_lstm_wt

class ReduceState(nn.Module):
    """
        合并BiLSTM的双向hidden state
    """
    def __init__(self, config):
        super(ReduceState, self).__init__()

        self.config = config
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        init_linear_wt(self.reduce_c)
        init_linear_wt(self.reduce_h)

    def forward(self, hidden):
        
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim
        
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # attention
        self.config = config
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        """
            attn = W(tanh(src_feature + s.expand + coverage))
            src_feature 和 s 都过了线性变换...
        """
        
        config = self.config
           
        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2 * hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage