import re, os
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn

from models.utils import reorder_lstm_states, reorder_sequence

PAD = 0
UNK = 1
START = 2
END = 3

special_tokens  = ['<pad>', '<unk>', '<start>', '<end>']

def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    for i, t in enumerate(special_tokens):
        word2id[t] = i
        id2word[i] = t
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
        id2word[i] = w
    return word2id, id2word

def make_embedding(emb_file, vocab, emb_size=300):
    # make embedding from GloVe file
    emb_matrix = torch.tensor((vocab.size(), emb_size), dtype=torch.float)
    got_emb = [False] * vocab.size()
    with open(emb_file, 'r') as emb:
        for entry in emb:
            entry = entry.split()
            word = entry[0]
            idx = vocab.word2id(word)
            if idx!=UNK:
                emb = torch.tensor([float(v) for v in entry[1:]])
                emb_matrix[idx] = emb
                got_emb[idx] = True
    for i in range(vocab.size()):
        if got_emb[i] == False:
            emb_matrix[i] = torch.randn(emb_size)
    return emb_matrix
    

def article2ids(words, vocab):
    ids = []
    oovs = []
    for w in words:
        i = vocab.word2id(w)
        if i == UNK:
            if w not in oovs:
                oovs.append(w)
            ids.append(vocab.size() + oovs.index(w))
        else: ids.append(i)

    return ids, oovs

def abstract2ids(words, vocab, article_oovs):
    ids = []
    for w in words:
        i = vocab.word2id(w)
        if i == UNK:
            if w in article_oovs:
                ids.append(vocab.size() + article_oovs.index(w))
            else: ids.append(UNK)
        else: ids.append(i)
    return ids

def output2words(ids, vocab, art_oovs):
    words = []
    for i in ids:
        w = vocab.id2word(i) if i < vocab.size() else art_oovs[i - vocab.size()]
        words.append(w)
    return words

def show_art_oovs(article, vocab):
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==UNK else w for w in words]
    out_str = ' '.join(words)
    return out_str

def pad_sequence(data, padding_idx=0, length = 0):
    """
        Padder 
        输入：list状的 参差不齐的东东
        输出：list状的 整齐的矩阵
    """
    if length==0: length = max(len(entry) for entry in data)
    padded = [d + [padding_idx]*(length - len(d)) for d in data]
    return padded

def get_input_from_batch(batch, config, device):
    batch_size = batch.enc_inp.size(0)

    enc_batch = Variable(batch.enc_inp.long())
    enc_pad_mask = Variable(batch.enc_pad_mask).float()
    c_t_1 = Variable(torch.zeros(batch_size, 2 * config.hidden_dim))
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None
    coverage = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(batch.art_batch_extend_vocab.long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros(batch_size, batch.max_art_oovs))
    
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    # move tensors to device
    enc_batch = enc_batch.to(device)
    enc_pad_mask = enc_pad_mask.to(device)
    c_t_1 = c_t_1.to(device)
    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(device)
    if coverage is not None:
        coverage = coverage.to(device)

    return enc_batch, enc_pad_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, device):
    dec_batch = Variable(batch.dec_inp.long())
    dec_pad_mask = Variable(batch.dec_pad_mask).float()
    dec_lens = batch.dec_lens
    dec_lens_var = Variable(torch.tensor(dec_lens)).float()
    # 这个东东是用来规范化batch loss用的
    # 每一句的总loss除以它的词数

    max_dec_len = max(dec_lens)
    tgt_batch = Variable(batch.dec_tgt).long() 

    dec_batch = dec_batch.to(device)
    dec_pad_mask = dec_pad_mask.to(device)
    tgt_batch = tgt_batch.to(device)
    dec_lens_var = dec_lens_var.to(device)

    return dec_batch, dec_pad_mask, max_dec_len, dec_lens_var, tgt_batch

