import argparse
import os
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from data import CNNDMDataset, Collate, Vocab
from data_utils import get_input_from_batch, get_output_from_batch
from utils.variables import *
from utils.logging import logging
from models.model import Model
from models.utils import calc_running_avg_loss
from decode import BeamSearch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.vocab = Vocab(config.vocab_file, config.vocab_size)
        self.train_data = CNNDMDataset('train', config.data_path, config, self.vocab)
        self.validate_data = CNNDMDataset('val', config.data_path, config, self.vocab)
        # self.model = Model(config).to(device)
        # self.optimizer = None
        self.setup(config)

    def setup(self, config):
        
        model = Model(config)
        checkpoint = None
        if config.train_from != '':
            logging('Train from %s'%config.train_from)
            checkpoint = torch.load(config.train_from, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            self.step = checkpoint['step']
        
        self.model = model.to(device)
        self.optimizer = Adagrad(model.parameters(), lr = config.learning_rate, initial_accumulator_value = config.initial_acc)
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    def train_one(self, batch):

        config = self.config
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config, device)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, device)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(max_dec_len):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss

    def train(self):

        config = self.config
        train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True, collate_fn=Collate())

        running_avg_loss = 0
        self.model.train()

        for e in range(config.train_epoch):
            for batch in train_loader:
                self.step += 1
                self.optimizer.zero_grad()
                loss = self.train_one(batch)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                #print(loss.item())
                running_avg_loss = calc_running_avg_loss(loss.item(), running_avg_loss)

                if self.step % config.report_every == 0:
                    logging("Step %d Train loss %.3f"%(self.step, running_avg_loss))    
                if self.step % config.validate_every == 0:
                    self.validate()
                if self.step % config.save_every == 0:
                    self.save(self.step)
                if self.step % config.test_every == 0:
                    pass

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        validate_loader = DataLoader(self.validate_data, batch_size=self.config.batch_size, shuffle=False, collate_fn=Collate())
        losses = []
        for batch in validate_loader:
            loss = self.train_one(batch)
            losses.append(loss.item())
        self.model.train()
        ave_loss = sum(losses) / len(losses)
        logging('Validate loss : %f'%ave_loss)

    def save(self, step):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step
        }
        save_path = os.path.join(self.config.model_path, 'model_s%d.pt'%step)
        logging('Saving model step %d to %s...'%(step, save_path))
        torch.save(state, save_path)

def test(config, model=None, step = 0):
    
    if model is None and config.test_from != '':
        print('Testing model %s...'%config.test_from)
        saved_model = torch.load(config.test_from, map_location='cpu')
        model = Model(config, is_eval=True)
        model.load_state_dict(saved_model['model'])
        step = saved_model['step']

    predictor = BeamSearch(model, config, step)
    predictor.decode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='urara')
    sys.stderr = open('stderr.txt','w+')
    # data/ save
    parser.add_argument('-data_path', default=CNNDMPath, type=str)
    parser.add_argument('-glove_path', default=glovePath, type=str)
    parser.add_argument('-vocab_file', default=CNNDMPath + '/vocab_cnt.pkl', type=str)
    parser.add_argument('-model_path', default='../saved_models', type=str)
    parser.add_argument('-log_root', default='../results', type=str)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-test_from', default='', type=str)
    # model mode...
    parser.add_argument('-is_coverage', default=1, type=int)
    parser.add_argument('-pointer_gen', default=1, type=int)
    # Data preprocess
    parser.add_argument('-max_src_ntokens', default=400, type=int)
    parser.add_argument('-max_tgt_ntokens', default=100, type=int)
    parser.add_argument('-max_dec_steps', default=120, type=int)
    parser.add_argument('-min_dec_steps', default=35, type=int)
    parser.add_argument('-batch_size', default=30, type=int)
    # Hyper params
    parser.add_argument('-learning_rate', default=0.20, type=float)
    parser.add_argument('-cov_loss_wt', default=1.0, type=float)
    parser.add_argument('-initial_acc', default=0.1, type=float)
    parser.add_argument('-max_grad_norm', default=2.0, type=float)
    parser.add_argument('-vocab_size', default=50000, type=int)
    parser.add_argument('-emb_dim', default=128, type=int)
    parser.add_argument('-hidden_dim', default=256, type=int)
    parser.add_argument('-eps', default=1e-12, type=float)
    # Decode
    parser.add_argument('-beam_size', default=5, type=int)
    # Train params
    parser.add_argument('-validate_every', default=5000, type=int)
    parser.add_argument('-report_every', default=10, type=int)
    parser.add_argument('-save_every', default=5000, type=int)
    
    config_ = parser.parse_args()
    # trainer = Trainer(config_)
    # trainer.train()
    test(config_)

