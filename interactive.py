from nltk import word_tokenize
from data import Example, Batch, Vocab
from decode import BeamSearch
from models.model import Model
import torch

def play(config):
    vocab = Vocab(config.vocab_file, config.vocab_size)
    saved_model = torch.load(config.test_from, map_location='cpu')
    model = Model(config, is_eval=True)
    model.load_state_dict(saved_model['model'])
    laser_beam = BeamSearch(model, config, 'intractive')
    while True:
        article = input('Input something to summarize:')
        data = {'article' : article, 'abstract' : 'for test'}
        entry = Example(config, vocab, data)
        normalized_input = Batch([entry] * config.beam_size)
        best_hyp = laser_beam.beam_search(normalized_input)
        decoded_abstract = laser_beam.get_summary(best_hyp)
        print('Summary decoded:')
        print(decoded_abstract)

    