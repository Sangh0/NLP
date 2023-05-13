import spacy

import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


# tokenize Deutsch sentences and reverse the orders
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)][::-1]


# tokenize English sentences
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


# load dataset
def load_dataset(batch_size: int=128, min_freq: int=2, device):
    # load tokenizer of English and Deutsch in spacy packages
    spacy_en = spacy.blank('en')
    spacy_de = spacy.blank('de')

    # set the standard of tokenizer using Field
    # set the start point to <sos> and the end point to <eos>
    SRC = Field(tokenize = tokenize_de, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower=True)

    in_dim = len(SRC.vocab)
    out_dim = len(TRG.vocab)
    ignore_index = TRG.vocab.stoi[TRG.pad_token]
    
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

    # create vocab dictionary
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    
    batch_size = 128
    train_loader, valid_loader, test_loader = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device)

    return {
        'train_loader': train_loader,
        'vallid_loader': valid_loader,
        'test_loader': test_loader,
        'input_dim': in_dim,
        'output_dim': out_dim,
        'ignore_index': ignore_index,
    }