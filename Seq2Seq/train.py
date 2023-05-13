import time
import math
import parser

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
from model import Seq2Seq, Encoder, Decoder


class TrainModel(object):

    def __init__(
        device,
        ignore_index,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float=0.001,
        weight_decay: float=0.0005,
        epochs: int=100,
        clip: float=1.,
    ):
        self.model = Seq2Seq(encoder, decoder, device).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.epochs = epochs

    def fit(self, train_loader, valid_loader):
        
        for epoch in range(self.epochs):
            start_time = time.time()

            train_loss = self.train(train_loader, clip)
            valid_loss = self.validate(valid_loader)

            end_time = time.time()

            print(f'Epoch: {epoch + 1:02} | Time: {end_time-start:.3f}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
            print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')

        return self.model

    def train(self, train_loader, clip=1):
        self.model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            src = batch.src
            trg = batch.trg

            self.optimizer.zero_grad()

            output = self.model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = self.loss_func(output, trg)
            
            loss.backward()
            self.optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            train_loss += loss.item()

        return train_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, valid_loader):
        model.eval()
        valid_loss = 0
        
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = self.model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            
            trg = trg[1:].view(-1)
            
            loss = self.loss_func(output, trg)
            vali_loss += loss.item()

        return valid_loss / len(valid_loader)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Seq2Seq', add_help=False)
    parser.add_argument('--min_freq', default=2, type=int,
                        help='minimum frequency in dataset')
    parser.add_argument('--embed_dim', default=256, type=int,
                        help='The embedding dimension of encoder and decoder networks')
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help='The dimension of hidden layer')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='The number of layers in LSTM')
    parser.add_argument('--drop', default=0.5, type=float,
                        help='drop out rate')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='set device for faster model training')
    parser.add_argument('--clip', default=1, type=float,
                        help='gradient clipping')
    return parser
        

def main(args):
    info = load_dataset(batch_size=args.batch_size, min_freq=args.min_freq, device=args.device)

    train_loader, valid_loader = info['train_loader'], info['train_loader']
    in_dim, out_dim, ignore_index = info['input_dim'], info['output_dim'], info['ignore_index']

    encoder = Encoder(
        in_dim=in_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        drop=args.drop,
    )
    decoder = Decoder(
        out_dim=in_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        drop=args.drop,
    )

    model = TrainModel(
        ignore_index=ignore_index,
        encoder=encoder,
        decoder=decoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        clip=args.clip,
    )

    history = model.fit(train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Seq2Seq training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)