import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self, 
        in_dim: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        drop: float=0.,
    ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(in_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=drop)
        self.dropout = nn.Dropout(drop)

    def forward(self, src):
        embedded = self.dropout(self.embed(src))
        out, (hidden, cell) =  self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    
    def __init__(
        self,
        out_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        drop: float=0.,
    ):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(out_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=drop)
        self.dropout = nn.Dropout(drop)

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(dim=0)
        embedded = self.dropout(self.embed(inputs))
        output = self.rnn(embedded, (hidden, cell))
        pred = self.fc(output.squeeze(dim=0))
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.init_weights()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        hidden, cell = self.encoder(src)

        # define tensors to append the outputs of decoder
        trg_len = trg.shape[0] # number of vocabs
        batch_size = trg.shape[1] # batch size
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # first inputs must be <sos> token
        inputs = trg[0, :]

        # forwarding to decoder network repeat the number of vocabs of targets
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)

            outputs[t] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            inputs = trg[t] if teacher_force else top1
        
        return outputs

    def init_weights(self):
        for m in self.modules():
            nn.init.uniform_(m.data, -0.08, 0.08)