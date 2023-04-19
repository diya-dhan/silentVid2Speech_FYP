from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2D, LinearNorm


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.location_conv = Conv2D(2, 32, bias=False, stride=1,
                                      dilation=1)

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        return processed_attention


class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()
        self.layers =LinearNorm(256, 256, bias=False)

    def forward(self, x):
        x = F.dropout(F.relu(self.layers(x)), training=True)
        return x


class Postnet(nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv2D(88, 512,
                         5, stride=1,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 4):
            self.convolutions.append(
                nn.Sequential(
                    Conv2D(512,
                             512,
                             kernel_size=5, stride=1,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                Conv2D(512, 88,
                         kernel_size=5, stride=1,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(88))
            )

    def forward(self, x):
        for i in range(4):
            x = F.dropout(torch.tanh(self.convolutions[i](x)))
        x = F.dropout(self.convolutions[-1](x))

        return x



    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = 88
        self.n_frames_per_step = 1
        self.prenet_dim = 256

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_layer = Attention()

        self.lstm = nn.LSTM()

        self.postnet = Postnet()

    def forward(self, decoder_inputs):

        mel_outputs = []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output = self.attention_layer(
                decoder_input)
            mel_output = self.prenet(mel_output)
            mel_output = self.lstm(mel_output)
            mel_outputs += [mel_output.squeeze(1)]
            
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs_postnet