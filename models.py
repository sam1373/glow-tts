import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons
import attentions
import monotonic_align

import numpy as np

from extract_durs import DurationExtractor

from revnet_block import irevnet_block

#import matplotlib.pyplot as plt

class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = attentions.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = attentions.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class Encoder(nn.Module):
  def __init__(self, in_channels, filter_channels, out_channels, kernel_size, p_dropout):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.out_channels = out_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = nn.BatchNorm1d(filter_channels, affine=True)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = nn.BatchNorm1d(filter_channels, affine=True)
    self.conv_3 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_3 = nn.BatchNorm1d(filter_channels, affine=True)
    self.conv_4 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_4 = nn.BatchNorm1d(filter_channels, affine=True)
    self.conv_5 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_5 = nn.BatchNorm1d(filter_channels, affine=True)
    self.proj = nn.Conv1d(filter_channels, out_channels, 1)
    #self.final_norm = nn.BatchNorm1d(out_channels, affine=False)

  def forward(self, x, x_mask):
    x0 = x * x_mask
    x = self.conv_1(x0)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.conv_3(x * x_mask)
    x = torch.relu(x)
    x = self.norm_3(x)
    x = self.drop(x)
    x = self.conv_4(x * x_mask)
    x = torch.relu(x)
    x = self.norm_4(x)
    x = self.drop(x)
    x = self.conv_5(x * x_mask)
    x = torch.relu(x)
    x = self.norm_5(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    x = x0 + x
    return x * x_mask


class TextPreEncoder(nn.Module):
  def __init__(self, 
      n_vocab, 
      out_channels, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      n_heads, 
      n_layers, 
      kernel_size, 
      p_dropout, 
      window_size=None,
      block_length=None,
      mean_only=False,
      prenet=False,
      gin_channels=0):

    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.prenet = prenet
    self.gin_channels = gin_channels

    self.proj_w = DurationPredictor(out_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)
  
  def forward(self, x, x_lengths, g=None):
    x_oh = torch.zeros(x.shape[0], self.out_channels, x.shape[1]).cuda()
    x_oh.scatter_(1, x.unsqueeze(1), 1)

    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_oh.size(2)), 1).to(x.dtype)

    #how does this work with masks
    blanks = torch.zeros(x_oh.shape[0], x_oh.shape[1], x_oh.shape[2]).cuda()
    blanks[:, 0] = 1.

    x_lengths_blanks = x_lengths * 2 + 1

    #print(x_lengths_blanks)

    #print(x_oh.shape)

    #print(blanks.shape)

    x_oh_blanks = torch.stack([blanks, x_oh], dim=-1).view(x_oh.shape[0], x_oh.shape[1], -1)
    x_oh_blanks = torch.cat([x_oh_blanks, torch.zeros(x_oh.shape[0], x_oh.shape[1], 1).cuda()], dim=-1)

    x_oh_blanks[:, 0, -1] = 1.

    #print(x_oh_blanks.shape)

    x_mask_blanks = torch.unsqueeze(commons.sequence_mask(x_lengths_blanks, x_oh_blanks.size(2)), 1).to(x.dtype)

    #print(x_mask_blanks.shape)

    #plt.imshow(x_oh_blanks[0].cpu())
    #plt.show()

    if g is not None:
      g_exp = g.expand(-1, -1, x.size(-1))
      x_dp = torch.cat([torch.detach(x_oh_blanks), g_exp], 1)
    else:
      x_dp = torch.detach(x_oh_blanks)

    logw = self.proj_w(x_dp, x_mask_blanks).squeeze(1)

    #print(logw.shape)

    return x_oh, x_oh_blanks, logw, x_mask_blanks


class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels,
      out_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      #self.flows.append(irevnet_block(in_channels * n_sqz, in_channels * n_sqz // 2, stride=1, mult=4))

      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        attentions.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))


  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    #if reverse:
    #  x = torch.log(x) - torch.log(1 - x)

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)

    #if not reverse:
    #  x = F.sigmoid(x)

    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()


class FlowGenerator(nn.Module):
  def __init__(self, 
      n_vocab, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=6,
      p_dropout=0., 
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      gin_channels=0, 
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec
    self.prenet = prenet

    self.pre_encoder = TextPreEncoder(
        n_vocab, 
        out_channels, 
        hidden_channels_enc or hidden_channels, 
        filter_channels, 
        filter_channels_dp, 
        n_heads, 
        n_layers_enc, 
        kernel_size, 
        p_dropout, 
        window_size=window_size,
        block_length=block_length,
        mean_only=mean_only,
        prenet=prenet,
        gin_channels=gin_channels)

    self.encoder = Encoder(
        out_channels,
        hidden_channels_enc,
        out_channels,
        11,
        p_dropout
    )

    self.decoder = FlowSpecDecoder(
        out_channels,
        hidden_channels_dec or hidden_channels, 
        n_vocab * n_sqz,
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels)

    self.dur_extractor = DurationExtractor()

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
      nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    x_oh, x_oh_blanks, logw, x_mask = self.pre_encoder(x, x_lengths, g=g)
    #logw - log predicted durations



    if gen == False:

      y_max_length = y.size(2)
      y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
      y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x.dtype)

      ctc_out_padded, logdet = self.decoder(y, y_mask, g=g, reverse=False)

      #ctc_out_padded = F.tanh(ctc_out_padded) * 5.

      ctc_out = ctc_out_padded[:, :self.n_vocab]

      ctc_out_greedy = torch.argmax(ctc_out, dim=1)

      #"true" durations extracted from ctc
      durs = torch.zeros([x.shape[0], x.shape[1] * 2 + 1]).cuda()

      for i in range(len(ctc_out)):
        blanks, cnts = self.dur_extractor(x[i, :x_lengths[i]].cpu().numpy(), ctc_out_greedy[i, :y_lengths[i]].detach().cpu().numpy(),
                                        ctc_out[i, :y_lengths[i]].detach().cpu().numpy(), y_lengths[i].cpu().numpy())

        combined_durs = np.empty(blanks.size + cnts.size)
        combined_durs[::2] = blanks
        combined_durs[1::2] = cnts

        durs[i, :combined_durs.shape[0]] = torch.Tensor(combined_durs).cuda()

      logw_ = torch.log(durs + 1.)
      #logw_ - true log durations
    else:

      #convert predicted logw to durations

      durs = torch.round(torch.exp(logw) - 1.).long()

      y_lengths = torch.sum(durs, dim=-1)

      y_lengths = (y_lengths // self.n_sqz) * self.n_sqz

      y_max_length, _ = torch.max(y_lengths, dim=-1)

      print(durs.shape, y_lengths.shape, y_max_length.shape)
      print(y_lengths, y_max_length)

      y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x.dtype)

      print(y_mask.shape)

    """
    durs = torch.round(torch.exp(logw) - 1.).long()

    y_lengths = torch.sum(durs, dim=-1)

    y_max_length, _ = torch.max(y_lengths, dim=-1)

    print(durs.shape, y_lengths.shape, y_max_length.shape)
    print(y_lengths, y_max_length)

    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x.dtype)
    """

    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

    path = commons.generate_path(durs, attn_mask.squeeze(1)).unsqueeze(1).float()

    x_proj = torch.matmul(path.squeeze(1).transpose(1, 2), x_oh_blanks.transpose(1, 2)).transpose(1, 2)

    x_proj = (x_proj - 0.5) * 10.


    #x_proj_greedy = torch.argmax(x_proj, dim=1)

    #print(x_proj_greedy)

    ##plt.imshow(x_proj[0].cpu())
    ##plt.show()

    #at this point x_proj is multiplied one hot representations, contains both blanks and symbols
    #next feed into encoder to get ctc


    pred_ctc_out_padded = self.encoder(x_proj, y_mask)
    #print(pred_ctc_out_padded.min(), pred_ctc_out_padded.max(), pred_ctc_out_padded.mean())
    pred_ctc_out_padded = F.tanh(pred_ctc_out_padded) * 5.#torch.clamp(pred_ctc_out_padded, min=-5., max=5.)
    #print(pred_ctc_out_padded.min(), pred_ctc_out_padded.max(), pred_ctc_out_padded.mean())
    #print(x_proj.shape, y_mask.shape, pred_ctc_out_padded.shape, ctc_out_padded.shape)

    #pred_ctc_out_padded[:, self.n_vocab:] = ctc_out_padded[:, self.n_vocab:]
    #test if just predict correct ctc out + noise
    #pred_ctc_out_padded = ctc_out_padded + torch.randn(ctc_out_padded.shape).cuda() * 0.5

    #if gen == False:
    #  pred_ctc_out_padded = pred_ctc_out_padded + torch.randn(pred_ctc_out_padded.shape).cuda() * 0.3

    #predicted ctc output by encoder

    #print(ctc_out)
    #print(ctc_out.shape)
    #print(ctc_out[0].mean(dim=-2), ctc_out[0].min(dim=-2), ctc_out[0].max(dim=-2))
    #print(pred_ctc_output[0].mean(dim=-2), pred_ctc_output[0].min(dim=-2), pred_ctc_output[0].max(dim=-2))

    #plt.imshow(ctc_out[0].detach().cpu())
    #plt.show()

    #plt.imshow(pred_ctc_output[0].detach().cpu())
    #plt.show()


    #test predicting the padded values
    ##pred_ctc_out_padded = torch.cat([pred_ctc_output, torch.zeros(pred_ctc_output.shape[0], self.out_channels - self.n_vocab, pred_ctc_output.shape[2]).cuda()],
    ##                              dim=1)

    #padding to correct out channels for decoder

    #now we need to reconstruct the spectrogram
    #by reversed decoder
    #if gen == False:
    #  y_pred = y
    #else:
    y_pred, _ = self.decoder(pred_ctc_out_padded, y_mask, g=g, reverse=True)


    y_pred = y_pred * y_mask
    #tested that original padded ctc works

    #plt.imshow(y[0].cpu())
    #plt.show()

    #plt.imshow(y_pred[0].detach().cpu())
    #plt.show()

    #plt.figure()


    """fig, axs = plt.subplots(5)
    fig.suptitle('CTC and spectrograms')
    axs[0].imshow(ctc_out_padded[0].detach().cpu())
    axs[1].imshow(x_proj[0].detach().cpu())
    axs[2].imshow(pred_ctc_out_padded[0].detach().cpu())
    axs[3].imshow(y[0].detach().cpu())
    axs[4].imshow(y_pred[0].detach().cpu())
    plt.show()
    input()"""


    if gen == False:
      return ctc_out_padded, pred_ctc_out_padded, logw, logw_, y_pred, y_lengths, logdet
    else:
      return y_pred, y_lengths

  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()
