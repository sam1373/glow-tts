import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
from text.symbols import symbols

global_step = 0
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))
  #train_and_eval(0, n_gpus, hps)


def train_and_eval(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextMelCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  if rank == 0:
    val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

  print(len(train_dataset))
  print(len(train_loader))

  generator = models.FlowGenerator(
      n_vocab=len(symbols), 
      out_channels=hps.data.n_mel_channels, 
      **hps.model).cuda(rank)
  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  if hps.train.fp16_run:
    generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  #generator = DDP(generator)
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
    epoch_str += 1
    optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    optimizer_g._update_learning_rate()
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
    epoch_str = 1
    global_step = 0

  
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      if epoch % 50 == 0:
        utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
  train_loader.sampler.set_epoch(epoch)
  global global_step

  generator.train()
  for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)


    # Train Generator
    optimizer_g.zero_grad()

    ctc_out, pred_ctc_out, logw, logw_, y_pred, y_lengths, logdet = generator(x, x_lengths, y, y_lengths, gen=False)
    #print(ctc_out.shape, pred_ctc_out.shape)
    """print("real")
    print(ctc_out[:, :len(symbols)].mean(), pred_ctc_out[:, :len(symbols)].mean())
    print(ctc_out[:, :len(symbols)].min(), pred_ctc_out[:, :len(symbols)].min())
    print(ctc_out[:, :len(symbols)].max(), pred_ctc_out[:, :len(symbols)].max())
    print("padded")
    print(ctc_out[:, len(symbols):].mean(), pred_ctc_out[:, len(symbols):].mean())"""
    l_ctc = ctc_loss(F.log_softmax(ctc_out.permute(2, 0, 1), dim=-1), x, y_lengths, x_lengths)
    if hps.l1_loss:
      l_tts = torch.sum(torch.abs(y[:, :, :y_pred.shape[2]] - y_pred)) / (
                torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
    else:
      l_tts = torch.sum((y[:, :, :y_pred.shape[2]] - y_pred) ** 2) / (
              torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
    l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
    loss_gs = [l_ctc, l_tts, l_length]
    if hps.ctc_pred:
      if hps.l1_loss:
        l_ctc_pred = torch.sum(torch.abs(ctc_out[:, :, :pred_ctc_out.shape[2]] - pred_ctc_out)) / (
                 torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
      else:
        l_ctc_pred = torch.sum((ctc_out[:, :, :pred_ctc_out.shape[2]] - pred_ctc_out) ** 2) / (
                torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
      l_ctc_clamp = torch.mean(F.relu(torch.abs(ctc_out) - 5.)) * 1000.
      loss_gs.append(l_ctc_pred)
      loss_gs.append(l_ctc_clamp)
    if hps.log_det:
      l_logdet = -torch.sum(logdet) / (
                torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
      loss_gs.append(l_logdet)

    loss_g = sum(loss_gs)

    if hps.train.fp16_run:
      with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        scaled_loss.backward()
      grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
    else:
      loss_g.backward()
      grad_norm = commons.clip_grad_value_(generator.parameters(), hps.grad_clip)
    optimizer_g.step()

    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        #(y_gen, *_), *_ = generator.module(x[:1], x_lengths[:1], gen=True)
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        """utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)"""

    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        ctc_out, pred_ctc_out, logw, logw_, y_pred, y_lengths, logdet = generator(x, x_lengths, y, y_lengths, gen=False)
        print(ctc_out[:, :len(symbols)].mean(), pred_ctc_out[:, :len(symbols)].mean())
        print(ctc_out[:, :len(symbols)].min(), pred_ctc_out[:, :len(symbols)].min())
        print(ctc_out[:, :len(symbols)].max(), pred_ctc_out[:, :len(symbols)].max())
        print("padded")
        print(ctc_out[:, len(symbols):].mean(), pred_ctc_out[:, len(symbols):].mean())
        l_ctc = ctc_loss(F.log_softmax(ctc_out.permute(2, 0, 1), dim=-1), x, y_lengths, x_lengths)
        if hps.l1_loss:
          l_tts = torch.sum(torch.abs(y[:, :, :y_pred.shape[2]] - y_pred)) / (
                  torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
        else:
          l_tts = torch.sum((y[:, :, :y_pred.shape[2]] - y_pred) ** 2) / (
                  torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
        l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        loss_gs = [l_ctc, l_tts, l_length]
        if hps.ctc_pred:
          if hps.l1_loss:
            l_ctc_pred = torch.sum(torch.abs(ctc_out[:, :, :pred_ctc_out.shape[2]] - pred_ctc_out)) / (
                    torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
          else:
            l_ctc_pred = torch.sum((ctc_out[:, :, :pred_ctc_out.shape[2]] - pred_ctc_out) ** 2) / (
                    torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
          loss_gs.append(l_ctc_pred)
        if hps.log_det:
          l_logdet = -torch.sum(logdet) / (
                  torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
          loss_gs.append(l_logdet)
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        if batch_idx % hps.train.log_interval == 0:
          logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(val_loader.dataset),
            100. * batch_idx / len(val_loader),
            loss_g.item()))
          logger.info([x.item() for x in loss_gs])
           
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))

                           
if __name__ == "__main__":
  main()
