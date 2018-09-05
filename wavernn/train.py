import os

import numpy as np
import torch
import torch.nn as nn
from datasets.audio import save_wavernn_wav
from infolog import log
from torch import optim
from torch.utils.data import DataLoader, Dataset
from wavernn.model import Model

bits = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudiobookDataset(Dataset):
    def __init__(self, metadata, path):
        self.metadata = metadata
        self.path = path

    def __getitem__(self, index):
        file_set = self.metadata[index]
        x = np.load(f'{file_set[0]}')
        m = np.load(f'{file_set[2]}')
        return m, x

    def __len__(self):
        return len(self.metadata)


def collate(batch):
    pad = 2
    mel_win = seq_len // hop_length + 2 * pad

    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)

    x_input = 2 * coarse[:, :seq_len].float() / (2**bits - 1.) - 1.

    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse


def eval(step, ouput_dir, metadata, hparams, samples=3):
    file_list = metadata[:samples]
    gts = [np.load(f'{file_set[0]}') for file_set in file_list]
    mels = [np.load(f'{file_set[2]}') for file_set in file_list]

    for i, (gt, mel) in enumerate(zip(gts, mels)):
        log('Generating: %i/%i' % (i + 1, samples))

        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.

        save_wavernn_wav(f'{ouput_dir}{k}k_steps_{i}_target.wav', gt, sr=hparams.sample_rate)


def train(args, log_dir, input_path, hparams):
    save_dir = os.path.join(log_dir, 'wavernn_pretrained')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'wavernn_model.pyt')

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    with open(input_path, 'r') as f:
        metadata = [line.strip().split('|') for line in f]

    dataset = AudiobookDataset(metadata, input_path)
    log('dataset length: %i' % len(dataset))

    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=32,
                             num_workers=2, shuffle=True, pin_memory=True)

    model = Model(rnn_dims=512, fc_dims=512, bits=bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    if not os.path.exists(checkpoint_path):
        log('Create new model!!!', slack=True)
        torch.save(['state_dict': model.state_dict() 'global_step': 0], checkpoint_path)
    else:
        log('Loading model from {}'.format(checkpoint_path), slack=True)

    checkpoint = torch.load(checkpoint_path)
    step = checkpoint['global_step']
    model.load_state_dict(checkpoint['state_dict'])

    optimiser = optim.Adam(model.parameters())

    criterion = nn.NLLLoss().to(device)

    # Train
    for e in range(args.num_epochs):
        running_loss = 0.
        start = time.time()

        for i, (x, m, y) in enumerate(data_loader):
            x, m, y = x.to(device), m.to(device), y.to(device)

            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            step += 1

            speed = (i + 1) / (time.time() - start)

            item_loss = loss.item()
            running_loss += item_loss
            avg_loss = running_loss / (i + 1)

            message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(step, speed, item_loss, avg_loss)
            log(message, slack=(step % args.checkpoint_interval == 0))

            if item_loss > 100 or np.isnan(item_loss):
                log('Loss exploded to {:.5f} at step {}'.format(item_loss, step))
                raise Exception('Loss exploded')

            if step % args.eval_interval == 0:
                log('Running evaluation at step {}'.format(step))
                eval(step, eval_wav_dir, metadata, hparams)

        torch.save(['state_dict': model.state_dict(), 'global_step': step], checkpoint_path)


def wavernn_train(args, log_dir, input_path, hparams):
    return train(args, log_dir, input_path, hparams)
