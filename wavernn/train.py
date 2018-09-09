import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from datasets.audio import save_wavernn_wav
from hparams import hparams_debug_string
from infolog import log
from torch import optim
from torch.utils.data import DataLoader, Dataset
from wavernn.model import Model

_bits = 9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.ids = ids
        self.path = path

    def __getitem__(self, index):
        id = self.ids[index]
        m = np.load(f'{self.path}/mels/{id}.npy')
        x = np.load(f'{self.path}/quant/{id}.npy')
        return m, x

    def __len__(self):
        return len(self.ids)


def collate(batch):
    pad = 2
    hop_length = 275
    seq_len = hop_length * 5
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

    x_input = 2 * coarse[:, :seq_len].float() / (2**_bits - 1.) - 1.

    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse


def test_generate(args, model, step, ouput_dir, hparams, samples=3):
    test_dir = os.path.join(args.base_dir, 'tacotron_output', 'eval')

    mels_paths = [f for f in sorted(os.listdir(test_dir)) if f.endswith(".npy")]
    mels = [np.load(os.path.join(test_dir, m)).T for m in mels_paths]

    k = step // 1000

    for i, mel in enumerate(mels):
        log('Generating: %i/%i' % (i + 1, samples))
        model.generate(mel, f'{ouput_dir}/{k}k_steps_{i}.wav', sr=hparams.sample_rate)


def train(args, log_dir, input_dir, hparams):
    save_dir = os.path.join(log_dir, 'wavernn_pretrained')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'wavernn_model.pyt')

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_dir))
    log('Using model: {}'.format(args.model))

    log(hparams_debug_string())

    with open(f'{input_dir}/dataset_ids.pkl', 'rb') as f:
        dataset_ids = pickle.load(f)

    dataset = AudiobookDataset(dataset_ids, input_dir)
    log('dataset length: %i' % len(dataset))

    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=32, shuffle=True)

    model = Model(rnn_dims=512, fc_dims=512, bits=_bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    if not os.path.exists(checkpoint_path):
        log('Create new model!!!', slack=True)
        torch.save({'state_dict': model.state_dict(), 'global_step': 0}, checkpoint_path)
    else:
        log('Loading model from {}'.format(checkpoint_path), slack=True)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    step = checkpoint['global_step']
    log('starting from step: ' + str(step), slack=True)

    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.NLLLoss().to(device)

    # Train
    for e in range(args.wavernn_train_epochs):
        running_loss = 0.
        start = time.time()

        for i, (x, m, y) in enumerate(data_loader):
            optimiser.zero_grad()

            x, m, y = x.to(device), m.to(device), y.to(device)
            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)

            loss = criterion(y_hat, y)
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
                test_generate(args, model, step, eval_wav_dir, hparams)

            if step % args.checkpoint_interval == 0:
                log('Saving model at step {}'.format(step))
                torch.save({'state_dict': model.state_dict(), 'global_step': step}, checkpoint_path)


def wavernn_train(args, log_dir, hparams):
    input_dir = os.path.join(args.base_dir, 'wavernn_data')

    train(args, log_dir, input_dir, hparams)
