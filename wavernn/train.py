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
from tqdm import tqdm
from wavernn.model import Model

_batch_size = 32
_bits = 9
_pad = 2
_hop_len = 275
_seq_len = _hop_len * 5
_mel_win = _seq_len // _hop_len + 2 * _pad

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
    mels = []
    coarse = []
    for x in batch:
        max_offset = x[0].shape[-1] - (_mel_win + 2 * _pad)
        mel_offset = np.random.randint(0, max_offset)
        sig_offset = (mel_offset + _pad) * _hop_len
        mels.append(x[0][:, mel_offset:(mel_offset + _mel_win)])
        coarse.append(x[1][sig_offset:(sig_offset + _seq_len + 1)])

    mels = torch.FloatTensor(np.stack(mels).astype(np.float32))
    coarse = torch.LongTensor(np.stack(coarse).astype(np.int64))

    x_input = 2 * coarse[:, :_seq_len].float() / (2**_bits - 1.) - 1.
    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse


def test_generate(model, step, input_dir, ouput_dir, sr, samples=3):
    filenames = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npy')]
    for i in tqdm(range(samples)):
        mel = np.load(os.path.join(input_dir, filenames[i])).T
        model.generate(mel, f'{ouput_dir}/{step // 1000}k_steps_{i}.wav', sr)


def train(args, log_dir, input_dir, hparams):
    test_dir = os.path.join(args.base_dir, 'tacotron_output', 'eval')
    save_dir = os.path.join(log_dir, 'wavernn_pretrained')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'wavernn_model.pyt')

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_dir))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    # Load Dataset
    with open(f'{input_dir}/dataset_ids.pkl', 'rb') as f:
        dataset = AudiobookDataset(pickle.load(f), input_dir)

    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=_batch_size, shuffle=True, pin_memory=True)

    # Initialize Model
    model = Model(rnn_dims=512, fc_dims=512, bits=_bits, pad=_pad,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    # Load Model
    if not os.path.exists(checkpoint_path):
        log('Created new model!!!', slack=True)
        torch.save({'state_dict': model.state_dict(), 'global_step': 0}, checkpoint_path)
    else:
        log('Loading model from {}'.format(checkpoint_path), slack=True)

    # Load Parameters
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    step = checkpoint['global_step']
    log('Starting from {} step'.format(step), slack=True)

    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss().to(device)

    # Train
    for e in range(args.wavernn_train_epochs):
        running_loss = 0.
        start = time.time()

        for i, (x, m, y) in enumerate(data_loader):
            x, m, y = x.to(device), m.to(device), y.to(device).unsqueeze(-1)
            y_hat = model(x, m).transpose(1, 2).unsqueeze(-1)

            loss = criterion(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            item_loss = loss.item()
            running_loss += item_loss
            avg_loss = running_loss / (i + 1)

            step += 1
            speed = (i + 1) / (time.time() - start)

            message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(step, speed, item_loss, avg_loss)
            log(message, end='\r')

        # Save Checkpoint and Eval Wave
        if (e + 1) % 30 == 0:
            log('\nSaving model at step {}'.format(step), end='', slack=True)
            torch.save({'state_dict': model.state_dict(), 'global_step': step}, checkpoint_path)
            test_generate(model, step, test_dir, eval_wav_dir, hparams.sample_rate)

        log('\nFinished {} epoch. Starting next epoch...'.format(e + 1))


def wavernn_train(args, log_dir, hparams):
    input_dir = os.path.join(args.base_dir, 'wavernn_data')

    train(args, log_dir, input_dir, hparams)
