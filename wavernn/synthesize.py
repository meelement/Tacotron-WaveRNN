import os

import numpy as np
import torch
from infolog import log
from wavernn.model import Model
from wavernn.train import _bits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def synthesize(args, input_dir, output_dir, checkpoint_path, hparams):
    # Initialize Model
    model = Model(rnn_dims=512, fc_dims=512, bits=_bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    # Load Model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    log('Loading WaveRNN model from {}'.format(checkpoint_path))

    mels_paths = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".npy")]
    mels = [np.load(os.path.join(input_dir, m)).T for m in mels_paths]

    for i, mel in enumerate(mels):
        log('\nGenerating: %i/%i' % (i + 1, len(mels)))
        model.generate(mel, f'{output_dir}/{i}_generated.wav', sr=hparams.sample_rate)


def wavernn_synthesize(args, hparams, checkpoint_path):
    input_dir = os.path.join(args.base_dir, 'tacotron_output', 'eval')
    output_dir = os.path.join(args.base_dir, 'wavernn_output')

    synthesize(args, input_dir, output_dir, checkpoint_path, hparams)
