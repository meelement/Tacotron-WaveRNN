import glob
import os
import pickle

import numpy as np
from infolog import log
from wavernn.train import _bits


def get_files(path, extension='.npy'):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames.append(filename)
    return sorted(filenames)


def convert_gta_audio(audio_path):
    audio = np.load(audio_path)
    quant = (audio + 1.) * (2**_bits - 1) / 2
    return quant.astype(np.int)


def convert_gta_mels(mels_path):
    mels = np.load(mels_path).T
    return mels.astype(np.float32)


def preprocess(args, audio_dir, taco_dir, hparams):
    output_dir = os.path.join(args.base_dir, 'wavernn_data')
    quant_dir = os.path.join(output_dir, 'quant')
    mels_dir = os.path.join(output_dir, 'mels')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quant_dir, exist_ok=True)
    os.makedirs(mels_dir, exist_ok=True)

    audio_files = get_files(audio_dir)
    mels_files = get_files(taco_dir)

    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(zip(audio_files, mels_files)):
        audio_id = path[0].split('/')[-1][6:-4]
        mels_id = path[1].split('/')[-1][4:-4]

        assert(mels_id == audio_id)

        dataset_ids.append(audio_id)

        np.save(f'{quant_dir}/{audio_id}.npy', convert_gta_audio(path[0]))
        np.save(f'{mels_dir}/{mels_id}.npy', convert_gta_mels(path[1]))

        log('%i/%i : audio: %s mel: %s' % (i + 1, len(audio_files), audio_id, mels_id))

    dataset_ids_unique = list(set(dataset_ids))

    with open(f'{output_dir}/dataset_ids.pkl', 'wb') as file:
        pickle.dump(dataset_ids_unique, file)


def wavernn_preprocess(args, hparams):
    audio_dir = os.path.join(args.base_dir, 'training_data', 'audio')
    taco_dir = os.path.join(args.base_dir, 'tacotron_output', 'gta')

    preprocess(args, audio_dir, taco_dir, hparams)
