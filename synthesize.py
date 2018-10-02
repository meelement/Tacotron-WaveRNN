import argparse
import os
from warnings import warn

import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from wavernn.synthesize import wavernn_synthesize


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    taco_checkpoint = os.path.join(args.base_dir, 'logs-' + args.model, 'taco_pretrained')
    wave_checkpoint = os.path.join(args.base_dir, 'logs-' + args.model, 'wavernn_pretrained', 'wavernn_model.pyt')

    return taco_checkpoint, wave_checkpoint, modified_hp


def get_sentences(args):
    if args.text_list != '':
        with open(args.text_list, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = hparams.sentences
    return sentences


def synthesize(args, hparams, taco_checkpoint, wave_checkpoint, sentences):
    log('Running End-to-End TTS Evaluation. Model: {}'.format(args.model))
    log('Synthesizing mel-spectrograms from text..')
    tacotron_synthesize(args, hparams, taco_checkpoint, sentences)

    # Delete Tacotron model from graph
    tf.reset_default_graph()

    # Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is synthesizing
    sleep(0.5)

    log('Synthesizing audio from mel-spectrograms.. (This may take a while)')
    wavernn_synthesize(args, hparams, wave_checkpoint)

    log('Tacotron-2 TTS synthesis complete!')


def main():
    accepted_modes = ['eval', 'synthesis', 'live']
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
    parser.add_argument('--GTA', type=bool, default=True, help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
    args = parser.parse_args()

    accepted_models = ['Tacotron', 'WaveRNN', 'Tacotron-2']

    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to synthesize with: {}'.format(accepted_models))

    if args.mode not in accepted_modes:
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

    if args.mode == 'live' and args.model == 'WaveRNN':
        raise RuntimeError('WaveRNN vocoder cannot be tested live due to its slow generation. Live only works with Tacotron!')

    if args.model == 'Tacotron-2':
        if args.mode == 'live':
            warn('Requested a live evaluation with Tacotron-2, WaveRNN will not be used!')
        if args.mode == 'synthesis':
            raise ValueError('I don\'t recommend running WaveRNN on entire dataset.. The world might end before the synthesis :) (only eval allowed)')

    taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
    sentences = get_sentences(args)

    if args.model == 'Tacotron':
        tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
    elif args.model == 'WaveRNN':
        wavernn_synthesize(args, hparams, wave_checkpoint)
    elif args.model == 'Tacotron-2':
        synthesize(args, hparams, taco_checkpoint, wave_checkpoint, sentences)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
    main()
