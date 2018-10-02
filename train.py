import argparse
import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from synthesize import get_sentences
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train
from wavernn.preprocess import wavernn_preprocess
from wavernn.train import wavernn_train

log = infolog.log


def save_seq(file, sequence):
    sequence = [s for s in sequence]
    with open(file, 'w') as f:
        f.write('|'.join(sequence))


def read_seq(file):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            sequence = f.read().split('|')
        return [int(s) for s in sequence[:-1]]
    else:
        return [0, 0, 0]


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(args.model))
    os.makedirs(log_dir, exist_ok=True)

    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), args.model, args.slack_url)

    return log_dir, modified_hp


def train(args, log_dir, hparams):
    state_file = os.path.join(log_dir, 'state_log')
    (taco_state, GTA_state, wave_state) = read_seq(state_file)

    if not taco_state:
        log('\n#############################################################\n')
        log('Tacotron Train\n')
        log('###########################################################\n')
        checkpoint = tacotron_train(args, log_dir, hparams)

        tf.reset_default_graph()

        # Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
        sleep(0.5)

        if checkpoint is None:
            raise('Error occured while training Tacotron, Exiting!')

        taco_state = 1
        save_seq(state_file, [taco_state, GTA_state, wave_state])
    else:
        checkpoint = os.path.join(log_dir, 'taco_pretrained')

    if not GTA_state:
        log('\n#############################################################\n')
        log('Tacotron GTA Synthesis\n')
        log('###########################################################\n')
        args.mode = 'synthesis'
        tacotron_synthesize(args, hparams, checkpoint)

        args.mode = 'eval'
        tacotron_synthesize(args, hparams, checkpoint, get_sentences(args))

        tf.reset_default_graph()

        # Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is training
        sleep(0.5)

        GTA_state = 1
        save_seq(state_file, [taco_state, GTA_state, wave_state])

    if not wave_state:
        log('\n#############################################################\n')
        log('WaveRNN Train\n')
        log('###########################################################\n')
        wavernn_preprocess(args, hparams)

        wavernn_train(args, log_dir, hparams)

        wave_state = 1
        save_seq(state_file, [taco_state, GTA_state, wave_state])

    if wave_state and GTA_state and taco_state:
        log('TRAINING IS ALREADY COMPLETE!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
    parser.add_argument('--GTA', type=bool, default=True, help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250, help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=10000, help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_steps', type=int, default=100000, help='total number of tacotron training steps')
    parser.add_argument('--wavernn_train_epochs', type=int, default=300, help='total number of wavenet training epochs')
    parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
    args = parser.parse_args()

    accepted_models = ['Tacotron', 'WaveRNN', 'Tacotron-2']

    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

    log_dir, hparams = prepare_run(args)

    if args.model == 'Tacotron':
        tacotron_train(args, log_dir, hparams)
    elif args.model == 'WaveRNN':
        wavernn_train(args, log_dir, hparams)
    elif args.model == 'Tacotron-2':
        train(args, log_dir, hparams)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
    main()
