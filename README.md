# tacorn
Tacotron + WaveRNN synthesis

Makes use of:
 - https://github.com/fatchord/WaveRNN
 - https://github.com/m-toman/tacorn
 - https://github.com/Rayhane-mamah/Tacotron-2

 You'll at least need python3, PyTorch 0.4.1, Tensorflow >= 1.9.0 and librosa.

## Synthesis
```
python3 synthesize.py --model='Tacotron-2' --text_list='[~~~ your text ~~~]'
```

## Training
```
python3 preprocess.py
```

```
python3 train.py --model='Tacotron-2'
```

If you would like to train separately...
```
# Tacotron
python3 train.py --model='Tacotron'

# Tacotron synth
python3 synthesize.py --model='Tacotron' --mode='synthesis'

# WaveRNN
python3 train.py --model='WaveRNN'
```
