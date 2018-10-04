# Tacotron-WaveRNN
Tacotron + WaveRNN synthesis

Makes use of:
 - Tacotron: https://github.com/Rayhane-mamah/Tacotron-2
 - WaveRNN: https://github.com/fatchord/WaveRNN

 You'll at least need python3, PyTorch 0.4.1, Tensorflow and librosa.

## Synthesis
```
python3 synthesize.py --model='Tacotron-2' --text_list='{your text file}'
```

## Training
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

## Pretrained Model
https://github.com/h-meru/Tacotron-WaveRNN/files/2444777/wavernn_model.zip

## Samples
https://github.com/h-meru/Tacotron-WaveRNN/files/2444792/Samples_730k.zip
