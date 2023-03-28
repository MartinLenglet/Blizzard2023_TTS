# Blizzard Challenge 2023 | Model Implementation | GIPSA-Lab (Grenoble, FRANCE)

This is the implementation GIPSA-Lab's model for the Blizzard Challenge 2023. This code is largely based on the [PyTorch implementation of FastSpeech2 shared by ming024](https://github.com/ming024/FastSpeech2).

# Original model and upgrades

Original FastSpeech2 is a parallel transformer-based architecture generally trained on phonetic inputs. Three prosodic predictors (pitch, energy and duration) are trained alongside the mel-spectrogram decoder. The duration predictor allows non-autoregressive decoding by predicting the letter-to-spectrogram alignment. The decoding process is parallel, using Feed-Forward-Transformer (FFT) layers similar to FastSpeech.

[ming024'implementation](https://github.com/ming024/FastSpeech2) follows an [early version of FastSpeech2](https://arxiv.org/abs/2006.04558v1), which uses F0 values as pitch features, instead of continuous wavelet transform in [later versions](https://arxiv.org/abs/2006.04558). On top that, a Tacotron2-like Postnet was added after the Decoder. Phoneme-level pitch and energy features replace original frame-level features.

# Our contribution

Our contribution focuses on the benefits of mixed representations training ([Kastner, Kyle, et al.](https://ieeexplore.ieee.org/abstract/document/8682880)). Following French Grapheme-to-Phoneme alignment proposed by [Lenglet, Martin, Olivier Perrotin, and Gérard Bailly](https://hal.science/hal-03727735/), we added orthographic input support to this implementation.

We also added a phonetic prediction layer similar to [Lenglet, Martin, Olivier Perrotin, and Gérard Bailly](https://hal.science/hal-03727735/). This additional layer predicts the phonetic output from the internal representations outputted by the text encoder. This layer is trained with cross-entropy. We found that phonetic prediction from orthographic inputs helped structuring the internal latent space of the model. Moreover, the phonetic prediction loss is retro-propagated through the text encoder, which enables to train the text encoder without audio data. Training the text encoder on rare words contexts can help synthesizing new utterances outside of the training database, as well as desambiguating homographs as shown by [Hajj, Maria-Loulou, et al.](https://link.springer.com/chapter/10.1007/978-3-031-20980-2_23).

We found that computing pitch in Semitones and energy in dB (SPL) produces better syntheses. Pitch and energy normalizations were modified to normalize by speaker instead of across all speakers. This forces the model to share pitch and energy embeddings between speakers with different mean fundamental frequencies, thus encoding relative prosodic features.

# FastSpeech2 Architecture

![](./img/model.png)

# Vocoder trained separately

This implementation is not intended to be trained jointly with a neural vocoder. To do that, refer to [ming024'implementation](https://github.com/ming024/FastSpeech2).

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```
This implementation does not require any additional dependencies compared to [ming024'implementation](https://github.com/ming024/FastSpeech2).

## Inference

You have to download the pretrained model and configuration files from this [Google Drive](). Put the pre-trained model in ``output/ckpt/Blizzard2023_Hub/`` and the config files at the root of this repository.

For French multi-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT"  --speaker_id SPEAKER_ID --restore_step 600000 --mode single -p config/Blizzard2023/preprocess_Blizzard2023.yaml -m config/Blizzard2023/model_Blizzard2023_Hub.yaml -t config/Blizzard2023/train_Blizzard2023_Hub.yaml
```
SPEAKER\_ID = ? for NEB, SPEAKER\_ID = ? for AD
The generated utterances will be put in ``output/audio/``.

## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/Blizzard2023/NEB_text.txt --restore_step 600000 --mode single -p config/Blizzard2023/preprocess_Blizzard2023.yaml -m config/Blizzard2023/model_Blizzard2023_Hub.yaml -t config/Blizzard2023/train_Blizzard2023_Hub.yaml
```
to synthesize all utterances in ``preprocessed_data/Blizzard2023/NEB_text.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration biais. Pitch (resp. Energy) control value is additive and specified in Semitones (resp. dB). Duration is controlled with a ratio.
For example, one can increase the speaking rate by 20 % with the argument ``--duration_control 0.8``, or decrease the pitch by 2 semitones with the argument ``--pitch_control -2.0``.

# Training

## Datasets

The [Blizzard2023](https://www.synsig.org/index.php/Blizzard_Challenge_2023#Test_set) dataset is supported.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/Blizzard2023/preprocess_Blizzard2023.yaml
```
to create audio samples by utterance from the audio samples by chapter given in the dataset.

Then generate TextGrid by utterance from the .csv given.
You have to unzip the files in ``preprocessed_data/Blizzard2023/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/Blizzard2023/preprocess_Blizzard2023.yaml
```
to compute mel, pitch, energy and duration by phoneme, saved as one .npy file by utterance.

## Training

Train your model with
```
python3 train.py -p config/Blizzard2023/preprocess_Blizzard2023.yaml -m config/Blizzard2023/model_Blizzard2023_Hub.yaml -t config/Blizzard2023/train_Blizzard2023_Hub.yaml
```
