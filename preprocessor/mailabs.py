import os

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
from tqdm import tqdm

from text import _clean_text

import re 

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    csv_dir = config["path"]["csv_path"]
    out_dir = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    with open(os.path.join(csv_dir, "NEB_train.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            start_utt = parts[1]
            start_utt = int(int(start_utt) * sampling_rate / 1000)
            end_utt = parts[2]
            end_utt = int(int(end_utt) * sampling_rate / 1000)
            text = parts[3]
            text = _clean_text(text, cleaners)
            align = parts[4]

            if re.match('.*_NEB_.*', base_name):

                print(base_name)

                if len(base_name.split("_")) >= 6:
                        speaker = base_name.split("_")[3]
                else:
                        speaker = base_name.split("_")[2]

                wav_path = os.path.join(in_dir, "{}.wav".format(base_name))

                number_utt_in_chapter = 1

                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav[start_utt:end_utt]
                    wav = wav / max(abs(wav)) * max_wav_value * 0.95

                    while os.path.exists(os.path.join(out_dir, speaker, "{}_{}.wav".format(base_name, number_utt_in_chapter))):
                        number_utt_in_chapter += 1

                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}_{}.wav".format(base_name, number_utt_in_chapter)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}_{}.lab".format(base_name, number_utt_in_chapter)),
                        "w",
                    ) as f1:
                        f1.write(text)
