import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

import re 


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler_all_speakers = StandardScaler()
        energy_scaler_all_speakers = StandardScaler()
        pitch_max_all_speakers = np.finfo(np.float64).min
        pitch_min_all_speakers = np.finfo(np.float64).max
        energy_max_all_speakers = np.finfo(np.float64).min
        energy_min_all_speakers = np.finfo(np.float64).max

        trim_silence = self.config["preprocessing"]["trim_silence"]
        skip_mean = False

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        stats_by_speaker = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i

            pitch_scaler = StandardScaler()
            energy_scaler = StandardScaler()

            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )

                dur_filename = "{}-duration-{}.npy".format(speaker, basename)
                pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
                energy_filename = "{}-energy-{}.npy".format(speaker, basename)
                mel_filename = "{}-mel-{}.npy".format(speaker, basename)

                if os.path.exists(os.path.join(self.out_dir, "duration", dur_filename)) and os.path.exists(os.path.join(self.out_dir, "pitch", pitch_filename)) and os.path.exists(os.path.join(self.out_dir, "energy", energy_filename)) and os.path.exists(os.path.join(self.out_dir, "mel", mel_filename)):
                    skip_mean = True
                    continue

                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename, trim_silence)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                        print(info)
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    pitch_scaler_all_speakers.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                    energy_scaler_all_speakers.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

            if not skip_mean:
                    print("Computing statistic quantities for speaker:{} ...".format(speaker))
                    # Perform normalization if necessary
                    if self.pitch_normalization:
                        pitch_mean = pitch_scaler.mean_[0]
                        pitch_std = pitch_scaler.scale_[0]
                        pitch_mean_all_speakers = pitch_scaler_all_speakers.mean_[0]
                        pitch_std_all_speakers = pitch_scaler_all_speakers.scale_[0]
                    else:
                        # A numerical trick to avoid normalization...
                        pitch_mean = 0
                        pitch_std = 1
                        pitch_mean_all_speakers = 0
                        pitch_std_all_speakers = 1
                    if self.energy_normalization:
                        energy_mean = energy_scaler.mean_[0]
                        energy_std = energy_scaler.scale_[0]
                        energy_mean_all_speakers = energy_scaler_all_speakers.mean_[0]
                        energy_std_all_speakers = energy_scaler_all_speakers.scale_[0]
                    else:
                        energy_mean = 0
                        energy_std = 1
                        energy_mean_all_speakers = 0
                        energy_std_all_speakers = 1

                    # Normalisation by speaker
                    pitch_min, pitch_max = self.normalize(
                        os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std, speaker
                    )
                    energy_min, energy_max = self.normalize(
                        os.path.join(self.out_dir, "energy"), energy_mean, energy_std, speaker
                    )
                    stats_by_speaker[speaker] = {
                        "pitch": [
                            float(pitch_min),
                            float(pitch_max),
                            float(pitch_mean),
                            float(pitch_std),
                        ],
                        "energy": [
                            float(energy_min),
                            float(energy_max),
                            float(energy_mean),
                            float(energy_std),
                        ],
                    }

                    with open(os.path.join(self.out_dir, "stats_{}.json".format(speaker)), "w") as f:
                        f.write(json.dumps(stats_by_speaker[speaker]))

                    pitch_min_all_speakers = min(pitch_min_all_speakers, pitch_min)
                    pitch_max_all_speakers = max(pitch_max_all_speakers, pitch_max)
                    energy_min_all_speakers = min(energy_min_all_speakers, energy_min)
                    energy_max_all_speakers = max(energy_max_all_speakers, energy_max)

        # Save files by speaker
        if not skip_mean:
            with open(os.path.join(self.out_dir, "stats_by_speaker.json"), "w") as f:
                f.write(json.dumps(stats_by_speaker))

            # Save files
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speakers))

            with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
                stats = {
                    "pitch": [
                        float(pitch_min_all_speakers),
                        float(pitch_max_all_speakers),
                        float(pitch_mean_all_speakers),
                        float(pitch_std_all_speakers),
                    ],
                    "energy": [
                        float(energy_min_all_speakers),
                        float(energy_max_all_speakers),
                        float(energy_mean_all_speakers),
                        float(energy_std_all_speakers),
                    ],
                }
                f.write(json.dumps(stats))

            print("Statistics updated")

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # # Write metadata
        # with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        #     for m in out[self.val_size :]:
        #         f.write(m + "\n")
        # with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
        #     for m in out[: self.val_size]:
        #         f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename, trim_silence):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones"),
            trim_silence
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]

        # Case seen in M_AILABS 
        if np.sum(pitch != 0) <= 1:
            print(basename)
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):

                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    # pitch[i] = 1 # was 0, use 1 for 0 semitones
                    pitch[i] = pitch[i-1] # use non-zero pitch to calculate coherent mean and std

                pos += d
            pitch = pitch[: len(duration)]

        # Compute pitch in semitones (ref 1Hz)
        pitch = 12*np.log2(pitch/1.0)

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):

                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    # energy[i] = 0
                    energy[i] = energy[i-1] # use non-zero energy to calculate coherent mean and std

                pos += d
            energy = energy[: len(duration)]

        # Save files
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            # "|".join([basename, speaker, text, raw_text]),
            "|".join([basename, speaker, raw_text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier, trim_silence):
        sil_phones = ["sil", "sp", "spn", "__", "_"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if trim_silence and (p in sil_phones):
                    continue
                else:
                    start_time = s

            if trim_silence:
                if p not in sil_phones:
                    # For ordinary phones
                    phones.append(p)
                    end_time = e
                    end_idx = len(phones)
                else:
                    # For silent phones
                    phones.append(p)
            else:
                # All phon are ordinary
                phones.append(p)
                end_time = e
                end_idx = len(phones)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        if trim_silence:
            phones = phones[:end_idx]
            durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std, speaker):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            if re.match(".*_{}_.*".format(speaker), filename):
                filename = os.path.join(in_dir, filename)
                print(filename)

                values = (np.load(filename) - mean) / std
                np.save(filename, values)

                max_value = max(max_value, max(values))
                min_value = min(min_value, min(values))

        return min_value, max_value
