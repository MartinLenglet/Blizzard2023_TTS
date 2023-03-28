import os
import json
import copy
import math
from collections import OrderedDict
from regex import B
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat

from utils.tools import get_mask_from_lengths, pad

from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"][
            "normalization"
        ]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"][
            "normalization"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            self.pitch_mean, self.pitch_std = stats["pitch"][2:4]
            energy_min, energy_max = stats["energy"][:2]
            self.energy_mean, self.energy_std = stats["energy"][2:4]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            # Pitch Prediction in Semitons => additive control
            if self.pitch_normalization:
                # self.pitch_std gives mean std from all speakers
                # prediction = prediction + control/self.pitch_std
                prediction = prediction + control/3.5701
            else:
                prediction = prediction + control

            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
            
            batch_size = prediction.size(dim=0)

        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            # Energy Prediction in dB => additive control
            if self.energy_normalization:
                # self.energy_std gives mean std from all speakers
                # prediction = prediction + control/self.energy_std
                prediction = prediction + control/8.4094
            else:
                prediction = prediction + control

            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=0.0,
        e_control=0.0,
        d_control=1.0,
        src_mask_noSpectro=None,
    ):

        # log_duration_prediction = self.duration_predictor(x, src_mask)
        log_duration_prediction = self.duration_predictor(x, src_mask_noSpectro)

        if self.pitch_feature_level == "phoneme_level":
            # pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            #     x, pitch_target, src_mask, p_control
            # )
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask_noSpectro, p_control
            )

            x = x + pitch_embedding
                
        if self.energy_feature_level == "phoneme_level":
            # energy_prediction, energy_embedding = self.get_energy_embedding(
            #     x, energy_target, src_mask, e_control
            # )
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask_noSpectro, e_control
            )

            x = x + energy_embedding
                
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            # duration_rounded = torch.clamp(
            #     (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            #     min=0,
            # )

            # Take residual duration into account
            predicted_duration = (torch.exp(log_duration_prediction) - 1) * d_control
            predicted_duration_compensated = predicted_duration

            for utt_in_batch in range(x.size()[0]):
                residual = 0.0
                for index_phon in range(x.size()[1]):
                    dur_phon = predicted_duration[utt_in_batch][index_phon]
                    dur_phon_rounded = torch.round(dur_phon + residual)
                    residual += dur_phon - dur_phon_rounded
                    predicted_duration_compensated[utt_in_batch][index_phon] = dur_phon_rounded

            # Add residual to compensate for round
            duration_rounded = torch.clamp(
                predicted_duration_compensated,
                min=0,
            )

            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )

            x = x + pitch_embedding

        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()

            # out.append(vec.expand(max(int(expand_size), 0), -1))
            out.append(vec.expand(max(int(np.round(expand_size)), 0), -1))

        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            # padding_mode='replicate',
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        # return F.softmax(self.linear_layer(x), dim=2)
        return self.linear_layer(x) # CrossEntropyLoss computes softmax internally
