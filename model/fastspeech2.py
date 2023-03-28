import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, LinearNorm
from utils.tools import get_mask_from_lengths, get_mask_from_lengths_noSpectro
from scipy.io import loadmat

from text.symbols import out_symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.compute_phon_prediction = model_config["compute_phon_prediction"]

        # Phonetic prediction from input
        if self.compute_phon_prediction:
            self.dim_out_symbols = len(out_symbols)
            self.phonetize = LinearNorm(model_config["transformer"]["encoder_hidden"], self.dim_out_symbols)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def freeze_encoder(self):
        print('Freeze_encoder')
        for name, child in self.encoder.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False

    def freeze_decoder(self):
        print('Freeze_decoder')
        for name, child in self.decoder.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # Freeze Mel Linear
        for param in self.mel_linear.parameters():
            param.requires_grad = False
    
    def freeze_decoder_visual(self):
        print('Freeze_decoder_visual')
        for name, child in self.decoder_visual.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # Freeze Mel Linear
        for param in self.au_linear.parameters():
            param.requires_grad = False

    def freeze_postnet(self):
        print('Freeze_postnet')
        for name, child in self.postnet.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        
    def freeze_postnet_visual(self):
        print('Freeze_postnet_visual')
        for name, child in self.postnet_visual.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
    
    def freeze_speaker_emb(self):
        print('Freeze_Speaker_Embeddings')
        for param in self.speaker_emb.parameters():
            param.requires_grad = False

    def freeze_phon_prediction(self):
        print('Freeze_Phon_Prediction')
        for name, child in self.phonetize.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        
    def freeze_variance_prediction(self):
        print('Freeze_Variance_Prediction')
        for name, child in self.variance_adaptor.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        phon_align_targets=None,
        p_control=0.0,
        e_control=0.0,
        d_control=1.0,
        no_spectro=False,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        src_masks_noSpectro = get_mask_from_lengths_noSpectro(src_lens, mel_lens, max_src_len)
        
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            
        if self.compute_phon_prediction:
            phon_outputs = self.phonetize(output).transpose(1,2)
        else:
            phon_outputs = None

        # If no_spectro, only train the encoder with the phonetic prediction, do not compute spectro
        if no_spectro:
            return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    phon_outputs,
                    src_masks_noSpectro,
                )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
            src_masks_noSpectro,
        )

        # Avoid decoding in case of only phonetic prediction in batch
        if mel_masks.nelement():
            output, mel_masks = self.decoder(output, mel_masks)

        output = self.mel_linear(output)
       
        # Avoid postnet in cas of only phonetic prediction in batch
        if mel_masks.nelement(): 
            postnet_output = self.postnet(output)
            postnet_output = postnet_output + output
        else:
            postnet_output = output
        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phon_outputs,
            src_masks_noSpectro,
        )
