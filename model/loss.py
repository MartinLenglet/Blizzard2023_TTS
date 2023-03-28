import torch
import torch.nn as nn
from utils.tools import device

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.compute_phon_prediction = model_config["compute_phon_prediction"]

    def forward(self, inputs, predictions, no_spectro=False):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            phon_align_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            phon_align_predictions,
            src_masks_noSpectro,
        ) = predictions
    
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        src_masks_noSpectro = ~src_masks_noSpectro

        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        phon_align_targets.requires_grad = False

        if no_spectro or (not torch.any(src_masks_noSpectro)):
            mel_loss = torch.Tensor([0]).long().to(device)
            postnet_mel_loss = torch.Tensor([0]).long().to(device)
            pitch_loss = torch.Tensor([0]).long().to(device)
            energy_loss = torch.Tensor([0]).long().to(device)
            duration_loss = torch.Tensor([0]).long().to(device)
            postnet_au_loss = torch.Tensor([0]).long().to(device)
        else:
            if self.pitch_feature_level == "phoneme_level":
                pitch_predictions = pitch_predictions.masked_select(src_masks_noSpectro)
                pitch_targets = pitch_targets.masked_select(src_masks_noSpectro)
            elif self.pitch_feature_level == "frame_level":
                pitch_predictions = pitch_predictions.masked_select(mel_masks)
                pitch_targets = pitch_targets.masked_select(mel_masks)
                
            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)

            if self.energy_feature_level == "phoneme_level":
                energy_predictions = energy_predictions.masked_select(src_masks_noSpectro)
                energy_targets = energy_targets.masked_select(src_masks_noSpectro)
            if self.energy_feature_level == "frame_level":
                energy_predictions = energy_predictions.masked_select(mel_masks)
                energy_targets = energy_targets.masked_select(mel_masks)

            energy_loss = self.mse_loss(energy_predictions, energy_targets)

            log_duration_predictions = log_duration_predictions.masked_select(src_masks_noSpectro)
            log_duration_targets = log_duration_targets.masked_select(src_masks_noSpectro)

            duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        if self.compute_phon_prediction:
            phon_align_loss = self.cross_entropy_loss(phon_align_predictions, phon_align_targets)
        else:
            phon_align_loss = torch.Tensor([0]).long().to(device)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + phon_align_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            phon_align_loss,
        )
