import argparse
import os

import torch
import yaml
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, output_wav, logger=None, vocoder=None, no_spectro=False):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        train_config["path"]["val_csv_path"], preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    nbr_loss_to_disp = 7
    loss_sums = [0 for _ in range(nbr_loss_to_disp)]

    nbr_batches_by_loss = [0 for _ in range(nbr_loss_to_disp)]
    loss_means = [0 for _ in range(nbr_loss_to_disp)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]), no_spectro=no_spectro)

                # Cal Loss
                losses = Loss(batch, output, no_spectro=no_spectro)

                for i in range(0, len(losses)):
                    if not losses[i].item() == 0:
                        loss_sums[i] += losses[i].item()
                        nbr_batches_by_loss[i] += 1

    for i_loss in range(nbr_loss_to_disp):
        if nbr_batches_by_loss[i_loss] == 0:
            loss_means[i_loss] = 0
        else:
            loss_means[i_loss] = loss_sums[i_loss]/nbr_batches_by_loss[i_loss]
    loss_sums[0] = np.sum(loss_means[1:])

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Phon Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None and output_wav:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    # Output of the model
    output_wav = train_config["output"]["wav"]

    message = evaluate(model, args.restore_step, configs, output_wav)
    print(message)
