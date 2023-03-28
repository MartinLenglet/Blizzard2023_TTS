import argparse
import os
from tokenize import Double

import torch
import yaml
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")
    torch.autograd.set_detect_anomaly(True)

    preprocess_config, model_config, train_config = configs

    # Output of the model
    output_wav = train_config["output"]["wav"]

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Get dataset
    dataset = Dataset(
        train_config["path"]["train_csv_path"], preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    if output_wav:
        # Load vocoder
        vocoder = get_vocoder(model_config, device)
    else:
        vocoder = None

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # Compute mean loss to print correct log
    nbr_loss_to_disp = 6
    mean_losses = np.zeros(nbr_loss_to_disp)
    nbr_batch_by_losses = np.zeros(nbr_loss_to_disp)

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                # print(batch)
                batch = to_device(batch, device)    
                #print(batch[0])

                # for i in range(len(batch[0])):
                #     print(batch[0][i])
                    # print(batch[1][i])
                    # print(batch[11][i])

                # Forward
                # print(batch[2:])
                output = model(
                    *(batch[2:]),
                    no_spectro=args.no_spectro,
                )

                # for i in range(len(output[0])):
                #     print(batch[0][i])
                #     print(batch[1][i])
                #     print(batch[11][i])

                # Cal Loss
                losses = Loss(batch, output, no_spectro=args.no_spectro)
                total_loss = losses[0]

                # Mean Loss instead of last loss before log
                index_loss = 0
                for l in losses[1:]:
                    if not l.item()==0:
                        mean_losses[index_loss] += l.item()
                        nbr_batch_by_losses[index_loss] += 1
                    index_loss += 1

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    # losses = [l.item() for l in losses]
                    for i_loss in range(nbr_loss_to_disp):
                        if nbr_batch_by_losses[i_loss] == 0:
                            mean_losses[i_loss] = 0
                        else:
                            mean_losses[i_loss] = mean_losses[i_loss]/nbr_batch_by_losses[i_loss]
                    total_loss = np.sum(mean_losses)
                    losses = np.concatenate(([total_loss], mean_losses), axis=0)
                    mean_losses = np.zeros(nbr_loss_to_disp)
                    nbr_batch_by_losses = np.zeros(nbr_loss_to_disp)

                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Phon Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0 and output_wav:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, output_wav, val_logger, vocoder, no_spectro=args.no_spectro)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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
    parser.add_argument('--freeze_encoder', action='store_true',
        help='freeze encoder for transfer learning')
    parser.add_argument('--freeze_decoder', action='store_true',
        help='freeze decoder for transfer learning')
    parser.add_argument('--freeze_decoder_visual', action='store_true',
        help='freeze visual decoder for transfer learning')
    parser.add_argument('--freeze_postnet', action='store_true',
        help='freeze postnet for transfer learning')
    parser.add_argument('--freeze_postnet_visual', action='store_true',
        help='freeze postnet visual for transfer learning')
    parser.add_argument('--freeze_speaker_emb', action='store_true',
        help='freeze speaker embeddings for transfer learning')
    parser.add_argument('--freeze_phon_prediction', action='store_true',
        help='freeze phon prediction for transfer learning')
    parser.add_argument('--freeze_variance_prediction', action='store_true',
        help='freeze variance prediction for transfer learning')

    parser.add_argument('--no_spectro', action='store_true',
        help='Train only until the phonetic predictor (no spectro)')

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
