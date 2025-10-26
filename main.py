import argparse
import os.path
import warnings
from itertools import cycle
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from peft import get_peft_model
from torch.utils.data import DataLoader

from dataset import ProteinDataset
from utils import get_protein_collate_fn, mcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=Path,
                        help='Path to the dataset.')
    parser.add_argument("-s", "--saving_path", type=Path,
                        help='Path where to save the results.')
    parser.add_argument("-c", "--config_name", type=str,
                        help='name of the config file in configs directory.')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    saving_path = args.saving_path
    config_name = args.config_name

    # this is not the best way to use hydra, but i did not want it to be overkilling wrt number of experiments
    initialize(version_base=None, config_path="configs", job_name="test_app")
    cfg = compose(config_name=config_name, return_hydra_config=True)
    print(OmegaConf.to_yaml(cfg))

    device = 0
    if device == 'cpu':
        warnings.warn("Device set to cpu.")
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(int(device))
    else:
        warnings.warn(f"Device not found {device} "
                      f"or CUDA {torch.cuda.is_available()}")

    device = torch.device(device)

    model, alphabet = hydra.utils.instantiate(cfg.model)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=
                                                   cfg.training.get('truncation_seq_length', None))

    train_dataset = hydra.utils.instantiate(cfg.dataset,
                                            dataset_path=dataset_path / 'train',
                                            truncation_seq_length=batch_converter.truncation_seq_length
                                            )

    test_dataset = hydra.utils.instantiate(cfg.dataset,
                                           sample=False,
                                           dataset_path=dataset_path / 'test',
                                           truncation_seq_length=None)

    bs = cfg.training.batch_size
    collate_fn = get_protein_collate_fn(batch_converter)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(train_dataset,
                                 batch_size=cfg.training.get('test_batch_size', bs),
                                 shuffle=False, collate_fn=collate_fn)

    if 'peft' in cfg:
        peft_config = hydra.utils.instantiate(cfg.peft)
        model = get_peft_model(model, peft_config)

    if 'model_wrapper' in cfg:
        model = hydra.utils.instantiate(cfg.model_wrapper, model=model)

    model = model.to(device)

    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())

    total_tokens = 0
    budget_tokens = cfg.training.budget_token

    progress_bar = tqdm.tqdm(cycle('looooop'))

    losses_stat = []
    budget_stat = []

    while progress_bar:
        model.train()
        progress_bar.set_postfix({'Percentage budget tokens used': (total_tokens / budget_tokens) * 100})

        epoch_losses = []
        epoch_budget = []

        for batch in tqdm.tqdm(train_dataloader, leave=False):
            proteins = batch['proteins'][-1].to(device)

            total_tokens += np.prod(proteins.shape)

            output = model(proteins)

            loss = loss_fn(output,
                           gt_matrix=batch.get('label_matrix'),
                           pos_res=batch.get('pos_res'),
                           neg_res=batch.get('neg_res'))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            epoch_losses.append(loss.item())

            epoch_budget.append(total_tokens / budget_tokens)

            if total_tokens > budget_tokens:
                break

        losses_stat.append(epoch_losses)
        budget_stat.append(epoch_budget)

        model.eval()
        with torch.no_grad():

            preds = []
            true = []

            for batch in test_dataloader:

                proteins = batch['proteins'][-1].to(device)

                contacts = model(proteins)

                for c, l, m in zip(torch.split(contacts, 1, 0),
                                   batch['length'], batch['label_matrix']):

                    y = (c[:, :l, :l] > 0.5).float().cpu().numpy()
                    preds.append(y)
                    true.append(m[:l, :l].cpu().numpy())

            # preds = np.concatenate(preds)
            # true = np.concatenate(true)

            global_score = mcc(true, preds, is_global=False)
            score = mcc(true, preds)

            print(global_score, score)

        if total_tokens > budget_tokens:
            break


if __name__ == "__main__":
    main()