import argparse
import json
import logging
import os.path
import warnings
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
import tqdm
from calflops import calculate_flops
from esm.rotary_embedding import RotaryEmbedding
from hydra import compose, initialize
from omegaconf import OmegaConf
from peft import get_peft_model
from torch.utils.data import DataLoader

from dataset import ProteinDataset
from model import OffsetRotaryEmbedding
from utils import get_protein_collate_fn, mcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=Path,
                        help='Path to the dataset.')
    parser.add_argument("-s", "--saving_path", type=Path,
                        help='Path where to save the results.')
    parser.add_argument("-c", "--config_name", type=str,
                        help='name of the config file in configs directory.')
    parser.add_argument("--device", type=Union[int, str], default=0, required=False,
                        help='The device to use (integer) or cpu.')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    saving_path = args.saving_path
    config_name = args.config_name
    device = args.device

    saving_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(saving_path / "output.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    # log = logging.getLogger(__name__)

    # this is not the best way to use hydra, but i did not want it to be overkilling wrt number of experiments
    initialize(version_base=None, config_path="configs", job_name="test_app")
    cfg = compose(config_name=config_name, return_hydra_config=True)

    logging.info(OmegaConf.to_yaml(cfg))

    device = 'cuda:{}'.format(int(device))

    try:
        device = torch.device(device)
    except Exception as e:
        logging.error(f"Device not found {device} "
                      f"or CUDA not available ({torch.cuda.is_available()})")
        raise e

    logging.info(f'Device in use: {device}')

    model, alphabet = hydra.utils.instantiate(cfg.model)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=
                                                   cfg.training.get('truncation_seq_length', None))

    train_dataset = hydra.utils.instantiate(cfg.dataset,
                                            dataset_path=dataset_path / 'train',
                                            truncation_seq_length=batch_converter.truncation_seq_length
                                            )

    logging.info(f'Train dataset loaded')

    test_dataset = hydra.utils.instantiate(cfg.dataset,
                                           sample=False,
                                           dataset_path=dataset_path / 'test',
                                           truncation_seq_length=None)

    logging.info(f'Test dataset loaded')

    batch_size = cfg.training.batch_size
    collate_fn = get_protein_collate_fn(batch_converter)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.get('test_batch_size', batch_size),
                                 shuffle=False, collate_fn=collate_fn)

    if 'peft' in cfg:
        peft_config = hydra.utils.instantiate(cfg.peft)
        model = get_peft_model(model, peft_config)

    if 'model_wrapper' in cfg:
        model = hydra.utils.instantiate(cfg.model_wrapper, model=model)

    model = model.to(device)

    flops_map = {}

    logging.info(f'Model loaded')

    accumulation_steps = cfg.training.get('accumulation_steps', 1)

    loss_fn = hydra.utils.instantiate(cfg.loss, average=accumulation_steps == 1)
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())

    total_tokens_used = 0
    budget_tokens = cfg.training.budget_token

    losses_stat = []
    budget_stat = []
    flops_stat = []

    b_i = 0
    all_scores = defaultdict(list)

    progress_bar = tqdm.tqdm(cycle('looooop'))
    while progress_bar:
        model.train()

        epoch_losses = []
        epoch_budget = []
        epoch_flops = []

        logging.info(f'Percentage budget tokens used so far: {(total_tokens_used / budget_tokens) * 100}')

        for batch in tqdm.tqdm(train_dataloader, leave=False):
            b_i += 1

            last_scores = {k: v[-1] for k, v in all_scores.items()} if len(all_scores) > 0 else None
            progress_bar.set_postfix({'Percentage budget tokens used': (total_tokens_used / budget_tokens) * 100,
                                      'last_scores': last_scores})

            proteins = batch['proteins'][-1].to(device)

            if proteins.shape[-1] not in flops_map:
                with torch.no_grad():
                    flops, macs, params = calculate_flops(model=model, args=[proteins[0:1]],
                                                          print_results=False, print_detailed=False,
                                                          output_as_string=False)
                    flops = flops / 1e9 # Giga flops
                    flops_map[proteins.shape[-1]] = flops

            flops = flops_map[proteins.shape[-1]] * len(proteins)
            epoch_flops.append(flops)

            total_tokens_used += np.prod(proteins.shape)

            output = model(proteins)

            loss = loss_fn(output,
                           gt_matrix=batch.get('gt_matrix'),
                           pos_indexes=batch.get('pos_indexes'),
                           neg_indexes=batch.get('neg_indexes'))

            if accumulation_steps > 1:
                if len(proteins) != batch_size:
                    continue

                loss = loss.sum() / accumulation_steps
                loss.backward()

                if (b_i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_losses.append(loss.item())
            else:
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                epoch_losses.append(loss.item())

            epoch_budget.append(total_tokens_used / budget_tokens)

            if total_tokens_used > budget_tokens:
                break

        losses_stat.append(epoch_losses)
        budget_stat.append(epoch_budget)
        flops_stat.append(epoch_flops)

        model.eval()
        with torch.no_grad():

            preds = []
            true = []

            for batch in test_dataloader:

                proteins = batch['proteins'][-1].to(device)

                contacts = model(proteins)

                for c, l, m in zip(torch.split(contacts, 1, 0),
                                   batch['length'], batch['gt_matrix']):
                    y = (c[:, :l, :l] > 0.5).float().cpu().numpy()
                    preds.append(y)
                    true.append(m[:l, :l].cpu().numpy())

            global_score = mcc(true, preds, is_global=False)
            score = mcc(true, preds)

            all_scores['global_mcc'].append(global_score)
            all_scores['score'].append(score)

            # print(global_score, score)

        if total_tokens_used > budget_tokens:
            break

    all_scores = dict(all_scores)

    with open(saving_path / f'test_results.json', 'w') as f:
        json.dump(all_scores, f, ensure_ascii=True, indent=4)

    with open(saving_path / f'train_results.json', 'w') as f:
        json.dump({'loss': losses_stat, 'budget_stat': budget_stat, 'flops_stat': flops_stat},
                  f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    main()
