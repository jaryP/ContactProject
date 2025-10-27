from collections import defaultdict
from typing import List, Sequence, Dict, Callable, Iterable

import numpy as np
import torch
import tqdm
from Bio.PDB import PDBParser
from esm import BatchConverter
from sklearn import metrics


MAPPING = {'ALA': 'A',
           'ARG': 'R',
           'ASP': 'D',
           'CYS': 'C',
           'CYX': 'C',
           'GLN': 'Q',
           'GLU': 'E',
           'GLY': 'G',
           'HIS': 'H',
           'HIE': 'H',
           'ILE': 'I',
           'LEU': 'L',
           'LYS': 'K',
           'MET': 'M',
           'PRO': 'P',
           'SER': 'S',
           'ASN': 'N',
           'PHE': 'F',
           'SEC': 'U',
           'THR': 'T',
           'TRP': 'W',
           'TYR': 'Y',
           'VAL': 'V'
           }


def get_protein_collate_fn(model_batch_converter: BatchConverter) -> Callable:
    """
    This function returns the collate function used to create the batches
    :param model_batch_converter: a batch converter object returned by doing alphabet.get_batch_converter()
    :return: a function that convert a set of data into a batch
    """

    # a closure is more practical because the returned function can be easily called
    # without moving around in the code the batch_converter class

    def protein_collate_fn(batch):
        """
        Collate function for protein batches.

        Each element in `batch` is a dict containing:
            - 'residues': sequence of residues defining the protein, str (sequence)
            - 'label_matrix': matrix containing the binary contacts, np.ndarray (LxL)
            - 'pos_res': index of positive residue,  np.ndarray (Lx1)
            - 'neg_res': index of negative residue, np.ndarray (Lx1)

        Check the class dataset.ProteinDataset for more information.

        Returns a dict with:
            - 'proteins': output of batch_converter
            - other tensors, padded accordingly
        """

        strings = [(f'prot{i}', b['residues']) for i, b in enumerate(batch)]

        strings = model_batch_converter(strings)

        padded_len = strings[-1].shape[-1]

        others: dict[str, List[torch.Tensor]] = defaultdict(list)

        # the following can be done using dictionary comprehension
        # or zip operation but this is clearer

        for b in batch:
            for k, v in b.items():
                if k == 'residues':
                    continue
                others[k].append(torch.as_tensor(v))

        for k, v in others.items():
            if k in ['length']:
                continue
            if k == 'offset':
                others[k] = torch.as_tensor(v)
            elif k != 'gt_matrix':
                others[k] = torch.nn.utils.rnn.pad_sequence(v, padding_value=-1, batch_first=True)
            else:
                others[k] = torch.stack([torch.nn.functional.pad(_v, (0, padded_len - _v.shape[-1] - 2,
                                                                      0, padded_len - _v.shape[-1] - 2), value=-1)
                                         for _v in v], 0)

        return {'proteins': strings, **others}

    return protein_collate_fn


@torch.no_grad()
def model_testing(model: torch.nn.Module, dataloader: Iterable) -> Dict[str, float]:
    """
    The function to test a model over a given dataset

    :param model: the Pytorch model to test
    :param dataloader: the dataloader which iterates the tested dataset
    :return: a dictionary [str, float], where the keys are the metrics and the values are the results
    """

    device = next(model.parameters()).device

    preds = []
    true = []

    model.eval()

    for batch in tqdm.tqdm(dataloader, leave=False, desc='Testing the model'):
        proteins = batch['proteins'][-1].to(device)

        contacts = model(proteins)

        for c, l, m in zip(torch.split(contacts, 1, 0),
                           batch['length'], batch['gt_matrix']):
            y = (c[:, :l, :l] > 0.5).cpu().float().numpy()
            preds.append(y)
            true.append(m[:l, :l].cpu().numpy())

    global_score = mcc(true, preds, is_global=False)
    local_score = mcc(true, preds)

    return {'global_mcc': None, 'local_score': local_score}


def mcc(y_true: Sequence[np.ndarray], y_pred: Sequence[np.ndarray], is_global=True, padding_value: int = -1) -> float:
    """
    Return the Matthews CorrCoef between the true and predicted values
    :param y_true: the ground truth values
    :param y_pred: the predicted values
    :param is_global: whether to calculate the metric over all the proteins or
    by averaging different results calculated over multiple proteins
    :param padding_value: the padding costant value (default: -1)
    :return: the metric score
    """

    # The advantages of the Matthews correlation coefficient (MCC)
    # over F1 score and accuracy in binary classification evaluation

    if is_global:
        y_true = np.concatenate([y.reshape(-1) for y in y_true])
        y_pred = np.concatenate([y.reshape(-1) for y in y_pred])

        # mask = y_true != padding_value

        score = metrics.matthews_corrcoef(y_true, y_pred)

    else:
        score = 0
        for yt, yp in zip(y_true, y_pred):
            yt, yp = yt.reshape(-1), yp.reshape(-1)

            # mask = yp != padding_value

            score += metrics.matthews_corrcoef(yt, yp)

        score = score / len(y_pred)

    return score.item()


class CustomParser(PDBParser):
    """
    A custom PDBParser parser function which loads also MODRES lines from a given file
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.modres = {}

    def _parse(self, header_coords_trailer):

        # Since the original parse function "consumes" the file,
        # we first search for  MODRES lines, the use it.
        # Probably, a better way will be to modify the _parse_coordinates function

        for line in header_coords_trailer:
            if not line.startswith("MODRES"):
                continue

            res_num = int(line[18:22])
            std_res = line[24:27].strip()
            self.modres[res_num] = std_res

        super()._parse(header_coords_trailer)
