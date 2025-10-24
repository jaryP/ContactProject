from collections import defaultdict
from typing import List

import numpy as np
import torch
from Bio.PDB import PDBParser
from sklearn import metrics


def get_protein_collate_fn(model_batch_converter):
    # a closure is more practical because the returned function can be easily called
    # without moving around the batch_converter

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
            if k != 'label_matrix':
                others[k] = torch.nn.utils.rnn.pad_sequence(v, padding_value=-1, batch_first=True)
            else:
                others[k] = torch.stack([torch.nn.functional.pad(_v, (0, padded_len - _v.shape[-1],
                                                                      0, padded_len - _v.shape[-1]), value=-1)
                                         for _v in v], 0)

        return {'proteins': strings, **others}

    return protein_collate_fn


def mcc(y_true: np.ndarray, y_pred: np.ndarray, is_global=True, padding_value: int = -1) -> float:
    # The advantages of the Matthews correlation coefficient (MCC)
    # over F1 score and accuracy in binary classification evaluation

    if is_global:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        mask = y_true[y_true != padding_value]

        score = metrics.matthews_corrcoef(y_true[mask], y_pred[mask])

    else:
        assert len(y_true.shape) == 4
        assert len(y_pred.shape) == 4

        n_samples, sequence, matrix_row, matrix_cols = y_true.shape

        y_true = y_true.reshape((n_samples, -1))
        y_pred = y_pred.reshape((n_samples, -1))

        score = 0
        for yt, yp in zip(y_true, y_pred):
            mask = yp[yp != padding_value]

            score += metrics.matthews_corrcoef(yt[mask], yp[mask])

        score = score / n_samples

    return score


class CustomParser(PDBParser):
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
