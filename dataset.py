import os
import pickle

from collections.abc import Generator
from typing import List, Tuple

import numpy as np
from Bio.PDB.Structure import Structure
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import CustomParser, get_protein_collate_fn


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


# https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb#scrollTo=n4FBkOUaWuXb&line=1&uniqifier=1
def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def calculate_contacts(positions: np.ndarray, contact_threshold: float = 8):
    idxs = []
    gt = []

    # iterating is more efficient for big molecules, since we don't need to store distances anyway

    for i, atom in enumerate(positions):
        dist = np.linalg.norm(atom - positions, axis=-1)

        dist = (dist < contact_threshold).astype(float)
        dist[i] = 0

        idxs.append(np.flatnonzero(dist))

        gt.append(dist)

    return idxs, np.asarray(gt)

class Protein:
    def __init__(self, chains: Generator[Structure],
                 modres: dict[int, str] = None,
                 contact_threshold: float = 8.0,
                 testing_protein: bool = False):

        assert contact_threshold > 0, 'contact_threshold must be positive'

        if modres is None:
            modres = {}

        self._testing_protein = testing_protein

        self._residues = ''

        self._residues_chain_id = []
        _residues_coord = {'ca': [], 'cb': []}

        for chain in chains:
            chain_residues = []

            chain_calpha_coords = []
            chain_cbeta_coords = []

            for ri, r in enumerate(chain):
                # skip hetero residues
                if 'CA' not in r or r.id[0].startswith('H_'):
                    continue

                r_name = r.resname

                if r_name not in MAPPING:
                    # let's skip non-standard residues from HET lines
                    # but not the modified amino acids (if modres line is present)
                    r_name = modres.get(ri + 1, None)

                if r_name is None:
                    continue

                c_alpha = r['CA'].coord

                if not all(atom in r for atom in ['CA', 'N', 'C']):
                    c_beta = c_alpha
                else:
                    CA = r['CA'].coord
                    N = r['N'].coord
                    C = r['C'].coord

                    c_beta = extend(C, N, CA, 1.522, 1.927, -2.143)

                # manages edge cases (e.g., file 5OL5.pdb has incomplete modres information)
                if r_name is not None and r_name != '':

                    self._residues += MAPPING[r_name]
                    _residues_coord['ca'].append(c_alpha)
                    _residues_coord['cb'].append(c_beta)

                    self._residues_chain_id.append(chain.id)

        self._gt_matrix = {}
        self._contacts_idx = {}

        for k, v in _residues_coord.items():
            ca_contacts_idx, ca_matrix = calculate_contacts(np.asarray(v))

            self._gt_matrix[k] = ca_matrix
            self._contacts_idx[k] = ca_contacts_idx

    def __len__(self):
        return len(self._residues)

    def sample_tuple(self, contact_type: str, n_pos: int = 1, n_neg: int = 1, residues_range: Tuple[int, int] = None):

        pos, neg = [], []

        if residues_range is not None:
            mn, mx = residues_range
        else:
            mn, mx = 0, len(self)

        mx = min(mx, len(self))
        indices = np.arange(mx - mn)

        for i, row in enumerate(self._contacts_idx[contact_type]):
            if not mn <= i < mx:
                continue

            # we only take residuals in the given range, and offset them
            new_row = [r - mn for r in row if mn <= r < mx]

            # If number of positives is less then total positive.
            # In that case we use evaluate the residual against itself
            if len(new_row) < n_pos:
                new_row += [i - mn] * (n_pos - len(new_row))
                # new_row += [-1] * (n_pos - len(new_row))

            pos_index = np.random.choice(new_row, n_pos)

            # zeroing the probability of positive indexes from the sampling list
            p = np.ones_like(indices)
            # applying the offset
            p[pos_index] = 0

            with_replace = True if n_neg > len(p) else False
            neg_index = np.random.choice(indices, n_neg, p=p / p.sum(), replace=with_replace)

            pos.append(pos_index)
            neg.append(neg_index)

        return np.asarray(pos), np.asarray(neg)

    def __call__(self,
                 contact_type: str = 'cb',
                 truncation_mode: str = None,
                 truncation_seq_length: int = None,
                 n_pos: int = 1, n_neg: int = 1,
                 to_sample: bool = True,
                 *args, **kwargs):

        assert contact_type in self._gt_matrix.keys(), f'contact_type must be in {list(self._gt_matrix.keys())}.'
        assert truncation_mode in ['random', 'cut']

        residues = self._residues
        label_matrix = self._gt_matrix[contact_type]

        if not to_sample:
            return {'residues': residues, 'label_matrix': label_matrix, 'length': len(self), 'offset': 0}
        else:

            res_range = None
            if truncation_seq_length is not None:
                if truncation_seq_length < len(self):
                    # mn, mx = 0, len(residues)
                    if truncation_mode == 'random':
                        mn = np.random.randint(0, min(len(self) - truncation_seq_length, len(self)))
                        mx = mn + truncation_seq_length
                    else:
                        mn, mx = 0, truncation_seq_length

                    residues = residues[mn:mx]
                    label_matrix = label_matrix[mn:mx, mn:mx]
                    res_range = (mn, mx)

                    assert len(residues) <= truncation_seq_length

            offset = 0 if res_range is None else res_range[0]

            pos_res, neg_res = self.sample_tuple(contact_type=contact_type,
                                                 n_pos=n_pos,
                                                 n_neg=n_neg,
                                                 residues_range=res_range)

            return {'residues': residues, 'label_matrix': label_matrix,
                    'pos_res': pos_res, 'neg_res': neg_res, 'length': len(self), 'offset': offset}


class ProteinDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 sample: bool,
                 contact_type: str = 'cb',
                 n_pos_sampled: int = 1,
                 n_neg_sampled: int = 1,
                 threshold: float = 8.0,
                 cache_dataset: bool = True,
                 n_files_to_use: int = -1,
                 truncation_seq_length: int = None,
                 truncation_mode: str = 'cut'):

        assert contact_type in ['ca', 'cb'], 'contact_type must be either "ca" or "cb"'

        self._truncation_seq_length = truncation_seq_length
        self._truncation_mode = truncation_mode
        self._n_pos_sampled = n_pos_sampled
        self._n_neg_sampled = n_neg_sampled
        self._to_sample = sample
        self._contact_type = contact_type

        self._proteins = []
        # save the files for debugging purposes
        self._files_names = []

        all_files = []

        for (dirpath, dirnames, filenames) in os.walk(dataset_path):
            all_files.extend([(dirpath, file)
                              for file in filenames if file.endswith('.pdb')])

        if n_files_to_use > 0:
            all_files = all_files[:n_files_to_use]

        self._files_names = all_files

        for dirpath, file in tqdm(all_files, leave=False, desc='Loading proteins...'):

            cached_path = os.path.join(dirpath, f'{file}'.replace('pdb', 'cache'))
            path = os.path.join(dirpath, file)

            protein = None

            if os.path.exists(cached_path) and cache_dataset:
                try:
                    with open(cached_path, 'rb') as file:
                        protein = pickle.load(file)
                except Exception:
                    protein = None

            if protein is None:
                parser = CustomParser(QUIET=True)
                structure = parser.get_structure("prot", path)
                chains = structure.get_chains()

                protein = Protein(chains, modres=parser.modres, contact_threshold=threshold)

                if cache_dataset:
                    with open(cached_path, 'wb') as file:
                        pickle.dump(protein, file)

            self._proteins.append(protein)

    @property
    def proteins(self) -> List[Protein]:
        return self._proteins

    def __getitem__(self, item):
        return self.proteins[item](truncation_mode=self._truncation_mode,
                                   truncation_seq_length=self._truncation_seq_length,
                                   n_pos=self._n_pos_sampled,
                                   n_neg=self._n_neg_sampled,
                                   to_sample=self._to_sample,
                                   contact_type=self._contact_type)

    def __len__(self) -> int:
        return len(self._proteins)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    all_files = []

    train_path = r'./dataset/train'

    dataset = ProteinDataset(train_path, sample=True)

    # parser = PDBParser(QUIET=True)
    # structure = parser.get_structure("prot", os.path.join(r'C:\Progetti\DeepOrigin\dataset\test\1DJA.pdb'))
    # chains = structure.get_chains()
    #
    # protein = Protein(chains)
    # pos, neg = protein.sample_tuple()
    # b1 = protein()
    #
    import esm

    # import torch
    #

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    loader = DataLoader(dataset, batch_size=8, collate_fn=get_protein_collate_fn(batch_converter))
    batch = next(iter(loader))

    import torch
    from loss import BinaryContrastiveLoss

    with torch.no_grad():
        results = model(batch['proteins'][-1], repr_layers=[model.num_layers], return_contacts=True)
        contact_map = results["contacts"][0]

        features = results['representations'][model.num_layers][:, 1:-1]

        loss = BinaryContrastiveLoss()(features, batch['pos_res'], batch['neg_res'])

        # I used [0:1, 0:1] to avoid removing dimensions that I need later
        zeros = torch.zeros_like(features[0:1, 0:1])

        pos_res = batch['pos_res'].expand(-1, -1, zeros.shape[-1])
        mask = (pos_res >= 0).unsqueeze(-1)

        safe_indices = pos_res.clamp(min=0)

        out = torch.gather(features, dim=1, index=safe_indices)  # (B, T, D)

        # pos_tokens = torch.gather(features, dim=1, index=batch['pos_res'].unsqueeze(0).expand(-1, -1, zeros.shape[-1])).detach()
        pos_tokens = torch.where(pos_res >= 0,
                                 # expand is more verbose but does not copy the data
                                 torch.gather(features, dim=1, index=pos_res.expand(-1, -1, zeros.shape[-1])),
                                 zeros).detach()

    a = 0

    #
    # # {'residues': self._residues, 'label_matrix': pos_matrix,
    # #  'pos_res': pos_res, 'neg_res': neg_res}
    #
    # batch = protein()
    #
    # data = [("protein1", batch['residues'])]
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    #
    # with torch.no_grad():
    #     results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=True)
    #     contact_map = results["contacts"][0].cpu()
    #
    #     # remove bos and eos
    #     features = results['representations'][model.num_layers][:, 1:-1]
#
#     pos_tokens = torch.gather(features, dim=1, index=torch.tensor(batch['pos_res']).unsqueeze(0).expand(-1, -1, features.shape[-1])).detach()
#     neg_tokens = torch.gather(features, dim=1, index=torch.tensor(batch['neg_res']).unsqueeze(0).expand(-1, -1, features.shape[-1])).detach()
#
#     pos_loss = torch.norm(features - pos_tokens, p=2, dim=-1)
#     neg_loss = 1 / (torch.norm(features - neg_tokens, p=2, dim=-1) + 1e-6)
#
#     loss = -(pos_loss + neg_loss).log()
#
#     a = 0
