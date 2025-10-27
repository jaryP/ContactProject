import os
import pickle

from collections.abc import Generator
from typing import List, Tuple, Union, Dict

import numpy as np
from Bio.PDB.Structure import Structure
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import CustomParser, MAPPING


def extend(a, b, c, L, A, D):
    """
    Taken from the official repository

    https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb#scrollTo=n4FBkOUaWuXb&line=1&uniqifier=1

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


def calculate_contacts(positions: np.ndarray, contact_threshold: float = 8) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    This function calculates the contact indices and the residues ground truth for each redisuals position
    :param positions: the positions in the space of the redisuals
    :param contact_threshold: the threshold (in Amstrong unit) for which two residues are considered in contact
    :return: the first returned value is a list containing the contact indices for each residue,
        while the second one is the binary ground truth matrix.
    """

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
                 contact_threshold: float = 8.0):
        """
        The Protein class. Using this class is possible to sample a subset of the stricture or all the chains.
        It calculates both Calpha and Cbeta distances. Which one to use can be selected when accessing the data.

        :param chains: a generator returning the chains in the Structure.
            For this project, we assume that the all the chains belong to the same protein
        :param modres:
        :param contact_threshold: the threshold (in Amstrong unit) for which two residues are considered in contact
        """

        assert contact_threshold > 0, 'contact_threshold must be positive'

        if modres is None:
            modres = {}

        self._residues = ''

        self._residues_chain_id = []
        _residues_coord = {'ca': [], 'cb': []}

        for chain in chains:
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
            ca_contacts_idx, ca_matrix = calculate_contacts(np.asarray(v), contact_threshold=contact_threshold)

            self._gt_matrix[k] = ca_matrix
            self._contacts_idx[k] = ca_contacts_idx

    def __len__(self):
        return len(self._residues)

    def sample_tuple(self,
                     contact_type: str,
                     n_pos: int = 1, n_neg: int = 1,
                     residues_range: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Allow for sampling positive and negative residues for each item in the chain.

        :param contact_type: Which one between Calpha (ca) or Cbeta (cb) to use
        :param n_pos: number of positive residues (with contact) to sample
        :param n_neg: number of positive residues (without contact) to sample
        :param residues_range: which subset of the chain must be extract. The indexes will be returned only within it.
        :return: two lists, containing respectively positive and negative indexes in the chain
        """

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

            # we only take residues in the given range, and offset them
            row_subset = [r - mn for r in row if mn <= r < mx]

            # If number of positives is less than total positive.
            # In that case we use evaluate the residue against itself
            if len(row_subset) < n_pos:
                row_subset += [-1] * (n_pos - len(row_subset))

            pos_index = np.random.choice(row_subset, n_pos)

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
                 *args, **kwargs) -> Dict[str, Union[np.ndarray, int, float]]:

        """
        The function that returns a dictionary containing all the information about the protein

        :param contact_type:  Which one between Calpha (ca) or Cbeta (cb) to use
        :param truncation_mode:  How the chain must be truncated.
            Can be cut or random. In the first case the chain is truncated at position truncation_seq_length,
            in the latter a random sub-chain of length truncation_seq_length is sampled.
        :param truncation_seq_length: the returned lenght of the chain, if the original one is longer.
        :param n_pos: number of positive residues (with contact) to sample
        :param n_neg: number of positive residues (without contact) to sample
        :param to_sample: if positive and negative indexes must be sampled
        :return: a dictionary so defined:
                    {'residues': the residues of the chain,
                    'gt_matrix': the binary ground truth matrix,
                    'length': the length of the sequence,
                    'offset': the offset of the sub-sequence, in case truncation_mode=random. Otherwise zero.}
        """

        assert contact_type in self._gt_matrix.keys(), f'contact_type must be in {list(self._gt_matrix.keys())}.'
        assert truncation_mode in ['random', 'cut']

        residues = self._residues
        label_matrix = self._gt_matrix[contact_type]

        res_range = None
        if truncation_seq_length is not None:
            if truncation_seq_length < len(self):

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

        ret = {'residues': residues, 'gt_matrix': label_matrix, 'length': len(self), 'offset': offset}

        if to_sample:
            pos_indexes, neg_indexes = self.sample_tuple(contact_type=contact_type,
                                                         n_pos=n_pos,
                                                         n_neg=n_neg,
                                                         residues_range=res_range)

            ret.update({'pos_indexes': pos_indexes, 'neg_indexes': neg_indexes})

        return ret


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
        """
        A dataset which is a collection of proteins. All the info used to get the proteins are stored
        whithin this class to avoid errors when moving a dataset around.

        :param dataset_path: The path of the protein files
        :param sample: if the positive-negative indexes must be sampled from the protein
        :param contact_type: Which one between Calpha (ca) or Cbeta (cb) to use
        :param n_pos_sampled:  number of positive residues (with contact) to sample
        :param n_neg_sampled: number of positive residues (without contact) to sample
        :param threshold: the threshold (in Amstrong unit) for which two residues are considered in contact
        :param cache_dataset: id the processed proteins must be stored on disk using Pickle
        :param n_files_to_use: the number of files to process. For debugging purposes.
        :param truncation_seq_length: the returned length of the chain, if the original one is longer.
        :param truncation_mode: How the chain must be truncated.
            Can be cut or random. In the first case the chain is truncated at position truncation_seq_length,
            in the latter a random sub-chain of length truncation_seq_length is sampled.
        """

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

    def __getitem__(self, item: int) -> Dict[str, Union[np.ndarray, int, float]]:
        """
        :param item: index of the protein to extract
        :return: See Protein.__call__ for details.
        """

        return self.proteins[item](truncation_mode=self._truncation_mode,
                                   truncation_seq_length=self._truncation_seq_length,
                                   n_pos=self._n_pos_sampled,
                                   n_neg=self._n_neg_sampled,
                                   to_sample=self._to_sample,
                                   contact_type=self._contact_type)

    def __len__(self) -> int:
        return len(self._proteins)
