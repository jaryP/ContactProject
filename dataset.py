import os
import pickle

from collections.abc import Generator
from typing import List, Tuple

import numpy as np
from Bio.PDB.Structure import Structure
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


class Protein:
    def __init__(self, chains: Generator[Structure],
                 # beta_sheets,
                 # alpha_helices,
                 modres: dict[int, str] = None,
                 contact_threshold: float = 8.0,
                 atom_to_use: str = 'CA',
                 testing_protein: bool = False):

        assert contact_threshold > 0, 'contact_threshold must be positive'
        assert atom_to_use == 'CA', 'Now only CA atom is supported.'

        if modres is None:
            modres = {}

        self._testing_protein = testing_protein

        self._residues = []
        self._residues_pos = []

        self._residues_chain_id = []

        for chain in chains:
            chain_residues = []
            chain_residues_pos = []

            for ri, r in enumerate(chain):
                if atom_to_use not in r:
                    continue

                r_coord = r[atom_to_use].coord
                r_name = r.resname

                if r_name not in MAPPING or r.id[0].startswith('H_'):
                    # let's skip non-standard residues from HET lines
                    # but not the modified amino acids (if modres line is present)

                    r_name = modres.get(ri + 1, None)

                # manages edge cases (e.g., file 5OL5.pdb has incomplete modres information)
                if r_name is not None and r_name != '':
                    chain_residues.append(MAPPING[r_name])
                    chain_residues_pos.append(r_coord)

            self._residues_chain_id.append((chain.id, len(chain_residues)))

            self._residues.extend(chain_residues)
            self._residues_pos.extend(chain_residues_pos)

        self._residues = ''.join(self._residues)
        self._residues_pos = np.asarray(self._residues_pos)

        dist_matrix = np.linalg.norm(self._residues_pos[:, None, :] - self._residues_pos[None, :, :], axis=-1)
        dist_matrix -= (contact_threshold + 1) * np.eye(len(dist_matrix))

        # we set to zero values that are higher than the contact_threshold
        # the next line should be the fastest way to do it
        dist_matrix[np.logical_or(dist_matrix < 0, dist_matrix > contact_threshold)] = 0
        # dist_matrix = np.maximum(dist_matrix, 0, dist_matrix)

        self._contact_indices = [np.nonzero(row)[0] for row in dist_matrix]
        self._gt_matrix = (dist_matrix > 0).astype(float)

    def __len__(self):
        return len(self._residues)

    def sample_tuple(self, n_pos: int = 1, n_neg: int = 1, residues_range: Tuple[int, int] = None):

        pos, neg = [], []

        if residues_range is not None:
            mn, mx = residues_range
        else:
            mn, mx = 0, len(self)

        mx = min(mx, len(self))
        indices = np.arange(mx - mn)

        for i, row in enumerate(self._contact_indices):
            if not mn <= i < mx:
                continue

            # we only take residuals in the given range, and offset them
            new_row = [r - mn for r in row if mn <= r < mx]

            # If number of positives is less then total positive.
            # In that case we use evaluate the residual against itself
            if len(new_row) < n_pos:
                new_row += [i - mn] * (n_pos - len(new_row))

            # if len(new_row) == 0:
            #     # there are no positives. In that case we use evaluate the residual against itself
            #     pos_index = np.asarray([i - mn])
            # else:
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
                 truncation_mode: str = None,
                 truncation_seq_length: int = None,
                 n_pos: int = 1, n_neg: int = 1,
                 *args, **kwargs):

        assert truncation_mode in ['random', 'cut']

        residues = self._residues
        label_matrix = self._gt_matrix

        res_range = None
        if truncation_seq_length is not None:
            if truncation_seq_length < len(residues):
                # mn, mx = 0, len(residues)
                if truncation_mode == 'random':
                    mn = np.random.randint(0, max(len(residues) - truncation_seq_length, len(residues)))
                    mx = mn + truncation_seq_length
                else:
                    mn, mx = 0, truncation_seq_length

                residues = residues[mn:mx]
                label_matrix = label_matrix[mn:mx, mn:mx]
                res_range = (mn, mx)

        if self._testing_protein:
            return {'residues': residues, 'label_matrix': label_matrix, 'length': len(residues)}
        else:
            pos_res, neg_res = self.sample_tuple(n_pos=n_pos, n_neg=n_neg, residues_range=res_range)
            return {'residues': residues, 'label_matrix': label_matrix,
                    'pos_res': pos_res, 'neg_res': neg_res, 'length': len(residues)}


class ProteinDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 training: bool,
                 n_pos_sampled: int = 1,
                 n_neg_sampled: int = 1,
                 threshold: float = 8.0,
                 cache_dataset: bool = True,
                 n_files_to_use: int = -1,
                 truncation_seq_length: int=None,
                 truncation_mode: str = 'cut'):

        self._truncation_seq_length = truncation_seq_length
        self._truncation_mode = truncation_mode
        self._n_pos_sampled = n_pos_sampled
        self._n_neg_sampled = n_neg_sampled

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

                protein = Protein(chains, modres=parser.modres, contact_threshold=threshold,
                                  testing_protein=not training)

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
                                   n_pose=self._n_pos_sampled,
                                   n_neg=self._n_neg_sampled)

    def __len__(self) -> int:
        return len(self._proteins)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    all_files = []

    train_path = r'./dataset/train'

    dataset = ProteinDataset(train_path, training=True)

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
