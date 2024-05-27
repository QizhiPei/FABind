import pandas as pd
from tqdm import tqdm
import torch
import os
import rdkit.Chem as Chem
import copy
fabind_src_folder_path = "../../fabind"
import sys
sys.path.insert(0, fabind_src_folder_path)
from utils.isomorphism import isomorphic_core

data_raw = 'fabind-neurips/dataset/processed/data.pt'
renumbered_mol_dir = 'fabind-neurips/renumber_atom_index_same_as_smiles'
data_to_save = 'fabind-neurips/dataset/processed/data_new.pt'

def read_mol(sdf_fileName, verbose=False):
    # Chem.WrapLogs()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True

    return mol, problem

data = torch.load(data_raw)

isomorphic_dict = {}
for idx in tqdm(range(len(data))):
    line = data.iloc[idx]
    if not line['use_compound_com']:
        isomorphic_dict[idx] = []
        continue
    pdb = line['pdb']
    mol_file = os.path.join(renumbered_mol_dir, f'{pdb}.sdf')
    rdkit_mol, _ = read_mol(mol_file)
    isomorphic_dict[idx] = isomorphic_core(rdkit_mol)

data_dict = data.to_dict(orient='dict')
data_dict.update({'isomorphics': isomorphic_dict})
data_new = pd.DataFrame(data_dict)

torch.save(data_new, data_to_save)