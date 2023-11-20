import torch
from tqdm import tqdm
import os
import argparse
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature


parser = argparse.ArgumentParser(description='Preprocess protein.')
parser.add_argument("--pdb_file_dir", type=str, default="../inference_examples/pdb_files",
                    help="Specify the pdb data path.")
parser.add_argument("--save_pt_dir", type=str, default="../inference_examples",
                    help="Specify where to save the processed pt.")
args = parser.parse_args()

esm2_dict = {}
protein_dict = {}

for pdb_file in tqdm(os.listdir(args.pdb_file_dir)):
    pdb = pdb_file.split(".")[0]

    pdb_filepath = os.path.join(args.pdb_file_dir, pdb_file)
    protein_structure = extract_protein_structure(pdb_filepath)
    protein_structure['name'] = pdb
    esm2_dict[pdb] = extract_esm_feature(protein_structure)
    protein_dict[pdb] = protein_structure

torch.save([esm2_dict, protein_dict], os.path.join(args.save_pt_dir, 'processed_protein.pt'))
