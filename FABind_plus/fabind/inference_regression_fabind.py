import os
import torch
from torch_geometric.loader import DataLoader
from utils.logging_utils import Logger
import sys
import argparse
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex
import time

from tqdm import tqdm

from utils.fabind_inference_dataset import InferenceDataset
from utils.inference_mol_utils import write_mol
from utils.post_optim_utils import post_optimize_compound_coords
import pandas as pd
from utils.parsing import parse_train_args


parser = argparse.ArgumentParser(description='Train your own TankBind model.')

parser.add_argument("--ckpt", type=str, default='../checkpoints/pytorch_model.bin')
parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                    help="Data path.")
parser.add_argument("--resultFolder", type=str, default="./result",
                    help="information you want to keep a record.")
parser.add_argument("--exp-name", type=str, default="",
                    help="data path.")
parser.add_argument('--seed', type=int, default=600,
                    help="seed to use.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size.")
parser.add_argument("--write-mol-to-file", action='store_true', default=False)
parser.add_argument('--sdf-to-mol2', action='store_true', default=False)
parser.add_argument("--infer-logging", action='store_true', default=False)
parser.add_argument("--use-clustering", action='store_true', default=False)
parser.add_argument("--dbscan-eps", type=float, default=9.0)
parser.add_argument("--dbscan-min-samples", type=int, default=2)
parser.add_argument("--choose-cluster-prob", type=float, default=0.5)
parser.add_argument("--save-rmsd-dir", type=str, default=None)
parser.add_argument("--infer-dropout", action='store_true', default=False)
parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
parser.add_argument("--command", type=str, default=None)

parser.add_argument("--post-optim", action='store_true', default=False)
parser.add_argument('--post-optim-mode', type=int, default=0)
parser.add_argument('--post-optim-epoch', type=int, default=1000)
parser.add_argument('--sdf-output-path-post-optim', type=str, default="")

parser.add_argument('--index-csv', type=str, default=None)
parser.add_argument('--pdb-file-dir', type=str, default="")
parser.add_argument('--preprocess-dir', type=str, default="")


infer_args = parser.parse_args()
_, train_parser = parse_train_args(test=True)

command = "fabind/main_fabind.py --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20"
command = shlex.split(command)

args = train_parser.parse_args(command[1:])
# print(vars(infer_args))
for attr in vars(infer_args):
    # Set the corresponding attribute in args
    setattr(args, attr, getattr(infer_args, attr))
# Overwrite or set specific attributes as needed
args.tqdm_interval = 0.1
args.disable_tqdm = False

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)

pre = f"{args.resultFolder}/{args.exp_name}"


os.makedirs(args.sdf_output_path_post_optim, exist_ok=True)
os.makedirs(pre, exist_ok=True)
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(16)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')


def post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_offset_per_sample_list, com_coord_per_sample_list, compound_batch, LAS_tmp, rigid=False):
    post_optim_device='cpu'
    for i in range(compound_batch.max().item()+1):
        i_mask = (compound_batch == i)
        com_coord_pred_i = com_coord_pred[i_mask]
        com_coord_i = data[i]['compound'].rdkit_coords

        com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)
        
        if args.post_optim:
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                # LAS_edge_index=(data[i]['complex', 'LAS', 'complex'].edge_index - data[i]['complex', 'LAS', 'complex'].edge_index.min()).to(post_optim_device),
                LAS_edge_index=LAS_tmp[i].to(post_optim_device),
                mode=args.post_optim_mode,
                total_epoch=args.post_optim_epoch,
            )
            predict_coord = predict_coord.to(accelerator.device)
            predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i
            com_coord_pred[i_mask] = predict_coord
        
        com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])
        com_coord_per_sample_list.append(com_coord_i)
        com_coord_offset_per_sample_list.append(data[i].coord_offset)
        
        mol_list.append(data[i].mol)
        uid_list.append(data[i].uid)
        smiles_list.append(data[i]['compound'].smiles)
        sdf_name_list.append(data[i].ligand_id + '.sdf')

    return


dataset = InferenceDataset(infer_args.index_csv, infer_args.pdb_file_dir, infer_args.preprocess_dir)
logger.log_message(f"data point: {len(dataset)}")
num_workers = 0
data_loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=num_workers)

device = 'cuda'
from models.model import *
model = get_model(args, logger)

model = accelerator.prepare(model)

model.load_state_dict(torch.load(args.ckpt))

set_seed(args.seed)

model.eval()

logger.log_message(f"Begin inference")
start_time = time.time()  # 记录开始时间

y_list = []
y_pred_list = []
com_coord_list = []
com_coord_pred_list = []
com_coord_per_sample_list = []

uid_list = []
smiles_list = []
sdf_name_list = []
mol_list = []
com_coord_pred_per_sample_list = []
com_coord_offset_per_sample_list = []

data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
for batch_id, data in enumerate(data_iter):
    try:
        data = data.to(device)
        LAS_tmp = []
        for i in range(len(data)):
            LAS_tmp.append(data[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())
        with torch.no_grad():
            
            com_coord_pred, compound_batch = model.inference(data)        
        post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_offset_per_sample_list, com_coord_per_sample_list, compound_batch, LAS_tmp=LAS_tmp)
    except Exception as e:
        print(e)
        continue

if args.sdf_to_mol2:
    from utils.sdf_to_mol2 import convert_sdf_to_mol2
    
if args.write_mol_to_file:
    info = pd.DataFrame({'uid': uid_list, 'smiles': smiles_list, 'sdf_name': sdf_name_list})
    info.to_csv(os.path.join(args.sdf_output_path_post_optim, f"uid_smiles_sdfname.csv"), index=False)
    for i in tqdm(range(len(info))):
        
        save_coords = com_coord_pred_per_sample_list[i] + com_coord_offset_per_sample_list[i]
        sdf_output_path = os.path.join(args.sdf_output_path_post_optim, info.iloc[i]['sdf_name'])
        mol = write_mol(reference_mol=mol_list[i], coords=save_coords, output_file=sdf_output_path)
        if args.sdf_to_mol2:
            convert_sdf_to_mol2(sdf_output_path, sdf_output_path.replace('.sdf', '.mol2'))        
        

end_time = time.time()  # 记录开始时间
logger.log_message(f"End infer, time spent: {end_time - start_time}")

