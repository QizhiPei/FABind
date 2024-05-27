import numpy as np
import os

import torch

from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from utils.logging_utils import Logger
import sys
import argparse
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex
from utils.training_confidence import validate
from utils.parsing import parse_train_args

parser = argparse.ArgumentParser(description='FABind model testing.')

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
parser.add_argument("--write-mol-to-file", type=str, default=None)
parser.add_argument("--infer-logging", action='store_true', default=False)
parser.add_argument("--use-clustering", action='store_true', default=False)
parser.add_argument("--dbscan-eps", type=float, default=9.0)
parser.add_argument("--dbscan-min-samples", type=int, default=2)
parser.add_argument("--choose-cluster-prob", type=float, default=0.5)
parser.add_argument("--save-rmsd-dir", type=str, default=None)
parser.add_argument("--infer-dropout", action='store_true', default=False)
parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
parser.add_argument("--command", type=str, default=None)
parser.add_argument("--sample-size", type=int, default=1)

test_args = parser.parse_args()
_, train_parser = parse_train_args(test=True)
train_parser.add_argument("--stack-mlp", action='store_true', default=False)
train_parser.add_argument("--confidence-dropout", type=float, default=0.1)
train_parser.add_argument("--confidence-use-ln-mlp", action='store_true', default=False)
train_parser.add_argument("--confidence-mlp-hidden-scale", type=int, default=2)
train_parser.add_argument("--ranking-loss", type=str, default='logsigmoid', choices=['logsigmoid', 'dynamic_hinge'])
train_parser.add_argument("--num-copies", type=int, default=1)
train_parser.add_argument("--keep-cls-2A", action='store_true', default=False)


if test_args.command is not None:
    command = test_args.command
else:
    command = 'fabind/main_fabind.py --stack-mlp --confidence-dropout 0.2 --confidence-mlp-hidden-scale 1 --confidence-use-ln-mlp --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20'
command = shlex.split(command)

args = train_parser.parse_args(command[1:])
# print(vars(test_args))
for attr in vars(test_args):
    # Set the corresponding attribute in args
    setattr(args, attr, getattr(test_args, attr))
# Overwrite or set specific attributes as needed
args.tqdm_interval = 0.1
args.disable_tqdm = False
args.confidence_inference = True
args.confidence_training = True

set_seed(args.seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)

pre = f"{args.resultFolder}/{args.exp_name}"

accelerator.wait_for_everyone()

os.makedirs(pre, exist_ok=True)
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(1)

torch.multiprocessing.set_sharing_strategy('file_system')


train, valid, test= get_data(args, logger, addNoise=args.addNoise, use_whole_protein=args.use_whole_protein, compound_coords_init_mode=args.compound_coords_init_mode, pre=args.data_path)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
num_workers = 10

test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)

test_unseen_pdb_list = [line.strip() for line in open("split_pdb_id/unseen_test_index")]
# test_unseen_pdb_list = [line.strip() for line in open("../split_pdb_id/sw_0.8_unseen_test_index")]
test_unseen_index = test.data.query("(group =='test') and (pdb in @test_unseen_pdb_list)").index.values
# double check
test_unseen_index_for_select = np.array([np.where(test._indices == i) for i in test_unseen_index]).reshape(-1)
test_unseen = test.index_select(test_unseen_index_for_select)
test_unseen_loader = DataLoader(test_unseen, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)


from models.model import *
device = 'cuda'

model = get_model(args, logger)

model = accelerator.prepare(model)

model.load_state_dict(torch.load(args.ckpt), strict=False)

compound_confidence_criterion = nn.BCEWithLogitsLoss()

if args.infer_dropout:
    model.train()
    for name, submodule in model.named_modules():
        if name.startswith('confidence') or name.startswith('ranking'):
            submodule.eval()
else:
    model.eval()


logger.log_message(f"Begin test")
if accelerator.is_main_process:
    for epoch in range(args.sample_size):
        metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), compound_confidence_criterion, accelerator.device, epoch=epoch, stage=2)
        logger.log_stats(metrics, epoch, args, prefix="Test_pp")
    
accelerator.wait_for_everyone()


# compute metrics
N=1
rmsds_list = []
centroid_diss_list = []
confidences_list = []
for epoch in range(args.sample_size):
    with open(f'{args.save_rmsd_dir}/epoch{epoch}_confidence.txt', 'r') as f:
        lines = f.readlines()
        cnt = 0
        rmsds = []
        centroid_diss = []
        confidences = []
        for line in lines:
            centroid_diss.append(float(line.split(' ')[-2]))
            rmsds.append(float(line.split(' ')[-3]))
            confidences.append(float(line.split(' ')[-1]))
            cnt += 1
            if cnt == 363:
                break
        rmsds_list.append(rmsds)
        centroid_diss_list.append(centroid_diss)
        confidences_list.append(confidences)

rmsds_list = np.array(rmsds_list)
centroid_diss_list = np.array(centroid_diss_list)
confidences_list = np.array(confidences_list)

topn_choice = confidences_list.argsort(axis=0)[::-1][:N]
topn_rmsds = []
topn_centroid_diss = []

for i in range(len(test)):
    topn_temp_rmsds = []
    topn_temp_centroid_diss = []
    for j in range(N): # N=1, only pick the minimum value
        topn_temp_centroid_diss.append(centroid_diss_list[topn_choice[j][i]][i])
        topn_temp_rmsds.append(rmsds_list[topn_choice[j][i]][i])    
    
    topn_rmsds.append(min(topn_temp_rmsds))
    topn_centroid_diss.append(min(topn_temp_centroid_diss))
    
rmsd_tensor = np.array(topn_rmsds)
centroid_dis_tensor = np.array(topn_centroid_diss)

rmsd_mean = np.mean(rmsd_tensor)
rmsd_2A = np.sum(rmsd_tensor < 2) / 363
rmsd_5A = np.sum(rmsd_tensor < 5) / 363
rmsd_25 = np.quantile(rmsd_tensor, 0.25)
rmsd_50 = np.quantile(rmsd_tensor, 0.50)
rmsd_75 = np.quantile(rmsd_tensor, 0.75)
centroid_dis_mean = np.mean(centroid_dis_tensor)
centroid_dis_2A = np.sum(centroid_dis_tensor < 2) / 363
centroid_dis_5A = np.sum(centroid_dis_tensor < 5) / 363
centroid_dis_25 = np.quantile(centroid_dis_tensor, 0.25)
centroid_dis_50 = np.quantile(centroid_dis_tensor, 0.50)
centroid_dis_75 = np.quantile(centroid_dis_tensor, 0.75)

logger.log_message(f'rmsd_mean: {rmsd_mean}')
logger.log_message(f'rmsd_2A: {rmsd_2A}')
logger.log_message(f'rmsd_5A: {rmsd_5A}')
logger.log_message(f'rmsd_25: {rmsd_25}')
logger.log_message(f'rmsd_50: {rmsd_50}')
logger.log_message(f'rmsd_75: {rmsd_75}')
logger.log_message(f'centroid_dis_mean: {centroid_dis_mean}')
logger.log_message(f'centroid_dis_2A: {centroid_dis_2A}')
logger.log_message(f'centroid_dis_5A: {centroid_dis_5A}')
logger.log_message(f'centroid_dis_25: {centroid_dis_25}')
logger.log_message(f'centroid_dis_50: {centroid_dis_50}')
logger.log_message(f'centroid_dis_75: {centroid_dis_75}')