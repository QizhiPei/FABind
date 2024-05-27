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
from utils.training import validate
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
parser.add_argument("--save-rmsd-dir", type=str, default=None)
parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
parser.add_argument("--command", type=str, default=None)

test_args = parser.parse_args()
_, train_parser = parse_train_args(test=True)

if test_args.command is not None:
    command = test_args.command
else:
    command = 'fabind/main_fabind.py --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20'
command = shlex.split(command)

args = train_parser.parse_args(command[1:])
# print(vars(test_args))
for attr in vars(test_args):
    # Set the corresponding attribute in args
    setattr(args, attr, getattr(test_args, attr))
# Overwrite or set specific attributes as needed
args.tqdm_interval = 0.1
args.disable_tqdm = False

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

if args.pred_dis:
    criterion = nn.MSELoss()
    pred_dis = True
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))

if args.permutation_invariant:
    if args.coord_loss_function == 'MSE':
        com_coord_criterion = nn.MSELoss(reduction='none')
    elif args.coord_loss_function == 'SmoothL1':
        com_coord_criterion = nn.SmoothL1Loss(reduction='none')
else:
    if args.coord_loss_function == 'MSE':
        com_coord_criterion = nn.MSELoss()
    elif args.coord_loss_function == 'SmoothL1':
        com_coord_criterion = nn.SmoothL1Loss()

if args.pocket_cls_loss_func == 'bce':
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
pocket_radius_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)


model.eval()

logger.log_message(f"Begin test")
if accelerator.is_main_process:
    metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_pp")

if accelerator.is_main_process:
    metrics = validate(accelerator, args, test_unseen_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_pp")
    
accelerator.wait_for_everyone()
