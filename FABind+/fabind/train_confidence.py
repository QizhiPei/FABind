import numpy as np
import os
import torch
from data import FABindDataSet
from utils.confidence_dataloader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger, log_to_wandb
import argparse
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from utils.parsing import parse_train_args
from models.model import get_model
import wandb
from utils.training_confidence import validate, train_one_epoch
import shlex

def parse_arguments():
    from utils.parsing import parse_train_args

    parser = argparse.ArgumentParser(description='FABind confidence model training.')
    
    # training args
    parser.add_argument("--reload", type=str, default='../checkpoints/pytorch_model.bin')
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
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument("--warmup-epochs", type=int, default=15,
                        help="used in combination with relative_k_mode.")
    parser.add_argument("--total-epochs", type=int, default=400,
                        help="option to switch training data after certain epochs.")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])
    parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--wandb", action='store_true', default=False)

    # sampling args
    parser.add_argument("--use-clustering", action='store_true', default=False)
    parser.add_argument("--dbscan-eps", type=float, default=9.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--choose-cluster-prob", type=float, default=0.5)
    parser.add_argument("--infer-dropout", action='store_true', default=False)
    
    # confidence args
    parser.add_argument("--confidence-training", action='store_true', default=False)
    parser.add_argument("--stack-mlp", action='store_true', default=False)
    parser.add_argument("--confidence-inference", action='store_true', default=False)
    parser.add_argument("--confidence-dropout", type=float, default=0.1)
    parser.add_argument("--confidence-use-ln-mlp", action='store_true', default=False)
    parser.add_argument("--confidence-mlp-hidden-scale", type=int, default=2)
    parser.add_argument("--num-copies", type=int, default=1)
    parser.add_argument("--ranking-loss", type=str, default='logsigmoid', choices=['logsigmoid', 'dynamic_hinge'])
    parser.add_argument("--keep-cls-2A", action='store_true', default=False)

    confidence_args = parser.parse_args()
    _, train_parser = parse_train_args(test=True)

    if confidence_args.command is not None:
        command = confidence_args.command
    else:
        command = 'fabind/main_fabind.py --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20'
    command = shlex.split(command)

    args = train_parser.parse_args(command[1:])
    for attr in vars(confidence_args):
        # Set the corresponding attribute in args
        setattr(args, attr, getattr(confidence_args, attr))
    # Overwrite or set specific attributes as needed
    args.confidence_training = True
    
    return args
        
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_accelerator(args):
    """Setup the accelerator for distributed training and return it along with other configurations."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    return accelerator

def setup_wandb_and_logging(accelerator, args, root_path, project_name="fabind-v2.2"):
    """Setup Weights & Biases logging."""
    if accelerator.is_main_process and args.wandb:
        wandb.init(
            project=project_name,
            name=args.exp_name,
            group=args.resultFolder.split('/')[-1],
            id=args.exp_name,
            config=args
        )
    accelerator.wait_for_everyone()
    
    """Setup logging and output directories."""
    logger = Logger(accelerator=accelerator, log_path=f'{root_path}/{datetime.now().strftime("%Y_%m_%d_%H_%M")}.log')
    return logger    

def prepare_data_loaders(args, logger):
    """Prepare data loaders for training, validation, and testing."""
    logger.log_message(f"Loading dataset")
    add_noise_to_com = float(args.addNoise) if args.addNoise else None
    pre = args.data_path

    new_dataset = FABindDataSet(f"{pre}/dataset", add_noise_to_com=add_noise_to_com, use_whole_protein=args.use_whole_protein, compound_coords_init_mode=args.compound_coords_init_mode, pocket_radius=args.pocket_radius, noise_for_predicted_pocket=args.noise_for_predicted_pocket, 
                                        test_random_rotation=args.test_random_rotation, pocket_idx_no_noise=args.pocket_idx_no_noise, use_esm2_feat=args.use_esm2_feat, seed=args.seed, pre=pre, args=args)
    # load compound features extracted using torchdrug.
    # c_length: number of atoms in the compound
    # This filter may cause some samples to be filtered out. So the actual number of samples is less than that in the original papers.        
    
    train_tmp = new_dataset.data.query("c_length < 150 and native_num_contact > 5 and protein_length < 1000 and group =='train' and use_compound_com").reset_index(drop=True)
    valid_test_tmp = new_dataset.data.query("(group == 'valid' or group == 'test') and use_compound_com").reset_index(drop=True)
    new_dataset.data = pd.concat([train_tmp, valid_test_tmp], axis=0).reset_index(drop=True)
    d = new_dataset.data
    train_val_index = d.query("group =='train' or group =='valid'").index.values
    train = new_dataset[train_val_index]
    valid_index = d.query("group =='valid'").index.values
    valid = new_dataset[valid_index]
    test_index = d.query("group =='test'").index.values
    test = new_dataset[test_index]

    num_workers = args.num_workers
    logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                                shuffle=True, pin_memory=False, num_workers=num_workers, num_copies=args.num_copies)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                              shuffle=False, pin_memory=False, num_workers=num_workers, num_copies=args.num_copies)
    test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                             shuffle=False, pin_memory=False, num_workers=num_workers, num_copies=args.num_copies)
    
    return train_loader, valid_loader, test_loader

def prepare_model_and_optimizer(args, accelerator, logger):
    """Prepare the model and optimizer for training."""
    model = get_model(args, logger)
    assert args.reload is not None, "Please provide a checkpoint to reload the model from."
    with accelerator.main_process_first():
        print("Training confidence model only. Loading pretrained model")
        print('ckpt is {}'.format(args.reload))
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.reload), strict=False)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")
        for name,p in model.named_parameters():
            p.requires_grad = False
            if "confidence" in name or "ranking" in name:
                p.requires_grad = True
    
    if accelerator.is_main_process:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"======= Number of trainable parameters: {num_trainable_params}")
    accelerator.wait_for_everyone()    
    
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return model, optimizer

def prepare_scheduler(args, optimizer, train_loader, logger):
    """Prepare the learning rate scheduler for training."""
    last_epoch = -1
    steps_per_epoch = len(train_loader)
    total_training_steps = args.total_epochs * len(train_loader)
    scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.warmup_epochs * len(train_loader) / args.gradient_accumulate_step,
        last_epoch=last_epoch,
    )
    if args.lr_scheduler == "constant":
        scheduler_post = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=(args.total_epochs - args.warmup_epochs)*len(train_loader)/args.gradient_accumulate_step, last_epoch=last_epoch)
    elif args.lr_scheduler == "poly_decay":
        scheduler_post = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=(args.total_epochs - args.warmup_epochs)*len(train_loader)/args.gradient_accumulate_step, last_epoch=last_epoch)
    elif args.lr_scheduler == "exp_decay":
        scheduler_post = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, last_epoch=last_epoch)
    elif args.lr_scheduler == "cosine_decay":
        scheduler_post = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.total_epochs - args.warmup_epochs)*len(train_loader), eta_min=1e-5, last_epoch=last_epoch)
    elif args.lr_scheduler == "cosine_decay_restart":
        scheduler_post = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.0001, last_epoch=last_epoch)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warm_up, scheduler_post],
        milestones=[args.warmup_epochs*len(train_loader)/args.gradient_accumulate_step],
    )
    
    logger.log_message(f"Total epochs: {args.total_epochs}")
    logger.log_message(f"Total training steps: {total_training_steps}")
    
    return scheduler, last_epoch, steps_per_epoch

def main():
    args = parse_arguments()
    assert args.batch_size == 1, "confidence training only support batch size 1"
    root_path = f"{args.resultFolder}/{args.exp_name}"
    os.system(f"mkdir -p {root_path}/models") # log files and save ckpts
    
    accelerator = setup_accelerator(args)
    logger = setup_wandb_and_logging(accelerator, args, root_path)
    train_loader, valid_loader, test_loader = prepare_data_loaders(args, logger)
    model, optimizer = prepare_model_and_optimizer(args, accelerator, logger)
    scheduler, last_epoch, steps_per_epoch = prepare_scheduler(args, optimizer, train_loader, logger)
    (
        model,
        optimizer,
        scheduler,
        train_loader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_loader,
    )

    output_last_epoch_dir = f"{root_path}/models/epoch_last"
    if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "pytorch_model.bin")):
        accelerator.load_state(output_last_epoch_dir)
        last_epoch = round(scheduler.state_dict()['last_epoch'] / steps_per_epoch) - 1
        logger.log_message(f'Load model from epoch: {last_epoch}')    
    
    compound_confidence_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(last_epoch+1, args.total_epochs):
        model.train()
        metrics = train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler, compound_confidence_criterion, accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Train")
        if accelerator.is_main_process and args.wandb:
            log_to_wandb(metrics, "train", epoch)
        
        torch.cuda.empty_cache()
        
        if epoch % args.test_interval == 0:
            for name, submodule in model.module.named_modules():
                if name.startswith('confidence') or name.startswith('ranking'):
                    # if accelerator.is_main_process:
                    #     print(name)
                    submodule.eval()
            if accelerator.is_main_process:
                logger.log_message(f"Begin validation with predicted pocket")            
                metrics = validate(accelerator, args, valid_loader, accelerator.unwrap_model(model), compound_confidence_criterion, accelerator.device, stage=2)
                logger.log_stats(metrics, epoch, args, prefix="Valid_pp")
                if args.wandb:
                    log_to_wandb(metrics, "valid_pp", epoch)
                    
                logger.log_message(f"Begin test")
                metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), compound_confidence_criterion, accelerator.device, stage=1)
                logger.log_stats(metrics, epoch, args, prefix="Test_gt")
                if args.wandb:
                    log_to_wandb(metrics, "test_gt", epoch)
                    
                logger.log_message(f"Begin test with predicted pocket")
                metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), compound_confidence_criterion, accelerator.device, stage=2)
                logger.log_stats(metrics, epoch, args, prefix="Test_pp")
                if args.wandb:
                    log_to_wandb(metrics, "test_pp", epoch)
                    
                # Save the model
                output_dir = f"{root_path}/models/epoch_{epoch}"
                accelerator.save_state(output_dir=output_dir, safe_serialization=False)
                accelerator.save_state(output_dir=output_last_epoch_dir, safe_serialization=False)
            
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
