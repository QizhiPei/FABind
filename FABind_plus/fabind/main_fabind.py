import numpy as np
import os
import torch
from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger, log_to_wandb
from torch.utils.data import RandomSampler
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from utils.parsing import parse_train_args
from models.model import get_model
import wandb
from utils.training import validate, train_one_epoch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_accelerator(args):
    """Setup the accelerator for distributed training and return it along with other configurations."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    train, valid, test = get_data(args, logger, addNoise=args.addNoise, use_whole_protein=args.use_whole_protein,
                                  compound_coords_init_mode=args.compound_coords_init_mode, pre=args.data_path)
    num_workers = args.num_workers
    logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    if args.sample_n > 0:
        sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
        train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], 
                                  sampler=sampler, pin_memory=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                                  shuffle=True, pin_memory=False, num_workers=num_workers)
    
    valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                              shuffle=False, pin_memory=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'],
                             shuffle=False, pin_memory=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader

def prepare_model_and_optimizer(args, accelerator, logger):
    """Prepare the model and optimizer for training."""
    model = get_model(args, logger)
    if accelerator.is_main_process:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"======= Number of trainable parameters: {num_trainable_params}")
    accelerator.wait_for_everyone()    
    if args.reload: # reload model from a checkpoint
        model.load_state_dict(torch.load(args.reload), strict=True)
    
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
    args, _ = parse_train_args()
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
    
    distmap_criterion = nn.MSELoss()
    if args.permutation_invariant:
        com_coord_criterion = nn.SmoothL1Loss(reduction='none')
    else:
        com_coord_criterion = nn.SmoothL1Loss()
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
    pocket_radius_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)

    for epoch in range(last_epoch+1, args.total_epochs):
        model.train()
        metrics = train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler, com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device)
        logger.log_stats(metrics, epoch, args, prefix="Train")
        if accelerator.is_main_process and args.wandb:
            log_to_wandb(metrics, "train", epoch)
        
        torch.cuda.empty_cache()
        
        if epoch % args.test_interval == 0:
            model.eval()
            if accelerator.is_main_process:
                logger.log_message(f"Begin validation with predicted pocket")            
                metrics = validate(accelerator, args, valid_loader, accelerator.unwrap_model(model), com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device, stage=2)
                logger.log_stats(metrics, epoch, args, prefix="Valid_pp")
                if args.wandb:
                    log_to_wandb(metrics, "valid_pp", epoch)
                    
                logger.log_message(f"Begin test")
                metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device, stage=1)
                logger.log_stats(metrics, epoch, args, prefix="Test_gt")
                if args.wandb:
                    log_to_wandb(metrics, "test_gt", epoch)
                    
                logger.log_message(f"Begin test with predicted pocket")
                metrics = validate(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, accelerator.device, stage=2)
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