import os
from accelerate.logging import get_logger
import logging
import wandb


class Logger:
    def __init__(self, accelerator, log_path):
        self.logger = get_logger('Main')

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s', ""))
        self.logger.logger.addHandler(handler)
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

    def log_stats(self, stats, epoch, args, prefix=''):
        msg_start = f'[{prefix}] Epoch {epoch} out of {args.total_epochs}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.5f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)
        
def log_to_wandb(metrics, phase, epoch):
    """
    Log metrics to wandb.

    Parameters:
    - metrics (dict): A dictionary containing the metrics to log.
    - phase (str): A string indicating the phase, e.g., 'train' or 'valid'.
    - epoch (int): The current epoch number.
    """
    log_dict = {}
    for key, value in metrics.items():
        log_dict[f"{phase}/{key}"] = value
    log_dict["epoch"] = epoch
    wandb.log(log_dict)
