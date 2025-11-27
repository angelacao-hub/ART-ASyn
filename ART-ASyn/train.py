import os
import re
import sys
import time
import argparse
from utils import logger
from pathlib import Path
from configs import config


import torch
from utils.train_util import TrainLoop
from utils.script_util import create_model
from utils.datasets import get_data_loader
sys.path.append(str(Path.cwd()))


def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)

    time_load_start = time.time()

    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = "checkpoints" + '_' + args.dataset

    logger.configure(Path(experiment_name), format_strs=["log", "stdout", "csv"])

    logger.log("Creating model ...")
    model = create_model(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device(config.device)
    model.to(device)
    logger.log(f"    Model number of parameters {pytorch_total_params}")

    start_step = 0
    if args.model_path is not None:
        logger.log(f"Loading checkpoint {args.model_path} ...")
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        match = re.search(r"model(\d+)\.pt", args.model_path)
        if match:
            start_step = int(match.group(1))
            logger.configure(Path(experiment_name), format_strs=["log", "stdout", "csv"], log_suffix=f"{start_step}+")

    logger.log("creating data loader...")
    train_loader = get_data_loader(args.data_dir, 'train', args.batch_size, generator=True)
    time_load_end = time.time()
    time_load = time_load_end - time_load_start
    logger.log("data loaded: time ", str(time_load))
    logger.log("training...")
    TrainLoop(
        model=model,
        data=train_loader,
        lr=config.model.training.lr,
        start_step=start_step,
        ema_rate=config.model.training.ema_rate,
        log_interval=config.model.training.log_interval,
        save_interval=args.save_interval,
        use_fp16=config.model.training.use_fp16,
        fp16_scale_growth=config.model.training.fp16_scale_growth,
        weight_decay=config.model.training.weight_decay,
        lr_decay_steps=config.model.training.lr_decay_steps,
        lr_decay_factor=config.model.training.lr_decay_factor,
        iterations=config.model.training.iterations,
        num_input_channels=config.model.num_input_channels,
        image_size=config.model.image_size,
        device=device,
        args=args
    ).run_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset to train", type=str, required=True)
    parser.add_argument("--data_dir", help="path to dataset", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)
    parser.add_argument("--save_interval", help="Save interval", type=int, default=5000)
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--model_path", help="checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", help="path to save model and log", type=str, default=None)
    args = parser.parse_args()
    main(args)