import os
import sys
import argparse
import random

import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path

from utils import logger
from configs import config
from sklearn import metrics
from utils.test_util import run_batch
from utils.script_util import create_model
from utils.datasets import get_data_loader
sys.path.append(str(Path.cwd()))

import warnings
warnings.filterwarnings("ignore")


def set_random_seed(seed=0, reproduce=False):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    random.seed(seed)

    if reproduce:
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
    else:
        th.backends.cudnn.benchmark = True

def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def main(args):
    set_random_seed(reproduce=True)

    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)

    logger.configure(Path("."), format_strs=["stdout"])
    logger.log(f"Loading data for {args.dataset}...")
    test_loader = get_data_loader(args.data_dir, 'test', args.batch_size, generator=False)
    
    logger.log(f"Loading checkpoint {args.model_path} ...")
    model = create_model(config)
    model.load_state_dict(
        th.load(args.model_path, map_location=th.device('cuda'), weights_only=True)
    )
    model.to(th.device('cuda'))

    if config.model.use_fp16:
        model.convert_to_fp16()

    model.eval()

    logger.log("Evaluating...")

    from torchmetrics.segmentation import DiceScore
    dice_metric = DiceScore(num_classes=2, include_background=False, average='micro')

    from torchmetrics.classification import BinaryAveragePrecision
    ap_metric = BinaryAveragePrecision()

    from torchmetrics.classification import BinaryAUROC
    auc_metric = BinaryAUROC()
    
    for i, (source, _, masks, labels, _) in enumerate(test_loader):
        print(f"\rBatch {i+1}/{len(test_loader)}", end="")
        test_data_input = source.cuda()
        test_data_seg = masks.cuda()
        test_data_label = labels

        output_mask, error_maps, reconstuction = run_batch(model, test_data_input)

        # Computing AP and AUC
        logits = th.mean(error_maps, dim=(1,2,3)).cpu()

        ap_metric.update(logits, test_data_label)
        ap = ap_metric.compute().item()

        auc_metric.update(logits, test_data_label)
        auc = auc_metric.compute().item()

        # Compute DICE
        dice_metric.update(output_mask.int(), test_data_seg.int())
        dice = (dice_metric.numerator[-1] / (dice_metric.denominator[-1] + 1e-8)).mean().item()
        print(f"  AP={ap:.4f}, AUC={auc:.4f}, DICE={dice:.4f}", end="")

    overall_ap = ap_metric.compute().item()
    overall_auc = auc_metric.compute().item()
    overall_dice = dice_metric.compute().item()
    print()
    print("Overall Result:")
    print(f"    AP={overall_ap:.4f}, AUC={overall_auc:.4f}, DICE={overall_dice:.4f}")


def reseed_random(seed):
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset to train", type=str, required=True)
    parser.add_argument("--data_dir", help="path to dataset", type=str, required=True)
    parser.add_argument("--model_path", help="checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size (WARNING: batch size of 4 means a maximum of 16 images are processed together)", type=int, default=4)
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    args = parser.parse_args()
    main(args)
