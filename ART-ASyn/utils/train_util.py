import copy
import os
import time
import numpy as np
import blobfile as bf

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from utils import logger
from utils.fp16_util import MixedPrecisionTrainer
from models.nn import update_ema
from utils.datasets import normalize_to_01

from torchmetrics.segmentation import DiceScore
dice_metric = DiceScore(num_classes=2, include_background=False, average='micro')


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            data,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            start_step=0,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_decay_steps=0,
            lr_decay_factor=1,
            iterations: int = 80e4,
            num_input_channels=None,
            image_size=None,
            device=None,
            args=None
    ):
        self.model = model
        self.data = data
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_factor = lr_decay_factor
        self.iterations = iterations
        self.num_input_channels = num_input_channels
        self.image_size = image_size

        '''timing'''
        self.args = args
        self.step = start_step
        self.time_iter_start = 0
        self.forward_backward_time = 0
        self.device = device
        self.x0_pred = None
        self.recursive_flag = 0
        self.sync_cuda = torch.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]

    def run_loop(self):
        while (
                not self.lr_decay_steps
                or self.step < self.iterations
        ):
            batch_data = next(self.data)
            self.run_step(batch_data)
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % self.log_interval == 0:
                self.time_iter_end = time.time()
                if self.time_iter_start == 0:
                    self.time_iter = 0
                else:
                    self.time_iter = self.time_iter_end - self.time_iter_start
                self.log_step()
                logger.dumpkvs()
                self.time_iter_start = time.time()
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_data):
        self.forward_backward(batch_data, phase="train")
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self.lr_decay()


    def forward_backward(self, batch_data, phase: str = "train"):
        assert phase in ["train", "val"]

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        self.mp_trainer.zero_grad()

        losses = self.training_losses(self.model, batch_data)
        loss = losses["loss"].mean()

        if phase == "train":
            self.mp_trainer.backward(loss)

        loss_ = loss.detach().cpu().numpy()
        self.loss = loss.item()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def lr_decay(self):
        if self.lr_decay_steps == 0 or self.step % self.lr_decay_steps != 0 or self.step == 0:
            return
        print('lr decay.....')
        n_decays = self.step // self.lr_decay_steps
        lr = self.lr * self.lr_decay_factor ** n_decays
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1))
        logger.logkv("time 100iter", self.time_iter)
        logger.logkv("loss", f"{self.loss:.4f}")
        
    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model ...")
            filename = f"model{(self.step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

    def training_losses(self, model, batch_data):
        source, target, masks, labels, idxs = batch_data

        source = source.to(self.device)
        target = target.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)

        batch_size = len(source)

        # 1. Push Encoder to encode synthetic anomaly feautre as normal feature
        source_features = model.get_encoded_feature(source).view(batch_size, -1)
        target_features = model.get_encoded_feature(target).view(batch_size, -1)

        source_features = F.normalize(source_features, dim=1)
        target_features = F.normalize(target_features, dim=1)

        cos_dist = 1 - F.cosine_similarity(source_features, target_features, dim=1)

        # 2. Global loss & local loss (focus on masked area)
        pred = model(source)
        mse = (target - pred) ** 2
        global_mse = mse.mean(dim=[1, 2, 3]) # shape: [batch_size]
        local_mse = (mse * masks).mean(dim=[1, 2, 3]) # shape: [batch_size]

        # 3. Dice loss
        pred_masks = torch.where((source - pred) ** 2 >= 0.2, 1.0, 0.0)
        dice = dice_metric(pred_masks, masks)
        
        # 4. Combine the losses together
        loss = (2 * local_mse + global_mse + cos_dist + dice).mean()
        # group_means = []
        # for i in torch.unique(idxs):
        #     group_loss = loss[idxs == i]
        #     group_means.append(group_loss.mean())
        # final_loss = torch.stack(group_means).mean()

        return {"loss": loss}


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
