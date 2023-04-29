from .train_util import TrainLoop

import copy
import functools
import os

# import random
import wandb

# import blobfile as bf
# import torch as th
import torch.distributed as dist

# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
# from torch.optim import AdamW

from . import dist_util, logger

# from .fp16_util import MixedPrecisionTrainer
# from .nn import update_ema
# from .resample import LossAwareSampler, UniformSampler
from .resample import LossAwareSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# INITIAL_LOG_LOSS_SCALE = 20.0


class DistillTrainLoop(TrainLoop):
    def __init__(
        self,
        *args,
        distillation_type=None,
        teacher_model=None,
        init_copy_teacher=False,
        **kwargs,
    ):
        assert teacher_model is not None, "distillation requires a teacher model"
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type

        if init_copy_teacher:
            self.model.load_state_dict(copy.deepcopy(self.teacher_model.state_dict()))

    def get_model_checkpoint(self, model_path):
        self.resume_step = parse_resume_step_from_filename(model_path)
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {model_path}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(model_path, map_location=dist_util.dev())
            )

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        logger.log("running with distillation (:")
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # t = random.uniform(0, self.diffusion.num_timesteps

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        wandb.log({key: values.mean().item()})
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            wandb.log({f"{key}_q{quartile}": sub_loss})
