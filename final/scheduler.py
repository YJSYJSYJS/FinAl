"""Cosine warmup/restart scheduler used by the paper configuration."""

import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        if first_cycle_steps <= 0:
            raise ValueError("first_cycle_steps must be positive")
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be smaller than first_cycle_steps")
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        for group in optimizer.param_groups:
            group["lr"] = min_lr
        super().__init__(optimizer, last_epoch)
        self.base_lrs = [min_lr for _ in optimizer.param_groups]

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / max(1, self.warmup_steps) + base_lr
                for base_lr in self.base_lrs
            ]
        progress = (self.step_in_cycle - self.warmup_steps) / max(
            1, self.cur_cycle_steps - self.warmup_steps
        )
        return [
            base_lr + (self.max_lr - base_lr) * (1.0 + math.cos(math.pi * progress)) / 2.0
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.cycle = int(epoch / self.first_cycle_steps)
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    cycle_span = epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1
                    self.cycle = int(math.log(cycle_span, self.cycle_mult))
                    consumed = self.first_cycle_steps * (
                        self.cycle_mult ** self.cycle - 1
                    ) / (self.cycle_mult - 1)
                    self.step_in_cycle = epoch - int(consumed)
                    self.cur_cycle_steps = int(
                        self.first_cycle_steps * self.cycle_mult ** self.cycle
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        self.max_lr = self.base_max_lr * self.gamma ** self.cycle
        self.last_epoch = math.floor(epoch)
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr
