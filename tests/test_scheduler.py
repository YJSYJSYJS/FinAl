"""Low-load checks for the FinAl learning-rate scheduler."""

import math
import unittest

import torch

from final.scheduler import CosineAnnealingWarmupRestarts


def make_scheduler(cycle_mult=1.0, gamma=1.0):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=10,
        cycle_mult=cycle_mult,
        max_lr=2e-5,
        min_lr=1e-6,
        warmup_steps=2,
        gamma=gamma,
    )
    return parameter, optimizer, scheduler


class SchedulerTests(unittest.TestCase):
    def test_warmup_cosine_and_restart_boundaries(self):
        parameter, optimizer, scheduler = make_scheduler()
        observed = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
            observed.append(optimizer.param_groups[0]["lr"])

        minimum = 1e-6
        maximum = 2e-5
        self.assertEqual(observed[0], minimum)
        self.assertEqual(observed[1], minimum + (maximum - minimum) / 2.0)
        self.assertEqual(observed[2], maximum)
        expected_step_three = minimum + (maximum - minimum) * (
            1.0 + math.cos(math.pi / 8.0)
        ) / 2.0
        self.assertAlmostEqual(observed[3], expected_step_three, places=15)
        self.assertEqual(observed[10], minimum)
        self.assertEqual(scheduler.cycle, 1)
        self.assertEqual(scheduler.step_in_cycle, 0)

    def test_gamma_changes_only_the_cycle_maximum(self):
        parameter, optimizer, scheduler = make_scheduler(gamma=0.5)
        for _ in range(12):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()

        self.assertEqual(scheduler.cycle, 1)
        self.assertEqual(scheduler.step_in_cycle, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)

    def test_state_dict_resume_preserves_next_step(self):
        parameter, optimizer, scheduler = make_scheduler()
        for _ in range(7):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        resumed_parameter, resumed_optimizer, resumed_scheduler = make_scheduler()
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)

        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        resumed_parameter.grad = torch.zeros_like(resumed_parameter)
        resumed_optimizer.step()
        resumed_scheduler.step()

        self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())
        self.assertEqual(
            optimizer.param_groups[0]["lr"],
            resumed_optimizer.param_groups[0]["lr"],
        )

    def test_invalid_cycle_lengths_are_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5)
        with self.assertRaisesRegex(ValueError, "first_cycle_steps"):
            CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=0)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            CosineAnnealingWarmupRestarts(
                optimizer, first_cycle_steps=2, warmup_steps=2
            )


if __name__ == "__main__":
    unittest.main()
