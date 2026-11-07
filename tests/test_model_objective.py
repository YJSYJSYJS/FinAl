"""Low-load numerical checks for the FinAl global objective."""

import unittest

import torch
import torch.nn.functional as F

from final.model import FinAlModel


def reference_adaptive_temperature(
    image_similarity, report_similarity, base_temperature
):
    """Formula from mgca_module_rank_ord_only_vis_mae_atemp.py."""
    batch_size = image_similarity.size(0)
    report_ranks = torch.argsort(
        torch.argsort(report_similarity, dim=1, descending=True), dim=1
    ).float()
    image_ranks = torch.argsort(
        torch.argsort(image_similarity, dim=1, descending=True), dim=1
    ).float()

    mask = ~torch.eye(
        batch_size, dtype=torch.bool, device=image_similarity.device
    )
    image_masked = image_ranks[mask].reshape(batch_size, batch_size - 1)
    report_masked = report_ranks[mask].reshape(batch_size, batch_size - 1)
    image_centered = image_masked - image_masked.mean(dim=1, keepdim=True)
    report_centered = report_masked - report_masked.mean(dim=1, keepdim=True)
    numerator = (image_centered * report_centered).sum(dim=1)
    denominator = (
        image_centered.pow(2).sum(dim=1).sqrt()
        * report_centered.pow(2).sum(dim=1).sqrt()
        + 1e-8
    )
    similarity_diff = (1.0 - numerator / denominator).mean()
    normalized_diff = torch.clamp(similarity_diff / 2.0, max=1.0)
    temperature = base_temperature * (1.0 + normalized_diff)
    return temperature, similarity_diff, report_ranks


def reference_clipped_weights(weights):
    """Attention-weight clipping from the pretraining source."""
    rows = []
    for source_row in weights:
        row = source_row.clone()
        nonzero = row.nonzero().squeeze()
        low = torch.quantile(row[nonzero], 0.1)
        high = torch.quantile(row[nonzero], 0.9)
        row[nonzero] = row[nonzero].clip(low, high)
        rows.append(row)
    clipped = torch.stack(rows)
    return clipped / clipped.sum(dim=1, keepdim=True)


class FinAlObjectiveTests(unittest.TestCase):
    def test_attention_weight_clipping_matches_pretraining_source(self):
        weights = torch.tensor(
            [
                [0.60, 0.20, 0.10, 0.05, 0.05, 0.00],
                [0.05, 0.15, 0.50, 0.25, 0.05, 0.00],
            ]
        )
        expected = reference_clipped_weights(weights)
        actual = FinAlModel._clipped_normalized_weights(weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_adaptive_temperature_matches_pretraining_source(self):
        image_similarity = torch.tensor(
            [
                [1.00, 0.20, 0.40, -0.10],
                [0.20, 1.00, 0.10, 0.30],
                [0.40, 0.10, 1.00, 0.50],
                [-0.10, 0.30, 0.50, 1.00],
            ]
        )
        report_similarity = torch.tensor(
            [
                [1.00, 0.45, 0.15, -0.20],
                [0.45, 1.00, 0.35, 0.05],
                [0.15, 0.35, 1.00, 0.25],
                [-0.20, 0.05, 0.25, 1.00],
            ]
        )

        expected = reference_adaptive_temperature(
            image_similarity, report_similarity, 0.07
        )
        actual = FinAlModel._adaptive_temperature(
            image_similarity, report_similarity, 0.07
        )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_identical_rankings_use_base_temperature(self):
        similarity = torch.tensor(
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]]
        )
        temperature, distance, ranks = FinAlModel._adaptive_temperature(
            similarity, similarity, 0.07
        )
        torch.testing.assert_close(distance, torch.tensor(0.0), rtol=0, atol=2e-7)
        torch.testing.assert_close(temperature, torch.tensor(0.07), rtol=0, atol=1e-7)

        scores = similarity / temperature
        labels = torch.softmax(-ranks, dim=1)
        loss = F.cross_entropy(scores, labels) + F.cross_entropy(scores.t(), labels)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
