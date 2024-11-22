from torchmetrics import Metric
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class UnsupervisedAccuracy(Metric):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        # Add state for confusion matrix with proper reduction in distributed training
        self.add_state(
            "stats",
            default=torch.zeros(n_classes, n_classes, dtype=torch.int64),
            dist_reduce_fx="sum"  # Sum across devices for distributed training
        )
        self.cluster_to_class_map = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the confusion matrix (stats) based on the predicted clusters and actual labels.
        """
        with torch.no_grad():
            actual = target.reshape(-1)  # Flatten the target
            preds = preds.reshape(-1)  # Flatten the predictions

            # Ensure predictions and targets are valid for the given number of classes
            mask = (
                (actual >= 0)
                & (actual < self.n_classes)
                & (preds >= 0)
                & (preds < self.n_classes)
            )
            actual = actual[mask]
            preds = preds[mask]

            # Update the confusion matrix
            self.stats += (
                torch.bincount(
                    self.n_classes * actual + preds, minlength=self.n_classes**2
                )
                .reshape(self.n_classes, self.n_classes)
                .t()
                .to(self.stats.device)  # Ensure the update happens on the correct device
            )

    def compute(self):
        """
        Compute the accuracy based on the current confusion matrix (stats).
        """
        # In distributed settings, ensure the confusion matrix is synchronized across devices
        stats = self.stats.detach()

        # Use linear sum assignment to find the optimal cluster-to-class mapping
        assignments = linear_sum_assignment(stats.cpu(), maximize=True)
        self.cluster_to_class_map = torch.tensor(assignments[1], device=self.stats.device)

        # Reorder confusion matrix according to the optimal mapping
        histogram = stats[torch.argsort(self.cluster_to_class_map), :]

        # Calculate the true positives along the diagonal
        tp = torch.diag(histogram)
        overall_precision = torch.sum(tp) / torch.sum(histogram)

        return 100 * overall_precision

    def reset(self):
        """
        Reset the confusion matrix and cluster-to-class mapping.
        """
        self.stats.zero_()  # Reset stats to zero
        self.cluster_to_class_map = None

    def map_clusters(self, clusters: torch.Tensor):
        """
        Map clusters to class labels using the cluster-to-class mapping.
        """
        if self.cluster_to_class_map is None:
            raise ValueError(
                "No cluster-to-class mapping available. Please call compute() first."
            )
        return self.cluster_to_class_map[clusters]

    def get_confusion_matrix(self):
        """
        Get the confusion matrix, sorted according to the cluster-to-class mapping.
        """
        if self.cluster_to_class_map is None:
            raise ValueError(
                "No cluster-to-class mapping available. Please call compute() first."
            )
        return self.stats[torch.argsort(self.cluster_to_class_map), :]



if __name__ == "__main__":
    num_classes = 5
    metric = UnsupervisedAccuracy(n_classes=num_classes)

    preds = (torch.tensor([0, 0, 2, 2, 4, 4, 0, 1, 2, 3, 0]) + 3) % 5
    target = torch.tensor([0, 0, 2, 2, 4, 4, 0, 1, 2, 3, 4])

    metric.update(preds, target)
    result = metric.compute()

    assert int(result) == 80

    print("Test passed!")
