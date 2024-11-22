from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from io import BytesIO
from PIL import Image

@dataclass
class PlotConfig:
    """Configuration for plot settings."""
    show_plots: bool = False
    selected_plots: List[str] = None
    figure_size: Tuple[int, int] = (5, 5)
    dpi: int = 150
    cmap: str = 'tab10'
    
    def __post_init__(self):
        if self.selected_plots is None:
            self.selected_plots = ['neighborhood_dist', 'embeddings']

class BasePlot:
    """Base class for all plot types."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        
    def setup_axis(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Set up or get matplotlib axis."""
        return ax if ax is not None else plt.gca()
    
    def plot(self, data: Dict, ax: Optional[plt.Axes] = None) -> None:
        """Plot implementation to be overridden by subclasses."""
        raise NotImplementedError

class EmbeddingsPlot(BasePlot):
    """2D embeddings visualization."""
    
    def plot(self, data: Dict, ax: Optional[plt.Axes] = None) -> None:
        ax = self.setup_axis(ax)
        embeddings, labels = data['embeddings'], data['labels']
        
        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1],
            hue=labels,
            palette=sns.color_palette(self.config.cmap),
            edgecolor='none',
            ax=ax,
            alpha=0.7,
            s=10
        )
        
        ax.set_title("2D Embeddings Visualization", fontsize=16)
        ax.set_xlabel("Embedding Dimension 1", fontsize=14)
        ax.set_ylabel("Embedding Dimension 2", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)

class NeighborhoodDistPlot(BasePlot):
    """Neighborhood distribution visualization."""
    
    def plot(self, data: Dict, ax: Optional[plt.Axes] = None) -> None:
        ax = self.setup_axis(ax)
        learned_kernel = F.normalize(data['learned_kernel'], p=1, dim=-1)
        target_kernel = F.normalize(data['target_kernel'], p=1, dim=-1)
        
        self._plot_distributions(learned_kernel, target_kernel, ax)
        
    def _plot_distributions(self, learned_kernel: torch.Tensor, 
                          target_kernel: torch.Tensor, ax: plt.Axes) -> None:
        """Plot the distribution comparison."""
        _, indices = torch.sort(target_kernel+0.01*learned_kernel, dim=-1)
        probs_sorted = self._gather_and_process(learned_kernel, indices)
        target_sorted = self._gather_and_process(target_kernel, indices)
        
        x_values = np.arange(probs_sorted[0].size(0))[::-1]
        
        self._plot_distribution_line(x_values, target_sorted, 'Target Distribution $P_i$', 
                                   'orange', ax)
        self._plot_distribution_line(x_values, probs_sorted, 'Learned Distribution $Q_i$', 
                                   'blue', ax)
        
        ax.set_yscale('log')
        ax.set_xlabel('Neighbors Ordered by Proximity', fontsize=14)
        ax.set_ylabel('Selection Probability', fontsize=14)
        ax.set_title('Neighbor Selection Probability Distributions', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    @staticmethod
    def _gather_and_process(kernel: torch.Tensor, 
                           indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather and process kernel statistics."""
        gathered = torch.gather(kernel, 1, indices)
        mean = gathered.mean(dim=0)
        std = gathered.std(dim=0)
        return mean.flip(0), std.flip(0)
    
    def _plot_distribution_line(self, x_values: np.ndarray, 
                              stats: Tuple[torch.Tensor, torch.Tensor],
                              label: str, color: str, ax: plt.Axes) -> None:
        """Plot a single distribution line with confidence interval."""
        mean, std = stats
        ax.plot(x_values, mean.numpy(), label=label, color=color, linewidth=2)
        ax.fill_between(
            x_values,
            (mean - std).numpy(),
            (mean + std).numpy(),
            color=color,
            alpha=0.3
        )

class ProbabilitiesStarPlot(BasePlot):
    """Star-shaped probability visualization."""
    
    def plot(self, data: Dict, ax: Optional[plt.Axes] = None) -> None:
        ax = self.setup_axis(ax)
        probs, labels = data['probabilities'], data['labels']
        n_clusters = probs.shape[1]
        
        vertices = self._compute_vertices(n_clusters)
        points = self._compute_points(probs, vertices)
        self._plot_star(points, labels, vertices, n_clusters, ax)
        
    def _compute_vertices(self, n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute vertex positions for the star plot."""
        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1)[:-1]
        return (torch.cos(theta), torch.sin(theta))
    
    def _compute_points(self, probs: torch.Tensor, 
                       vertices: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute point positions from probabilities."""
        x = probs @ vertices[0]
        y = probs @ vertices[1]
        return x, y
    
    def _plot_star(self, points: Tuple[torch.Tensor, torch.Tensor], 
                   labels: torch.Tensor, vertices: Tuple[torch.Tensor, torch.Tensor], 
                   n_clusters: int, ax: plt.Axes) -> None:
        """Create the star plot visualization."""
        colors = plt.get_cmap(self.config.cmap)(torch.linspace(0, 1, n_clusters))
        x, y = points
        vx, vy = vertices
        
        for i in range(n_clusters):
            mask = (labels == i)
            ax.scatter(x[mask], y[mask], color=colors[i], s=250, alpha=0.15, 
                      label=f'cluster {i}')
        
        # Plot vertices
        ax.scatter(vx, vy, c='black', s=400, marker='*', alpha=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

class ClusterSizesPlot(BasePlot):
    """Cluster sizes visualization with uncertainty."""
    
    def plot(self, data: Dict, ax: Optional[plt.Axes] = None) -> None:
        ax = self.setup_axis(ax)
        probabilities = data['probabilities']
        
        cluster_sizes, uncertainty = self._compute_statistics(probabilities)
        self._create_bar_plot(cluster_sizes, uncertainty, ax)
    
    def _compute_statistics(self, probabilities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cluster sizes and uncertainties."""
        cluster_sizes = probabilities.sum(dim=0)
        uncertainty = torch.sqrt((probabilities * (1 - probabilities)).sum(dim=0) / 
                               probabilities.size(0))
        
        # Sort by size
        cluster_sizes, indices = torch.sort(cluster_sizes, descending=True)
        uncertainty = uncertainty[indices]
        
        return cluster_sizes, uncertainty
    
    def _create_bar_plot(self, cluster_sizes: torch.Tensor, 
                        uncertainty: torch.Tensor, ax: plt.Axes) -> None:
        """Create bar plot with error bars."""
        classes = np.arange(len(cluster_sizes))
        ax.bar(classes, cluster_sizes.cpu().numpy(), 
               yerr=uncertainty.cpu().numpy(), 
               capsize=5, color='skyblue')
        
        ax.set_title('Cluster Sizes with Uncertainty', fontsize=16)
        ax.set_xlabel('Cluster Index', fontsize=14)
        ax.set_ylabel('Number of Points', fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5)

class PlotLogger(pl.Callback):
    """PyTorch Lightning callback for visualization logging."""
    
    PLOT_CLASSES = {
        'embeddings': EmbeddingsPlot,
        'neighborhood_dist': NeighborhoodDistPlot,
        'probabilities_star': ProbabilitiesStarPlot,
        'cluster_sizes': ClusterSizesPlot
    }
    
    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__()
        self.config = config or PlotConfig()
        self.plots = {name: plot_class(self.config) 
                     for name, plot_class in self.PLOT_CLASSES.items() 
                     if name in self.config.selected_plots}
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create and log plots at the end of validation epoch."""
        plot_data = self._gather_plot_data(pl_module)
        self._create_and_log_plots(plot_data, trainer, pl_module)
    
    def _gather_plot_data(self, pl_module: pl.LightningModule) -> Dict:
        """Gather data needed for plotting from the model."""
        embeddings, logits, labels, learned_kernel, target_kernel = pl_module.aggregated_val_outputs
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'learned_kernel': learned_kernel,
            'target_kernel': target_kernel,
            'probabilities': torch.softmax(logits, dim=-1) if pl_module.config.linear_probe else logits
        }
    
    def _create_and_log_plots(self, plot_data: Dict, 
                             trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create and log all selected plots."""
        n_plots = len(self.plots)
        fig_width = self.config.figure_size[0] * n_plots
        fig_height = self.config.figure_size[1]
        
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
        if n_plots == 1:
            axes = [axes]
        
        for ax, (plot_name, plot) in zip(axes, self.plots.items()):
            plot.plot(plot_data, ax)
        
        plt.tight_layout()
        
        if self.config.show_plots:
            plt.show()
        else:
            self._log_to_tensorboard(fig, trainer, pl_module)
        
        plt.close(fig)
    
    def _log_to_tensorboard(self, fig: plt.Figure, 
                           trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log figure to TensorBoard."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        
        image = np.array(Image.open(buffer).convert('RGB'))
        for plot_name in self.plots:
            trainer.logger.experiment.add_image(
                plot_name,
                image,
                global_step=trainer.global_step,
                dataformats='HWC'
            )