"""Matplotlib plotting backend for RDF and other properties."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_rdf_matplotlib(
    results: dict[str, dict[str, Any]],
    pairs: list[str],
    output: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 150,
    show_coordination: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot RDF using matplotlib.

    Args:
        results: Dictionary of RDF results
        pairs: List of pairs to plot
        output: Output file path (None = display only)
        figsize: Figure size (width, height)
        dpi: DPI for saved figure
        show_coordination: Also plot coordination number
        **kwargs: Additional matplotlib parameters

    Returns:
        Tuple of (figure, axes)
    """
    # Create subplots
    n_plots = 2 if show_coordination else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = np.array([axes])

    # Plot g(r)
    ax_gr = axes[0]

    for pair in pairs:
        if pair not in results:
            print(f"Warning: No results for pair {pair}")
            continue

        data = results[pair]
        ax_gr.plot(
            data["r"],
            data["gr"],
            label=f"{pair} (N={data['n_frames']} frames)",
            linewidth=2,
        )

    ax_gr.set_xlabel("r (Å)", fontsize=12)
    ax_gr.set_ylabel("g(r)", fontsize=12)
    ax_gr.set_title("Radial Distribution Function", fontsize=14, fontweight="bold")
    ax_gr.legend(fontsize=10)
    ax_gr.grid(True, alpha=0.3)
    ax_gr.set_xlim(left=0)

    # Plot coordination number if requested
    if show_coordination:
        ax_cn = axes[1]

        for pair in pairs:
            if pair not in results:
                continue

            data = results[pair]
            ax_cn.plot(data["r"], data["coordination"], label=pair, linewidth=2)

        ax_cn.set_xlabel("r (Å)", fontsize=12)
        ax_cn.set_ylabel("Coordination Number", fontsize=12)
        ax_cn.set_title("Running Coordination Number", fontsize=14, fontweight="bold")
        ax_cn.legend(fontsize=10)
        ax_cn.grid(True, alpha=0.3)
        ax_cn.set_xlim(left=0)

    plt.tight_layout()

    # Save or show
    if output:
        output_path = Path(output)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return fig, axes


def plot_multiple_rdfs(
    results_list: list[dict[str, dict[str, Any]]],
    labels: list[str],
    pairs: list[str],
    output: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple RDF results on the same figure (comparison plot).

    Args:
        results_list: List of RDF result dictionaries
        labels: Labels for each result set
        pairs: List of pairs to plot
        output: Output file path
        figsize: Figure size
        dpi: DPI for saved figure
        **kwargs: Additional matplotlib parameters

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    for i, (results, label) in enumerate(zip(results_list, labels, strict=False)):
        for pair in pairs:
            if pair not in results:
                continue

            data = results[pair]
            ax.plot(
                data["r"],
                data["gr"],
                label=f"{label} - {pair}",
                color=colors[i],
                linewidth=2,
            )

    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_title("RDF Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output:
        output_path = Path(output)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return fig, ax
