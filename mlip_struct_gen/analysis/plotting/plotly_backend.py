"""Plotly plotting backend for RDF and other properties."""

from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required for interactive plots. " "Install it with: pip install plotly"
    ) from None


def plot_rdf_plotly(
    results: dict[str, dict[str, Any]],
    pairs: list[str],
    output: str | Path | None = None,
    show_coordination: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Plot RDF using plotly (interactive).

    Args:
        results: Dictionary of RDF results
        pairs: List of pairs to plot
        output: Output HTML file path (None = display only)
        show_coordination: Also plot coordination number
        **kwargs: Additional plotly parameters

    Returns:
        Plotly figure object
    """
    # Create subplots
    if show_coordination:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Radial Distribution Function", "Running Coordination Number"),
            horizontal_spacing=0.12,
        )
    else:
        fig = go.Figure()

    # Plot g(r)
    for pair in pairs:
        if pair not in results:
            print(f"Warning: No results for pair {pair}")
            continue

        data = results[pair]

        # Add g(r) trace
        if show_coordination:
            fig.add_trace(
                go.Scatter(
                    x=data["r"],
                    y=data["gr"],
                    mode="lines",
                    name=f"{pair}",
                    legendgroup=pair,
                    line={"width": 2},
                    hovertemplate="r: %{x:.2f} Å<br>g(r): %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data["r"],
                    y=data["gr"],
                    mode="lines",
                    name=f"{pair} (N={data['n_frames']} frames)",
                    line={"width": 2},
                    hovertemplate="r: %{x:.2f} Å<br>g(r): %{y:.3f}<extra></extra>",
                )
            )

    # Plot coordination number if requested
    if show_coordination:
        for pair in pairs:
            if pair not in results:
                continue

            data = results[pair]

            fig.add_trace(
                go.Scatter(
                    x=data["r"],
                    y=data["coordination"],
                    mode="lines",
                    name=f"{pair}",
                    legendgroup=pair,
                    showlegend=False,
                    line={"width": 2},
                    hovertemplate="r: %{x:.2f} Å<br>CN: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Update layout
    if show_coordination:
        fig.update_xaxes(title_text="r (Å)", row=1, col=1)
        fig.update_xaxes(title_text="r (Å)", row=1, col=2)
        fig.update_yaxes(title_text="g(r)", row=1, col=1)
        fig.update_yaxes(title_text="Coordination Number", row=1, col=2)
    else:
        fig.update_xaxes(title_text="r (Å)")
        fig.update_yaxes(title_text="g(r)")

    fig.update_layout(
        title="Radial Distribution Function Analysis",
        hovermode="x unified",
        template="plotly_white",
        height=500 if not show_coordination else 500,
        width=800 if not show_coordination else 1200,
    )

    # Save or show
    if output:
        output_path = Path(output)
        if output_path.suffix == ".html":
            fig.write_html(str(output_path))
        elif output_path.suffix in [".png", ".jpg", ".pdf", ".svg"]:
            fig.write_image(str(output_path))
        else:
            # Default to HTML
            fig.write_html(str(output_path.with_suffix(".html")))
        print(f"Plot saved to {output_path}")
    else:
        fig.show()

    return fig


def plot_multiple_rdfs_plotly(
    results_list: list[dict[str, dict[str, Any]]],
    labels: list[str],
    pairs: list[str],
    output: str | Path | None = None,
    **kwargs: Any,
) -> go.Figure:
    """
    Plot multiple RDF results on the same figure (comparison plot).

    Args:
        results_list: List of RDF result dictionaries
        labels: Labels for each result set
        pairs: List of pairs to plot
        output: Output file path
        **kwargs: Additional plotly parameters

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    for results, label in zip(results_list, labels, strict=False):
        for pair in pairs:
            if pair not in results:
                continue

            data = results[pair]

            fig.add_trace(
                go.Scatter(
                    x=data["r"],
                    y=data["gr"],
                    mode="lines",
                    name=f"{label} - {pair}",
                    line={"width": 2},
                    hovertemplate="r: %{x:.2f} Å<br>g(r): %{y:.3f}<extra></extra>",
                )
            )

    fig.update_layout(
        title="RDF Comparison",
        xaxis_title="r (Å)",
        yaxis_title="g(r)",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=900,
    )

    # Save or show
    if output:
        output_path = Path(output)
        if output_path.suffix == ".html":
            fig.write_html(str(output_path))
        elif output_path.suffix in [".png", ".jpg", ".pdf", ".svg"]:
            fig.write_image(str(output_path))
        else:
            fig.write_html(str(output_path.with_suffix(".html")))
        print(f"Plot saved to {output_path}")
    else:
        fig.show()

    return fig
