"""Plotting backends for physical properties."""

from .matplotlib_backend import plot_multiple_rdfs, plot_rdf_matplotlib

try:
    from .plotly_backend import plot_multiple_rdfs_plotly, plot_rdf_plotly

    __all__ = [
        "plot_rdf_matplotlib",
        "plot_multiple_rdfs",
        "plot_rdf_plotly",
        "plot_multiple_rdfs_plotly",
    ]
except ImportError:
    # Plotly not available
    __all__ = ["plot_rdf_matplotlib", "plot_multiple_rdfs"]
