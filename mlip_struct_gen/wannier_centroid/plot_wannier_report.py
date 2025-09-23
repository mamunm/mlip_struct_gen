"""Generate wannier centroid distribution plots and statistical reports."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_wc_file(filepath: Path) -> list[dict]:
    """
    Parse a wc_out.txt file to extract wannier centroid data.

    Args:
        filepath: Path to wc_out.txt file

    Returns:
        List of dictionaries containing atom data
    """
    results = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for atom entries
        if line.startswith("Atom") and "(" in line and ")" in line:
            # Parse: "Atom 0 (O)"
            match = re.match(r"Atom\s+(\d+)\s+\((\w+)\)", line)
            if match:
                atom_index = int(match.group(1))
                atom_symbol = match.group(2)

                # Initialize result dict
                result = {
                    "atom_index": atom_index,
                    "atom_symbol": atom_symbol,
                    "file_path": str(filepath.parent),
                }

                # Look for centroid norm in next few lines
                for j in range(i + 1, min(i + 20, len(lines))):
                    if "Centroid norm:" in lines[j]:
                        norm_match = re.search(r"Centroid norm:\s+([\d.]+)", lines[j])
                        if norm_match:
                            result["centroid_norm"] = float(norm_match.group(1))
                            results.append(result)
                            break

        i += 1

    return results


def find_wc_files(root_dir: Path) -> list[Path]:
    """
    Recursively find all wc_out.txt files in directory tree.

    Args:
        root_dir: Root directory to search

    Returns:
        List of paths to wc_out.txt files
    """
    return list(root_dir.rglob("wc_out.txt"))


def calculate_dipole_moment(centroid_norm: float, atom_symbol: str) -> float:
    """
    Calculate dipole moment in Debye from centroid norm and atom type.

    Args:
        centroid_norm: Wannier centroid norm in Angstroms
        atom_symbol: Atom symbol (O, Na, Cl, K, Li, Cs)

    Returns:
        Dipole moment in Debye
    """
    # Charge assignments (in elementary charge units)
    atom_charges = {"O": 6, "Na": 9, "Cl": 7, "K": 9, "Li": 3, "Cs": 9}
    wc_charges = {"O": -8, "Na": -8, "Cl": -8, "K": -8, "Li": -2, "Cs": -8}

    # Conversion factor: 1 e*Å = 4.803 Debye
    e_angstrom_to_debye = 4.803

    if atom_symbol not in atom_charges:
        return 0.0

    # Total charge on wannier centers
    q_wc = wc_charges[atom_symbol]

    # Dipole moment = charge * distance
    # The wannier centroid represents the effective position of the wannier charge
    dipole_moment = abs(q_wc) * centroid_norm * e_angstrom_to_debye

    return dipole_moment


def collect_data(
    root_dir: Path, verbose: bool = False
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Collect all wannier centroid data from directory tree.

    Args:
        root_dir: Root directory to search
        verbose: Print progress information

    Returns:
        Tuple of (centroid_norms_dict, dipole_moments_dict) both mapping atom symbols to lists
    """
    wc_files = find_wc_files(root_dir)

    if verbose:
        print(f"Found {len(wc_files)} wc_out.txt files")

    centroid_data = {}
    dipole_data = {}
    file_count = {}

    for filepath in wc_files:
        if verbose:
            print(f"Processing: {filepath}")

        try:
            results = parse_wc_file(filepath)

            for result in results:
                symbol = result["atom_symbol"]
                norm = result["centroid_norm"]
                dipole = calculate_dipole_moment(norm, symbol)

                if symbol not in centroid_data:
                    centroid_data[symbol] = []
                    dipole_data[symbol] = []
                    file_count[symbol] = set()

                centroid_data[symbol].append(norm)
                dipole_data[symbol].append(dipole)
                file_count[symbol].add(result["file_path"])

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    if verbose:
        print("\nData collection summary:")
        for symbol in sorted(centroid_data.keys()):
            print(
                f"  {symbol}: {len(centroid_data[symbol])} data points from {len(file_count[symbol])} directories"
            )

    return centroid_data, dipole_data


def generate_statistics(
    centroid_data: dict[str, list[float]], dipole_data: dict[str, list[float]]
) -> str:
    """
    Generate statistical summary report including dipole moments.

    Args:
        centroid_data: Dictionary mapping atom symbols to centroid norms
        dipole_data: Dictionary mapping atom symbols to dipole moments (Debye)

    Returns:
        Formatted statistics report string
    """
    report = []
    report.append("=" * 90)
    report.append("WANNIER CENTROID AND DIPOLE MOMENT STATISTICAL REPORT")
    report.append("=" * 90)
    report.append("")

    # Charge configuration information
    report.append("CHARGE CONFIGURATION (for dipole calculation)")
    report.append("-" * 40)
    report.append("Atom   Atom Charge   WC Charge   Total WC")
    report.append("O      +6e           -2e         -8e (4 WCs)")
    report.append("Na     +9e           -2e         -8e (4 WCs)")
    report.append("Cl     +7e           -2e         -8e (4 WCs)")
    report.append("K      +9e           -2e         -8e (4 WCs)")
    report.append("Li     +3e           -2e         -2e (1 WC)")
    report.append("Cs     +9e           -2e         -8e (4 WCs)")
    report.append("")

    # Overall centroid statistics
    all_centroid_values = []
    all_dipole_values = []
    for symbol in centroid_data:
        all_centroid_values.extend(centroid_data[symbol])
        all_dipole_values.extend(dipole_data[symbol])

    if all_centroid_values:
        report.append("OVERALL CENTROID STATISTICS")
        report.append("-" * 40)
        report.append(f"Total data points: {len(all_centroid_values)}")
        report.append("Centroid norm (Å):")
        report.append(f"  Minimum:    {np.min(all_centroid_values):.6f}")
        report.append(f"  Maximum:    {np.max(all_centroid_values):.6f}")
        report.append(f"  Mean:       {np.mean(all_centroid_values):.6f}")
        report.append(f"  Std dev:    {np.std(all_centroid_values):.6f}")
        report.append("")

        report.append("OVERALL DIPOLE MOMENT STATISTICS")
        report.append("-" * 40)
        report.append("Dipole moment (Debye):")
        report.append(f"  Minimum:    {np.min(all_dipole_values):.3f}")
        report.append(f"  Maximum:    {np.max(all_dipole_values):.3f}")
        report.append(f"  Mean:       {np.mean(all_dipole_values):.3f}")
        report.append(f"  Std dev:    {np.std(all_dipole_values):.3f}")
        report.append("")

    # Per-atom centroid statistics
    report.append("PER-ATOM CENTROID STATISTICS (Å)")
    report.append("-" * 40)
    report.append(
        f"{'Atom':<6} {'Count':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12} {'Median':<12}"
    )
    report.append("-" * 80)

    for symbol in sorted(centroid_data.keys()):
        values = centroid_data[symbol]
        if values:
            report.append(
                f"{symbol:<6} {len(values):<8} "
                f"{np.min(values):<12.6f} {np.max(values):<12.6f} "
                f"{np.mean(values):<12.6f} {np.std(values):<12.6f} "
                f"{np.median(values):<12.6f}"
            )

    report.append("-" * 80)
    report.append("")

    # Per-atom dipole statistics
    report.append("PER-ATOM DIPOLE MOMENT STATISTICS (Debye)")
    report.append("-" * 40)
    report.append(
        f"{'Atom':<6} {'Count':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12} {'Median':<12}"
    )
    report.append("-" * 80)

    for symbol in sorted(dipole_data.keys()):
        values = dipole_data[symbol]
        if values:
            report.append(
                f"{symbol:<6} {len(values):<8} "
                f"{np.min(values):<12.3f} {np.max(values):<12.3f} "
                f"{np.mean(values):<12.3f} {np.std(values):<12.3f} "
                f"{np.median(values):<12.3f}"
            )

    report.append("-" * 80)
    report.append("")

    # Centroid distribution percentiles
    report.append("CENTROID DISTRIBUTION PERCENTILES (Å)")
    report.append("-" * 40)
    report.append(f"{'Atom':<6} {'5%':<12} {'25%':<12} {'50%':<12} {'75%':<12} {'95%':<12}")
    report.append("-" * 70)

    for symbol in sorted(centroid_data.keys()):
        values = centroid_data[symbol]
        if values:
            percentiles = np.percentile(values, [5, 25, 50, 75, 95])
            report.append(
                f"{symbol:<6} {percentiles[0]:<12.6f} {percentiles[1]:<12.6f} "
                f"{percentiles[2]:<12.6f} {percentiles[3]:<12.6f} {percentiles[4]:<12.6f}"
            )

    report.append("-" * 70)
    report.append("")

    # Dipole distribution percentiles
    report.append("DIPOLE MOMENT DISTRIBUTION PERCENTILES (Debye)")
    report.append("-" * 40)
    report.append(f"{'Atom':<6} {'5%':<12} {'25%':<12} {'50%':<12} {'75%':<12} {'95%':<12}")
    report.append("-" * 70)

    for symbol in sorted(dipole_data.keys()):
        values = dipole_data[symbol]
        if values:
            percentiles = np.percentile(values, [5, 25, 50, 75, 95])
            report.append(
                f"{symbol:<6} {percentiles[0]:<12.3f} {percentiles[1]:<12.3f} "
                f"{percentiles[2]:<12.3f} {percentiles[3]:<12.3f} {percentiles[4]:<12.3f}"
            )

    report.append("-" * 70)

    return "\n".join(report)


def plot_distributions(
    centroid_data: dict[str, list[float]],
    dipole_data: dict[str, list[float]],
    output_dir: Path,
    bins: int = 50,
    verbose: bool = False,
) -> None:
    """
    Create distribution plots for wannier centroids and dipole moments.

    Args:
        centroid_data: Dictionary mapping atom symbols to centroid norms
        dipole_data: Dictionary mapping atom symbols to dipole moments
        output_dir: Directory to save output files
        bins: Number of bins for histograms
        verbose: Print progress information
    """
    atom_types = sorted(centroid_data.keys())
    n_types = len(atom_types)

    if n_types == 0:
        print("No data to plot")
        return

    # --- CENTROID PLOTS ---
    # Create figure with subplots for centroids
    fig_height = max(4, 3 * ((n_types + 1) // 2))
    plt.figure(figsize=(12, fig_height))

    # Individual histograms for each atom type
    n_cols = 2 if n_types > 1 else 1
    n_rows = (n_types + 1) // 2 if n_types > 1 else 1

    for idx, symbol in enumerate(atom_types):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        values = centroid_data[symbol]

        if values:
            # Create histogram
            counts, bin_edges, patches = ax.hist(
                values, bins=bins, alpha=0.7, edgecolor="black", density=True
            )

            # Add statistics lines
            mean_val = np.mean(values)
            median_val = np.median(values)

            ax.axvline(
                mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}", linewidth=2
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                label=f"Median: {median_val:.4f}",
                linewidth=2,
            )

            ax.set_xlabel("Wannier Centroid Norm (Å)")
            ax.set_ylabel("Density")
            ax.set_title(f"{symbol} (n={len(values)})")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle("Wannier Centroid Norm Distributions by Atom Type", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save centroid histogram plot
    hist_path = output_dir / "centroid_histograms.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved centroid histogram plot to: {hist_path}")

    # --- DIPOLE MOMENT PLOTS ---
    # Create figure with subplots for dipole moments
    plt.figure(figsize=(12, fig_height))

    for idx, symbol in enumerate(atom_types):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        values = dipole_data[symbol]

        if values:
            # Create histogram
            counts, bin_edges, patches = ax.hist(
                values, bins=bins, alpha=0.7, edgecolor="black", density=True, color="orange"
            )

            # Add statistics lines
            mean_val = np.mean(values)
            median_val = np.median(values)

            ax.axvline(
                mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f} D", linewidth=2
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                label=f"Median: {median_val:.2f} D",
                linewidth=2,
            )

            ax.set_xlabel("Dipole Moment (Debye)")
            ax.set_ylabel("Density")
            ax.set_title(f"{symbol} (n={len(values)})")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle("Dipole Moment Distributions by Atom Type", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save dipole histogram plot
    dipole_hist_path = output_dir / "dipole_histograms.png"
    plt.savefig(dipole_hist_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved dipole histogram plot to: {dipole_hist_path}")

    # Create combined centroid distribution plot
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_types))

    for idx, symbol in enumerate(atom_types):
        values = centroid_data[symbol]
        if values:
            # Create kernel density estimate for smoother curve
            from scipy import stats

            density = stats.gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            plt.plot(
                x_range,
                density(x_range),
                label=f"{symbol} (n={len(values)})",
                color=colors[idx],
                linewidth=2,
            )

    plt.xlabel("Wannier Centroid Norm (Å)")
    plt.ylabel("Probability Density")
    plt.title("Wannier Centroid Norm Distributions - Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save combined centroid plot
    combined_path = output_dir / "centroid_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved combined centroid plot to: {combined_path}")

    # Create combined dipole distribution plot
    plt.figure(figsize=(10, 6))

    for idx, symbol in enumerate(atom_types):
        values = dipole_data[symbol]
        if values:
            # Create kernel density estimate for smoother curve
            from scipy import stats

            density = stats.gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            plt.plot(
                x_range,
                density(x_range),
                label=f"{symbol} (n={len(values)})",
                color=colors[idx],
                linewidth=2,
            )

    plt.xlabel("Dipole Moment (Debye)")
    plt.ylabel("Probability Density")
    plt.title("Dipole Moment Distributions - Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save combined dipole plot
    dipole_combined_path = output_dir / "dipole_combined.png"
    plt.savefig(dipole_combined_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved combined dipole plot to: {dipole_combined_path}")

    # Create box plot for centroid comparison
    plt.figure(figsize=(10, 6))

    centroid_boxplot_data = []
    labels_for_boxplot = []

    for symbol in atom_types:
        if centroid_data[symbol]:
            centroid_boxplot_data.append(centroid_data[symbol])
            labels_for_boxplot.append(f"{symbol}\n(n={len(centroid_data[symbol])})")

    plt.boxplot(
        centroid_boxplot_data,
        labels=labels_for_boxplot,
        showfliers=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    plt.ylabel("Wannier Centroid Norm (Å)")
    plt.title(
        "Wannier Centroid Norm Distribution - Box Plot Comparison", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3, axis="y")

    # Save centroid box plot
    box_path = output_dir / "centroid_boxplot.png"
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved centroid box plot to: {box_path}")

    # Create box plot for dipole comparison
    plt.figure(figsize=(10, 6))

    dipole_boxplot_data = []
    labels_for_boxplot = []

    for symbol in atom_types:
        if dipole_data[symbol]:
            dipole_boxplot_data.append(dipole_data[symbol])
            labels_for_boxplot.append(f"{symbol}\n(n={len(dipole_data[symbol])})")

    plt.boxplot(
        dipole_boxplot_data,
        labels=labels_for_boxplot,
        showfliers=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    plt.ylabel("Dipole Moment (Debye)")
    plt.title("Dipole Moment Distribution - Box Plot Comparison", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")

    # Save dipole box plot
    dipole_box_path = output_dir / "dipole_boxplot.png"
    plt.savefig(dipole_box_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Saved dipole box plot to: {dipole_box_path}")

    plt.close("all")


def generate_report(
    root_dir: Path, output_name: str = "wc_report", bins: int = 50, verbose: bool = False
) -> None:
    """
    Generate complete wannier centroid analysis report.

    Args:
        root_dir: Root directory to search for wc_out.txt files
        output_name: Name of the output directory to create
        bins: Number of bins for histograms
        verbose: Print progress information
    """
    print(f"Searching for wc_out.txt files in: {root_dir}")

    # Collect data
    centroid_data, dipole_data = collect_data(root_dir, verbose)

    if not centroid_data:
        print("No wannier centroid data found!")
        return

    # Create output directory
    output_dir = Path(output_name)
    output_dir.mkdir(exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    # Generate text report
    report_text = generate_statistics(centroid_data, dipole_data)

    # Save text report
    txt_path = output_dir / "statistics.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)

    print(f"\nSaved statistical report to: {txt_path}")
    print("\n" + report_text)

    # Generate plots
    print("\nGenerating distribution plots...")
    plot_distributions(centroid_data, dipole_data, output_dir, bins, verbose)

    print("\nWannier centroid and dipole moment analysis complete!")
    print(f"Output directory: {output_dir}")
    print("Output files:")
    print("  Statistics:")
    print(f"    - {txt_path}")
    print("  Centroid plots:")
    print(f"    - {output_dir / 'centroid_histograms.png'}")
    print(f"    - {output_dir / 'centroid_combined.png'}")
    print(f"    - {output_dir / 'centroid_boxplot.png'}")
    print("  Dipole plots:")
    print(f"    - {output_dir / 'dipole_histograms.png'}")
    print(f"    - {output_dir / 'dipole_combined.png'}")
    print(f"    - {output_dir / 'dipole_boxplot.png'}")
