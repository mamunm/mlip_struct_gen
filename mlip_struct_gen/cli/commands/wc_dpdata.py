"""Unified OUTCAR + Wannier centroid -> dpdata conversion command."""

import argparse
from pathlib import Path

from mlip_struct_gen.generate_dpdata.wc_dpdata_converter import WCDPDataConverter
from mlip_struct_gen.utils.logger import get_logger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "wc-dpdata",
        help="Convert VASP OUTCARs + Wannier centroids to dpdata (with atomic dipoles)",
        description=(
            "Walk OUTCARs recursively and, for each snapshot directory that also "
            "contains wannier90_centres.xyz (or wc_out.npy), produce a dpdata tree "
            "with energy/force/coord/box/virial and atomic_dipole arrays aligned "
            "under a single type_map."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Pt-water snapshots
  mlip-struct-gen wc-dpdata \\
      --input-dir ./snapshots \\
      --output-dir DATA/pt_water \\
      --type-map Pt O H

  # Override which atom types get dipole labels (defaults to O/Na/Cl/K/Li/Cs
  # filtered against type-map)
  mlip-struct-gen wc-dpdata \\
      --input-dir ./snapshots \\
      --output-dir DATA/salt_water \\
      --type-map O H Na Cl \\
      --sel-type O Na Cl

Requirements per OUTCAR directory:
  - OUTCAR (single-frame/single-point calculations only)
  - wannier90_centres.xyz

Output structure (per composition):
  <output-dir>/<composition>/
    set.000/
      coord.npy, energy.npy, force.npy, box.npy[, virial.npy]
      atomic_dipole.npy    # shape (n_frames, 3 * n_sel_atoms)
    type.raw, type_map.raw
  <output-dir>/metadata.json
        """,
    )

    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, default="./DATA")
    parser.add_argument(
        "--type-map",
        "-t",
        nargs="+",
        type=str,
        required=True,
        help="Element type map (e.g., Pt O H). Order matters for DeePMD.",
    )
    parser.add_argument(
        "--sel-type",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Atom types that receive dipole labels. Defaults to "
            "{O, Na, Cl, K, Li, Cs} intersected with --type-map."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories; only */OUTCAR under input-dir",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-file-loc",
        type=str,
        help="File path to save processed OUTCAR parent directory locations",
    )


def handle_command(args: argparse.Namespace) -> int:
    logger = get_logger()

    verbose = getattr(args, "verbose", False)
    if verbose:
        import logging

        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Unified OUTCAR + Wannier -> DPData Conversion")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Type map: {args.type_map}")
    if args.sel_type:
        logger.info(f"Sel type: {args.sel_type}")
    logger.info(f"Recursive search: {not args.no_recursive}")

    if not Path(args.input_dir).exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    if args.dry_run:
        logger.info("\nDRY RUN MODE - No conversion will be performed")
        converter = WCDPDataConverter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            type_map=args.type_map,
            sel_type=args.sel_type,
            recursive=not args.no_recursive,
            verbose=verbose,
            save_file_loc=args.save_file_loc,
        )
        outcars = converter.find_outcars()
        logger.info(f"Would process {len(outcars)} OUTCARs")
        for p in outcars[:10]:
            has_wc = (p.parent / "wannier90_centres.xyz").exists()
            tag = "WC" if has_wc else "no-WC"
            logger.info(f"  [{tag}] {p}")
        if len(outcars) > 10:
            logger.info(f"  ... and {len(outcars) - 10} more")
        return 0

    try:
        converter = WCDPDataConverter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            type_map=args.type_map,
            sel_type=args.sel_type,
            recursive=not args.no_recursive,
            verbose=verbose,
            save_file_loc=args.save_file_loc,
        )
        converter.run()
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(
        prog="mlip-wc-dpdata",
        description=(
            "Convert VASP OUTCARs + Wannier centroids to dpdata format with "
            "type-map-aligned atomic dipoles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, default="./DATA")
    parser.add_argument("--type-map", "-t", nargs="+", type=str, required=True)
    parser.add_argument("--sel-type", nargs="+", type=str, default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-file-loc", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    sys.exit(handle_command(args))
