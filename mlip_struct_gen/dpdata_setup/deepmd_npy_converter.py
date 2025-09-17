"""Converter for VASP OUTCAR files to dpdata/DeepMD npy format."""

import re
import shutil
import tempfile
from pathlib import Path

import dpdata
import numpy as np

from mlip_struct_gen.utils.logger import get_logger


def natural_sort_key(text: str) -> list:
    """
    Convert a string into a list of mixed integers and strings for natural sorting.

    Args:
        text: String to be sorted

    Returns:
        List of integers and strings for sorting
    """

    def convert(match):
        return int(match) if match.isdigit() else match

    return [convert(c) for c in re.split(r"(\d+)", text)]


def validate_dpdata_consistency(
    first_data: dpdata.LabeledSystem,
    current_data: dpdata.LabeledSystem,
    first_dir: str,
    current_dir: str,
) -> None:
    """
    Validate that two dpdata systems have consistent structure.

    Args:
        first_data: First dpdata system
        current_data: Current dpdata system to compare
        first_dir: Directory name of first system
        current_dir: Directory name of current system

    Raises:
        ValueError: If systems are inconsistent
    """
    if first_data.get_natoms() != current_data.get_natoms():
        raise ValueError(
            f"Inconsistent number of atoms: {first_dir} has {first_data.get_natoms()}, "
            f"but {current_dir} has {current_data.get_natoms()}"
        )

    first_types = first_data["atom_types"]
    current_types = current_data["atom_types"]

    if not np.array_equal(first_types, current_types):
        raise ValueError(f"Inconsistent atom types between {first_dir} and {current_dir}")


def convert_to_dpdata(
    input_dir: str,
    output_dir: str,
    keep_temp: bool = False,
    dry_run: bool = False,
    set_name: str | None = None,
) -> None:
    """
    Convert VASP OUTCAR files from multiple directories to consolidated dpdata format.

    Args:
        input_dir: Directory containing subdirectories with OUTCAR files
        output_dir: Output directory for consolidated dpdata
        keep_temp: Keep temporary deepmd_data folders (default: False)
        dry_run: Preview what will be processed without conversion (default: False)
        set_name: Name of the output set directory. If None, uses the set name from first conversion

    Raises:
        ValueError: If no OUTCAR files found or data inconsistency
        RuntimeError: If conversion fails
    """
    logger = get_logger()

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")

    # Find all subdirectories with OUTCAR files
    outcar_dirs = []
    for subdir in sorted(input_path.iterdir(), key=lambda x: natural_sort_key(x.name)):
        if subdir.is_dir():
            outcar_path = subdir / "OUTCAR"
            if outcar_path.exists():
                outcar_dirs.append(subdir)

    if not outcar_dirs:
        raise ValueError(f"No OUTCAR files found in subdirectories of {input_path}")

    logger.info(f"Found {len(outcar_dirs)} directories with OUTCAR files")

    if dry_run:
        logger.info("Dry run mode - listing directories to process:")
        for dir_path in outcar_dirs:
            logger.info(f"  - {dir_path.name}")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to store all numpy arrays by filename
    all_data = {}
    first_data = None
    first_dir_name = None
    actual_set_name = set_name  # Will be determined from first conversion if not provided
    successful_conversions = 0
    failed_conversions = 0
    failed_structures = []

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Process each OUTCAR
        for idx, outcar_dir in enumerate(outcar_dirs):
            logger.step(f"Processing {outcar_dir.name} ({idx+1}/{len(outcar_dirs)})")

            try:
                # Convert OUTCAR to dpdata
                outcar_path = outcar_dir / "OUTCAR"
                ls = dpdata.LabeledSystem(str(outcar_path), fmt="vasp/outcar")

                # Validate consistency
                if first_data is None:
                    first_data = ls
                    first_dir_name = outcar_dir.name
                else:
                    validate_dpdata_consistency(first_data, ls, first_dir_name, outcar_dir.name)

                # Create temporary deepmd directory
                temp_deepmd = temp_path / f"deepmd_data_{outcar_dir.name}"
                ls.to("deepmd/npy", str(temp_deepmd))

                # Get the set name from the first conversion if not provided
                if successful_conversions == 0 and actual_set_name is None:
                    # Find the set directory created by dpdata
                    set_dirs = [
                        d for d in temp_deepmd.iterdir() if d.is_dir() and d.name.startswith("set.")
                    ]
                    if not set_dirs:
                        raise RuntimeError(f"No set directory created for {outcar_dir.name}")
                    actual_set_name = set_dirs[0].name
                    logger.info(f"Using set name from first conversion: {actual_set_name}")

                # Use the determined set name
                temp_set = temp_deepmd / actual_set_name
                if not temp_set.exists():
                    raise RuntimeError(f"{actual_set_name} not created for {outcar_dir.name}")

                # Collect all npy files
                for npy_file in temp_set.glob("*.npy"):
                    file_name = npy_file.name
                    data = np.load(npy_file)

                    if file_name not in all_data:
                        all_data[file_name] = []

                    all_data[file_name].append(data)

                # Copy raw files from first successful directory only
                if successful_conversions == 0:
                    for raw_file in temp_deepmd.glob("*.raw"):
                        shutil.copy2(raw_file, output_path / raw_file.name)
                    logger.debug(f"Copied raw files from {outcar_dir.name}")

                # Keep temporary folder if requested
                if keep_temp:
                    keep_path = output_path / f"temp_{outcar_dir.name}"
                    shutil.copytree(temp_deepmd, keep_path)

                successful_conversions += 1

            except Exception as e:
                logger.error(f"Failed to process {outcar_dir.name}: {str(e)}")
                failed_conversions += 1
                failed_structures.append(outcar_dir.name)
                continue  # Continue with next structure instead of raising

        # Check if any conversions were successful
        if successful_conversions == 0:
            raise RuntimeError(
                f"All {len(outcar_dirs)} OUTCAR conversions failed. No dpdata output created."
            )

        # Create the output set directory now that we know the actual set name
        set_path = output_path / actual_set_name
        set_path.mkdir(exist_ok=True)

        # Concatenate all data
        logger.step("Concatenating all snapshot data...")
        for file_name, data_list in all_data.items():
            # Stack along first dimension (frame dimension)
            concatenated = np.concatenate(data_list, axis=0)
            output_file = set_path / file_name
            np.save(output_file, concatenated)
            logger.debug(f"Saved {file_name}: shape {concatenated.shape}")

    # Report conversion statistics
    logger.info("")
    logger.info("Conversion completed:")
    logger.info(f"  - Successful: {successful_conversions}/{len(outcar_dirs)} structures")
    logger.info(f"  - Failed: {failed_conversions}/{len(outcar_dirs)} structures")

    if failed_structures:
        logger.warning(f"Failed structures: {', '.join(failed_structures)}")

    if successful_conversions > 0:
        logger.success(
            f"Successfully converted {successful_conversions} snapshots to {output_path}"
        )
        logger.info("Output directory structure:")
        logger.info(f"  {output_path}/")
        logger.info("    ├── type.raw")
        logger.info("    ├── type_map.raw")
        logger.info(f"    └── {actual_set_name}/")
        logger.info("        ├── coord.npy")
        logger.info("        ├── energy.npy")
        logger.info("        ├── force.npy")
        logger.info("        └── box.npy (if periodic)")


def convert_to_dpdata_batch(input_dirs: list[str], output_base_dir: str, **kwargs) -> None:
    """
    Convert multiple VASP directories to dpdata format in batch.

    Args:
        input_dirs: List of input directories to convert
        output_base_dir: Base output directory
        **kwargs: Additional arguments passed to convert_to_dpdata
    """
    logger = get_logger()
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for input_dir in input_dirs:
        input_path = Path(input_dir)
        output_name = f"{input_path.name}_dpdata"
        output_path = output_base / output_name

        logger.info(f"Converting {input_path.name} -> {output_name}")
        try:
            convert_to_dpdata(str(input_path), str(output_path), **kwargs)
        except Exception as e:
            logger.error(f"Failed to convert {input_path.name}: {str(e)}")
            continue
