"""Converter for VASP OUTCAR files to dpdata raw format."""

import shutil
from pathlib import Path
from typing import List, Optional
import re

import numpy as np
import dpdata

from mlip_struct_gen.utils.logger import get_logger


def natural_sort_key(text: str) -> List:
    """
    Convert a string into a list of mixed integers and strings for natural sorting.
    
    Args:
        text: String to be sorted
        
    Returns:
        List of integers and strings for sorting
    """
    def convert(match):
        return int(match) if match.isdigit() else match
    
    return [convert(c) for c in re.split(r'(\d+)', text)]


def validate_dpdata_consistency(first_data: dpdata.LabeledSystem, 
                                current_data: dpdata.LabeledSystem,
                                first_dir: str, 
                                current_dir: str) -> None:
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
    
    first_types = first_data['atom_types']
    current_types = current_data['atom_types']
    
    if not np.array_equal(first_types, current_types):
        raise ValueError(
            f"Inconsistent atom types between {first_dir} and {current_dir}"
        )


def convert_to_raw(input_dir: str, 
                   output_dir: str,
                   dry_run: bool = False) -> None:
    """
    Convert VASP OUTCAR files from multiple directories to consolidated dpdata raw format.
    
    Args:
        input_dir: Directory containing subdirectories with OUTCAR files
        output_dir: Output directory for consolidated raw data
        dry_run: Preview what will be processed without conversion (default: False)
        
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
    
    # Lists to store all data
    all_coords = []
    all_forces = []
    all_energies = []
    all_boxes = []
    all_virials = []
    
    first_data = None
    first_dir_name = None
    
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
            
            # Extract data for each frame
            nframes = ls.get_nframes()
            for frame_idx in range(nframes):
                # Get coordinates (flatten to 1D)
                coords = ls['coords'][frame_idx].flatten()
                all_coords.append(coords)
                
                # Get forces (flatten to 1D)
                forces = ls['forces'][frame_idx].flatten()
                all_forces.append(forces)
                
                # Get energy (scalar)
                energy = ls['energies'][frame_idx]
                all_energies.append(energy)
                
                # Get box/cell (flatten to 1D)
                if 'cells' in ls.data:
                    box = ls['cells'][frame_idx].flatten()
                    all_boxes.append(box)
                
                # Get virial if available (flatten to 1D)
                if 'virials' in ls.data:
                    virial = ls['virials'][frame_idx].flatten()
                    all_virials.append(virial)
                    
        except Exception as e:
            logger.error(f"Failed to process {outcar_dir.name}: {str(e)}")
            raise
    
    # Save data in raw format
    logger.step("Saving data in raw format...")
    
    # Save atom types and type map from first system
    if first_data is not None:
        # Save type.raw (atom types)
        atom_types = first_data['atom_types']
        np.savetxt(output_path / "type.raw", atom_types, fmt='%d')
        
        # Save type_map.raw (element names)
        if 'atom_names' in first_data.data:
            atom_names = first_data['atom_names']
            with open(output_path / "type_map.raw", 'w') as f:
                for name in atom_names:
                    f.write(f"{name}\n")
    
    # Save coord.raw
    coord_array = np.array(all_coords)
    np.savetxt(output_path / "coord.raw", coord_array, fmt='%.18e')
    logger.debug(f"Saved coord.raw: shape {coord_array.shape}")
    
    # Save force.raw
    force_array = np.array(all_forces)
    np.savetxt(output_path / "force.raw", force_array, fmt='%.18e')
    logger.debug(f"Saved force.raw: shape {force_array.shape}")
    
    # Save energy.raw
    energy_array = np.array(all_energies)
    np.savetxt(output_path / "energy.raw", energy_array, fmt='%.18e')
    logger.debug(f"Saved energy.raw: shape {energy_array.shape}")
    
    # Save box.raw if periodic
    if all_boxes:
        box_array = np.array(all_boxes)
        np.savetxt(output_path / "box.raw", box_array, fmt='%.18e')
        logger.debug(f"Saved box.raw: shape {box_array.shape}")
    
    # Save virial.raw if available
    if all_virials:
        virial_array = np.array(all_virials)
        np.savetxt(output_path / "virial.raw", virial_array, fmt='%.18e')
        logger.debug(f"Saved virial.raw: shape {virial_array.shape}")
    
    logger.success(f"Successfully converted {len(outcar_dirs)} snapshots to raw format in {output_path}")
    logger.info(f"Output files:")
    logger.info(f"  - type.raw (atom types)")
    logger.info(f"  - type_map.raw (element names)")
    logger.info(f"  - coord.raw (coordinates)")
    logger.info(f"  - force.raw (forces)")
    logger.info(f"  - energy.raw (energies)")
    if all_boxes:
        logger.info(f"  - box.raw (cell parameters)")
    if all_virials:
        logger.info(f"  - virial.raw (virial tensors)")


def convert_to_raw_batch(input_dirs: List[str],
                        output_base_dir: str,
                        **kwargs) -> None:
    """
    Convert multiple VASP directories to raw format in batch.
    
    Args:
        input_dirs: List of input directories to convert
        output_base_dir: Base output directory
        **kwargs: Additional arguments passed to convert_to_raw
    """
    logger = get_logger()
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        output_name = f"{input_path.name}_raw"
        output_path = output_base / output_name
        
        logger.info(f"Converting {input_path.name} -> {output_name}")
        try:
            convert_to_raw(str(input_path), str(output_path), **kwargs)
        except Exception as e:
            logger.error(f"Failed to convert {input_path.name}: {str(e)}")
            continue