#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
import os
from checkpoint_metadata import read_checkpoint_metadata


###################################################################
### REPLACE THE FUNCTIONS BELOW WITH WHATEVER YOU WANT TO WRITE ###
###################################################################

def eval_scalar(pos):
    """
    pos: (N, 3)
    returns: (N,)
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.sqrt(x * x + y * y + z * z)


def eval_vec3(pos):
    """
    pos: (N, 3)
    returns: (N, 3)
    """
    # Example: normalize position (with epsilon safety)
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    return pos / np.maximum(r, 1e-12)


###################################################################
###################################################################


def dtype_from_bytes(nbytes: int):
    if nbytes == 4:
        return np.float32
    elif nbytes == 8:
        return np.float64
    else:
        raise ValueError("Float size must be 4 or 8 bytes")


def overwrite_checkpoint(checkpoint_directory: Path, checkpoint_out_basename: str):
    try:
        metadata = read_checkpoint_metadata(os.path.join(checkpoint_directory, "checkpoint_metadata.bin"))
    except Exception as e:
        raise RuntimeError(str(e))

    data_file_name_with_step = checkpoint_out_basename
    data_file_name_without_step = data_file_name_with_step.rsplit("_", 1)[0]

    data_files = [f for f in metadata.grid_data_files if f.grid_name == data_file_name_without_step]

    if len(data_files) != 1:
        raise RuntimeError(
            "Could not find data file name in checkpoint metadata or found multiple checkpoint outfile data names.")

    data_file = data_files[0]

    pos_float_bytes = metadata.grid_data_bytes

    out_float_bytes = data_file.scalar_bytes

    out_vec_dim = data_file.vec_dim

    pos_dtype = dtype_from_bytes(pos_float_bytes)
    out_dtype = dtype_from_bytes(out_float_bytes)

    pos_file = (checkpoint_directory / "geometry.bin")
    out_file = (checkpoint_directory / (data_file_name_with_step + ".bin"))

    pos_bytes = pos_file.stat().st_size
    out_bytes = out_file.stat().st_size

    if pos_bytes % (3 * pos_float_bytes) != 0:
        raise RuntimeError("Position file size is not a multiple of vec3")

    num_points = pos_bytes // (3 * pos_float_bytes)
    expected_out_bytes = num_points * out_vec_dim * out_float_bytes

    if out_bytes != expected_out_bytes:
        raise RuntimeError(
            f"Output file size mismatch:\n"
            f"  expected {expected_out_bytes} bytes\n"
            f"  got      {out_bytes} bytes"
        )

    pos = np.memmap(
        pos_file,
        dtype=pos_dtype,
        mode="r",
        shape=(num_points, 3),
    )

    out = np.memmap(
        out_file,
        dtype=out_dtype,
        mode="r+",
        shape=(num_points,) if out_vec_dim == 1 else (num_points, out_vec_dim),
    )

    if out_vec_dim == 1:
        out[:] = eval_scalar(pos)
    elif out_vec_dim == 3:
        out[:] = eval_vec3(pos)
    else:
        raise RuntimeError(f"No function available for vec dim {out_vec_dim}")

    out.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a function on the vec3 position grid and "
            "overwrite a scalar or vec3 checkpoint data file (in-place).\n\n"
            "Modify this script by editing the function that you want to use to overwrite the data."
        )
    )
    parser.add_argument("checkpoint_directory", type=Path, help="Checkpoint directory")
    parser.add_argument("checkpoint_out_basename", type=str,
                        help="Checkpoint data name (with out the '.bin' suffix) to overwrite (but with the step number). "
                             "The format (scalar of vectorial) will be detected from the metadata and the respective "
                             "function will be called for evaluation.")

    args = parser.parse_args()

    overwrite_checkpoint(args.checkpoint_directory, args.checkpoint_out_basename)


if __name__ == "__main__":
    main()
