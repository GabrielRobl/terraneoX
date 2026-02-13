#!/usr/bin/env python3

import argparse
import struct
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import pprint
from typing import List, Optional


@dataclass(frozen=True)
class SubdomainInfo:
    diamond_id: int
    subdomain_x: int
    subdomain_y: int
    subdomain_r: int

    @staticmethod
    def from_global_id(global_id: int) -> "SubdomainInfo":
        diamond_id = (global_id >> 57) & ((1 << 7) - 1)
        subdomain_x = (global_id >> 0) & ((1 << 19) - 1)
        subdomain_y = (global_id >> 19) & ((1 << 19) - 1)
        subdomain_r = (global_id >> 38) & ((1 << 19) - 1)

        info = SubdomainInfo(
            diamond_id=diamond_id,
            subdomain_x=subdomain_x,
            subdomain_y=subdomain_y,
            subdomain_r=subdomain_r,
        )

        if global_id != info.global_id():
            raise ValueError("Invalid global ID conversion.")

        return info

    def global_id(self) -> int:
        return (
                (self.diamond_id << 57)
                | (self.subdomain_r << 38)
                | (self.subdomain_y << 19)
                | (self.subdomain_x << 0)
        )


@dataclass
class GridDataFile:
    grid_name: str
    scalar_data_type: int
    scalar_bytes: int
    vec_dim: int


@dataclass
class CheckpointMetadata:
    version: int
    num_subdomains_per_diamond_lateral_direction: int
    num_subdomains_per_diamond_radial_direction: int
    size_x: int
    size_y: int
    size_r: int
    radii: List[float]
    grid_data_bytes: int
    grid_data_files: List[GridDataFile]
    checkpoint_subdomain_ordering: int
    subdomain_ids: Optional[List[SubdomainInfo]]


class BinaryReader:
    def __init__(self, f, byteorder="little"):
        self.f = f
        self.endian = "<" if byteorder == "little" else ">"

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("Failed to read from input stream.")
        return b

    def i32(self) -> int:
        return struct.unpack(self.endian + "i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack(self.endian + "q", self.read(8))[0]

    def f64(self) -> float:
        return struct.unpack(self.endian + "d", self.read(8))[0]

    def string(self, n: int) -> str:
        return self.read(n).decode("utf-8", errors="replace")


def read_checkpoint_metadata(path: str) -> CheckpointMetadata:
    with open(path, "rb") as f:
        r = BinaryReader(f)

        version = r.i32()
        num_lat = r.i32()
        num_rad = r.i32()
        size_x = r.i32()
        size_y = r.i32()
        size_r = r.i32()

        # ---- radii ----
        n_radii = num_rad * (size_r - 1) + 1
        radii: List[float] = []

        for i in range(n_radii):
            val = r.f64()
            if i > 0 and val <= radii[-1]:
                raise ValueError("Radii are not sorted correctly in checkpoint.")
            radii.append(val)

        if version > 0:
            grid_data_bytes = r.i32()

        # ---- grid data files ----
        num_grid_data_files = r.i32()
        grid_data_files: List[GridDataFile] = []

        for _ in range(num_grid_data_files):
            name_len = r.i32()
            name = r.string(name_len)

            scalar_data_type = r.i32()
            scalar_bytes = r.i32()
            vec_dim = r.i32()

            grid_data_files.append(
                GridDataFile(
                    grid_name=name,
                    scalar_data_type=scalar_data_type,
                    scalar_bytes=scalar_bytes,
                    vec_dim=vec_dim,
                )
            )

        # ---- ordering ----
        checkpoint_subdomain_ordering = r.i32()

        ordering_ids: Optional[List[SubdomainInfo]] = None
        if checkpoint_subdomain_ordering == 0:
            num_global_subdomains = (
                    10 * num_lat * num_lat * num_rad
            )

            ordering_ids = [
                SubdomainInfo.from_global_id(r.i64())
                for _ in range(num_global_subdomains)
            ]

        return CheckpointMetadata(
            version=version,
            num_subdomains_per_diamond_lateral_direction=num_lat,
            num_subdomains_per_diamond_radial_direction=num_rad,
            size_x=size_x,
            size_y=size_y,
            size_r=size_r,
            radii=radii,
            grid_data_bytes=grid_data_bytes,
            grid_data_files=grid_data_files,
            checkpoint_subdomain_ordering=checkpoint_subdomain_ordering,
            subdomain_ids=ordering_ids,
        )


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint_metadata.bin in a checkpoint directory"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to checkpoint directory containing checkpoint_metadata.bin",
    )
    parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Do not print global subdomain ID list (can be very large)",
    )

    args = parser.parse_args()

    metadata_path = args.checkpoint_dir / "checkpoint_metadata.bin"

    if not metadata_path.is_file():
        parser.error(f"File not found: {metadata_path}")

    try:
        metadata = read_checkpoint_metadata(metadata_path)
    except Exception as e:
        parser.error(str(e))

    if args.no_ids:
        metadata = CheckpointMetadata(
            **{
                **metadata.__dict__,
                "checkpoint_ordering_0_global_subdomain_ids": None,
            }
        )

    pprint.pprint(metadata)
    print()
    print("(scalar_data_type: 0 = int, 1 = uint, 2 = float)")


if __name__ == "__main__":
    main()
