# Checkpoint post-processing

This directory contains scripts to post-process checkpoints "offline".

## `checkpoint_metadata.py`

Cmdline tool to dump checkpoint metadata to `stdout`. Can also be used as a module to use the underlying reader in other scripts.

## `overwrite_checkpoint.py`

Cmdline tool to overwrite checkpoint data files with functions of the checkpoint grid node positions.
Given a python (numpy) function in the script that takes cartesian positions `(x, y, z)` and returns a scalar or vector
per node, a checkpoint/XDMF binary data file can be overwritten, and then read back in with the checkpoint reader.

This can be useful to generate/modify data with other tools and hand them into the framework.