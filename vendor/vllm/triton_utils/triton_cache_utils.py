# SPDX-License-Identifier: Apache-2.0
"""
Utilities to resolve Triton .so inode conflicts after CRIU restore.

After CRIU restore, old .so file mappings may still exist in process memory
with stale inodes. When Triton tries to load a new .so, it may find the old
mapping and use it instead of loading from disk, causing inode conflicts.

This module provides functions to detect and resolve such conflicts by creating
copies of .so files with new inodes.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from vllm.logger import init_logger

logger = init_logger(__name__)


def _find_triton_so_in_maps() -> list[tuple[str, int]]:
    """
    Find Triton-related .so files in /proc/self/maps.
    
    Returns:
        List of (path, inode) tuples for Triton-related .so files.
    """
    triton_sos: list[tuple[str, int]] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                # Parse /proc/self/maps format:
                # address-range perms offset dev inode pathname
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                path = parts[5] if len(parts) > 5 else ""
                if not path:
                    continue
                
                # Check if it's a Triton-related .so
                if (
                    ".so" in path
                    and (
                        "triton" in path.lower()
                        or "/root/.triton" in path
                        or "/dev/shm/torchinductor_root/triton" in path
                        or "__triton_launcher.so" in path
                        or "cuda_utils.so" in path
                    )
                ):
                    try:
                        # Extract inode (5th field, index 4)
                        inode = int(parts[4])
                        triton_sos.append((path, inode))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.debug("Failed to parse /proc/self/maps: %s", e)
    
    return triton_sos


def _create_so_copy(so_path: str) -> str:
    """
    Create a copy of .so file with a new inode.
    
    Args:
        so_path: Path to the .so file to copy.
    
    Returns:
        Path to the new copy with a new inode.
    """
    if not os.path.exists(so_path):
        return so_path
    
    try:
        so_dir = os.path.dirname(so_path)
        so_name = os.path.basename(so_path)
        so_base, so_ext = os.path.splitext(so_name)
        
        # Use tempfile to generate unique name
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f"{so_base}_", suffix=so_ext, dir=so_dir
        )
        os.close(temp_fd)
        
        # Copy file content (this creates new inode)
        shutil.copy2(so_path, temp_path)
        
        return temp_path
    
    except Exception as e:
        logger.debug("Failed to create copy for %s: %s", so_path, e)
        return so_path
