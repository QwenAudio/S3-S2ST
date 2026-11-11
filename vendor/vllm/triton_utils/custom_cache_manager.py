# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
import torch.distributed
from triton.runtime.cache import FileCacheManager

import vllm.envs as envs
from vllm.logger import init_logger


def get_home_dir():
    return os.getenv("TRITON_HOME", Path.home())


try:
    from triton.runtime.cache import default_cache_dir
except ImportError:

    def default_cache_dir():
        return os.path.join(get_home_dir(), ".triton", "cache")


try:
    from triton.runtime.cache import default_dump_dir
except ImportError:

    def default_cache_dir():
        return os.path.join(get_home_dir(), ".triton", "dump")


try:
    from triton.runtime.cache import default_override_dir
except ImportError:

    def default_cache_dir():
        return os.path.join(get_home_dir(), ".triton", "override")


logger = init_logger(__name__)

# Process-level inode mapping: inode -> set of file paths
# Used to detect conflicts where different paths have the same inode
_inode_to_paths: dict[int, set[str]] = {}


def maybe_set_triton_cache_manager() -> None:
    """Set environment variable to tell Triton to use a
    custom cache manager"""
    cache_manger = os.environ.get("TRITON_CACHE_MANAGER", None)
    if cache_manger is None:
        manager = "vllm.triton_utils.custom_cache_manager:CustomCacheManager"
        logger.info("Setting Triton cache manager to: %s", manager)
        os.environ["TRITON_CACHE_MANAGER"] = manager


class CustomCacheManager(FileCacheManager):
    """Re-implements Triton's cache manager, ensuring that a
    unique cache directory is created for each process. This is
    needed to avoid collisions when running with tp>1 and
    using multi-processing as the distributed backend.

    Note this issue was fixed by triton-lang/triton/pull/4295,
    but the fix is not yet included in triton==v3.0.0. However,
    it should be included in the subsequent version.
    """

    # Copy count per file path (max 3 copies per file)
    _MAX_COPIES: int = 3

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = (
                os.getenv("TRITON_CACHE_DIR",
                          os.path.join(envs.VLLM_CACHE_ROOT, "triton")
                ).strip() or default_cache_dir()
            )
            if self.cache_dir:
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    self.cache_dir = f"{self.cache_dir}_{rank}"
                else:
                    self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def put_group(self, filename: str, group: dict[str, str]) -> str:
        if envs.VLLM_PRINT_TRITON_CACHE_COMPILATIONS:
            logger.info(
                "Updating triton cache group: %s -> %r", filename, group
            )  # type: ignore
        return super().put_group(filename, group)

    def put(self, data, filename, binary=True) -> str:
        """
        Store data in cache and return the file path.
        
        This is called by Triton after compiling a .so file. We intercept this
        to check for inode conflicts BEFORE the file is loaded. If a conflict
        is detected, we create a copy with a new inode and REPLACE the original
        file with the copy, ensuring that get_file() will also return the correct path.
        
        Flow:
        1. Triton compiles and generates .so file
        2. Triton calls cache.put() to store the file
        3. We write the file using parent class
        4. If it's a .so file, we check for inode conflicts
        5. If conflict detected, create copy and REPLACE original with copy
        6. Return original path (which now points to the copy if conflict was detected)
        7. Triton calls _load_module_from_path() with the returned path
        8. dlopen loads the file (copy if conflict, original if no conflict)
        
        IMPORTANT: We replace the original file with the copy to ensure consistency.
        This way, get_file() will return the same path that put() returned, and
        both will point to the conflict-resolved file.
        
        NOTE: If record_triton_cache_inode() has not been called, _inode_to_paths
        will be empty, and conflict detection will be skipped (returns original path).
        """
        # Call parent class to write the file
        filepath = super().put(data, filename, binary)
        
        # Skip conflict detection if _inode_to_paths is not initialized
        # (record_triton_cache_inode() was not called)
        if _inode_to_paths == {}:
            return filepath

        # Check for inode conflicts if this is a .so file
        if filename.endswith('.so') and os.path.exists(filepath):
            resolved_path = self._check_and_resolve_inode_conflict(filepath)
            
            # If a copy was created (resolved_path != filepath),
            # replace original with copy. This ensures that get_file() will return
            # the same path that put() returned
            if resolved_path != filepath:
                logger.info(
                    "[RUND_TEMPLATE] in put: Replacing original file with copy: "
                    "%s -> %s",
                    filepath,
                    resolved_path,
                )
                # Replace original file with copy (atomic operation)
                os.replace(resolved_path, filepath)

        
        return filepath

    def get_file(self, filename) -> str | None:
        """
        Retrieve a cached file path.
        
        This is called when Triton needs to load a .so file directly.
        We check for inode conflicts and fix them BEFORE returning the path,
        ensuring that dlopen receives a path with the correct inode.
        
        Flow:
        1. get_file is called by Triton
        2. We get the file path from parent class
        3. If it's a .so file, we check for inode conflicts
        4. If conflict detected, create copy and REPLACE original file with copy
        5. Return original path (which now points to the copy if conflict was detected)
        6. Triton calls dlopen with the returned path
        
        IMPORTANT: If a conflict is detected and a copy is created, we REPLACE the
        original file with the copy using os.replace(). This ensures:
        - The fixup is persisted in cache (the file at the original path is now
          the copy)
        - Next time get_file() is called, it will return the same path pointing to
          the conflict-resolved file (no need to create another copy)
        - Consistency: put() and get_file() both work with the same file path
        
        NOTE: If record_triton_cache_inode() has not been called, _inode_to_paths
        will be empty, and conflict detection will be skipped (returns original path).
        """
        result = super().get_file(filename)
        
        # Skip conflict detection if _inode_to_paths is not initialized
        # (record_triton_cache_inode() was not called)
        if _inode_to_paths == {}:
            return result

        if result is not None:
            # Check if this is a .so file that needs inode conflict resolution
            if result.endswith('.so'):
                resolved_path = self._check_and_resolve_inode_conflict(result)

                # If a copy was created (resolved_path != result),
                # replace original with copy. This ensures that the fixup is
                # persisted in cache, and next time get_file() will return the same
                # path pointing to the conflict-resolved file
                if resolved_path != result:
                    logger.warning(
                        "[RUND_TEMPLATE] in get_file: Replacing original file "
                        "with copy: %s -> %s",
                        result,
                        resolved_path,
                    )
                    # Replace original file with copy (atomic operation)
                    # This ensures the fixup is saved in cache
                    os.replace(resolved_path, result)

        
        return result

    def _check_and_resolve_inode_conflict(self, so_path: str) -> str:
        """
        Check if the given .so file path has an inode conflict with already
        loaded files in the current process. A conflict occurs when:
        - The file exists on disk
        - Its inode matches an inode of an already loaded file
          (recorded in _inode_to_paths)
        - But the paths are different
        
        This method only checks the current process's _inode_to_paths mapping,
        which was populated by record_triton_cache_inode() during initialization.
        
        If conflict is detected, create a copy with a new inode (max 3 copies per file).
        New copies are NOT recorded in _inode_to_paths.
        """
        if not os.path.exists(so_path):
            return so_path

        try:
            file_stat = os.stat(so_path)
            file_inode = file_stat.st_ino
            
            # Check if this inode is already mapped to a different path
            if file_inode in _inode_to_paths:
                existing_paths = _inode_to_paths[file_inode]
                # Check if any existing path is different from current path
                if so_path not in existing_paths:
                    # Conflict detected: same inode, different path
                    conflicting_path = next(iter(existing_paths))
                    logger.warning(
                        "[RUND_TEMPLATE] Inode conflict detected: %s (inode=%d) "
                        "conflicts with %s. Creating copy...",
                        so_path,
                        file_inode,
                        conflicting_path,
                    )
                    from vllm.triton_utils.triton_cache_utils import _create_so_copy
                    
                    # Try to create a copy with a new inode (max 3 attempts)
                    for attempt in range(1, self._MAX_COPIES + 1):
                        resolved_path = _create_so_copy(so_path)
                        new_stat = os.stat(resolved_path)
                        new_inode = new_stat.st_ino

                        if new_inode not in _inode_to_paths:
                            logger.info(
                                "[RUND_TEMPLATE] Created copy with new inode "
                                "(attempt %d/%d): %s -> %s (inode: %d -> %d)",
                                attempt,
                                self._MAX_COPIES,
                                so_path,
                                resolved_path,
                                file_inode,
                                new_inode,
                            )
                            return resolved_path

                    # All attempts failed
                    logger.warning(
                        "[RUND_TEMPLATE] Inode conflict detected but max "
                        "attempts (%d) reached: %s (inode=%d) conflicts "
                        "with %s. Using original path.",
                        self._MAX_COPIES,
                        so_path,
                        file_inode,
                        conflicting_path,
                    )
                    return so_path

            
            # No conflict or new inode, return original path
            return so_path

        except Exception as e:
            logger.warning(
                "[RUND_TEMPLATE] Failed to check inode conflict for %s: %s",
                so_path,
                e,
            )
            return so_path

    @classmethod
    def record_triton_cache_inode(cls) -> None:
        """
        Scan current process memory maps (/proc/self/maps)
        and record inode -> file path mappings.
        
        This should be called once during process initialization
        (in Worker or EngineCore) to build the inode conflict
        detection map for the current process.
        
        Each process has its own independent _inode_to_paths mapping, so this must be
        called in each process that may use Triton compilation.
        """
        from vllm.triton_utils.triton_cache_utils import _find_triton_so_in_maps
        
        triton_sos = _find_triton_so_in_maps()
        for path, inode in triton_sos:
            if inode not in _inode_to_paths:
                _inode_to_paths[inode] = set()
            _inode_to_paths[inode].add(path)
        
        if triton_sos:
            logger.info(
                "[RUND_TEMPLATE] in record_triton_cache_inode, "
                "Recorded %d Triton .so files with %d unique inodes",
                len(triton_sos),
                len(_inode_to_paths),
            )
