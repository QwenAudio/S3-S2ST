import json
import os
from contextlib import suppress

# ------------------------------------------
#  Import Stria
# ------------------------------------------
_stria_imported = False
try:
    import stria

    _stria_imported = True
except ImportError as e:
    print(f"[stria] Failed to import stria. Error:{e}")

# ------------------------------------------
#  CUPTI Configuration Parsing from JSON
# ------------------------------------------
_ar_cfg_tplt = {}
_pcs_cfg_tplt = {}
_active_mode = None  # None, 'AR', or 'PCS'

if _stria_imported:
    config_str = os.environ.get("STRIA_CONFIG")
    if not config_str:
        print("[stria] STRIA_CONFIG is not set. Profiling disabled.")
    else:
        try:
            user_config = json.loads(config_str)
            if not isinstance(user_config, dict):
                raise TypeError("[stria] STRIA_CONFIG must be a JSON dictionary.")
            
            _dump_dir_root = user_config.get("dump_dir", "/path/to/logs")
            # -------------------------------------------------
            # 互斥逻辑判断
            # -------------------------------------------------
            has_ar = "activity_record" in user_config
            has_pcs = "pc_sampling" in user_config

            if has_ar and has_pcs:
                print("[stria] Both AR and PCS are configured.")
                _active_mode = None
            elif has_pcs:
                _active_mode = 'PCS'
            elif has_ar:
                _active_mode = 'AR'
            else:
                _active_mode = None
                print("[stria] No valid section found in STRIA_CONFIG.")
            
            # -------------------------------------------------
            # Parse config
            # -------------------------------------------------
            if _active_mode == 'AR':
                ar_cfg = user_config["activity_record"]    
                # Activity Level Parsing
                cupti_level = ar_cfg.get("level", 0)
                level_to_activities = {
                    1: "CONCURRENT_KERNEL",
                    2: "CONCURRENT_KERNEL,MARKER",
                    3: "CONCURRENT_KERNEL,MEMCPY",
                    4: "CONCURRENT_KERNEL,MEMCPY,MARKER",
                    5: "CONCURRENT_KERNEL,MEMCPY,RUNTIME",
                    6: "CONCURRENT_KERNEL,MEMCPY,RUNTIME,MARKER",
                    7: "CONCURRENT_KERNEL,MEMCPY,RUNTIME,DRIVER",
                    8: "CONCURRENT_KERNEL,MEMCPY,RUNTIME,DRIVER,MARKER",
                }
                
                if cupti_level not in level_to_activities:
                    print(f"[stria] Invalid AR level {cupti_level}. Disabled.")
                    _active_mode = None
                else:
                    _ar_cfg_tplt["enabled_acts"] = level_to_activities[cupti_level]
                    _ar_cfg_tplt["enable_lat_ts"] = ar_cfg.get(
                        "enable_lat_ts", 0)
                    _ar_cfg_tplt["rotate_bytes"] = ar_cfg.get(
                        "rotate_bytes", 100 * 1024 * 1024)
                    _ar_cfg_tplt["rotate_files"] = ar_cfg.get(
                        "rotate_files", 50)
                    _ar_cfg_tplt["buf_blk_size"] = ar_cfg.get(
                        "buf_blk_size", 32 * 1024 * 1024)
                    _ar_cfg_tplt["buf_blk_pre_alloc"] = ar_cfg.get(
                        "buf_blk_pre_alloc", 50)
                    _ar_cfg_tplt["buf_blk_limit"] = ar_cfg.get(
                        "buf_blk_limit", 200)
                    _ar_cfg_tplt["buf_loc"] = ar_cfg.get(
                        "buf_loc", 0)
                
                print("[stria] CUPTI Mode: Activity Record (AR)")
            
            elif _active_mode == 'PCS':
                pcs_cfg = user_config["pc_sampling"]

                _pcs_cfg_tplt["sampling_period"] = pcs_cfg.get(
                    "period", 5)
                _pcs_cfg_tplt["stall_reasons"] = pcs_cfg.get(
                    "stall_reasons", True)
                _pcs_cfg_tplt["buffer_size"] = pcs_cfg.get(
                    "buffer_size", 10 * 1024 * 1024)
                _pcs_cfg_tplt["rotate_bytes"] = pcs_cfg.get(
                    "rotate_bytes", 100 * 1024 * 1024)
                _pcs_cfg_tplt["rotate_files"] = pcs_cfg.get(
                    "rotate_files", 50)
                
                print("[stria] CUPTI Mode: PC Sampling (PCS)")

            # -------------------------------------------------
            # Create Directories
            # -------------------------------------------------
            if _active_mode:
                if not os.path.exists(_dump_dir_root):
                    try:
                        os.makedirs(_dump_dir_root, exist_ok=True)
                    except OSError:
                        print(f"[stria] Cannot create root {_dump_dir_root}")
                        _active_mode = None

                # AR -> logs/activity_record/
                # PCS -> logs/pc_sampling/
                if _active_mode == 'AR':
                    sub_dir = os.path.join(_dump_dir_root, "activity_record")
                else:
                    sub_dir = os.path.join(_dump_dir_root, "pc_sampling")
                
                try:
                    os.makedirs(sub_dir, exist_ok=True)
                except OSError:
                    print(f"[stria] Cannot create subdir {sub_dir}")
                    _active_mode = None

        except (json.JSONDecodeError, TypeError) as e:
            print(
                f"[stria] Failed to parse STRIA_CONFIG JSON. Error: {e}"
            )
            _active_mode = None


def init_stria_on_vllm_worker(rank: int):
    """
    Initializes EITHER Activity Record OR PC Sampling based on config.
    """
    if _active_mode is None:
        return

    worker_pid = os.getpid()
    if _active_mode == 'AR':
        try:
            ar_config = _ar_cfg_tplt.copy()
            base_dir = os.path.join(_dump_dir_root, "activity_record")
            ar_config["log_base"] = os.path.join(
                base_dir, f"cupti_worker_{worker_pid}.log"
            )
            ar_config["rank_id"] = rank
            ar_config["pid"] = worker_pid
            stria.cupti.activity_record.init(ar_config)
            print(f"[stria][AR] Initialized for rank {rank}, and pid: {worker_pid}")
        except Exception as e:
            print(f"[stria][AR] Init failed: {e}")
    elif _active_mode == 'PCS':
        try:
            pcs_config = _pcs_cfg_tplt.copy()
            base_dir = os.path.join(_dump_dir_root, "pc_sampling")
            pcs_config["log_base"] = os.path.join(
                base_dir, f"cupti_worker_{worker_pid}.log")
            pcs_config["rank_id"] = rank
            pcs_config["pid"] = worker_pid
            stria.cupti.pc_sampling.init(pcs_config)
            print(f"[stria][PCS] Initialized for rank {rank}, and pid: {worker_pid}")
        except Exception as e:
             print(f"[stria][PCS] Init failed: {e}")


def cupti_flush():
    if _active_mode == 'AR':
        with suppress(Exception):
            stria.cupti.activity_record.flush()
    elif _active_mode == 'PCS':
        with suppress(Exception):
            if hasattr(stria, 'pc_sampling'):
                stria.cupti.pc_sampling.flush()


def cupti_shutdown(rank: int):
    if _active_mode is None:
        return

    print(f"[stria] Shutting down stria ({_active_mode}) for rank {rank}")
    
    if _active_mode == 'AR':
        try:
            stria.cupti.activity_record.shutdown(rank)
        except Exception as e:
            print(f"[stria][AR] Shutdown exception: {e}")
            
    elif _active_mode == 'PCS':
        try:
            if hasattr(stria, 'pc_sampling'):
                stria.cupti.pc_sampling.shutdown(rank)
        except Exception as e:
            print(f"[stria][PCS] Shutdown exception: {e}")