#!/usr/bin/env python3
import os
import sys
import json
import random
import datetime
import argparse
import subprocess
import hashlib
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

# ---------- Utility ----------
def now_str_with_ms():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def parse_token_count(token_spec):
    """
    Parse token count specification.
    Can be:
    - int/float: number of tokens
    - str "all": consume all available tokens
    - str with suffix: "1M" (million), "2B" (billion), "3T" (trillion), base-10
    Returns: (target_tokens, is_all) where is_all=True means consume all
    """
    if isinstance(token_spec, (int, float)):
        return int(token_spec), False
    if isinstance(token_spec, str):
        token_spec = token_spec.strip().upper()
        if token_spec == "ALL":
            return None, True
        # Parse suffix
        multipliers = {"M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        for suffix, mult in multipliers.items():
            if token_spec.endswith(suffix):
                num_str = token_spec[:-len(suffix)]
                try:
                    return int(float(num_str) * mult), False
                except ValueError:
                    raise ValueError(f"Invalid token count format: {token_spec}")
        # Try parsing as plain number
        try:
            return int(float(token_spec)), False
        except ValueError:
            raise ValueError(f"Invalid token count format: {token_spec}")
    raise ValueError(f"Token count must be int, float, or str, got {type(token_spec)}")


def parse_duration(duration_spec):
    """
    Parse duration specification.
    Can be:
    - int/float: duration in seconds (e.g. 400 -> 400 seconds)
    - str "400h": duration in hours (e.g. 400 * 3600 seconds)
    - str "100wh": 万 hours = 100 * 10_000 hours
    - str "all": use all available duration (no cap)
    - str "all*2", "all * 0.5": use multiplier of total available duration (spaces ignored)
    Returns: (value, mode) where mode is:
      - "seconds": value is duration in seconds (float)
      - "all": value is None (consume all)
      - "all_scale": value is multiplier (float), target = total_available_duration * multiplier
    """
    if isinstance(duration_spec, (int, float)):
        return float(duration_spec), "seconds"
    if isinstance(duration_spec, str):
        s = duration_spec.strip().lower().replace(" ", "")
        # "all*2", "all*0.5" -> multiplier
        if s.startswith("all*"):
            try:
                mult = float(s[4:])
                return mult, "all_scale"
            except ValueError:
                raise ValueError(f"Invalid duration format: {duration_spec}")
        if s == "all":
            return None, "all"
        # "100wh" -> 100 * 10000 * 3600 (万 = 10,000)
        if s.endswith("wh"):
            try:
                num = float(s[:-2])
                return num * 10_000 * 3600, "seconds"
            except ValueError:
                raise ValueError(f"Invalid duration format: {duration_spec}")
        # "400h" -> 400 * 3600
        if s.endswith("h"):
            try:
                num = float(s[:-1])
                return num * 3600, "seconds"
            except ValueError:
                raise ValueError(f"Invalid duration format: {duration_spec}")
        # plain number (seconds)
        try:
            return float(s), "seconds"
        except ValueError:
            raise ValueError(f"Invalid duration format: {duration_spec}")
    raise ValueError(f"Duration must be int, float, or str, got {type(duration_spec)}")


def json_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def json_save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def list_all_matching(root, suffix):
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(suffix):
                matches.append(os.path.join(dirpath, fn))
    return sorted(matches)


def get_file_num_lines(file_path):
    """
    Return number of lines for a file. Reads from .stats file if present (e.g. file.index.stats
    with content {"num_lines": N}), otherwise counts lines in the file.
    """
    stats_path = file_path + ".stats"
    if os.path.exists(stats_path):
        try:
            data = json_load(stats_path)
            return int(data["num_lines"])
        except (KeyError, TypeError, ValueError):
            pass
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def _file_num_lines_item(file_path):
    """Return (file_path, num_lines) for use with multiprocessing."""
    return (file_path, get_file_num_lines(file_path))


def build_groups_by_num_lines(files, max_lines_per_group):
    """
    Group files so each group's total line count <= max_lines_per_group.
    A file with more than max_lines_per_group lines is in a group by itself.
    Returns list of groups (each group is a list of file paths).
    """
    with Pool(processes=cpu_count()) as pool:
        file_lines = list(tqdm(
            pool.imap(_file_num_lines_item, files),
            total=len(files),
            desc="Counting lines",
            unit="file",
        ))
    groups = []
    current_group = []
    current_lines = 0
    for f, num_lines in file_lines:
        if num_lines > max_lines_per_group:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_lines = 0
            groups.append([f])
            continue
        if current_lines + num_lines > max_lines_per_group and current_group:
            groups.append(current_group)
            current_group = []
            current_lines = 0
        current_group.append(f)
        current_lines += num_lines
    if current_group:
        groups.append(current_group)
    return groups

def process_group(group, preproc_dir, preproc_script, command_only=False):
    group_hash = hashlib.sha256(str(group).encode()).hexdigest()
    output_fn = f"group_{group_hash}.jsonl"
    output_path = os.path.join(preproc_dir, output_fn)
    command = [preproc_script, output_path, *group]
    if not command_only:
        subprocess.run(command, check=True)
    return command, output_path, group

def process_file_index(file_path, stride_lines, use_offset_cache=True):
    """
    Process a single file to build its index.
    
    Args:
        file_path: Path to the jsonl file
        stride_lines: Number of lines per block
        use_offset_cache: If True, check for .offset file and use it if available
    
    Returns:
        Tuple of (file_path, offsets) where offsets is a list of dicts with:
        - offset: start byte offset
        - end_offset: end byte offset
        - tokens: total tokens in block
        - lines: number of lines in block
        - duration: total duration in seconds (optional; for duration-based run configs)
    """
    # Check for .offset cache file
    offset_file = file_path + '.offset'
    if use_offset_cache and os.path.exists(offset_file):
        try:
            offsets = json_load(offset_file)
            return file_path, offsets
        except Exception as e:
            print(f"Warning: Failed to load offset cache from {offset_file}: {e}. Rebuilding...")
    
    # Build offsets from scratch
    offsets = []
    with open(file_path, 'rb') as fh:
        line_count = 0
        tokens_sum = 0
        duration_sum = 0    # in seconds
        block_start_offset = fh.tell()
        block_line_count = 0
        while True:
            line = fh.readline()
            if not line:
                if line_count > 0:
                    pos = fh.tell()
                    offsets.append({
                        "offset": block_start_offset,
                        "end_offset": pos,
                        "tokens": tokens_sum,
                        "lines": block_line_count,
                        "duration": duration_sum
                    })
                break
            # try:
            obj = json.loads(line.decode('utf-8'))
            # except:
            #     print(f"Error loading line: {line[:100]}")
            #     continue
            tokens_sum += obj['extra']['source_len']
            if 'items' in obj:
                for item in obj['items']:
                    if 'extra' in item:
                        if 'merged_slices' in item['extra']:
                            for slice in item['extra']['merged_slices']:
                                duration_sum += slice['speech_duration']
                        else:
                            duration_sum += item['extra']['speech_duration']
            line_count += 1
            # Count newline characters (not lines) to match original behavior
            if line.endswith(b'\n'):
                block_line_count += 1
            if line_count % stride_lines == 0:
                pos = fh.tell()
                offsets.append({
                    "offset": block_start_offset,
                    "end_offset": pos,
                    "tokens": tokens_sum,
                    "lines": block_line_count,
                    "duration": duration_sum
                })
                tokens_sum = 0
                duration_sum = 0
                block_start_offset = pos
                block_line_count = 0
    return file_path, offsets

# ---------- DataSource ----------
class DataSource:
    def __init__(self, name, cfg, proj_root):
        self.name = name
        self.raw_dir = cfg['raw_dir']
        self.suffix = cfg['suffix']
        self.weight = cfg['weight']
        self.is_base = cfg.get('base', False)
        self.preproc_required = cfg.get('preproc_required', False)
        self.preproc_group_size = cfg.get('preproc_group_size', 0)
        self.preproc_group_num_lines = cfg.get('preproc_group_num_lines', None)
        if self.preproc_required:
            has_size = self.preproc_group_size and self.preproc_group_size > 0
            has_num_lines = self.preproc_group_num_lines is not None and self.preproc_group_num_lines > 0
            if has_size and has_num_lines:
                raise ValueError(f"DataSource '{name}': preproc_group_size and preproc_group_num_lines are mutually exclusive")
            if not has_size and not has_num_lines:
                raise ValueError(f"DataSource '{name}': when preproc_required=True, set either preproc_group_size or preproc_group_num_lines")
        self.preproc_script = cfg.get('preproc_script', None)
        self.preproc_dir = os.path.join(proj_root, cfg.get('preproc_dir', f"{self.raw_dir}_preproc"))
        self.stride_lines = cfg.get('index_stride', 100)
        self.cache_file = os.path.join(proj_root, 'data_sources', self.name, 'cache.json')
        self.progress_file = os.path.join(proj_root, 'data_sources', self.name, 'progress.json')
        self.index_file = os.path.join(proj_root, 'data_sources', self.name, 'index.json')
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        if self.preproc_required:
            os.makedirs(self.preproc_dir, exist_ok=True)

    def load_cache(self):
        try:
            return json_load(self.cache_file)
        except FileNotFoundError:
            return {"files": {}}

    def save_cache(self, cache):
        json_save(self.cache_file, cache)

    def scan_and_update_cache(self):
        cache = self.load_cache()
        raw_dirs = []
        files_found = []
        if isinstance(self.raw_dir, str):
            raw_dirs.append(self.raw_dir)
        else:
            raw_dirs = self.raw_dir
        for raw_dir in raw_dirs:
            files_found.extend(list_all_matching(raw_dir, self.suffix))
            print(f'#files_found: {len(files_found)}')
        for f in files_found:
            if f not in cache["files"]:
                cache["files"][f] = {"preprocessed": False}
        self.save_cache(cache)

    def run_preprocessor(self, generate_preprocessor_command_list_only=False, ignore_unprocessed=False):
        if not self.preproc_required:
            return

        cache = self.load_cache()
        if ignore_unprocessed:
            print(f"[Preproc] {self.name}: ignoring unprocessed files")
            unprocessed = []
        else:
            unprocessed = [f for f, meta in cache["files"].items() if not meta["preprocessed"]]
        print(f"[Preproc] {self.name}: {len(unprocessed)} unprocessed files")

        if not unprocessed:
            return

        random.shuffle(unprocessed)
        if self.preproc_group_num_lines is not None and self.preproc_group_num_lines > 0:
            groups = build_groups_by_num_lines(unprocessed, self.preproc_group_num_lines)
        else:
            groups = [
                unprocessed[i:i + self.preproc_group_size]
                for i in range(0, len(unprocessed), self.preproc_group_size)
            ]

        print(f"[Preproc] {self.name}: {len(groups)} groups to process")

        command_list_path = os.path.join(self.preproc_dir, 'command_list.txt')
        if generate_preprocessor_command_list_only:
            # Write to task list file
            process_group_partial = partial(process_group, preproc_dir=self.preproc_dir, preproc_script=self.preproc_script, command_only=True)
            results = list(map(process_group_partial, groups))
            with open(command_list_path, 'w') as f:
                for command, output_path, group in results:
                    f.write(f"{' '.join(command)}\n")
        else:
            # Run in parallel
            process_group_partial = partial(process_group, preproc_dir=self.preproc_dir, preproc_script=self.preproc_script)
            # with Pool(processes=int(os.cpu_count() * 0.75)) as pool:
            with Pool(processes=int(os.cpu_count())) as pool:
                results = pool.map(process_group_partial, groups)

        # Update cache after processing
        for command, output_path, group in results:
            for f in group:
                cache["files"][f]["preprocessed"] = True
                cache["files"][f]["output_file"] = output_path

        self.save_cache(cache)

        return command_list_path

    def get_ready_files(self):
        if self.preproc_required:
            return list_all_matching(self.preproc_dir, ".jsonl")
        else:
            return list_all_matching(self.raw_dir, self.suffix)

    def build_index(self):
        """Build block index for each file (offset, end_offset, cumulative tokens so far in block)."""
        try:
            index_data = json_load(self.index_file)
        except FileNotFoundError:
            index_data = {"indexes": {}}
        
        files = self.get_ready_files()
        # Filter out files that already have indexes
        files_to_process = [f for f in files if f not in index_data["indexes"]]
        
        if not files_to_process:
            print(f"[Index] All files already indexed for {self.name}")
            return
        
        print(f"[Index] Building index for {len(files_to_process)} files in {self.name}")
        
        # Create partial function with stride_lines parameter and use_offset_cache=True
        process_file_partial = partial(process_file_index, stride_lines=self.stride_lines, use_offset_cache=True)
        
        # Use multiprocessing with imap_unordered and tqdm
        with Pool(processes=cpu_count()) as pool:
            # Use imap_unordered for better performance and tqdm for progress
            results = list(tqdm(
                pool.imap_unordered(process_file_partial, files_to_process),
                total=len(files_to_process),
                desc=f"Indexing {self.name}",
                unit="file"
            ))
        
        # Update index_data with results
        for file_path, offsets in results:
            index_data["indexes"][file_path] = offsets
        
        json_save(self.index_file, index_data)

    def load_progress(self):
        try:
            return json_load(self.progress_file)
        except FileNotFoundError:
            return {"consumed_blocks": {}}

    def save_progress(self, progress):
        json_save(self.progress_file, progress)

    def get_total_duration(self) -> float:
        """Sum of all block durations in the index (seconds). Used for 'all*multiplier' duration spec."""
        index_data = json_load(self.index_file)
        total = 0.0
        for f in self.get_ready_files():
            for b in index_data["indexes"].get(f, []):
                total += b.get("duration", 0.0)
        return total

    def consume_blocks(self, target_tokens, allow_repeat=False, target_duration=None) -> Tuple[List[Tuple[str,int,int,int]], int, int, float]:
        """
        Return list of (file, offset_start, offset_end, lines) blocks meeting token and/or duration target.
        Stops when total_tokens >= target_tokens, or when target_duration is set and total_duration >= target_duration.
        Blocks may include a 'duration' (seconds) field from .offset; missing duration is treated as 0.
        Returns: (selected, total_tokens, num_epochs, total_duration).
        """
        index_data = json_load(self.index_file)
        progress = self.load_progress()
        selected = []
        total_tokens = 0
        total_duration = 0.0
        num_epochs = 0
        while True:
            for f in self.get_ready_files():
                blocks = index_data["indexes"].get(f, [])
                consumed_idx = progress["consumed_blocks"].get(f, 0)
                for b in blocks[consumed_idx:]:
                    selected.append((f, b['offset'], b['end_offset'], b['lines']))
                    total_tokens += b['tokens']
                    total_duration += b.get('duration', 0.0)
                    consumed_idx += 1
                    if total_tokens >= target_tokens:
                        break
                    if target_duration is not None and total_duration >= target_duration:
                        break
                progress["consumed_blocks"][f] = consumed_idx
                if total_tokens >= target_tokens:
                    break
                if target_duration is not None and total_duration >= target_duration:
                    break
            done = total_tokens >= target_tokens or (target_duration is not None and total_duration >= target_duration)
            if allow_repeat and not done:
                for f in self.get_ready_files():
                    progress["consumed_blocks"][f] = 0
            else:
                break
            num_epochs += 1

        self.save_progress(progress)
        return selected, total_tokens, num_epochs, total_duration

# ---------- Project Manager ----------
class ProjectManager:
    def __init__(self, proj_root):
        self.proj_root = proj_root
        self.config_path = os.path.join(proj_root, 'project.json')
        self.sources: Dict[str, DataSource] = {}
        self.target_lines = 50000  # default value
        self.runs = None
        self.load_project()

    def load_project(self):
        cfg = json_load(self.config_path)
        self.target_lines = cfg.get('target_lines', 50000)
        self.runs = cfg.get('runs', None)
        self.sources.clear()
        for name, scfg in cfg['sources'].items():
            self.sources[name] = DataSource(name, scfg, self.proj_root)

    def get_source(self, name) -> DataSource:
        return self.sources[name]

# ---------- Run Manager ----------
class RunManager:
    def __init__(self, proj_root):
        self.proj_root = proj_root

    def start_run(self):
        run_base_dir = os.path.join(self.proj_root, 'runs')
        os.makedirs(run_base_dir, exist_ok=True)
        run_id = len(os.listdir(run_base_dir)) + 1
        run_dir = os.path.join(run_base_dir, f"{run_id:03d}_{now_str_with_ms()}")
        os.makedirs(run_dir, exist_ok=True)
        output_dir = os.path.join(run_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        return run_dir, output_dir

def copy_block(args):
    fpath, start, end, out_file = args
    with open(fpath, 'rb') as src, open(out_file, 'ab') as dst:
        src.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = src.read(min(1024*1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    return out_file

def _write_one_copy_job(args):
    """Write one copy job JSON file; returns command line string. Used with ThreadPoolExecutor."""
    idx, blocks, out_file, copy_jobs_dir, copy_script = args
    job_path = os.path.join(copy_jobs_dir, f'job_{idx:05d}.json')
    job = {
        'blocks': [list(b) for b in blocks],
        'output_file': os.path.abspath(out_file),
    }
    json_save(job_path, job)
    return f'python {copy_script} {os.path.abspath(job_path)}'


def copy_blocks_to_file(args):
    """Copy multiple blocks to a single output file."""
    blocks, out_file = args
    with open(out_file, 'wb') as dst:
        for fpath, start, end, lines in blocks:
            with open(fpath, 'rb') as src:
                src.seek(start)
                remaining = end - start
                while remaining > 0:
                    chunk = src.read(min(1024*1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
    return out_file

def process_run(pm: ProjectManager, run_id: str, run_cfg: Dict, run_dir: str, out_dir: str, generate_copy_command_list_only: bool = False):
    """
    Process a single run configuration.
    run_cfg should have a "sources" dict with source names as keys and configs with:
    - num_tokens: int/float/str (number, "1M"/"2B"/"3T", or "all")
    - duration: int/float/str (seconds, "400h", "100wh"); "all" for all available; "all*2" or "all*0.5" for multiple of total
    - weight_to_others: float (target tokens = (tokens from num_tokens sources + tokens from duration sources) * weight)
    - duration_to_others: float (target duration = total duration from duration sources * weight)

    Mixing token-based and duration-based sources in one run is supported. Duration sources
    contribute both duration (for duration_to_others) and tokens (for weight_to_others). For
    weight_to_others to be non-zero, at least one source must use num_tokens or duration. For
    duration_to_others to be non-zero, at least one source must use duration.
    """
    print(f"\n[Run {run_id}] Processing run configuration")
    
    run_sources_cfg = run_cfg.get('sources', {})
    if not run_sources_cfg:
        raise ValueError(f"Run {run_id} has no sources configured")
    
    # Separate sources into those with num_tokens and those with weight_to_others
    sources_with_tokens = {}
    sources_with_weight = {}
    
    sources_with_duration = {}
    sources_with_duration_weight = {}

    for source_name, source_run_cfg in run_sources_cfg.items():
        if source_name not in pm.sources:
            raise ValueError(f"Source '{source_name}' not found in project sources")

        if 'num_tokens' in source_run_cfg:
            target_tokens, is_all = parse_token_count(source_run_cfg['num_tokens'])
            sources_with_tokens[source_name] = {
                'target_tokens': target_tokens,
                'is_all': is_all
            }
        elif 'duration' in source_run_cfg:
            value, mode = parse_duration(source_run_cfg['duration'])
            sources_with_duration[source_name] = {'value': value, 'mode': mode}
        elif 'weight_to_others' in source_run_cfg:
            sources_with_weight[source_name] = {
                'weight': source_run_cfg['weight_to_others']
            }
        elif 'duration_to_others' in source_run_cfg:
            sources_with_duration_weight[source_name] = {
                'weight': source_run_cfg['duration_to_others']
            }
        else:
            raise ValueError(
                f"Source '{source_name}' must have one of: 'num_tokens', 'duration', 'weight_to_others', 'duration_to_others'"
            )
    
    # Step 1: Consume sources with explicit token counts
    consumed_blocks_list = []
    total_tokens_from_num_tokens_sources = 0
    total_tokens_from_duration_sources = 0
    total_duration_from_duration_sources = 0.0

    for source_name, cfg in sources_with_tokens.items():
        ds = pm.sources[source_name]
        if cfg['is_all']:
            blocks, tokens, num_epochs, duration = ds.consume_blocks(target_tokens=1 << 60, allow_repeat=False)
            print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, {tokens} tokens (all available)")
        else:
            blocks, tokens, num_epochs, duration = ds.consume_blocks(target_tokens=cfg['target_tokens'], allow_repeat=True)
            target_reached_ratio = tokens / cfg['target_tokens'] if cfg['target_tokens'] > 0 else 0.0
            print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, epochs {num_epochs}, {tokens} tokens (target {cfg['target_tokens']}, ratio {target_reached_ratio:.3f})")
        consumed_blocks_list.append((ds, blocks, tokens, num_epochs, duration))
        total_tokens_from_num_tokens_sources += tokens

    # Step 2: Consume sources with explicit duration
    for source_name, cfg in sources_with_duration.items():
        ds = pm.sources[source_name]
        mode = cfg['mode']
        value = cfg['value']
        if mode == "all":
            blocks, tokens, num_epochs, duration = ds.consume_blocks(
                target_tokens=1 << 60, allow_repeat=False
            )
            print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, {tokens} tokens, {duration:.1f}s (all available)")
        elif mode == "all_scale":
            total_available = ds.get_total_duration()
            target_duration = total_available * value
            blocks, tokens, num_epochs, duration = ds.consume_blocks(
                target_tokens=1 << 60, allow_repeat=True, target_duration=target_duration
            )
            ratio = duration / target_duration if target_duration > 0 else 0.0
            print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, epochs {num_epochs}, {tokens} tokens, {duration:.1f}s (target {target_duration:.1f}s = all*{value}, ratio {ratio:.3f})")
        else:
            # mode == "seconds"
            blocks, tokens, num_epochs, duration = ds.consume_blocks(
                target_tokens=1 << 60, allow_repeat=True, target_duration=value
            )
            ratio = duration / value if value > 0 else 0.0
            print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, epochs {num_epochs}, {tokens} tokens, {duration:.1f}s (target {value:.1f}s, ratio {ratio:.3f})")
        consumed_blocks_list.append((ds, blocks, tokens, num_epochs, duration))
        total_duration_from_duration_sources += duration
        total_tokens_from_duration_sources += tokens

    # Step 3: Consume sources with weight_to_others (tokens)
    # Base = tokens from num_tokens sources + tokens from duration sources (both are in the index)
    total_tokens_for_weight = total_tokens_from_num_tokens_sources + total_tokens_from_duration_sources
    if sources_with_weight and total_tokens_for_weight == 0:
        raise ValueError(
            f"Run {run_id}: sources with 'weight_to_others' require at least one source with 'num_tokens' or 'duration' in the same run. "
            "No tokens were consumed from those sources, so weight_to_others would yield target_tokens=0."
        )
    for source_name, cfg in sources_with_weight.items():
        ds = pm.sources[source_name]
        target_tokens = int(total_tokens_for_weight * cfg['weight'])
        blocks, tokens, num_epochs, duration = ds.consume_blocks(target_tokens=target_tokens, allow_repeat=True)
        target_reached_ratio = tokens / target_tokens if target_tokens > 0 else 0.0
        print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, epochs {num_epochs}, {tokens} tokens (target {target_tokens} based on weight {cfg['weight']} * {total_tokens_for_weight}, ratio {target_reached_ratio:.3f})")
        consumed_blocks_list.append((ds, blocks, tokens, num_epochs, duration))

    # Step 4: Consume sources with duration_to_others
    if sources_with_duration_weight and total_duration_from_duration_sources == 0:
        raise ValueError(
            f"Run {run_id}: sources with 'duration_to_others' require at least one source with 'duration' in the same run. "
            "No duration-based sources were consumed, so duration_to_others would yield target_duration=0."
        )
    for source_name, cfg in sources_with_duration_weight.items():
        ds = pm.sources[source_name]
        target_duration = total_duration_from_duration_sources * cfg['weight']
        blocks, tokens, num_epochs, duration = ds.consume_blocks(
            target_tokens=1 << 60, allow_repeat=True, target_duration=target_duration
        )
        ratio = duration / target_duration if target_duration > 0 else 0.0
        print(f"[Run {run_id}] {source_name}: {len(blocks)} blocks, epochs {num_epochs}, {tokens} tokens, {duration:.1f}s (target {target_duration:.1f}s from duration_to_others, ratio {ratio:.3f})")
        consumed_blocks_list.append((ds, blocks, tokens, num_epochs, duration))

    # Total tokens consumed (for metadata)
    total_tokens_consumed = sum(tokens for _, _, tokens, _, _ in consumed_blocks_list)
    
    # Step 5: Collect all blocks
    all_blocks = []
    for ds, blocks, tokens, num_epochs, duration in consumed_blocks_list:
        for b in blocks:
            all_blocks.append(b)
    
    # Step 6: Group blocks by target lines
    grouped_jobs = []
    current_group = []
    current_lines = 0
    part_idx = 0
    
    print(f"[Run {run_id}] Grouping {len(all_blocks)} blocks with target {pm.target_lines} lines per file")
    
    for block in tqdm(all_blocks, desc=f"Grouping blocks for run {run_id}"):
        block_lines = block[3]  # lines count is the 4th element
        
        if current_lines + block_lines > pm.target_lines and current_group:
            out_file = os.path.join(out_dir, f"part-{part_idx:05d}.jsonl")
            grouped_jobs.append((current_group.copy(), out_file))
            part_idx += 1
            current_group = []
            current_lines = 0
        
        current_group.append(block)
        current_lines += block_lines
    
    # Add the last group if it has any blocks
    if current_group:
        out_file = os.path.join(out_dir, f"part-{part_idx:05d}.jsonl")
        grouped_jobs.append((current_group, out_file))
        part_idx += 1
    
    # Step 7: Copy blocks to files (or generate command list for distributed copy)
    if generate_copy_command_list_only:
        copy_jobs_dir = os.path.join(run_dir, 'copy_jobs')
        os.makedirs(copy_jobs_dir, exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        copy_script = os.path.join(script_dir, 'copy_blocks_to_file.py')
        write_args = [
            (idx, blocks, out_file, copy_jobs_dir, copy_script)
            for idx, (blocks, out_file) in enumerate(grouped_jobs)
        ]
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            command_lines = list(tqdm(
                executor.map(_write_one_copy_job, write_args),
                total=len(write_args),
                desc=f"Writing copy jobs for run {run_id}",
                unit="job",
            ))
        command_list_path = os.path.join(run_dir, 'copy_command_list.txt')
        with open(command_list_path, 'w') as f:
            f.write('\n'.join(command_lines))
        print(f"[Run {run_id}] Copy command list saved ({len(command_lines)} jobs): {command_list_path}")
    else:
        print(f"[Run {run_id}] Copying {len(grouped_jobs)} grouped files to {out_dir}")
        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(copy_blocks_to_file, grouped_jobs),
                              total=len(grouped_jobs), desc=f"Copying files for run {run_id}"):
                pass

    # Step 8: Save metadata
    meta = {
        "run_id": run_id,
        "total_tokens": total_tokens_consumed,
        "sources": {}
    }

    for ds, blocks, tokens, num_epochs, duration in consumed_blocks_list:
        source_cfg = run_sources_cfg[ds.name]
        source_meta = {
            "tokens_used": tokens,
            "duration_used_sec": duration,
            "blocks_used": len(blocks),
            "epochs": num_epochs,
            "weight": ds.weight
        }

        if 'num_tokens' in source_cfg:
            target_tokens, is_all = parse_token_count(source_cfg['num_tokens'])
            if is_all:
                source_meta["target_tokens"] = "all"
            else:
                source_meta["target_tokens"] = target_tokens
                source_meta["target_reached_ratio"] = tokens / target_tokens if target_tokens > 0 else 0.0
        elif 'duration' in source_cfg:
            val, mode = parse_duration(source_cfg['duration'])
            if mode == "all":
                source_meta["target_duration"] = "all"
            elif mode == "all_scale":
                source_meta["target_duration"] = f"all*{val}"
                total_avail = ds.get_total_duration()
                computed = total_avail * val
                source_meta["computed_target_duration_sec"] = computed
                source_meta["duration_target_reached_ratio"] = duration / computed if computed > 0 else 0.0
            else:
                source_meta["target_duration_sec"] = val
                source_meta["duration_target_reached_ratio"] = duration / val if val > 0 else 0.0
        elif 'weight_to_others' in source_cfg:
            source_meta["weight_to_others"] = source_cfg['weight_to_others']
            computed_target = int(total_tokens_for_weight * source_cfg['weight_to_others'])
            source_meta["computed_target_tokens"] = computed_target
            source_meta["total_tokens_base_for_weight"] = total_tokens_for_weight
            source_meta["target_reached_ratio"] = tokens / computed_target if computed_target > 0 else 0.0
        elif 'duration_to_others' in source_cfg:
            source_meta["duration_to_others"] = source_cfg['duration_to_others']
            computed_dur = total_duration_from_duration_sources * source_cfg['duration_to_others']
            source_meta["computed_target_duration_sec"] = computed_dur
            source_meta["duration_target_reached_ratio"] = duration / computed_dur if computed_dur > 0 else 0.0

        meta["sources"][ds.name] = source_meta
    
    json_save(os.path.join(run_dir, 'run_metadata.json'), meta)
    print(f"[Run {run_id}] Complete: {run_dir}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Index-based streaming mixer")
    parser.add_argument('project_dir', help='Project root directory')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--generate-preprocessor-command-list-only', action='store_true', default=False)
    parser.add_argument('--generate-copy-command-list-only', action='store_true', default=False,
                        help='Write copy_jobs/*.json and copy_command_list.txt for distributed copy; do not run copy locally')
    parser.add_argument('--ignore-unprocessed', action='store_true', default=False)
    parser.add_argument('--no-build-cache', action='store_true', default=False)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    pm = ProjectManager(args.project_dir)

    # Step1: scan & preprocess
    for ds in pm.sources.values():
        if args.no_build_cache:
            print(f"[Scan] {ds.name}: skipping cache build")
        else:
            ds.scan_and_update_cache()
        ret = ds.run_preprocessor(generate_preprocessor_command_list_only=args.generate_preprocessor_command_list_only,
                                  ignore_unprocessed=args.ignore_unprocessed)
        if args.generate_preprocessor_command_list_only:
            if ds.preproc_required:
                command_list_path = ret
                print(f"[Run] Command list saved to {command_list_path}")
            continue
        ds.build_index()
    if args.generate_preprocessor_command_list_only:
        print(f"[Run] Command list(s) saved. Exiting.")
        exit(0)

    # Step2: Check if runs configuration exists
    if pm.runs is not None:
        # New logic: process each run
        rm = RunManager(args.project_dir)
        for run_id, run_cfg in pm.runs.items():
            run_dir, out_dir = rm.start_run()
            process_run(pm, run_id, run_cfg, run_dir, out_dir, generate_copy_command_list_only=args.generate_copy_command_list_only)
    else:
        # Old logic: base + secondaries
        # Step2: pick base & secondaries
        base_ds = next(ds for ds in pm.sources.values() if ds.is_base)
        secondaries = [ds for ds in pm.sources.values() if not ds.is_base]

        # Step3: consume base
        base_blocks, base_tokens, num_epochs, _ = base_ds.consume_blocks(target_tokens=1 << 60)  # all blocks
        print(f"[Base] {base_ds.name}: {len(base_blocks)} blocks, {base_tokens} tokens")

        # Step4: consume secondaries on ratio
        sec_blocks_list = []
        for sec in secondaries:
            target_tokens = int(base_tokens * (sec.weight / base_ds.weight))
            blocks, sec_tokens, num_epochs, _ = sec.consume_blocks(target_tokens, allow_repeat=True)
            target_reached_ratio = sec_tokens / target_tokens
            print(f"[Secondary] {sec.name}: {len(blocks)} blocks, epochs {num_epochs}, {sec_tokens} tokens (target {target_tokens}, target reached ratio {target_reached_ratio})")
            sec_blocks_list.append((sec, blocks, sec_tokens, num_epochs, target_tokens, target_reached_ratio))

        # Step5: Group blocks by target lines and prepare copy jobs
        rm = RunManager(args.project_dir)
        run_dir, out_dir = rm.start_run()
        
        # Collect all blocks with their source info
        all_blocks = []
        # Base blocks first, then secondaries in sequence by weight
        sources_seq = [(base_ds.weight, base_blocks, base_ds)] + [(sec.weight, blks, sec) for sec, blks, _, _, _, _ in sec_blocks_list]
        for weight, blocks, ds in sources_seq:
            for b in blocks:
                all_blocks.append(b)
        
        # Group blocks by target lines
        grouped_jobs = []
        current_group = []
        current_lines = 0
        part_idx = 0
        
        print(f"[Group] Grouping {len(all_blocks)} blocks with target {pm.target_lines} lines per file")
        
        for block in tqdm(all_blocks):
            # Use pre-computed line count from index
            block_lines = block[3]  # lines count is now the 4th element
            
            # If adding this block would exceed target, start a new group
            if current_lines + block_lines > pm.target_lines and current_group:
                out_file = os.path.join(out_dir, f"part-{part_idx:05d}.jsonl")
                grouped_jobs.append((current_group.copy(), out_file))
                part_idx += 1
                current_group = []
                current_lines = 0
            
            current_group.append(block)
            current_lines += block_lines
        
        # Add the last group if it has any blocks
        if current_group:
            out_file = os.path.join(out_dir, f"part-{part_idx:05d}.jsonl")
            grouped_jobs.append((current_group, out_file))
            part_idx += 1

        # Step6: Multiprocessing copy (or generate command list for distributed copy)
        if args.generate_copy_command_list_only:
            copy_jobs_dir = os.path.join(run_dir, 'copy_jobs')
            os.makedirs(copy_jobs_dir, exist_ok=True)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            copy_script = os.path.join(script_dir, 'copy_blocks_to_file.py')
            write_args = [
                (idx, blocks, out_file, copy_jobs_dir, copy_script)
                for idx, (blocks, out_file) in enumerate(grouped_jobs)
            ]
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                command_lines = list(tqdm(
                    executor.map(_write_one_copy_job, write_args),
                    total=len(write_args),
                    desc="Writing copy jobs",
                    unit="job",
                ))
            command_list_path = os.path.join(run_dir, 'copy_command_list.txt')
            with open(command_list_path, 'w') as f:
                f.write('\n'.join(command_lines))
            print(f"[Copy] Copy command list saved ({len(command_lines)} jobs): {command_list_path}")
        else:
            print(f"[Copy] Copying {len(grouped_jobs)} grouped files to {out_dir}")
            with Pool(processes=cpu_count()) as pool:
                for result in tqdm(pool.imap_unordered(copy_blocks_to_file, grouped_jobs), total=len(grouped_jobs), desc="Copying files"):
                    pass

        # Step7: Metadata
        meta = {
            "base_source": base_ds.name,
            "base_tokens": base_tokens,
            "base_weight": base_ds.weight,
            "sources": {
                sec.name: {
                    "tokens_used": tokens,
                    "tokens_target": target_tokens,
                    "tokens_target_reached_ratio": target_reached_ratio,
                    "epochs": epochs,
                    "weight": sec.weight
                } for sec, _, tokens, epochs, target_tokens, target_reached_ratio in sec_blocks_list
            }
        }
        json_save(os.path.join(run_dir, 'run_metadata.json'), meta)
        print(f"[Run] Complete: {run_dir}")

if __name__ == "__main__":
    main()
