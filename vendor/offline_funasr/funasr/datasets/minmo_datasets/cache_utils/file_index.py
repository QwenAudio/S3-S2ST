#!/usr/bin/env python3

'''
Load index for a big jsonl file, then seek into it.
For each line in a jsonl file, the index contains:
1. the file path, "file_path"
2. the line number, "line_no"
3. the offset and length of the line in bytes, "offset", "length"
4. extra customizable fields, "extra"
'''

import os
try:
    import ujson as json
except ImportError:
    import json
import mmap
import gc
import tqdm
import time
from multiprocessing import Pool

class FileIndex:
    '''
    This class keeps a index for many files.

    Instead of loading index files fully into memory as Python strings,
    we support mmap mode to map each index file and store just the offsets + lengths.
    '''
    def __init__(self, max_file_handles_to_keep=4, max_file_handle_idle_time=600, use_mmap=True):
        self.index_entries = []       # list of tuples: (index_file_id, offset, length)
        self._index_maps = []         # list of (file_path, mmap_obj)
        self.file_handles = {}        # data file handles for payloads
        self.max_file_handles_to_keep = max_file_handles_to_keep
        self.max_file_handle_idle_time = max_file_handle_idle_time
        self.use_mmap = use_mmap
        print(f"FileIndex use mmap: {use_mmap}")

    def __del__(self):
        # Ensure mmaps are closed
        for _, mm in self._index_maps:
            mm.close()
        self._index_maps.clear()
        # Close payload file handles
        for file_path, file_handle in self.file_handles.items():
            file_handle[0].close()
        self.file_handles.clear()
        self.index_entries.clear()
        gc.collect()

    def _load_index_file(self, index_file: str):
        # print(f"loading {index_file}")
        with open(index_file, 'r', encoding='utf-8') as fp_in:
            entries = fp_in.readlines()
        # print(f"loaded {index_file}")
        return index_file, entries

    def _mmap_index_file(self, index_file: str):
        """Open index file and scan lines; store offset/length for each."""
        f = open(index_file, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        index_id = len(self._index_maps)
        self._index_maps.append((index_file, mm))

        offset = 0
        entries = []
        while True:
            line = mm.readline()
            if not line:
                break
            # strip newline only when decoding later
            length = len(line)
            entries.append((index_id, offset, length))
            offset += length
        f.close()  # safe: mmap holds independent fd
        return entries

    def add_index_file(self, index_file: str):
        if self.use_mmap:
            entries = self._mmap_index_file(index_file)
        else:
            index_file, entries = self._load_index_file(index_file)
        self.index_entries.extend(entries)

    def add_index_files(self, index_files: list[str], num_workers=0):
        if self.use_mmap:
            if num_workers == 0:
                for index_file in tqdm.tqdm(index_files, desc="Loading index files (mmap)"):
                    entries = self._mmap_index_file(index_file)
                    self.index_entries.extend(entries)
                    print(f"added {len(entries)} entries from {index_file}")
            else:
                # We map in main thread because mmap objects aren't picklable
                raise ValueError("mmap mapping must be done in main thread; set num_workers=0")
        else:
            if num_workers == 0:
                for index_file in tqdm.tqdm(index_files, desc="Loading index files"):
                    index_file, entries = self._load_index_file(index_file)
                    # print(f"add {len(entries)} entries from {index_file}")
                    self.index_entries.extend(entries)
                    print(f"added {len(entries)} entries from {index_file}")
            else:
                with Pool(num_workers) as pool:
                    for index_file, entries in pool.imap(self._load_index_file, index_files):
                        # print(f"add {len(entries)} entries from {index_file}")
                        self.index_entries.extend(entries)
                        print(f"added {len(entries)} entries from {index_file}")

    def __len__(self):
        return len(self.index_entries)
    
    def get_index_entry(self, index):
        if self.use_mmap:
            file_id, offset, length = self.index_entries[index]
            mm = self._index_maps[file_id][1]
            raw_line = mm[offset:offset+length].rstrip(b'\n\r')
            return json.loads(raw_line)
        else:
            return json.loads(self.index_entries[index])

    def _manage_file_handles(self):
        # for file_path, file_handle in self.file_handles.items():
        #     if time.time() - file_handle[1] > self.max_file_handle_idle_time:
        #         print(f"close file handle {file_path} because of idle time")
        #         file_handle[0].close()
        #         del self.file_handles[file_path]
        # sort and close
        sorted_file_handles = sorted(self.file_handles.items(), key=lambda x: x[1][1])
        for file_path, file_handle in sorted_file_handles:
            if time.time() - file_handle[1] > self.max_file_handle_idle_time:
                # print(f"close file handle {file_path} because of idle time")
                file_handle[0].close()
                del self.file_handles[file_path]
        if len(self.file_handles) > self.max_file_handles_to_keep:
            for file_path, file_handle in sorted_file_handles[:-self.max_file_handles_to_keep]:
                # print(f"close file handle {file_path} because of max file handles to keep")
                file_handle[0].close()
                del self.file_handles[file_path]

    def _get_file_handle(self, file_path):
        if file_path not in self.file_handles:
            if len(self.file_handles) >= self.max_file_handles_to_keep:
                self._manage_file_handles()
            # print(f"open file handle {file_path}")
            self.file_handles[file_path] = [open(file_path, 'rb'), time.time()]
        file_handle = self.file_handles[file_path]
        return file_handle

    def __getitem__(self, index):
        lines = []
        entry = self.get_index_entry(index)
        if 'items' not in entry:
            # text mode
            entry['extra']['pt_mode'] = 'unimodal-text'
            entry = {
                'items': [entry],
                'extra': entry['extra']
            }
        for item in entry['items']:
            try:
                file_path = item['file_path']
                file_handle = self._get_file_handle(file_path)
                fp_in = file_handle[0]
                fp_in.seek(item['offset'])
                line = fp_in.read(item['length']).decode('utf-8').strip()
                file_handle[1] = time.time()
                pt_mode = entry['extra']['pt_mode']
                lines.append((line, item, pt_mode))
            except Exception as e:
                print(f"error reading file {file_path}: {e}, item {item}")

        return lines

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_files', type=str, required=True, nargs='+')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-file-handles-to-keep', type=int, default=8)
    parser.add_argument('--max-file-handle-idle-time', type=int, default=600)
    parser.add_argument('--not-use-mmap', action='store_true', default=False)
    args = parser.parse_args()
    file_index = FileIndex(max_file_handles_to_keep=args.max_file_handles_to_keep, max_file_handle_idle_time=args.max_file_handle_idle_time, use_mmap=not args.not_use_mmap)
    # for index_file in args.index_files:
    #     print(f"add index file {index_file}")
    #     file_index.add_index(index_file)
    file_index.add_index_files(args.index_files, num_workers=args.num_workers)
    print(f"total entries: {len(file_index.index_entries)}")

    if args.shuffle:
        print("shuffle entries")
        from random import shuffle
        shuffle(file_index.index_entries)

    # print(len(file_index.index_entries))
    # print(len(file_index[0]))
    import tqdm
    for i in tqdm.tqdm(range(len(file_index))):
        items = file_index[i]
        for line, slice_modes, pt_mode in items:
            # print(pt_mode)
            pass