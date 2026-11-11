#!/usr/bin/env python3
"""Join mapping, text chunks and forced alignment into streaming S2TT turns."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, re, sqlite3
from contextlib import ExitStack
from pathlib import Path

GRID_SECONDS = 0.640

def progress(stage, count):
    if count and count % 100000 == 0:
        print(f"[{stage}] {count:,}", flush=True)

def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def mapping_pair_id(row):
    value = row.get("pair_id") or row.get("en_wav_id")
    if not value:
        raise KeyError("mapping has neither pair_id nor en_wav_id")
    return str(value)

def normalized(text):
    return "".join(re.findall(r"[\w]+", str(text).casefold(), flags=re.UNICODE))

def timestamp_entries(segment):
    raw = (((segment.get("asr_gemini") or {}).get("alignments") or {}).get("timestamps") or [])
    out = []
    if isinstance(raw, str):
        for token in raw.split():
            parts = token.rsplit("_", 3)
            if len(parts) != 4:
                continue
            try:
                out.append((parts[0], float(parts[1]), float(parts[2])))
            except ValueError:
                pass
        return out
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text") or item.get("word") or item.get("token") or ""
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            text, start, end = item[:3]
        else:
            continue
        if start is not None and end is not None:
            out.append((str(text), float(start), float(end)))
    return out

def make_absolute(entries, segment_start, segment_end):
    if not entries:
        return []
    duration = max(0.0, segment_end - segment_start)
    if segment_start > 0 and max(x[2] for x in entries) <= duration + 1.0:
        return [(t, s + segment_start, e + segment_start) for t, s, e in entries]
    return entries

def prefix_end_times(source_parts, entries):
    targets, current = [], ""
    for part in source_parts:
        current += part
        targets.append(normalized(current))
    if not targets or not targets[-1]:
        raise ValueError("empty normalized source chunks")
    out, aligned, target_idx = [], "", 0
    for text, _start, end in entries:
        aligned += normalized(text)
        while target_idx < len(targets) and len(aligned) >= len(targets[target_idx]):
            target = targets[target_idx]
            if aligned[:len(target)] != target:
                raise ValueError("alignment tokens do not reconstruct labeled source prefix")
            out.append(end)
            target_idx += 1
    if target_idx != len(targets):
        raise ValueError(f"only matched {target_idx}/{len(targets)} source boundaries")
    return out

def speech_token_length(seconds):
    frames = max(1, int(round(float(seconds) * 100)))
    length = 1 + (frames - 3 + 2) // 2
    length = 1 + (length - 3 + 2) // 2
    return max(1, (length - 1) // 2 + 1)

def quantize_boundaries(end_times, segment_start, segment_end):
    out = [
        min(segment_end, segment_start + math.ceil(max(0.001, end-segment_start)/GRID_SECONDS)*GRID_SECONDS)
        for end in end_times
    ]
    out[-1] = segment_end
    return out

def build_record(mapping, label, entries, direction):
    source_key = "en" if direction == "en2zh" else "zh"
    target_key = "zh" if direction == "en2zh" else "en"
    chunks = label.get("chunks") or []
    if not chunks:
        raise ValueError("empty LLM chunks")
    source_parts = [str(c[1]) for c in chunks] if direction == "en2zh" else [str(c[0]) for c in chunks]
    target_parts = [str(c[0]) for c in chunks] if direction == "en2zh" else [str(c[1]) for c in chunks]
    if "".join(source_parts) != str(mapping[f"{source_key}_text"]):
        raise ValueError("LLM source chunks do not reconstruct mapping text")
    if "".join(target_parts) != str(mapping[f"{target_key}_text"]):
        raise ValueError("LLM target chunks do not reconstruct mapping text")
    segment_start = float(mapping.get(f"{source_key}_start_time", 0.0))
    segment_end = float(mapping.get(f"{source_key}_end_time", -1.0))
    if segment_end < 0:
        duration = float(mapping.get("source_duration_seconds") or 0)
        if duration <= 0:
            raise ValueError("source end_time and duration unavailable")
        segment_end = segment_start + duration
    if segment_end <= segment_start:
        raise ValueError("non-positive source duration")
    ends = prefix_end_times(source_parts, make_absolute(entries, segment_start, segment_end))
    quantized = quantize_boundaries(ends, segment_start, segment_end)
    collapsed = []
    for end, target in zip(quantized, target_parts):
        if collapsed and end <= collapsed[-1][0] + 1e-6:
            collapsed[-1] = (collapsed[-1][0], collapsed[-1][1] + target)
        else:
            collapsed.append((end, target))
    pair_id = mapping_pair_id(mapping)
    segment_id = str(mapping[f"{source_key}_segment_wav_id"])
    audio_path = str(mapping[f"{source_key}_audio_path"])
    prompt = "Translate into Chinese:" if direction == "en2zh" else "Translate into English:"
    system_text="You are a helpful assistant."
    messages = [{"role":"system","text":system_text,"content":system_text,"mode":"tc","audio":"","speech_length":0}]
    previous = segment_start
    for turn, (end, target) in enumerate(collapsed):
        duration = end - previous
        final = turn + 1 == len(collapsed)
        if duration <= 0:
            raise ValueError("non-increasing quantized boundary")
        if duration < GRID_SECONDS*0.90 and not final:
            raise ValueError("sub-640ms non-final audio turn")
        user_text=(prompt if turn == 0 else "") + f"<|startofspeech|>!{audio_path}<|endofspeech|>"
        messages.append({
            "role":"user",
            "text":user_text,"content":user_text,
            "mode":"tc","audio":"","speech_length":speech_token_length(duration),
            "start_time":round(previous,6),"end_time":round(end,6),
            "segment_id":segment_id,"slice_mode":"absolute",
        })
        messages.append({"role":"assistant","text":target,"content":target,"mode":"t","audio":"","speech_length":0})
        previous = end
    return {
        "key":pair_id,"messages":messages,"task":"streaming_s2tt",
        "src_lang":source_key.upper(),"tgt_lang":target_key.upper(),
        "streaming":{"decision_ms":640,"interval_audio_tokens":8,"end_token":"<|im_end|>",
                     "direction":direction,"source_segment_id":segment_id},
    }

def ingest_mapping(path, direction, conn, limit=0):
    source_key = "en" if direction == "en2zh" else "zh"
    # A source segment may legitimately be reused by multiple bilingual pairs.
    conn.execute("CREATE TABLE mapping(pair_id TEXT PRIMARY KEY,source_sid TEXT NOT NULL,payload TEXT NOT NULL)")
    batch, source_ids, pair_ids, count = [], set(), set(), 0
    for row in rows(path):
        pid, sid = mapping_pair_id(row), str(row[f"{source_key}_segment_wav_id"])
        batch.append((pid,sid,json.dumps(row,ensure_ascii=False,separators=(",",":"))))
        source_ids.add(sid); pair_ids.add(pid); count += 1
        progress("mapping", count)
        if len(batch) >= 10000:
            conn.executemany("INSERT OR REPLACE INTO mapping VALUES (?,?,?)",batch); batch.clear()
        if limit and count >= limit:
            break
    if batch: conn.executemany("INSERT OR REPLACE INTO mapping VALUES (?,?,?)",batch)
    conn.execute("CREATE INDEX mapping_source_sid ON mapping(source_sid)")
    conn.commit(); return count, source_ids, pair_ids

def ingest_labels(path, needed, conn):
    conn.execute("CREATE TABLE labels(pair_id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
    batch, count = [], 0
    for row in rows(path):
        pair_id = str(row["pair_id"])
        if pair_id not in needed:
            continue
        batch.append((pair_id,json.dumps(row,ensure_ascii=False,separators=(",",":")))); count += 1
        progress("labels", count)
        if len(batch) >= 10000:
            conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)",batch); batch.clear()
        if count >= len(needed):
            break
    if batch: conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)",batch)
    conn.commit(); return count

def ingest_alignments(path, needed, conn):
    conn.execute("CREATE TABLE align(segment_id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
    batch, found_ids, missing, next_progress = [], set(), 0, 100000
    for row in rows(path):
        for segment in row.get("segments") or []:
            sid = str(segment.get("segment_wav_id") or "")
            if sid not in needed: continue
            entries = timestamp_entries(segment)
            if entries:
                batch.append((sid,json.dumps(entries,ensure_ascii=False,separators=(",",":")))); found_ids.add(sid)
            else: missing += 1
        if len(batch) >= 10000:
            conn.executemany("INSERT OR REPLACE INTO align VALUES (?,?)",batch); batch.clear()
        if len(found_ids) >= len(needed):
            break
        if len(found_ids) >= next_progress:
            print(f"[alignments] {len(found_ids):,}", flush=True)
            next_progress += 100000
    if batch: conn.executemany("INSERT OR REPLACE INTO align VALUES (?,?)",batch)
    conn.commit(); return len(found_ids), missing

def split_name(pair_id, valid_basis_points):
    bucket = int.from_bytes(hashlib.sha1(pair_id.encode()).digest()[:4],"big") % 10000
    return "valid" if bucket < valid_basis_points else "train"

class ShardedWriter:
    def __init__(self, root, train_shards, valid_shards):
        self.root=Path(root).resolve(); self.counts=collections.Counter()
        self.shard_counts={"train":train_shards,"valid":valid_shards}; self.stack=ExitStack(); self.handles={}
        for split,count in self.shard_counts.items():
            (self.root/split).mkdir(parents=True,exist_ok=True)
            for i in range(count):
                self.handles[(split,i)]=self.stack.enter_context((self.root/split/f"data.{i:05d}.jsonl").open("w",encoding="utf-8"))
    def write(self,split,pair_id,record):
        i=int.from_bytes(hashlib.sha1(pair_id.encode()).digest()[4:8],"big")%self.shard_counts[split]
        self.handles[(split,i)].write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
        self.counts[split]+=1
    def close(self):
        self.stack.close()
        for split,count in self.shard_counts.items():
            with (self.root/f"{split}.data.list").open("w",encoding="utf-8") as f:
                for i in range(count):
                    path=self.root/split/f"data.{i:05d}.jsonl"
                    if path.stat().st_size: f.write(str(path)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--direction",choices=["en2zh","zh2en"],required=True)
    ap.add_argument("--mapping",type=Path,required=True); ap.add_argument("--text-labels",type=Path,required=True)
    ap.add_argument("--word-alignments",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--sqlite",type=Path,required=True); ap.add_argument("--train-shards",type=int,default=128)
    ap.add_argument("--valid-shards",type=int,default=4); ap.add_argument("--valid-basis-points",type=int,default=100)
    ap.add_argument("--limit",type=int,default=0); args=ap.parse_args()
    args.sqlite.parent.mkdir(parents=True,exist_ok=True)
    if args.sqlite.exists(): args.sqlite.unlink()
    conn=sqlite3.connect(args.sqlite); conn.execute("PRAGMA journal_mode=WAL")
    mapping_count,source_ids,pair_ids=ingest_mapping(args.mapping,args.direction,conn,args.limit)
    label_count=ingest_labels(args.text_labels,pair_ids,conn)
    del pair_ids
    aligned_count,alignment_without_timestamps=ingest_alignments(args.word_alignments,source_ids,conn)
    del source_ids
    writer=ShardedWriter(args.output_dir,args.train_shards,args.valid_shards)
    reject_file=(args.output_dir/"rejects.jsonl").open("w",encoding="utf-8")
    rejected=collections.Counter(); processed=written=0
    query=("SELECT m.payload,l.payload,a.payload FROM mapping m LEFT JOIN labels l USING(pair_id) "
           "LEFT JOIN align a ON a.segment_id=m.source_sid ORDER BY m.rowid")
    try:
        for mapping_json,label_json,align_json in conn.execute(query):
            if args.limit and processed>=args.limit: break
            processed+=1
            progress("join", processed)
            try:
                if label_json is None: raise ValueError("missing text label")
                if align_json is None: raise ValueError("missing source word alignment")
                record=build_record(json.loads(mapping_json),json.loads(label_json),json.loads(align_json),args.direction)
                split=split_name(record["key"],args.valid_basis_points); writer.write(split,record["key"],record); written+=1
            except Exception as exc:
                reason=str(exc); rejected[reason]+=1
                reject_file.write(json.dumps({"pair_id":mapping_pair_id(json.loads(mapping_json)),"reason":reason},ensure_ascii=False)+"\n")
    finally:
        reject_file.close(); writer.close()
    summary={"direction":args.direction,"mapping_rows":mapping_count,"text_label_rows":label_count,
             "source_alignments_with_timestamps":aligned_count,
             "source_alignment_segments_without_timestamps":alignment_without_timestamps,
             "processed_pairs":processed,"written_pairs":written,"split_counts":dict(writer.counts),
             "rejected_pairs":processed-written,"reject_reasons":dict(rejected.most_common()),
             "decision_ms":640,"interval_audio_tokens":8,"sqlite":str(args.sqlite.resolve()),
             "output_dir":str(args.output_dir.resolve())}
    (args.output_dir/"merge_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); conn.close()
if __name__=="__main__": main()
