#!/usr/bin/env python3
"""Check packed/unpacked sample conservation and explain known packer filters."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

def paths(data_list: Path) -> list[Path]:
    return [Path(x.strip()) for x in data_list.open(encoding="utf-8") if x.strip()]

def signature(row: dict) -> str:
    fields=[]
    for message in row.get("messages") or []:
        fields.append((message.get("role"),message.get("text"),message.get("start_time"),
                       message.get("end_time"),message.get("segment_id")))
    return json.dumps(fields,ensure_ascii=False,separators=(",",":"))

def reason(row: dict) -> str:
    messages=row.get("messages") or []
    if len(messages)>2:
        text=str(messages[2].get("text") or messages[2].get("content") or "")
        if (text.startswith("(") or text.startswith("（")) and "笑" in text:
            return "offline_packer_laugh_filter"
    return "unknown_packer_drop"

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--unpacked-list",type=Path,required=True)
    ap.add_argument("--packed-list",type=Path,required=True)
    ap.add_argument("--allow-laugh-filter",action="store_true")
    args=ap.parse_args()
    raw_paths,packed_paths=paths(args.unpacked_list),paths(args.packed_list)
    if len(raw_paths)!=len(packed_paths): raise ValueError("packed/unpacked shard count mismatch")
    counts=collections.Counter(); exclusions=[]
    for raw_path,packed_path in zip(raw_paths,packed_paths):
        packed=collections.Counter()
        for line in packed_path.open(encoding="utf-8"):
            for row in json.loads(line): packed[signature(row)]+=1; counts["packed_records"]+=1
        for line_no,line in enumerate(raw_path.open(encoding="utf-8"),1):
            row=json.loads(line); counts["unpacked_records"]+=1; sig=signature(row)
            if packed[sig]: packed[sig]-=1
            else:
                why=reason(row); counts[f"excluded_{why}"]+=1
                exclusions.append({"key":row.get("key"),"reason":why,
                                   "source_shard":str(raw_path),"source_line":line_no})
        counts["unmatched_packed_records"]+=sum(packed.values())
    unknown=counts["excluded_unknown_packer_drop"]
    allowed=counts["excluded_offline_packer_laugh_filter"] if args.allow_laugh_filter else 0
    passed=(counts["unmatched_packed_records"]==0 and unknown==0 and
            counts["unpacked_records"]-counts["packed_records"]==allowed)
    result={"unpacked_list":str(args.unpacked_list.resolve()),"packed_list":str(args.packed_list.resolve()),
            **counts,"exclusions":exclusions,"passed":passed}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if passed else 1)

if __name__=="__main__": main()
