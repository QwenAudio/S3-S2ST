#!/usr/bin/env python3
"""CPU-only structural audit for unpacked streaming S2TT shards."""
import argparse, collections, json, math, sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from streaming.data.build_interleaved_data import GRID_SECONDS, speech_token_length

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-list",type=Path,required=True)
    ap.add_argument("--max-rows",type=int,default=0)
    ap.add_argument("--examples",type=int,default=3)
    args=ap.parse_args()
    shards=[Path(x.strip()) for x in args.data_list.open(encoding="utf-8") if x.strip()]
    counts=collections.Counter(); examples=[]
    for shard in shards:
        with shard.open(encoding="utf-8") as source:
            for line in source:
                row=json.loads(line); counts["rows"]+=1
                messages=row["messages"]
                assert messages[0]["role"]=="system"
                assert (len(messages)-1)%2==0
                users=messages[1::2]; assistants=messages[2::2]
                assert all(x["role"]=="user" for x in users)
                assert all(x["role"]=="assistant" for x in assistants)
                assert all(x.get("content")==x.get("text") for x in messages)
                assert all(x["text"] for x in assistants)
                for index,user in enumerate(users):
                    duration=float(user["end_time"])-float(user["start_time"])
                    assert duration>0
                    declared=int(user["speech_length"])
                    expected=speech_token_length(duration)
                    delta=declared-expected
                    # Large absolute offsets lose a tiny amount of precision when
                    # start/end are serialized.  At an exact 10 ms rounding tie this
                    # can change the packing estimate by one adaptor token.  The
                    # packer already reserves two audio tokens of tolerance; record
                    # the event, but reject anything larger than the known bound.
                    if delta:
                        counts["speech_length_rounding_mismatch"]+=1
                        counts[f"speech_length_delta_{delta:+d}"]+=1
                    assert abs(delta)<=1
                    if index+1<len(users):
                        units=duration/GRID_SECONDS
                        assert math.isclose(units,round(units),abs_tol=2e-5)
                counts["turns"]+=len(users)
                counts["audio_tokens"]+=sum(int(x["speech_length"]) for x in users)
                counts["assistant_chars"]+=sum(len(x["text"]) for x in assistants)
                if len(examples)<args.examples:
                    examples.append({"key":row["key"],"turns":[
                        {"audio":[u["start_time"],u["end_time"]],"speech_tokens":u["speech_length"],"translation":a["text"]}
                        for u,a in zip(users,assistants)]})
                if args.max_rows and counts["rows"]>=args.max_rows:
                    break
        if args.max_rows and counts["rows"]>=args.max_rows:
            break
    if not counts["rows"]:
        raise RuntimeError("no records found")
    result={"data_list":str(args.data_list.resolve()),"shards":len(shards),**counts,
            "average_turns":counts["turns"]/counts["rows"],"examples":examples}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
