#!/usr/bin/env python3
"""CPU-only packing for preformatted multi-turn streaming S2TT JSONL."""
import argparse, json, os, re
from pathlib import Path


_AUDIO_PATTERN = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
_WRAPPERS = {
    "system_prefix": "<|im_start|>system\n",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "assistant_suffix": "<|im_end|>\n",
    "audio_prefix": "<|audio_start|>",
    "audio_suffix": "<|audio_end|>",
}


def apply_exact_streaming_lengths(sample, tokenizer):
    """Set packing lengths to the exact sequence constructed at runtime.

    FunASR's generic preformatted converter assumes one system/user/assistant
    turn.  Streaming S2TT has many user/audio + assistant/text turns, so the
    generic estimate tokenizes every OSS URI while accounting for the ChatML
    wrappers only once.  That under-fills an 8192-token pack by roughly half
    after the runtime replaces each URI with audio embeddings.
    """
    wrapper_lengths = {
        key: len(tokenizer.encode(value)) for key, value in _WRAPPERS.items()
    }
    system_count = 0
    for message in sample["messages"]:
        role = message["role"]
        text = message.get("content", message.get("text", "")) or ""
        if role == "system":
            system_count += 1
            text_length = (
                wrapper_lengths["system_prefix"]
                + len(tokenizer.encode(text))
                + wrapper_lengths["system_suffix"]
            )
        elif role == "user":
            text_length = (
                wrapper_lengths["user_prefix"] + wrapper_lengths["user_suffix"]
            )
            for part in _AUDIO_PATTERN.split(text):
                if not part:
                    continue
                if part.startswith("<|startofspeech|>"):
                    text_length += (
                        wrapper_lengths["audio_prefix"]
                        + wrapper_lengths["audio_suffix"]
                    )
                else:
                    text_length += len(tokenizer.encode(part))
        elif role == "assistant":
            text_length = (
                len(tokenizer.encode(text)) + wrapper_lengths["assistant_suffix"]
            )
        else:
            raise ValueError(f"unsupported streaming message role: {role}")
        message["text_length"] = int(text_length)
        message["speech_length"] = int(message.get("speech_length") or 0)
    if system_count != 1:
        raise ValueError(f"streaming sample must contain one system message, got {system_count}")
    sample["text_length"] = sum(
        int(message["text_length"]) for message in sample["messages"]
    )
    sample["speech_length"] = sum(
        int(message["speech_length"]) for message in sample["messages"]
    )
    return sample

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-list",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--funasr-root",required=True)
    ap.add_argument("--tokenizer-model",required=True)
    ap.add_argument("--cutoff-len",type=int,default=8192)
    ap.add_argument("--num-workers",type=int,default=16)
    ap.add_argument("--seed",type=int,default=20260810)
    args=ap.parse_args()
    import ray
    sources=[Path(x.strip()) for x in open(args.input_list,encoding="utf-8") if x.strip()]
    if not sources: raise ValueError("input list is empty")
    output=Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    funasr_root=str(Path(args.funasr_root).resolve())
    tokenizer_model=str(Path(args.tokenizer_model).resolve())
    @ray.remote(num_cpus=1)
    class Actor:
        def __init__(self,index):
            import sys
            sys.path.insert(0,funasr_root)
            from offline_pack.format_converter import Converter
            from offline_pack.sft_packer import SFTDataPacker
            self.index=index
            class StreamingConverter(Converter):
                def convert_format(converter_self, sample):
                    sample = apply_exact_streaming_lengths(sample, converter_self.tokenizer)
                    converted = super(StreamingConverter, converter_self).convert_format(sample)
                    # The generic converter adds its legacy one-turn 19 text +
                    # 2 audio-token allowance after summing message metadata.
                    # Every wrapper and audio marker is already counted above.
                    converted["text_length"] = sum(
                        int(message["text_length"])
                        for message in converted["messages"]
                    )
                    converted["speech_length"] = sum(
                        int(message["speech_length"])
                        for message in converted["messages"]
                    )
                    return converted
            self.converter=StreamingConverter(tokenizer_model_path=tokenizer_model,
                dataset_type="MinMo_S2S_Channels_SFT_Pack_Mega",packing_mode="s2t")
            self.packer=SFTDataPacker(cutoff_len=args.cutoff_len)
        def pack(self,index,source,destination):
            import random
            random.seed(args.seed+index)
            stats=self.packer.pack_from_jsonl(input_file=source,output_file=destination,
                format_converter=self.converter,verbose=False,shuffle=True)
            return index,destination,stats
    ray.init(log_to_driver=False)
    actors=[Actor.remote(i) for i in range(min(args.num_workers,len(sources)))]
    pending=[]
    for i,source in enumerate(sources):
        destination=output/f"data.{i:05d}.jsonl"
        pending.append(actors[i%len(actors)].pack.remote(i,str(source),str(destination)))
    results=ray.get(pending)
    ray.shutdown()
    total_samples=total_bins=0; destinations=[]
    for i,destination,stats in sorted(results):
        path=Path(destination)
        if not path.is_file() or path.stat().st_size==0:
            raise RuntimeError(f"empty packed shard: {path}")
        destinations.append(path); total_samples+=int(stats["total_samples"]); total_bins+=int(stats["packed_bins"])
    tmp=output/"data.list.tmp"
    with tmp.open("w",encoding="utf-8") as f:
        for path in destinations: f.write(str(path)+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,output/"data.list")
    summary={"source_shards":len(sources),"packed_shards":len(destinations),
             "original_samples":total_samples,"packed_bins":total_bins,
             "cutoff_len":args.cutoff_len,"packing_mode":"s2t","seed":args.seed,
             "length_accounting":"exact_streaming_runtime_v1"}
    (output/"packing_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
