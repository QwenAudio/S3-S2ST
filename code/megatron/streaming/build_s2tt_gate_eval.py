"""Extract the validated S2ST text loop and add pre-threshold gate tracing."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
src = root / "s2st_deferred_talker_eval"
dst = root / "s2tt_gate_eval"
dst.mkdir(exist_ok=True)
(dst / "vllm_overrides").mkdir(exist_ok=True)
manifest = {}
for name in ["streaming_core.py", "true_streaming_encoder.py", "block640_lookahead160_encoder.py", "vllm_overrides/omnidrca2_next_thinker.py"]:
    content = (src / name).read_text()
    manifest[name] = hashlib.sha256(content.encode()).hexdigest()
    (dst / name).write_text(content)
override = dst / "vllm_overrides/omnidrca2_next_thinker.py"
s = override.read_text()
needle = "            logits = gate(hidden_states)\n"
assert s.count(needle) == 1
s = s.replace(needle, needle + '''            # Capture BEFORE threshold adjustment and sampler processing.
            trace_path = os.environ.get("S2TT_GATE_TRACE")
            if trace_path:
                import json
                from vllm.distributed import get_tensor_model_parallel_rank
                if get_tensor_model_parallel_rank() == 0:
                    raw = logits.detach().float().cpu()
                    probs = raw.softmax(dim=-1)
                    with open(trace_path, "a") as trace:
                        for z, p in zip(raw.tolist(), probs.tolist()):
                            trace.write(json.dumps({"logits": z, "p_write": p[1]}) + "\\n")
                            trace.flush()
''')
override.write_text(s)
core = dst / "streaming_core.py"
s = core.read_text()
needle = "        self.args = args\n"
s = s.replace(needle, needle + '''        self.trace_path = str(Path(args.output).resolve()) + ".gate_trace.jsonl"
        if Path(args.output).exists() or Path(self.trace_path).exists():
            raise FileExistsError("Use a new output path; this evaluator does not silently resume")
        Path(self.trace_path).touch()
        self.trace_reader = open(self.trace_path)
        os.environ["S2TT_GATE_TRACE"] = self.trace_path
''', 1)
needle = "        return token_ids[0] == 1\n"
s = s.replace(needle, '''        records = [json.loads(line) for line in self.trace_reader.readlines()]
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one MLP score per gate request, got {len(records)}")
        self.last_gate_score = records[0]
        p = self.last_gate_score["p_write"]
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise RuntimeError(f"Invalid MLP probability: {p}")
        # Tie behavior follows the original greedy argmax, choosing READ.
        expected = p > self.args.threshold
        if abs(p - self.args.threshold) > 0.002 and expected != (token_ids[0] == 1):
            raise RuntimeError("Raw MLP score and gate decision disagree")
        return token_ids[0] == 1
''', 1)
start = s.index("def evaluate_one_streaming_thinker_deferred_talker(")
end = s.index("\n\ndef evaluate_one(", start)
loop = s[start:end].replace("evaluate_one_streaming_thinker_deferred_talker", "evaluate_one_s2tt")
loop = loop.replace("backend.generate_text(prompt_text, segments)", "backend.generate(prompt_text, segments)")
a = loop.index("        replayed_text = backend.generate_forced_speech")
b = loop.index("        pieces.append(text)", a)
loop = loop[:a] + loop[b:]
loop = loop.replace('            "forced_eos": forced,', '''            "p_write": None if forced else backend.last_gate_score["p_write"],
            "logits": None if forced else backend.last_gate_score["logits"],
            "forced_eos": forced,''')
a = loop.index("    result.update({")
loop = loop[:a] + '''    result["inference_contract"] = "s2tt_block640_lookahead160_raw_gate_v1"
    return result
'''
# Preserve final empty-WRITE handling without invoking Talker.
loop = loop.replace('end_of_audio and pieces and hasattr(\n                backend, "finalize_without_new_write"\n            )', 'end_of_audio and pieces')
loop = loop.replace('                backend.finalize_without_new_write(prompt_text, segments)\n', '')
s = s[:start] + loop + s[end:]
s = s.replace("result = evaluate_one(index, obj, encoder, backend, args)", "result = evaluate_one_s2tt(index, obj, encoder, backend, args)")
s = s.replace('    args = ap.parse_args()\n', '''    args = ap.parse_args()
    if not args.true_streaming_encoder or args.offline_thinker or args.encoder_chunk_ms != 640:
        ap.error("Requires --true-streaming-encoder --encoder-chunk-ms 640")
    if not 0 < args.threshold < 1:
        ap.error("threshold must be in (0, 1)")
''')
core.write_text(s)
wrapper = (src / "run_block640_lookahead160.py").read_text()
wrapper = wrapper[:wrapper.index("# Import only after")]
wrapper += '\nif __name__ == "__main__":\n    s2tt.main()\n'
(dst / "run_s2tt.py").write_text(wrapper)
(dst / "SOURCE_MANIFEST.json").write_text(json.dumps({"source": str(src), "sha256": manifest}, indent=2))
print(dst)
