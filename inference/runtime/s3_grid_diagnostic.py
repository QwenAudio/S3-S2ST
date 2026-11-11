"""Run-local READ hidden delivery repair and stage-by-stage latency audit.

No model/checkpoint/source mutation. The control only records events. The READ
arm reuses the projected hidden states from the existing causal gate prefill;
it never creates text, a WRITE, a codec BOS, or future source padding.
"""
import inspect
import json
import os
import pickle
import textwrap
import time
import uuid
from pathlib import Path

import s3_stream_entry as entry

m, s = entry.m, entry.s
torch = m.torch
cfg = entry.cfg
deliver_reads = bool(cfg.get('deliver_read_hidden', False))
fm_chunk = int(cfg.get('fm_chunk', 15))
assert fm_chunk in (1, 4, 7, 8, 15)
rpc = entry.native.prefix.capture.rpc
PCM = entry.PCMRuntime

# Check the expanded multimodal token sequence, not an unexpanded text prompt.
old_hidden_export = entry.native.prefix.get_cached_hidden
hidden_source = inspect.getsource(old_hidden_export)
hidden_anchor = '        actual_prompt_ids=list(output.prompt_token_ids)'
assert hidden_source.count(hidden_anchor) == 1
hidden_source = hidden_source.replace(hidden_anchor, hidden_anchor + '''
        previous = getattr(engine, '_read_prefix_ids', None)
        if previous is not None:
            assert actual_prompt_ids[:len(previous)] == previous, 'WRITE changed a delivered READ prefix'
''')
hidden_ns = {}
exec(compile(textwrap.dedent(hidden_source), __file__, 'exec'),
     old_hidden_export.__globals__, hidden_ns)
m.CodecHistoryReplayEngine._run_global_alignment.__globals__['get_cached_hidden'] = hidden_ns['get_cached_hidden']

# Keep decoder semantics unchanged except for the explicitly named chunk arm.
if fm_chunk != 15:
    for name in ('on_codec', 'decode'):
        old = inspect.unwrap(getattr(PCM, name))
        source = textwrap.dedent(inspect.getsource(old)).replace('@torch.inference_mode()\n', '')
        source = source.replace('n%15', 'n%fm_chunk').replace('c+15', 'c+fm_chunk')
        source = source.replace('(n-c)%15', '(n-c)%fm_chunk')
        source = source.replace('block=15*ratio', 'block=fm_chunk*ratio')
        source = source.replace('//15)', '//fm_chunk)')
        source = source.replace('chunk_size=15,', 'chunk_size=fm_chunk,')
        scope = dict(old.__globals__)
        scope['fm_chunk'] = fm_chunk
        ns = {}
        exec(compile(textwrap.dedent(source), __file__, 'exec'), scope, ns)
        setattr(PCM, name, torch.inference_mode()(ns[name]))


def record(pcm, kind, **extra):
    if pcm.active:
        pcm.audit.append(dict(kind=kind, wall_s=time.monotonic()-pcm.play.start,
                              source_ms=pcm.source_ms, **extra))


old_begin = PCM.begin
def begin(self, *args, **kwargs):
    old_begin(self, *args, **kwargs)
    self.audit = []
    self.engine._read_prefix_ids = None
PCM.begin = begin

old_init = m.StreamingS2STBackend.__init__
def init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    self.pcm.engine = self.s2st_engine
    for obj, name, kind in [(self.pcm.fm, '_inference', 'flow_matching'),
                            (self.pcm.voc.generator, 'inference_cache', 'vocoder')]:
        original = getattr(obj, name)
        def timed(*a, _fn=original, _kind=kind, **kw):
            torch.cuda.synchronize(self.pcm.device)
            started = time.monotonic()
            result = _fn(*a, **kw)
            torch.cuda.synchronize(self.pcm.device)
            record(self.pcm, _kind, seconds=time.monotonic()-started,
                   finalize=bool(kw.get('finalize', False)))
            return result
        setattr(obj, name, timed)
m.StreamingS2STBackend.__init__ = init

old_codec = PCM.on_codec
def on_codec(self, token):
    e = self.engine
    record(self, 'codec', token=int(token), position=e._global_processed_position,
           available_end=e._global_available_end)
    return old_codec(self, token)
PCM.on_codec = on_codec

old_request = m.StreamingS2STBackend._request
def request(view, *args, **kwargs):
    result = old_request(view, *args, **kwargs)
    if view.mode == 'gate':
        view.engine._latency_gate_output = result[-1]
    return result
m.StreamingS2STBackend._request = staticmethod(request)


def start_pending_bos(e):
    """Start an already queued WRITE once real READ positions reach its BOS."""
    first = e._next_injected_bos()
    if first is None or first[0] >= e._global_available_end:
        return
    pos, bos_id, final_write = first
    hidden = e._global_available_hidden
    prompt = hidden[:pos+1].detach().cpu().clone().contiguous()
    flat = prompt.reshape(prompt.size(0), -1)
    emb = e._replay_codec_embedding[bos_id:bos_id+1]
    flat[pos:pos+1] += emb.to(dtype=flat.dtype)
    rid = 'global_codec_' + uuid.uuid4().hex
    e._global_talker_request_id = rid
    e._global_processed_position = pos
    e._global_expect_codec_output = True
    e._global_active_bos_id = bos_id
    e._global_active_write_is_final = final_write
    e._global_next_bos_index += 1
    e._global_codec_tokens.append(bos_id)
    e._global_audio_by_position[pos] = bos_id
    sampling = e._global_write_specs[0]['sampling']
    sampling.stop_token_ids = []
    sampling.ignore_eos = True
    e.output_queue[rid]
    e.thinker_output_len[rid] = 0
    e.thinker_generate_speech[rid] = True
    e._start_cosyvoice_talker_from_hidden(
        request_id=rid, hidden_states=flat, sampling_params=sampling,
        blank_count=e._global_write_specs[0]['blank_count'],
        prompt_frame_count=pos+1)
    record(e._s3_pcm, 'start_queued_bos_on_read', position=pos)


def export_gate_hidden(e):
    meta = rpc(e, 'export_cached_alignment')
    rebuild, args = pickle.loads(meta['ipc'])
    args = list(args)
    args[6] = int(os.environ['SINGLE_GPU_PHYSICAL_ID'])
    shared = rebuild(*args)
    hidden = shared.clone()
    torch.cuda.synchronize(hidden.device)
    del shared
    rpc(e, 'release_cached_alignment')
    ids = list(e._latency_gate_output.prompt_token_ids)
    assert meta['start'] == 0, meta
    assert len(hidden) == meta['prompt'] == len(ids)
    previous = e._read_prefix_ids
    if previous is not None:
        assert ids[:len(previous)] == previous, 'READ changed a delivered token prefix'
    e._read_prefix_ids = ids
    return hidden


old_gate = m.StreamingS2STBackend.gate_decision
def gate(self, *args, **kwargs):
    e, pcm = self.s2st_engine, self.pcm
    active = bool(e._global_write_specs)
    record(pcm, 'gate_begin', available_end=e._global_available_end,
           position=e._global_processed_position,
           waiting=bool(e._global_waiting_after_output), active=active)
    if active and deliver_reads:
        rpc(e, 'begin_cached_alignment')
    decision = old_gate(self, *args, **kwargs)
    if active and deliver_reads:
        hidden = export_gate_hidden(e)
        with e._global_lock:
            old_end = e._global_available_end
            assert len(hidden) >= old_end, (len(hidden), old_end)
            difference = (hidden[:old_end].float() - e._global_available_hidden[:old_end].float()).abs()
            old_prefix_max = float(difference.max()) if old_end else 0.
            old_prefix_mean = float(difference.mean()) if old_end else 0.
            e._global_available_hidden = hidden
            e._global_available_end = len(hidden)
            e._global_real_hidden_len = len(hidden)
            record(pcm, 'deliver_read_hidden', old_end=old_end,
                   new_end=len(hidden), write_decision=bool(decision),
                   position=e._global_processed_position,
                   old_prefix_max=old_prefix_max, old_prefix_mean=old_prefix_mean)
            if e._global_talker_request_id is None:
                start_pending_bos(e)
            else:
                e._resume_global_talker_if_waiting()
    record(pcm, 'gate_end', write=bool(decision), available_end=e._global_available_end,
           position=e._global_processed_position,
           waiting=bool(e._global_waiting_after_output))
    return decision
m.StreamingS2STBackend.gate_decision = gate

old_align = m.CodecHistoryReplayEngine._run_global_alignment
def align(self, request_id, output):
    result = old_align(self, request_id, output)
    record(self._s3_pcm, 'write_alignment', prefix_len=result[1], real_end=result[2],
           padded_end=result[3], blank_count=result[4], final=bool(result[5]))
    return result
m.CodecHistoryReplayEngine._run_global_alignment = align

old_finish = PCM.finish
def finish(self, *args, **kwargs):
    result = old_finish(self, *args, **kwargs)
    result['latency_audit'] = list(self.audit)
    result['diagnostic_contract'] = dict(deliver_read_hidden=deliver_reads,
                                         fm_chunk=fm_chunk, model_unchanged=True)
    if result.get('wav'):
        (Path(result['wav']).parent/'trace.json').write_text(json.dumps(result, indent=2))
    return result
PCM.finish = finish

# Join the previous utterance listener before clearing shared state.
# This executes in start_sample, outside the source/PCM timing clock.
from sample_cleanup_guard import install
install(m.CodecHistoryReplayEngine)

if __name__ == '__main__':
    m.main()
