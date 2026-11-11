"""Phase C: remove forced text generation, preserving the canonical alignment oracle."""
import copy,inspect,textwrap
import capture_gpu_entry as capture
m=capture.m
original=capture.shared.original_speech
source=inspect.getsource(original)
anchor='        self._close_consumed_writer_shm_handles()'
assert source.count(anchor)==2
source=source.replace(anchor,'''        rpc(self.s2st_engine, "set_single_gpu_mode", "text")
        if not record_segment or not getattr(self.s2st_engine, "_captured_text", None):
            return original_speech(self, prompt_text, segments, forced_text, record_segment=record_segment)
'''+anchor,1)
start=source.index('        output_queue = self.s2st_engine.add_request(')
end=source.index('        decoded_text =',start)
source=source[:start]+'''        # Keep the exact canonical text IDs used by the original alignment.
        # No second autoregressive request is needed to manufacture them.
        captured = self.s2st_engine._captured_text
        last = copy.copy(captured["output"])
        last.outputs = [copy.copy(last.outputs[0])]
        last.outputs[0].token_ids = list(forced_token_ids)
        last.outputs[0].text = forced_text
        last.finished = True
        self.s2st_engine.request_prompts[request_id] = prompt
        self.s2st_engine.talker_requests[request_id] = talker_params
        self.s2st_engine._start_global_talker_from_alignment(request_id, last)
        assert self.s2st_engine._global_alignment_epoch >= expected_alignment_epoch
'''+source[end:]
scope=original.__globals__.copy()
scope.update(copy=copy,rpc=capture.rpc,original_speech=original)
namespace={}
exec(compile(textwrap.dedent(source),__file__,"exec"),scope,namespace)
m.StreamingS2STBackend.generate_forced_speech=namespace["generate_forced_speech"]
if __name__=="__main__":m.main()
