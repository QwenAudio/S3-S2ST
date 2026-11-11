"""Wait for the old listener before reusing its shared per-sample state."""
import asyncio
import threading


def install(engine_type, timeout=60.0):
    original = engine_type.reset_global_timeline

    def reset(self):
        request_id = self._global_talker_request_id
        listener = self.talker_listener.get(request_id) if request_id is not None else None
        if listener is not None and listener.is_alive():
            if listener is threading.current_thread():
                raise RuntimeError('Cannot reset a sample from its Talker listener')
            # The original reset also aborts this request, but immediately
            # clears shared EOS/failure flags. Wait for the listener first.
            future = asyncio.run_coroutine_threadsafe(self.talker_engine.abort(request_id), self._loop)
            future.result(timeout=timeout)
            listener.join(timeout=timeout)
            if listener.is_alive():
                raise RuntimeError('Previous sample Talker listener did not exit')
        return original(self)

    engine_type.reset_global_timeline = reset

