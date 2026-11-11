# worker.py
import json
import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from vllm.logger import init_logger

logger = init_logger(__name__)


class LoggingSpanExporter(SpanExporter):
    def __init__(
        self,
        log_dir: str,
        base_name: str = "spans",
        *,
        max_bytes: int = 20 * 1024 * 1024,
        backup_count: int = 200,
    ):
        pid = os.getpid()
        log_name = f"otel_{base_name}_{pid}"
        log_file = os.path.join(log_dir, f"{log_name}.log")

        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        same_file = lambda h: isinstance(h, RotatingFileHandler) and getattr(
            h, "baseFilename", None
        ) == str(log_file)

        if not any(same_file(h) for h in self._logger.handlers):
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._own_handler = handler
        else:
            # 复用现有 handler
            self._own_handler = None

    def export(self, spans) -> "SpanExportResult":
        try:
            batch = []
            for s in spans:
                ctx = s.context
                batch.append(
                    {
                        "trace_id": ctx.trace_id,
                        "span_id": ctx.span_id,
                        "parent_id": s.parent.span_id if s.parent is not None else None,
                        "name": s.name,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "attributes": dict(s.attributes),
                    }
                )

            self._logger.info("%s", json.dumps(batch, ensure_ascii=False))
            return SpanExportResult.SUCCESS
        except Exception:
            traceback.print_exc()
            return SpanExportResult.FAILURE

    def shutdown(self):
        try:
            if self._own_handler is not None:
                self._own_handler.flush()
                self._logger.removeHandler(self._own_handler)
                self._own_handler.close()
        except Exception:
            traceback.print_exc()
