# wired. move hybrid_connector.utils under a common utils directory.
import asyncio
import contextlib
import os
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Optional

import msgspec
import uvloop
from msgspec import msgpack

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.hybrid_connector.utils import (
    RpcServer,
    kill_me_if_exception,
    start_asyncio_thread,
)

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc

logger = init_logger(__name__)


class _ParticipantInfo(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    dprank: int = 0
    host: str = ""
    port: int = 0


# Body:
# +------+
# | step |
# +------+
# step: 4bytes
_START_STEP_REQ = 0x20181218
# No Body
_START_STEP_RESP = 0x81218102

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded _ParticipantInfo
_REG_PART_REQ = 0x20181219
# No Body
_REQ_PART_RESP = 0x91218102


async def _start_step_rpc(host: str, port: int, step: int):
    reader, writer = await asyncio.open_connection(host, port)
    reqbuf = struct.pack("=II", _START_STEP_REQ, step)
    writer.write(reqbuf)
    await writer.drain()

    respbuf = await reader.readexactly(4)
    respcode = struct.unpack("=I", respbuf)[0]
    assert respcode == _START_STEP_RESP

    writer.close()
    await writer.wait_closed()
    return


def _start_step_rpc_sync(host: str, port: int, step: int):
    # loop = asyncio.get_event_loop()
    # coro = _start_step_rpc(self._coord_host, self._coord_port,
    # self._local_step)
    # loop.run_until_complete(coro)
    with socket.create_connection((host, port)) as sock:
        reqbuf = struct.pack("=II", _START_STEP_REQ, step)
        sock.sendall(reqbuf)
        respbuf = sock.recv(4)
        respcode = struct.unpack("=I", respbuf)[0]
        assert respcode == _START_STEP_RESP
    return


# Move to DPCoordinator process
class _Coordinator:
    def __init__(self, dpsize: int, port: int, loop):
        self._loop = loop
        self._port = port
        self._dpsize = dpsize
        self._step = 0
        self._parts: list[Optional[_ParticipantInfo]] = [None for _ in range(dpsize)]
        self._info_decoder = msgpack.Decoder(_ParticipantInfo)

        self._rpcsrv = RpcServer(self._port, "Coordinator")
        self._rpcsrv.register_method(_START_STEP_REQ, self._start_step)
        self._rpcsrv.register_method(_REG_PART_REQ, self._reg_part)

    async def _part_start_step(self, partidx: int):
        pinfo = self._parts[partidx]
        assert pinfo is not None  # may be sleep if pinfo is None
        with contextlib.suppress(Exception):
            await _start_step_rpc(pinfo.host, pinfo.port, self._step)
        return

    @kill_me_if_exception
    async def _do_start_step(self, step: int):
        if step <= self._step:
            return
        self._step = step + 24
        futs = [self._part_start_step(idx) for idx in range(self._dpsize)]
        await asyncio.gather(*futs)
        return

    async def _start_step(self, reader, writer):
        bodybuf = await reader.readexactly(4)
        step = int(struct.unpack("=I", bodybuf)[0])

        if step > self._step:
            self._loop.create_task(self._do_start_step(step))

        resp = struct.pack("=I", _START_STEP_RESP)
        writer.write(resp)
        await writer.drain()
        return

    async def _reg_part(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        info: _ParticipantInfo = self._info_decoder.decode(reqbuf)

        assert self._parts[info.dprank] is None
        self._parts[info.dprank] = info

        resp = struct.pack("=I", _REQ_PART_RESP)
        writer.write(resp)
        await writer.drain()
        return

    def start(self, loop):
        self._rpcsrv.start(loop)
        return


def _start_coord(port: int, dpsize: int):
    loop = start_asyncio_thread("coord")
    coord = _Coordinator(dpsize, port, loop)
    coord.start(loop)
    return


class FakeParticipant:
    def __init__(self):
        return

    def new_step(self):
        return


class Participant:
    def __init__(self, vllmcfg: VllmConfig, core: "DPEngineCoreProc"):
        # Core Thread
        self._core = core
        self._info_encoder = msgpack.Encoder()
        self._dprank = vllmcfg.parallel_config.data_parallel_rank
        self._coord_port = vllmcfg.parallel_config.get_next_dp_init_port()
        self._coord_host = vllmcfg.parallel_config.data_parallel_master_ip
        dpsize = vllmcfg.parallel_config.data_parallel_size

        # R/W only on Core Thread
        self._local_step = 0
        self._coord_step = 0

        if self._dprank == 0:
            _start_coord(self._coord_port, dpsize)

        loop = start_asyncio_thread("part")
        if "AQUILA_RPC_PROTOCOL" in os.environ and envs.VLLM_DP_MASTER_PORT>0:
            self._port = envs.VLLM_DP_MASTER_PORT + 10 + self._dprank
        else:
            self._port = get_open_port()
        self._rpcsrv = RpcServer(self._port, "Participant")
        self._rpcsrv.register_method(_START_STEP_REQ, self._start_step)
        self._rpcsrv.start(loop)

        regcoro = self._reg()
        asyncio.run_coroutine_threadsafe(regcoro, loop)
        return

    def new_step(self):
        # Core Thread
        self._local_step += 1
        if self._local_step <= self._coord_step:
            return

        with contextlib.suppress(Exception):
            _start_step_rpc_sync(self._coord_host, self._coord_port, self._local_step)
        return

    def engines_running(self) -> bool:
        # Core Thread
        return self._local_step < self._coord_step

    def core_start_step(self, step):
        # Core Thread
        if step > self._coord_step:
            self._coord_step = step
        return

    async def _start_step(self, reader, writer):
        bodybuf = await reader.readexactly(4)
        step = int(struct.unpack("=I", bodybuf)[0])

        self._core.start_step_threadsafe(step)

        resp = struct.pack("=I", _START_STEP_RESP)
        writer.write(resp)
        await writer.drain()
        return

    async def _reg_part(self, info: _ParticipantInfo):
        reader, writer = await asyncio.open_connection(
            self._coord_host, self._coord_port
        )

        infobuf = self._info_encoder.encode(info)
        reqbuf = struct.pack("=II", _REG_PART_REQ, len(infobuf))
        reqbuf += infobuf
        writer.write(reqbuf)
        await writer.drain()

        respbuf = await reader.readexactly(4)
        respcode = struct.unpack("=I", respbuf)[0]
        assert respcode == _REQ_PART_RESP

        writer.close()
        await writer.wait_closed()
        return

    @kill_me_if_exception
    async def _reg(self):
        info = _ParticipantInfo(dprank=self._dprank, host=get_ip(), port=self._port)
        logger.info("partinfo=%s", info)

        last_log_time = 0.0
        while True:
            try:
                await self._reg_part(info)
                break
            except ConnectionRefusedError:
                # If the coordinator is not ready yet, it will expect a
                # ConnectionRefusedError, so we retry after a short sleep.
                if time.time() - last_log_time > envs.VLLM_LOG_STATS_INTERVAL:
                    logger.warning(
                        "Connection to coordinator at %s:%d refused. Retrying...",
                        self._coord_host, self._coord_port,
                    )
                    last_log_time = time.time()
                await asyncio.sleep(0.1)
            except Exception:
                # Log other unexpected exceptions with full traceback
                logger.exception("Unexpected error during registration")
                await asyncio.sleep(0.1)
        return
