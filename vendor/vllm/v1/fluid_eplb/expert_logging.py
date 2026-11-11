import os
import threading
from queue import Queue
from typing import Any

import nvtx
import torch
import torch.distributed as dist

from vllm.v1.fluid_eplb import utils
from vllm.v1.fluid_eplb.options import eplb_options


def build(
    num_moe_layers: int,
    num_experts: int,
    num_redundant_experts: int,
    device: torch.device,
    ep_group: dist.ProcessGroup,
    rank: int,
    ranks: int,
) -> "ExpertLogging":
    return ExpertLogging(
        num_moe_layers,
        num_experts,
        num_redundant_experts,
        device,
        ep_group,
        rank,
        ranks,
    )

class ExpertLogging:
    def __init__(self, num_moe_layers: int,
        num_experts: int,
        num_redundant_experts: int,
        device: torch.device,
        ep_group: dist.ProcessGroup,
        rank: int,
        ranks: int,
    ):
        self.origin_expert_weights = torch.zeros(
            num_moe_layers,
            num_experts,
            dtype=torch.int64,
            device=device,
        )
        self.expert_weights = torch.zeros(
            num_moe_layers,
            num_experts + num_redundant_experts,
            dtype=torch.int64,
            device=device,
        )
        self.worker_thread: threading.Thread | None = None
        self.worker_queue: Queue[Any] = Queue()
        self.latest_record_step = torch.tensor(
            [-1], dtype=torch.int64, device=device
        )
        self.steps = 0
        self.ep_group = ep_group
        self.rank = rank
        self.ranks = ranks
        self.device = device
        self.init_worker()
        self.latest_imbalance_ratio: float = 0.0
        self.latest_intra_imbalance_ratio: float = 0.0

    def init_worker(self):
        assert dist.is_initialized(), "dist should be initialized"
        if self.worker_thread is not None:
            return
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="expert_logging_worker",
        )
        self.worker_thread.start()

    def _worker(self):
        while True:
            task = self.worker_queue.get()
            if isinstance(task, str) and task == "stop":
                break
            elif isinstance(task, tuple) and len(task) == 4:
                step, origin_expert_weights, expert_weights, compute_stream = task
                if torch.cuda.current_stream() != compute_stream:
                    torch.cuda.current_stream().wait_stream(compute_stream)
                dump_step = step // eplb_options.dump_batch_size
                if eplb_options.enable_expert_cnt_dump and self.rank == 0:
                    from safetensors.torch import save_file
                    tensors = {
                        "origin_expert_weights": origin_expert_weights,
                        "remapped_expert_cnts": expert_weights,
                    }
                    path = os.path.join(
                        eplb_options.log_dir,
                        f"eplb-step{dump_step}.safetensors",
                    )
                    save_file(tensors, path)
                    
                need_rebalance, \
                imbalance_ratio, \
                intra_imbalance_ratio = utils.check_and_gen_eplb(
                    origin_expert_weights,
                    expert_weights,
                    step,
                    self.rank,
                    self.ranks,
                )
                del origin_expert_weights
                del expert_weights
                if need_rebalance:
                    self.latest_record_step[0] = step 
                self.latest_imbalance_ratio = imbalance_ratio
                self.latest_intra_imbalance_ratio = intra_imbalance_ratio

    @nvtx.annotate("expert-imbalance-ratio", color="green")
    def record(self, origin_expert_weights: torch.Tensor, expert_weights: torch.Tensor):
        self.origin_expert_weights.add_(origin_expert_weights)
        self.expert_weights.add_(expert_weights)
        self.steps += 1
        if self.steps % eplb_options.dump_batch_size == 0:
            cloned_origin_expert_weights = self.origin_expert_weights.clone()
            cloned_expert_weights = self.expert_weights.clone()
            dist.all_reduce(cloned_origin_expert_weights, group=self.ep_group)
            dist.all_reduce(cloned_expert_weights, group=self.ep_group)
            self.worker_queue.put(
                (
                    self.steps,
                    cloned_origin_expert_weights,
                    cloned_expert_weights,
                    torch.cuda.current_stream(),
                ),
                block=False,
            )
            self.origin_expert_weights.fill_(0)
            self.expert_weights.fill_(0)