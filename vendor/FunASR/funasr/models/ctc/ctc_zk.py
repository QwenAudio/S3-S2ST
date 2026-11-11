import logging

import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
            self,
            odim: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
            ctc_type: str = "builtin",
            reduce: bool = True,
            ignore_nan_grad: bool = True,
            blank_id: int = 0,
            **kwargs

    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad
        self.blank_id = blank_id

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none", blank=blank_id)
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            if ignore_nan_grad:
                logging.warning("ignore_nan_grad option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        else:
            raise ValueError(f'ctc_type must be "builtin" or "warpctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = F.log_softmax(th_pred, dim=2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # 使用 torch.autograd.grad 计算梯度
                grad = torch.autograd.grad(loss, th_pred, grad_outputs=torch.ones_like(loss), retain_graph=True,
                                           create_graph=False)[0]

                # 聚合梯度：对时间步和输出类别维度求和，得到每个样本的梯度
                ctc_grad = grad.sum(dim=[0, 2])  # shape: (B,)

                indices = torch.isfinite(ctc_grad)
                size = indices.sum()

                if size == 0:
                    # 所有样本的梯度都是 NaN
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    # 部分样本的梯度是 NaN
                    invalid = th_pred.size(1) - size
                    logging.warning(
                        f"{invalid}/{th_pred.size(1)} samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # 创建每个样本的起始索引和长度
                    batch_size = th_olen.size(0)
                    device = th_target.device
                    starts = torch.cat([torch.tensor([0], device=device), th_olen.cumsum(0)[:-1]])
                    lengths = th_olen

                    # 创建一个与 th_target 相同长度的索引
                    mask = torch.zeros_like(th_target, dtype=torch.bool, device=device)

                    # 使用向量化操作设置有效样本的掩码
                    # 计算每个时间步的样本索引
                    sample_indices = torch.arange(mask.size(0), device=device).unsqueeze(0).expand(batch_size, -1)
                    sample_starts = starts.unsqueeze(1)
                    sample_ends = (starts + lengths).unsqueeze(1)
                    time_steps = torch.arange(th_target.size(0), device=device).unsqueeze(0)

                    # 创建样本掩码矩阵
                    sample_mask = (time_steps >= sample_starts) & (time_steps < sample_ends)

                    # 根据有效样本过滤
                    valid_mask = sample_mask & indices.unsqueeze(1)

                    # 扁平化 mask
                    mask = valid_mask.view(-1)

                    # 使用掩码过滤 th_pred 和 th_target
                    th_pred_filtered = th_pred[:, indices, :]
                    th_target_filtered = th_target[mask]
                    th_ilen_filtered = th_ilen[indices]
                    th_olen_filtered = th_olen[indices]

                    # 重新计算 CTC 损失
                    loss = self.ctc_loss(
                        th_pred_filtered,
                        th_target_filtered,
                        th_ilen_filtered,
                        th_olen_filtered,
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                # 按批量大小平均
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # NOTE: sum() is needed to keep consistency since warpctc
                # return as tensor w/ shape (1,)
                # but builtin return as tensor w/o shape (scalar).
                loss = loss.sum()
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
