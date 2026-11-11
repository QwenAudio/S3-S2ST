# -*- coding: utf-8 -*-

"""Network related utility tools."""

from typing import List, Tuple

import torch


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def get_minimal_ctc_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Transducer loss I/O.

    Args:
        labels: Label ID sequences. (B, L)
        encoder_out_lens: Encoder output lengths. (B,)
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Returns:
        decoder_in: Decoder inputs. (B, U)
        target: Target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        u_len: Label lengths. (B,)

    """

    def pad_list(labels: List[torch.Tensor], padding_value: int = 0):
        """Create padded batch of labels from a list of labels sequences.

        Args:
            labels: Labels sequences. [B x (?)]
            padding_value: Padding value.

        Returns:
            labels: Batch of padded labels sequences. (B,)

        """
        batch_size = len(labels)

        padded = labels[0].new(batch_size, max(x.size(0) for x in labels), *labels[0].size()[1:]).fill_(padding_value)

        for i in range(batch_size):
            padded[i, : labels[i].size(0)] = labels[i]

        return padded

    device = labels.device

    labels_unpad = [y[y != ignore_id] for y in labels]
    blank = labels[0].new([blank_id])

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)

    encoder_out_lens = list(map(int, encoder_out_lens))
    t_len = torch.IntTensor(encoder_out_lens).to(device)

    u_len = torch.IntTensor([y.size(0) for y in labels_unpad]).to(device)

    return target, t_len, u_len
