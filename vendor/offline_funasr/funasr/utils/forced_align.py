from itertools import groupby

import torch
import torchaudio.functional as F


def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    """
    Perform CTC alignment on the given log probabilities and targets.
    Args:
        log_probs (torch.Tensor): Log probabilities of CTC emission output. Tensor of shape (T, C), where T is the input length, and C is the number of characters in the alphabet including blank.
        targets (torch.Tensor): Target sequence. Tensor of shape (L,), where L is the target length.
        blank (int): The index of the blank symbol in CTC emission. Default is 0.
    Returns:
        List[Dict]: A list of alignment items representing the alignments.
    """
    # The current version only supports batch_size==1.
    log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
    alignments, scores = F.forced_align(log_probs, targets, blank=blank)
    alignments, scores = alignments[0], scores[0]

    items = []
    # use enumerate to keep track of the original indices, then group by token value
    for token, group in groupby(enumerate(alignments), key=lambda item: item[1]):
        if token == blank:
            continue
        group = list(group)
        start = group[0][0]
        items.append({"text": token.item(), "start_time": start, "end_time": start + len(group)})
    return items
