from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class MMMetrics(object):
    def __init__(self, mm_num_times=10):
        super().__init__()

        self.name = "MultiModality scores"
        self.mm_num_times = mm_num_times

    def init(self):
        self.count = torch.tensor(0)
        self.count_seq = torch.tensor(0)

        self.MultiModality = torch.tensor(0.)
        self.metrics = ["MultiModality"]

        self.mm_motion_embeddings = []

    def compute(self):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # cat all embeddings
        all_mm_motions = torch.cat(self.mm_motion_embeddings, axis=0).cpu().numpy()
        metrics['MultiModality'] = calculate_multimodality_np(all_mm_motions, self.mm_num_times)

        return {**metrics}

    def update(
        self,
        mm_motion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # store all mm motion embeddings
        self.mm_motion_embeddings.append(mm_motion_embeddings)
