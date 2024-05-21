from typing import List

import torch
from torch import Tensor
# from torchmetrics import Metric
# from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class TM2TMetrics(object):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 **kwargs):

        self.name = "matching, fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.init()

    def init(self):
        self.count = torch.tensor(0)
        self.count_seq = torch.tensor(0)

        self.metrics = []
        # Matching scores
        self.Matching_score = torch.tensor(0.)
        self.gt_Matching_score = torch.tensor(0.)
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]

        # R-precision top 3
        self.R_precision_top_1 = torch.tensor(0.)
        self.R_precision_top_2 = torch.tensor(0.)
        self.R_precision_top_3 = torch.tensor(0.)

        self.Matching_metrics.append(f"R_precision_top_1")
        self.Matching_metrics.append(f"R_precision_top_2")
        self.Matching_metrics.append(f"R_precision_top_3")

        # Ground Truth R-precision top 3
        self.gt_R_precision_top_1 = torch.tensor(0.)
        self.gt_R_precision_top_2 = torch.tensor(0.)
        self.gt_R_precision_top_3 = torch.tensor(0.)

        self.Matching_metrics.append(f"gt_R_precision_top_1")
        self.Matching_metrics.append(f"gt_R_precision_top_2")
        self.Matching_metrics.append(f"gt_R_precision_top_3")

        self.metrics.extend(self.Matching_metrics)

        # Fid
        self.FID = torch.tensor(0.0)
        self.metrics.append("FID")

        # Diversity
        self.Diversity = torch.tensor(0.0)
        self.gt_Diversity = torch.tensor(0.0)
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.text_embeddings = []
        self.recmotion_embeddings = []
        self.gtmotion_embeddings = []

    def compute(self):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.text_embeddings, axis=0).cpu()[shuffle_idx, :]
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0).cpu()[shuffle_idx, :]

        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]

            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size:(i + 1) * self.R_size]

            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()

            self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)

            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = self.Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                          self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()

            # match score
            self.gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
            
        metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions, self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        recmotion_embeddings = torch.flatten(recmotion_embeddings, start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings, start_dim=1).detach()

        # store all texts and motions
        self.text_embeddings.append(text_embeddings)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)