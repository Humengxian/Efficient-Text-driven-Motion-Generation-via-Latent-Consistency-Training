from typing import List
import torch
from torch import Tensor
# from torchmetrics import Metric
# from torchmetrics.functional import pairwise_euclidean_distance
import os
from .utils import *

from ..model.network import humanact12_gru


class HUMANACTMetrics(object):
    full_state_update = True

    def __init__(self,
                 datapath,
                 num_labels=12,
                 diversity_times=200,
                 multimodality_times=20,
                 **kwargs):
        super().__init__()

        self.name = "matching, fid, and diversity scores"

        self.num_labels = num_labels
        self.diversity_times = diversity_times
        self.multimodality_times = multimodality_times

        # init classifier module
        self.gru_classifier = humanact12_gru.MotionDiscriminator(
            input_size=72, hidden_size=128, hidden_layer=2, output_size=12)

        self.gru_classifier_for_fid = humanact12_gru.MotionDiscriminatorForFID(
            input_size=72, hidden_size=128, hidden_layer=2, output_size=12)
        # load pretrianed
        a2m_checkpoint = torch.load(datapath)
        self.gru_classifier.load_state_dict(a2m_checkpoint["model"])
        self.gru_classifier_for_fid.load_state_dict(a2m_checkpoint["model"])
        # freeze params
        self.gru_classifier.eval()
        self.gru_classifier_for_fid.eval()
        for p in self.gru_classifier.parameters():
            p.requires_grad = False
        for p in self.gru_classifier_for_fid.parameters():
            p.requires_grad = False

        self.init()

    def init(self):
        # add metrics
        self.count = torch.tensor(0)
        self.count_seq = torch.tensor(0)

        self.metrics = []
        # Accuracy
        self.accuracy = torch.tensor(0)
        self.gt_accuracy = torch.tensor(0)
        self.metrics.extend(["accuracy", "gt_accuracy"])
        
        # Fid
        self.FID = torch.tensor(0)
        self.gt_FID = torch.tensor(0)
        self.metrics.extend(["FID", "gt_FID"])

        # Diversity
        self.Diversity = torch.tensor(0)
        self.gt_Diversity = torch.tensor(0)
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # Multimodality
        self.Multimodality = torch.tensor(0)
        self.gt_Multimodality = torch.tensor(0)
        self.metrics.extend(["Multimodality", "gt_Multimodality"])

        # chached batches
        self.confusion = torch.zeros(self.num_labels, self.num_labels, dtype=torch.long)
        self.gt_confusion = torch.zeros(self.num_labels, self.num_labels, dtype=torch.long)

        self.label_embeddings = []
        self.recmotion_embeddings = []
        self.gtmotion_embeddings = []

    def compute(self):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Accuracy
        self.accuracy = torch.trace(self.confusion) / torch.sum(self.confusion)
        self.gt_accuracy = torch.trace(self.gt_confusion) / torch.sum(
            self.gt_confusion)

        # cat all embeddings
        all_labels = torch.cat(self.label_embeddings, axis=0)
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0)
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0)
        all_gtmotions2 = all_gtmotions.clone()[
            torch.randperm(all_gtmotions.shape[0]), :]
        genstats = calculate_activation_statistics(all_genmotions)
        gtstats = calculate_activation_statistics(all_gtmotions)
        gtstats2 = calculate_activation_statistics(all_gtmotions2)

        all_labels = all_labels.cpu()

        # calculate diversity and multimodality
        self.Diversity, self.Multimodality = calculate_diversity_multimodality(
            all_genmotions,
            all_labels,
            self.num_labels,
            diversity_times=self.diversity_times,
            multimodality_times=self.multimodality_times)

        self.gt_Diversity, self.gt_Multimodality = calculate_diversity_multimodality(
            all_gtmotions, all_labels, self.num_labels)

        metrics.update(
            {metric: getattr(self, metric)
             for metric in self.metrics})

        # Compute Fid
        metrics["FID"] = calculate_fid(gtstats, genstats)
        metrics["gt_FID"] = calculate_fid(gtstats, gtstats2)

        return {**metrics}

    def update(
        self,
        label: Tensor,
        recmotion: Tensor,
        gtmotion: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        labels = label.squeeze().long()
        # Accuracy
        batch_prob = self.gru_classifier(recmotion, lengths=lengths)
        gt_batch_prob = self.gru_classifier(gtmotion, lengths=lengths)
        batch_pred = batch_prob.max(dim=1).indices

        for label, pred in zip(labels, batch_pred):
            self.confusion[label][pred] += 1

        gt_batch_pred = gt_batch_prob.max(dim=1).indices
        for label, pred in zip(labels, gt_batch_pred):
            self.gt_confusion[label][pred] += 1

        # Compute embeddings
        recmotion_embeddings = self.gru_classifier_for_fid(recmotion,
                                                           lengths=lengths)
        gtmotion_embeddings = self.gru_classifier_for_fid(gtmotion,
                                                          lengths=lengths)

        # store all texts and motions
        self.label_embeddings.append(labels)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)
