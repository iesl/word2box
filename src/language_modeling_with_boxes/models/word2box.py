import argparse
import torchtext, random, torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import wandb

from ..box.box_wrapper import DeltaBoxTensor, BoxTensor
from ..box.modules import BoxEmbedding
from .BaseModule import BaseModule, MaskedAvgPoolingLayer

global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1


class Word2Box(BaseModule):
    def __init__(
        self,
        TEXT=None,
        embedding_dim=50,
        batch_size=10,
        n_gram=4,
        volume_temp=1.0,
        intersection_temp=1.0,
        box_type="BoxTensor",
        **kwargs
    ):
        super(Word2Box, self).__init__()

        # Model
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.itos)
        self.embedding_dim = embedding_dim

        # Box features
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.box_type = box_type

        # Create embeddings
        self.embeddings_word = BoxEmbedding(
            self.vocab_size, self.embedding_dim, box_type=box_type
        )
        self.embedding_context = BoxEmbedding(
            self.vocab_size, self.embedding_dim, box_type=box_type
        )

    def forward(self, idx_word, idx_context, train=True):
        context_boxes = self.embedding_context(idx_context)  # Batch_size * 2 * dim
        word_boxes = self.embeddings_word(idx_word)  # Batch_size * ns+1 * 2 * dim
        if train == True:
            word_boxes.data.unsqueeze_(
                1
            )  # Braodcast the word vector to the the context + negative_samples.

        if self.intersection_temp == 0.0:
            score = word_boxes.intersection_log_soft_volume(
                context_boxes, temp=self.volume_temp
            )
        else:
            score = word_boxes.gumbel_intersection_log_volume(
                context_boxes,
                volume_temp=self.volume_temp,
                intersection_temp=self.intersection_temp,
            )

        return score

    def word_similarity(self, w1, w2):
        with torch.no_grad():
            word1 = self.embeddings_word(w1)
            word2 = self.embeddings_word(w2)
            if self.intersection_temp == 0.0:
                score = word1.intersection_log_soft_volume(word2, temp=self.volume_temp)
            else:
                score = word1.gumbel_intersection_log_volume(
                    word2,
                    volume_temp=self.volume_temp,
                    intersection_temp=self.intersection_temp,
                )
            return score

    def conditional_similarity(self, w1, w2):
        with torch.no_grad():
            word1 = self.embeddings_word(w1)
            word2 = self.embeddings_word(w2)
            if self.intersection_temp == 0.0:
                score = word1.intersection_log_soft_volume(word2, temp=self.volume_temp)
            else:
                score = word1.gumbel_intersection_log_volume(
                    word2,
                    volume_temp=self.volume_temp,
                    intersection_temp=self.intersection_temp,
                )
            #  Word1 Word2  queen   royalty 5.93
            # Word2 is more geenral P(royalty | queen) = 1
            # Thus we need p(w2 | w1)
            score -= word1._log_soft_volume_adjusted(
                word1.z,
                word1.Z,
                temp=self.volume_temp,
                gumbel_beta=self.intersection_temp,
            )
            return score


class Word2BoxConjunction(Word2Box):
    def intersect_multiple_box(self, boxes, mask):
        beta = self.intersection_temp
        z = boxes.z.clone()
        Z = boxes.Z.clone()

        z[~mask] = float("-inf")
        Z[~mask] = float("inf")
        z = beta * torch.logsumexp(z / beta, dim=1, keepdim=True)
        Z = -beta * torch.logsumexp(-Z / beta, dim=1, keepdim=True)

        return BoxTensor.from_zZ(z, Z)

    def forward(self, idx_word, idx_context, mask_context, train=True):
        context_boxes = self.embedding_context(idx_context)  # Batch_size * 2 * dim
        # Notce that the context is not masked yet. Need to mask them as well.

        word_boxes = self.embeddings_word(idx_word)  # Batch_size * ns+1 * 2 * dim
        pooled_context = self.intersect_multiple_box(context_boxes, mask_context)

        if self.intersection_temp == 0.0:
            score = word_boxes.intersection_log_soft_volume(
                pooled_context, temp=self.volume_temp
            )
        else:
            score = word_boxes.gumbel_intersection_log_volume(
                pooled_context,
                volume_temp=self.volume_temp,
                intersection_temp=self.intersection_temp,
            )
        return score
