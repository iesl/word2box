import argparse
import torchtext, random, torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import wandb

from .BaseModule import BaseModule, MaskedAvgPoolingLayer


global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1


class Word2Vec(BaseModule):
    def __init__(self, TEXT=None, embedding_dim=50, batch_size=10, n_gram=4, **kwargs):
        super(Word2Vec, self).__init__()

        # Model
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.itos)
        self.embedding_dim = embedding_dim

        # Create embeddings
        self.embeddings_word = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings_context = nn.Embedding(self.vocab_size, self.embedding_dim)

    def get_masked_embeddings(self, embeddings, idx, mask):
        return embeddings(idx) * mask.unsqueeze(-1)

    def word_similarity(self, w1, w2):
        with torch.no_grad():
            word1 = self.embeddings_word(w1)
            word2 = self.embeddings_word(w2)
            return torch.sum(word1 * word2)

    def conditional_similarity(self, w1, w2):
        return self.word_similarity(w1, w2)

    def forward(self, idx_word, idx_context, context_mask, train=True):
        context = self.get_masked_embeddings(
            self.embeddings_context, idx_context, context_mask.unsqueeze(-1)
        )  # Batch_size * 2xWindow * ns+1 * dim
        word = self.embeddings_word(idx_word)  # Batch_size * dim
        score = torch.sum(word.unsqueeze(1).unsqueeze(1) * context, dim=-1)
        return score


class Word2VecPooled(Word2Vec):
    def __init__(
        self, TEXT=None, embedding_dim=50, batch_size=10, n_gram=4, pooling="avg_pool"
    ):
        super(Word2VecPooled, self).__init__(
            TEXT=TEXT,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            n_gram=n_gram,
        )

        # Pooling
        self.pooling = pooling
        if self.pooling == "avg_pool":
            self.avg_pool_layer = MaskedAvgPoolingLayer()

    def forward(self, idx_word, idx_context, context_mask, train=True):
        context = self.embeddings_context(idx_context)
        # Batch_size * 2xWindow * dim
        # Pool all the context words
        if self.pooling == "avg_pool":
            context = self.avg_pool_layer(context, context_mask, dim=1)
        elif self.pooling == "avg_pool_unmasked":
            context = torch.mean(context, dim=1)
        elif self.pooling == "max_pool":
            context = torch.max(context, dim=1)
        else:
            raise ValueError(f"Pool type {self.pooling} is not allowed for vectors")
        # Compute the similarity with the target word and negatives as well.
        word = self.embeddings_word(idx_word)
        score = torch.sum(word * context.unsqueeze(1), dim=-1)
        return score
