import torch
import torch.nn as nn

from .BaseModule import BaseModule

eps = 1e-10


class Word2Gauss(BaseModule):
    def __init__(self, TEXT=None, embedding_dim=50, batch_size=10, n_gram=4, **kwargs):
        super(Word2Gauss, self).__init__()
        # Model
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.itos)
        self.embedding_dim = embedding_dim

        # Create embeddings
        self.embeddings_word_mu = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings_word_sigma = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings_context_mu = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings_context_sigma = nn.Embedding(
            self.vocab_size, self.embedding_dim
        )

    def word_similarity(self, w1, w2):
        with torch.no_grad():
            word1_mu = self.embeddings_word_mu(w1)
            word1_var = self.embeddings_word_sigma(w2) ** 2 + eps

            word2_mu = self.embeddings_word_mu(w1)
            word2_var = self.embeddings_word_sigma(w2) ** 2 + eps

            self_similarity = (word1_mu * word1_var * word1_mu).sum(dim=-1) + (
                word2_mu * word2_var * word2_mu
            ).sum(dim=-1)
            resultant_var = word1_var + word2_var
            resultant_mu = word1_var * word1_mu + word2_var * word2_mu
            mutual_similarity = (resultant_mu * (1 / resultant_var) * resultant_mu).sum(
                dim=-1
            )
            score = (
                mutual_similarity
                - self_similarity
                - torch.log(resultant_var).sum(dim=-1)
            )
            return score

    def forward(self, idx_word, idx_context, mask_context, train=True):
        context_mu = self.embeddings_context_mu(idx_context).unsqueeze(
            1
        )  # batch_size x 1 x ngram x embedding size
        context_var = (self.embeddings_context_sigma(idx_context) ** 2 + eps).unsqueeze(
            1
        )  # batch_size x 1 x ngram x embedding size

        word_mu = self.embeddings_word_mu(idx_word).unsqueeze(
            2
        )  # batch_size x negative_sample x 1 x embedding size
        word_var = (self.embeddings_word_sigma(idx_word) ** 2 + eps).unsqueeze(
            2
        )  # batch_size x negative_sample x 1 x embedding size

        self_similarity = (
            (context_mu * context_var * context_mu).sum(dim=-1).sum(dim=-1)
        ) + (word_mu * word_var * word_mu).sum(dim=-1).sum(dim=-1)

        resultant_var = context_var.sum(dim=2, keepdim=True) + word_var
        resultant_mu = (context_var * context_mu + word_var * word_mu).sum(
            dim=2, keepdim=True
        )
        mutual_similarity = (
            (resultant_mu * (1 / resultant_var) * resultant_mu).sum(dim=-1).sum(dim=-1)
        )
        score = (
            mutual_similarity
            - self_similarity
            - torch.log(resultant_var).sum(dim=-1).squeeze()
        )

        return score
