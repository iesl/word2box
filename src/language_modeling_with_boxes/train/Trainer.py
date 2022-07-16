import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import csv
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
from xopen import xopen
import os
import pprint

from .loss import nll, nce, max_margin
from .negative_sampling import RandomNegativeCBOW, RandomNegativeSkipGram


global use_cuda
use_cuda = torch.cuda.is_available()
device = torch.cuda.current_device() if use_cuda else "cpu"
torch.autograd.set_detect_anomaly(True)

criterions = {"nll": nll, "nce": nce, "max_margin": max_margin}


class Trainer:
    def __init__(
        self,
        train_iter,
        val_iter,
        vocab,
        lr=0.001,
        n_gram=4,
        loss_fn=None,
        negative_samples=5,
        log_frequency=1000,
    ):
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.n_gram = n_gram
        self.lr = lr
        self.vocab = vocab
        self.vocab_size = len(self.vocab.itos)
        self.negative_samples = negative_samples
        self.log_frequency = log_frequency
        self.vocab.freqs["<pad>"] = 0  # Don't want to negaive sample pads.
        sorted_freqs = torch.tensor(
            [self.vocab.freqs.get(key, 0) for key in self.vocab.itos]
        )
        self.sampling = torch.pow(sorted_freqs, 0.75)
        self.sampling = self.sampling / torch.sum(self.sampling)
        if use_cuda:
            self.sampling = self.sampling.cuda()

    def train_model(self, model, num_epochs=100, path="./", save_model=False):
        pass

    def load_checkpoint(self, path):
        self.load_state_dict(
            torch.load(os.path.join(path), map_location=torch.device("cpu"))
        )
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()


class TrainerWordSimilarity(Trainer):
    """docstring for TrainerWordSimilarity"""

    def __init__(
        self,
        train_iter,
        val_iter,
        vocab,
        lr=0.001,
        n_gram=4,
        loss_fn="max_margin",
        negative_samples=5,
        model_mode="CBOW",
        log_frequency=1000,
        margin=0.0,
        similarity_datasets_dir=None,
        subsampling_prob=None,
    ):
        super(TrainerWordSimilarity, self).__init__(
            train_iter,
            val_iter,
            vocab,
            lr=lr,
            n_gram=n_gram,
            loss_fn=loss_fn,
            negative_samples=negative_samples,
            log_frequency=log_frequency,
        )

        self.similarity_datasets_dir = similarity_datasets_dir
        self.margin = margin
        self.loss_fn = criterions[loss_fn]
        # If subsampling has been done earlier then word count must have been changed
        # This is an expected word count based on the subsampling prob parameters.
        if subsampling_prob != None:
            self.sampling = (
                torch.min(torch.tensor(1.0).to(device), 1 - subsampling_prob.to(device))
                * self.sampling
            )
        if model_mode == "CBOW":
            self.add_negatives = RandomNegativeCBOW(negative_samples, self.sampling)
        elif model_mode == "SkipGram":
            self.add_negatives = RandomNegativeSkipGram(negative_samples, self.sampling)

    def to(self, batch, device):
        for k, v in batch.items():
            batch[k] = batch[k].to(device)
        return batch

    def train_model(
        self, model, num_epochs=100, path="./checkpoints", save_model=False
    ):
        ## Setting up the optimizers
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=self.lr)
        metric = {}
        best_simlex_ws = -1
        ## Setting Up the loss function
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
            model.train()

            for i, batch in enumerate(tqdm(self.train_iter)):
                # Create negative samples for the batch
                batch = self.to(batch, device)
                batch = self.add_negatives(batch)

                # Start the optimization
                optimizer.zero_grad()
                score = model.forward(
                    batch["center_word"],
                    batch["context_words"],
                    batch["context_mask"],
                    train=True,
                )
                assert (
                    score.shape[-1] == self.negative_samples + 1
                )  # check the shape of the score

                # Score log_intersection_volume (un-normalised) for Word2Box
                pos_score = score[..., 0].reshape(
                    -1, 1
                )  # The first element correspond to the Positive
                neg_score = score[..., 1:].reshape(
                    -1, self.negative_samples
                )  # The rest of the elements are for negative samples
                # Calculate Loss
                loss = self.loss_fn(
                    pos_score, neg_score, margin=self.margin
                )  # Margin is not required for nll or nce
                # Handled through kwargs in loss.py
                total_loss = torch.sum(loss)
                avg_loss = torch.mean(loss)
                if torch.isnan(loss).any():
                    raise RuntimeError("Loss value is nan :(")

                # Back-propagation
                total_loss.backward()
                optimizer.step()

                for param in model.parameters():
                    if torch.isinf(param).any():
                        raise RuntimeError("parameters went to infinity")
                    if torch.isnan(param).any():
                        raise RuntimeError("parameters went to nan")
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            raise RuntimeError("Gradient went to nan")

                epoch_loss.append(avg_loss.data.item())

                # Intermediate eval
                if i % int(len(self.train_iter) / self.log_frequency) == 0:
                    # Start model eval
                    model.eval()
                    ws_metric = self.model_eval(
                        model
                    )  # This ws_metric contains correlations
                    metric.update({"epoch_loss": np.mean(epoch_loss)})
                    # Update the metric for wandb login
                    metric.update(ws_metric)

                    simlex_ws = metric["En-Simlex-999.Txt"]
                    best_simlex_ws = max(metric["En-Simlex-999.Txt"], best_simlex_ws)
                    metric.update({"best_simlex_ws": best_simlex_ws})
                    print(
                        "Epoch {0} | Loss: {1}| spearmanr: {2}".format(
                            epoch + 1, np.mean(epoch_loss), simlex_ws
                        )
                    )

                    if save_model:
                        model.save_checkpoint(Path(path) / "model.ckpt")
                        # savethe best hyperparameter
                        if simlex_ws == best_simlex_ws:
                            model.save_checkpoint(Path(path) / "best_model.ckpt")

                    model.train()

            # Logging training loss
            metric.update({"epoch_loss": np.mean(epoch_loss)})

            model.eval()
            ws_metric = self.model_eval(model)  # This ws_metric contains correlations

            # Update the metric
            metric.update(ws_metric)

            simlex_ws = metric["En-Simlex-999.Txt"]
            best_simlex_ws = max(simlex_ws, best_simlex_ws)
            metric.update({"best_simlex_ws": best_simlex_ws})
            print(
                "Epoch {0} | Loss: {1}| spearmanr: {2}".format(
                    epoch + 1, np.mean(epoch_loss), simlex_ws
                )
            )

            if save_model:
                model.save_checkpoint(Path(path) / "model.ckpt")
                # savethe best hyperparameter
                if simlex_ws > best_simlex_ws:
                    model.save_checkpoint(Path(path) / "best_model.ckpt")

        print("Model trained.")
        print("Output saved.")

    def model_eval(self, model):
        if self.similarity_datasets_dir == None:
            return 0

        metrics = {}
        correlation = 0.0

        # similarity_file is expected to be in the format word1\tword2\tscore
        if self.similarity_datasets_dir is not None and self.vocab is not None:
            file_list = os.listdir(self.similarity_datasets_dir)
            for file in file_list:
                with xopen(os.path.join(self.similarity_datasets_dir, file)) as f:
                    reader = csv.reader(f, delimiter="\t")
                    real_scores = []
                    predicted_scores = []
                    missing_count = 0
                    total_count = 0
                    for row in reader:
                        row[0] = row[0].lower()
                        row[1] = row[1].lower()
                        if (
                            self.vocab.stoi.get(row[0], "<unk>") != "<unk>"
                            and self.vocab.stoi.get(row[1], "<unk>") != "<unk>"
                        ):
                            word1 = (
                                torch.tensor(self.vocab.stoi[row[0]], dtype=int)
                                .unsqueeze(0)
                                .to(device)
                            )
                            word2 = (
                                torch.tensor(self.vocab.stoi[row[1]], dtype=int)
                                .unsqueeze(0)
                                .to(device)
                            )
                            score = model.word_similarity(word1, word2)
                            if file.title() == "Hyperlex-Dev.Txt":
                                score = model.conditional_similarity(word1, word2)

                            predicted_scores.append(score.item())
                            real_scores.append(float(row[2]))
                        else:
                            missing_count += 1
                        total_count += 1
                    # print(f"{file.title()} missing data point: {missing_count} out of {total_count}")
                    # Calculate spearman's rank correlation coefficient between predicted scores and real scores
                    correlation = spearmanr(predicted_scores, real_scores)[0]
                    metrics[file.title()] = correlation

        pprint.pprint(metrics, width=1)

        return metrics
