import json
import os
from os import path
import pickle
import multiprocessing as mp
from multiprocessing import Manager
import torch
import itertools

import torchtext
from torchtext.datasets import PennTreebank, WikiText2, WikiText103
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets.word2vecgpu import LazyDatasetLoader, Word2VecDatasetOnDevice

from pathlib import Path
from typing import *


global use_cuda
use_cuda = torch.cuda.is_available()
device = torch.cuda.current_device() if use_cuda else "cpu"
max_window = 10


def load_lines(dataset):
    eos_idxs = [i for i, token in enumerate(dataset) if token == "<eos>"]
    dataset_lines = [
        dataset[i + 1 : j + 1] for i, j in zip([-1] + eos_idxs, eos_idxs + [-2])
    ]
    return dataset_lines


def get_token_ids(dataset, vocab):
    dataset_tokenized = [
        [vocab.get(x, vocab["<unk>"]) for x in line if len(x.strip()) != 0]
        for line in dataset
    ]
    return dataset_tokenized


def load_vocab(data_dir: Union[str, Path]):
    vocab_tsv = Path(data_dir) / "vocab.tsv"
    if vocab_tsv.exists():
        vocab_stoi = {}
        vocab_freq = {}
        with vocab_tsv.open() as vocab_file:
            next(vocab_file)  # skips header line
            for token_id, line in enumerate(vocab_file):
                token, frequency = line.split()
                vocab_stoi[token] = int(token_id)
                vocab_freq[token] = int(frequency)
    elif path.isfile(data_dir + "vocab_stoi.json") and path.isfile(
        data_dir + "vocab_freq.json"
    ):
        vocab_stoi = json.load(open(data_dir + "vocab_stoi.json", "r"))
        vocab_freq = json.load(open(data_dir + "vocab_freq.json", "r"))
    else:
        TEXT = torchtext.data.Field()
        train_split = torchtext.datasets.LanguageModelingDataset.splits(
            path=data_dir,
            train="train.txt",
            validation=None,
            test=None,
            text_field=TEXT,
        )
        TEXT.build_vocab(train_split[0])
        vocab_stoi_file = open(data_dir + "vocab_stoi.json", "w")
        vocab_freq_file = open(data_dir + "vocab_freq.json", "w")
        json.dump(TEXT.vocab.stoi, vocab_stoi_file)
        json.dump(TEXT.vocab.freqs, vocab_freq_file)
        vocab_stoi_file.close()
        vocab_freq_file.close()
    return vocab_stoi, vocab_freq


def load_tokenizer(dataset):
    data_dir = "./data/" + dataset + "/"
    if path.isfile(data_dir + "train_tokenized.pkl"):
        train_tokenized = pickle.load(open(data_dir + "train_tokenized.pkl", "rb"))
    else:
        train_tokenized = []
        vocab_stoi = json.load(open(data_dir + "vocab_stoi.json", "r"))
        vocab_freq = json.load(open(data_dir + "vocab_freq.json", "r"))

        with open(data_dir + "train.txt", "r") as f:
            for line in f:
                words = line.split()
                train_tokenized.append(
                    [vocab_stoi[ele] for ele in words] + [vocab_stoi["<eos>"]]
                )

        pickle.dump(train_tokenized, open(data_dir + "train_tokenized.pkl", "wb"))
    return train_tokenized


def load_train_data_as_tensor(dataset):
    data_dir = "./data/" + dataset + "/"
    tensor_file = Path(data_dir + "train.pt")
    if tensor_file.exists():
        return torch.load(tensor_file)
    else:
        train_tensor = torch.tensor(
            list(itertools.chain.from_iterable(load_tokenizer(dataset)))
        )
        torch.save(train_tensor, tensor_file)
    return train_tensor


def get_iter_on_device(
    batch_size,
    dataset,
    model_type,
    n_gram,
    subsample_thresh,
    data_device,
    add_pad,
    eos_mask,
):
    print("Loading VOCAB & Tokenized Training files ...")
    vocab_stoi, vocab_freq = load_vocab("./data/" + dataset)
    train_tokenized = load_train_data_as_tensor(dataset)

    ## Create Vocabulary properties
    print("Creating iterable dataset ...")
    TEXT = torchtext.data.Field()
    TEXT.stoi = vocab_stoi
    TEXT.freqs = vocab_freq
    TEXT.itos = [k for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])]

    # Since we won't train on <pad> and <eos>. These should not come in any sort of
    # subsampling and negative sampling part.
    TEXT.freqs["<pad>"] = 0
    TEXT.freqs["<unk>"] = 0

    if eos_mask:
        TEXT.freqs["<eos>"] = 0
    # We want to pad max window length pad tokens and eos to the start
    # and to the end of the corpus and remove <unk> tokens

    # if add_pad:
    #     paddings = torch.tensor([TEXT.stoi['<eos>']] * max_window)
    #     train_tokenized = torch.cat(
    #                         (paddings,
    #                         train_tokenized[train_tokenized != TEXT.stoi['<unk>']],
    #                         paddings))

    ## Create data on the device
    print("Creating iterable dataset on GPU/CPU...")
    if data_device == "gpu":
        data_device = device
    train_iter = LazyDatasetLoader(
        training_tensor=train_tokenized,
        n_splits=1000,
        window_size=n_gram,
        vocab=TEXT,
        subsample_thresh=subsample_thresh,
        eos_mask=eos_mask,
        device=data_device,
        batch_size=batch_size,
    )

    val_iter, test_iter = None, None
    return TEXT, train_iter, val_iter, test_iter, None
