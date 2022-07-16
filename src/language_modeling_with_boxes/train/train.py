import torch
import random

from .Trainer import Trainer, TrainerWordSimilarity

from ..models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from ..datasets.utils import get_iter_on_device

global use_cuda
use_cuda = torch.cuda.is_available()
device = torch.cuda.current_device() if use_cuda else "cpu"


def training(config):

    # Set the seed
    if config["seed"] is None:
        config["seed"] = random.randint(0, 2**32)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    TEXT, train_iter, val_iter, test_iter, subsampling_prob = get_iter_on_device(
        config["batch_size"],
        config["dataset"],
        config["model_type"],
        config["n_gram"],
        config["subsample_thresh"],
        config["data_device"],
        config["add_pad"],
        config["eos_mask"],
    )

    if config["model_type"] == "Word2Box":
        model = Word2Box(
            TEXT=TEXT,
            embedding_dim=config["embedding_dim"],
            batch_size=config["batch_size"],
            n_gram=config["n_gram"],
            intersection_temp=config["int_temp"],
            volume_temp=config["vol_temp"],
            box_type=config["box_type"],
            pooling=config["pooling"],
        )

    elif config["model_type"] == "Word2Vec":
        model = Word2Vec(
            TEXT=TEXT,
            embedding_dim=config["embedding_dim"],
            batch_size=config["batch_size"],
            n_gram=config["n_gram"],
        )

    elif config["model_type"] == "Word2VecPooled":
        model = Word2VecPooled(
            TEXT=TEXT,
            embedding_dim=config["embedding_dim"],
            batch_size=config["batch_size"],
            n_gram=config["n_gram"],
            pooling=config["pooling"],
        )
    elif config["model_type"] == "Word2BoxConjunction":
        model = Word2BoxConjunction(
            TEXT=TEXT,
            embedding_dim=config["embedding_dim"],
            batch_size=config["batch_size"],
            n_gram=config["n_gram"],
            intersection_temp=config["int_temp"],
            volume_temp=config["vol_temp"],
            box_type=config["box_type"],
        )
    elif config["model_type"] == "Word2Gauss":
        model = Word2Gauss(
            TEXT=TEXT,
            embedding_dim=config["embedding_dim"],
            batch_size=config["batch_size"],
            n_gram=config["n_gram"],
        )
    else:
        raise ValueError("Model type is not valid. Please enter a valid model type")

    if use_cuda:
        model.cuda()

    # Instance of trainer
    if config["model_type"] == "Word2Box" or config["model_type"] == "Word2Vec":
        trainer = TrainerWordSimilarity(
            train_iter=train_iter,
            val_iter=val_iter,
            vocab=TEXT,
            lr=config["lr"],
            n_gram=config["n_gram"],
            loss_fn=config["loss_fn"],
            negative_samples=config["negative_samples"],
            model_mode="SkipGram",
            log_frequency=config["log_frequency"],
            margin=config["margin"],
            similarity_datasets_dir=config["eval_file"],
            subsampling_prob=None,  # pass: subsampling_prob, when you want to adjust neg_sampling distn
        )
    elif (
        config["model_type"] == "Word2BoxPooled"
        or config["model_type"] == "Word2VecPooled"
        or config["model_type"] == "Word2BoxConjunction"
        or config["model_type"] == "Word2Gauss"
    ):
        trainer = TrainerWordSimilarity(
            train_iter=train_iter,
            val_iter=val_iter,
            vocab=TEXT,
            lr=config["lr"],
            n_gram=config["n_gram"],
            loss_fn=config["loss_fn"],
            negative_samples=config["negative_samples"],
            model_mode="CBOW",
            log_frequency=config["log_frequency"],
            margin=config["margin"],
            similarity_datasets_dir=config["eval_file"],
            subsampling_prob=None,  # pass: subsampling_prob, when you want to adjust neg_sampling distn
        )

    trainer.train_model(
        model=model,
        num_epochs=config["num_epochs"],
        path=config.get("save_dir", False),
        save_model=config.get("save_model", False),
    )
