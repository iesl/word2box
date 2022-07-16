import click


class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


@click.command(
    context_settings=dict(show_default=True),
)

########## Model Properties ##################
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "Word2Vec",
            "GloVe",
            "lbl",
            "box_affine",
            "Word2Box",
            "Word2VecPooled",
            "Word2BoxPooled",
            "Word2BoxConjunction",
            "Word2BoxConjunctionConditional",
            "Word2BoxConjunctionBounded",
            "Word2Gauss",
        ],
        case_sensitive=False,
    ),
    default="lbl",
    help="model architecture to use",
)
@click.option(
    "--embedding_dim",
    type=int,
    default=300,
    help="dimension for embedding space",
)
@click.option(
    "--sep_output", type=int, default=0, help="Use seperate output representation."
)
@click.option(
    "--diag_context",
    type=int,
    default=0,
    help="Context matrix type would be diagonal if 1 else full matrix.",
)

#######  Dataset Properties #########
@click.option(
    "--dataset",
    type=click.Choice(
        [
            "ptb",
            "hyper_text",
            "wackydata",
            "wackydata_padded",
            "ptb_padded",
            "wackypedia",
            "wackypedia_lemma",
            "wikipedia_large",
            "wikipedia_lemma_large",
        ]
    ),
    default="ptb",
    help="Dataset to train and evaluate the language model",
)
@click.option(
    "--eval_file",
    default="./data/similarity_datasets/",
    help="Dataset to evaluate the language model",
)
@click.option(
    "--subsample_thresh", type=float, default=1e-3, help="window size for context words"
)
@click.option("--data_device", type=str, default="cpu", help="Data load device")
@click.option(
    "--add_pad / --no_add_pad",
    default=True,
    help="Enable/ Disable padding at begin/end",
)
@click.option(
    "--eos_mask / --no_eos_mask", default=True, help="Enable/ Disable eos mask"
)

######  Training parameters #########
@click.option("--n_gram", type=int, default=4, help="window size for context words")
@click.option(
    "--negative_samples",
    type=int,
    default=5,
    help="no. of negative samples for training word2vec",
)
@click.option(
    "--batch_size",
    type=int,
    default=1024,
    help="batch size for training will be BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--lr",
    type=float,
    default=0.01,
    help="learning rate",
)
@click.option(
    "--num_epochs", type=int, default=2000, help="maximum number of epochs to train"
)

######### Loss Function ############
@click.option(
    "--loss_fn",
    type=click.Choice(
        ["max_margin", "nce", "nll"],
        case_sensitive=False,
    ),
    default="max_margin",
    help="Loss Function. nce for word2vec, max_margin for Word2Box",
)
@click.option(
    "--margin",
    type=float,
    default=10,
    help="margin is applicable for max_margin loss",
)


######### Box properties ############
@click.option(
    "--box_type",
    type=click.Choice(
        [
            "BoxTensor",
            "BoxTensorLearntTemp",
            "DeltaBoxTensor",
            "SigmoidBoxTensor",
        ],
        case_sensitive=False,
    ),
    default="DeltaBoxTensor",
    help="model architecture to use",
)
@click.option(
    "--int_temp",
    type=float,
    default=0.01,
    help="Intersection temperature",
)
@click.option(
    "--vol_temp",
    type=float,
    default=1.0,
    help="Volume temperature",
)

############ Pooling options #########
@click.option(
    "--pooling",
    type=click.Choice(
        ["avg_pool", "learnt_pool"],
        case_sensitive=False,
    ),
    default="avg_pool",
    help="model architecture to use",
)
@click.option(
    "--alpha_dim",
    type=int,
    default=32,
    help="dimension for alpha parameters in learnt pool",
)

####### System configs ##############
@click.option(
    "--seed",
    type=int,
    help="seed for random number generator",
)
@click.option("--cuda / --no_cuda", default=True, help="Enable/ Disable GPU")
@click.option(
    "--save_model / --no_save_model",
    default=False,
    help="whether or not to save the best model",
)
@click.option(
    "--save_dir",
    default="./checkpoint",
    help="Dataset to evaluate the language model",
)
@click.option(
    "--log_frequency", type=int, default=1000, help="Number of steps between evals"
)
def train(**config):
    """Train a languge model"""
    from .train import training

    training(config)
