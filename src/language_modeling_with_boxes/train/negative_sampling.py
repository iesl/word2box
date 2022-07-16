import torch
from torch import LongTensor


class RandomNegativeCBOW:
    """
    This augments a batch of data to include randomly sampled target center words.
    Appends the sampled words with 'center_words' of the batch
    """

    def __init__(self, number_of_samples: int = 5, sampling_distn: LongTensor = None):
        self.number_of_samples = number_of_samples
        self.sampling_distn = sampling_distn

    def __call__(self, batch) -> LongTensor:
        x, y = batch["context_words"].shape
        negatives = torch.multinomial(
            self.sampling_distn,
            num_samples=self.number_of_samples * x,
            replacement=True,
        ).resize(x, self.number_of_samples)
        batch["center_word"] = torch.cat(
            (batch["center_word"].unsqueeze(1), negatives), dim=-1
        )
        return batch


class RandomNegativeSkipGram(RandomNegativeCBOW):
    """
    This augments a batch of data to include randomly sampled target context words.
    Appends the sampled words with 'context_words' of the batch
    """

    def __call__(self, batch) -> LongTensor:
        x, y = batch["context_words"].shape
        negatives = torch.multinomial(
            self.sampling_distn,
            num_samples=self.number_of_samples * x * y,
            replacement=True,
        ).resize(x, y, self.number_of_samples)
        batch["context_words"] = torch.cat(
            (batch["context_words"].unsqueeze(-1), negatives), dim=-1
        )
        return batch
