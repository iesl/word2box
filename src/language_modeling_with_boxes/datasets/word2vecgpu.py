import torch
from torch import LongTensor, BoolTensor, Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Union

from pytorch_utils import TensorDataLoader

import attr


class Word2VecDatasetOnDevice(Dataset):
    def __init__(
        self,
        corpus: LongTensor,
        window_size: int = 10,
        vocab: Union[Dict, Any] = None,
        subsample_thresh: float = 1e-3,
        eos_mask: bool = True,
        device: Union[int, str] = None,
    ):
        self.corpus = corpus
        self.window_size = window_size
        self.vocab = vocab
        self.subsample_thresh = subsample_thresh
        self.eos_mask = eos_mask
        self.pad_id = torch.tensor(self.vocab.stoi["<pad>"]).to(self.corpus.device)
        self.eos_token = torch.tensor(self.vocab.stoi["<eos>"]).to(self.corpus.device)
        # pad this at the beginning and end with window_size number of padding
        self.pad_size = 10
        total_words = sum(self.vocab.freqs.values())
        unigram_prob = (
            torch.tensor([self.vocab.freqs.get(key, 0) for key in self.vocab.itos])
            / total_words
        )
        self.subsampling_prob = 1.0 - torch.sqrt(
            subsample_thresh / (unigram_prob + 1e-19)
        ).to(
            self.corpus.device
        )  #

    def __getitem__(
        self, idx: LongTensor
    ) -> Tuple[LongTensor, LongTensor, BoolTensor, BoolTensor]:
        # idx is a Tensor of indicies of the corpus, eg. [2342,12312312,34534,1]
        # we will interpret these as the id of the center word
        idx += self.pad_size
        # Idx is repeated to get the sliding window effect
        # For the sliding window part we add the range with idx
        window_range = torch.arange(-self.window_size, self.window_size + 1)
        idx = idx.unsqueeze(1) + window_range.unsqueeze(0)

        # idx = torch.transpose(idx.repeat(2*self.window_size+1,1), 0, 1)
        # idx = idx + torch.arange(-self.window_size, self.window_size+1)

        # Get the middle slice for the center
        # The rest of them are context
        center = self.corpus[idx[:, self.window_size]]
        context = self.corpus[
            torch.cat(
                (idx[:, : self.window_size], idx[:, self.window_size + 1 :]), dim=1
            )
        ]
        # Get do the subsampling.
        center = self.sub_sample_words(center)
        context = self.sub_sample_words(context)

        # Get rid of the dataset that has the center word as <pad>.
        # Or has all context words as <pad>.
        if not self.eos_mask:
            keep = (center != self.pad_id) & (context != self.pad_id).any(dim=-1)
            center = center[keep]
            context = context[keep]
            assert (center != self.pad_id).all()
            context_mask = torch.ones_like(context)
        else:
            keep = (
                (center != self.pad_id)
                & (context != self.pad_id).any(dim=-1)
                & (center != self.eos_token)
                & (context != self.eos_token).any(dim=-1)
            )
            center = center[keep]
            context = context[keep]
            assert (center != self.pad_id).all()
            context_mask = self.get_mask(context)
            # Mask might do away with the whole sentence. In that case remove that
            keep = (context_mask != False).any(dim=1).squeeze()
            center = center[keep]
            context = context[keep]
            context_mask = context_mask[keep]
        return {
            "center_word": center,
            "context_words": context,
            "context_mask": context_mask,
        }

    def __len__(self) -> int:
        return len(self.corpus) - 2 * self.pad_size

    def to(self, device: Union[torch.device, str]):
        return Word2VecDatasetOnDevice(
            self.corpus.to(device), self.window_size, self.vocab, self.subsample_thresh
        )

    def sub_sample_words(self, _input: LongTensor) -> BoolTensor:
        ## Mask out the subsampled words. We will do so by
        ## replacing them with pad ids.
        mask_prob = torch.rand(_input.shape).to(_input.device)
        _input[mask_prob < self.subsampling_prob[_input]] = self.pad_id
        return _input

    def get_mask(self, _input: LongTensor) -> BoolTensor:
        ## Get the mask for the contexts that has pad token.
        right_mask = ~(
            (_input[:, self.window_size :] == self.eos_token).cumsum(dim=-1) > 0
        )
        l_eos = _input[:, : self.window_size] == self.eos_token
        left_mask = ~(
            l_eos | (l_eos.any(dim=-1, keepdim=True) & (l_eos.cumsum(dim=-1) == 0))
        )
        mask = torch.cat((left_mask, right_mask), dim=-1)
        return (_input != self.pad_id) & mask


# @classmethod
#   def from_pkl(
#       cls,
#       file_location: Path
#   ) -> Word2VecDataset:
#       raise NotImplementedError
#       corpus = ....
#       return cls(corpus)

from math import ceil


@attr.s(auto_attribs=True)
class LazyDatasetLoader:
    training_tensor: Tensor
    n_splits: int
    window_size: int = 10
    vocab: Union[Dict, Any] = None
    subsample_thresh: float = 1e-3
    eos_mask: bool = True
    batch_size: int = 64
    device: Union[int, str] = None

    def __attrs_post_init__(self):
        lng = len(self.training_tensor)
        splits = [int(lng / 10) * i for i in range(10)]
        splits.append(lng)
        self.leng = sum(
            ceil((j - i) / self.batch_size) for i, j in zip(splits[:-1], splits[1:])
        )
        self.training_tensor_chunks = [
            self.training_tensor[splits[i] : splits[i + 1]] for i in range(10)
        ]

    def __iter__(self):
        for chunk in self.training_tensor_chunks:
            train_dataset = Word2VecDatasetOnDevice(
                corpus=chunk,
                window_size=self.window_size,
                vocab=self.vocab,
                subsample_thresh=self.subsample_thresh,
                eos_mask=self.eos_mask,
            ).to(self.device)
            train_iter = TensorDataLoader(train_dataset, self.batch_size, shuffle=True)
            yield from train_iter
            del train_dataset
            del train_iter

    def __len__(self):
        return self.leng
