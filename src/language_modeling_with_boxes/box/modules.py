import logging
from typing import Union, TypeVar

import torch
from torch.nn import Embedding

from .box_wrapper import BoxTensor, TBoxTensor, DeltaBoxTensor

logger = logging.getLogger(__name__)

TTensor = TypeVar("TTensor", bound="torch.Tensor")

# TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


def _uniform_init_using_minmax(weight, emb_dim, param1, param2, box_type):
    with torch.no_grad():
        temp = torch.zeros_like(weight)
        torch.nn.init.uniform_(temp, param1, param2)
        z, Z = (
            torch.min(temp[..., :emb_dim], temp[..., emb_dim:]),
            torch.max(temp[..., :emb_dim], temp[..., emb_dim:]),
        )
        w, W = box_type.get_wW(z, Z)
        weight[..., :emb_dim] = w
        weight[..., emb_dim:] = W


def _uniform_small(weight, emb_dim, param1, param2, box_type):
    with torch.no_grad():
        temp = torch.zeros_like(weight)
        torch.nn.init.uniform_(temp, 0.0 + 1e-7, 1.0 - 0.1 - 1e-7)
        # z = torch.min(temp[..., :emb_dim], temp[..., emb_dim:])
        z = temp[..., :emb_dim]
        Z = z + 0.1
        w, W = box_type.get_wW(z, Z)
        weight[..., :emb_dim] = w
        weight[..., emb_dim : emb_dim * 2] = W


def _uniform_big(weight, emb_dim, param1, param2, box_type):
    with torch.no_grad():
        temp = torch.zeros_like(weight)
        torch.nn.init.uniform_(temp, 0 + 1e-7, 0.01)
        z = torch.min(temp[..., :emb_dim], temp[..., emb_dim:])
        Z = z + 0.9
        w, W = box_type.get_wW(z, Z)
        weight[..., :emb_dim] = w
        weight[..., emb_dim : emb_dim * 2] = W


class BoxEmbedding(Embedding):
    box_types = {
        "DeltaBoxTensor": DeltaBoxTensor,
        "BoxTensor": BoxTensor,
    }

    def init_weights(self):
        _uniform_small(
            self.weight,
            self.box_embedding_dim,
            0.0 + 1e-7,
            1.0 - 1e-7,
            self.box_types[self.box_type],
        )

    def __init__(
        self,
        num_embeddings: int,
        box_embedding_dim: int,
        box_type="BoxTensor",
        init_interval_center=0.25,
        init_interval_delta=0.1,
    ) -> None:
        """Similar to allennlp embeddings but returns box
        tensor by splitting the output of usual embeddings
        into z and Z

        Arguments:
            box_embedding_dim: Embedding weight would be box_embedding_dim*2
                               if the temp and the
        """
        vector_emb_dim = box_embedding_dim * 2
        if box_type == "BoxTensorLearntTemp":
            vector_emb_dim = box_embedding_dim * 4

        super().__init__(num_embeddings, vector_emb_dim)
        self.box_type = box_type
        self.init_interval_delta = init_interval_delta
        self.init_interval_center = init_interval_center
        try:
            self.box = self.box_types[box_type]
        except KeyError as ke:
            raise ValueError("Invalid box type {}".format(box_type)) from ke
        self.box_embedding_dim = box_embedding_dim
        self.init_weights()

    def forward(self, inputs: torch.LongTensor):
        emb = super().forward(inputs)  # shape (**, self.box_embedding_dim*2)
        box_emb = self.box.from_split(emb)
        return box_emb

    def get_volumes(self, temp: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.all_boxes.log_soft_volume(temp=temp)

    @property
    def all_boxes(self) -> TBoxTensor:
        all_index = torch.arange(
            0, self.num_embeddings, dtype=torch.long, device=self.weight.device
        )
        all_ = self.forward(all_index)

        return all_
