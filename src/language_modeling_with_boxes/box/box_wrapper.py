from torch import Tensor
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Type, TypeVar
from .utils import log1mexp, ExpEi, reparam_trick, Bessel

tanh_eps = 1e-20
euler_gamma = 0.57721566490153286060


def _box_shape_ok(t: Tensor, learnt_temp=False) -> bool:
    if len(t.shape) < 2:
        return False
    if not learnt_temp:
        if t.size(-2) != 2:
            return False
        return True
    else:
        if t.size(-2) != 4:
            return False

        return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
    return "Shape of {} has to be {} but is {}".format(
        tensor_name, expected_shape, tuple(actual_shape)
    )


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):
    """A wrapper to which contains single tensor which
    represents single or multiple boxes.

    Have to use composition instead of inheritance because
    it is not safe to interit from :class:`torch.Tensor` because
    creating an instance of such a class will always make it a leaf node.
    This works for :class:`torch.nn.Parameter` but won't work for a general
    box_tensor.
    """

    def __init__(self, data: Tensor, learnt_temp: bool = False) -> None:
        """
        .. todo:: Validate the values of z, Z ? z < Z

        Arguments:
            data: Tensor of shape (**, zZ, num_dims). Here, zZ=2, where
                the 0th dim is for bottom left corner and 1st dim is for
                top right corner of the box
        """

        if _box_shape_ok(data, learnt_temp):
            self.data = data
        else:
            raise ValueError(_shape_error_str("data", "(**,2,num_dims)", data.shape))
        super().__init__()

    def __repr__(self):
        return "box_tensor_wrapper(" + self.data.__repr__() + ")"

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor"""

        return self.data[..., 0, :]

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor"""

        return self.data[..., 1, :]

    @property
    def box_type(self):
        return "BoxTensor"

    @property
    def centre(self) -> Tensor:
        """Centre coordinate as Tensor"""

        return (self.z + self.Z) / 2

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        """
        Creates a box by stacking z and Z along -2 dim.
        That is if z.shape == Z.shape == (**, num_dim),
        then the result would be box of shape (**, 2, num_dim)
        """

        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape
                )
            )
        box_val: Tensor = torch.stack((z, Z), -2)

        return cls(box_val)

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor, dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(t.size(dim))
            )
        split_point = int(len_dim / 2)
        z = t.index_select(
            dim,
            torch.tensor(list(range(split_point)), dtype=torch.int64, device=t.device),
        )

        Z = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)), dtype=torch.int64, device=t.device
            ),
        )

        return cls.from_zZ(z, Z)

    def _intersection(
        self: TBoxTensor,
        other: TBoxTensor,
        gumbel_beta: float = 1.0,
        bayesian: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        t1 = self
        t2 = other

        if bayesian:
            try:
                z = gumbel_beta * torch.logaddexp(
                    t1.z / gumbel_beta, t2.z / gumbel_beta
                )
                z = torch.max(z, torch.max(t1.z, t2.z))
                Z = -gumbel_beta * torch.logaddexp(
                    -t1.Z / gumbel_beta, -t2.Z / gumbel_beta
                )
                Z = torch.min(Z, torch.min(t1.Z, t2.Z))
            except Exception as e:
                print("Gumbel intersection is not possible")
                breakpoint()
        else:
            z = torch.max(t1.z, t2.z)
            Z = torch.min(t1.Z, t2.Z)

        return z, Z

    def gumbel_intersection_log_volume(
        self: TBoxTensor,
        other: TBoxTensor,
        volume_temp=1.0,
        intersection_temp: float = 1.0,
        scale=1.0,
    ) -> TBoxTensor:
        z, Z = self._intersection(other, gumbel_beta=intersection_temp, bayesian=True)
        vol = self._log_soft_volume_adjusted(
            z, Z, temp=volume_temp, gumbel_beta=intersection_temp, scale=scale
        )
        return vol

    def intersection(self: TBoxTensor, other: TBoxTensor) -> TBoxTensor:
        """Gives intersection of self and other.

        .. note:: This function can give fipped boxes, i.e. where z[i] > Z[i]
        """
        z, Z = self._intersection(other)

        return self.from_zZ(z, Z)

    def join(self: TBoxTensor, other: TBoxTensor) -> TBoxTensor:
        """Gives join"""
        z = torch.min(self.z, other.z)
        Z = torch.max(self.Z, other.Z)

        return self.from_zZ(z, Z)

    @classmethod
    def _log_soft_volume(
        cls, z: Tensor, Z: Tensor, temp: float = 1.0, scale: Union[float, Tensor] = 1.0
    ) -> Tensor:
        eps = torch.finfo(z.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        return torch.sum(
            torch.log(F.softplus(Z - z, beta=temp) + 1e-23), dim=-1
        ) + torch.log(
            s
        )  # need this eps to that the derivative of log does not blow

    def log_soft_volume(
        self, temp: float = 1.0, scale: Union[float, Tensor] = 1.0
    ) -> Tensor:
        res = self._log_soft_volume(self.z, self.Z, temp=temp, scale=scale)

        return res

    @classmethod
    def _log_soft_volume_adjusted(
        cls,
        z: Tensor,
        Z: Tensor,
        temp: float = 1.0,
        gumbel_beta: float = 1.0,
        scale: Union[float, Tensor] = 1.0,
    ) -> Tensor:
        eps = torch.finfo(z.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        return (
            torch.sum(
                torch.log(
                    F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=temp) + 1e-23
                ),
                dim=-1,
            )
            + torch.log(s)
        )

    def intersection_log_soft_volume(
        self,
        other: TBoxTensor,
        temp: float = 1.0,
        gumbel_beta: float = 1.0,
        bayesian: bool = False,
        scale: Union[float, Tensor] = 1.0,
    ) -> Tensor:
        z, Z = self._intersection(other, gumbel_beta, bayesian)
        vol = self._log_soft_volume(z, Z, temp=temp, scale=scale)

        return vol

    @classmethod
    def get_wW(cls, z, Z):
        return z, Z

    @classmethod
    def _weights_init(cls, weights: torch.Tensor):
        """An in-place weight initializer method
        which can be used to do sensible init
        of weights depending on box type.
        For this base class, this method does nothing"""
        pass


class DeltaBoxTensor(BoxTensor):
    """Same as BoxTensor but with a different parameterization: (**,wW, num_dims)

    z = w
    Z = z + delta(which is always positive)
    """

    @property
    def z(self) -> Tensor:
        return self.data[..., 0, :]

    @property
    def Z(self) -> Tensor:
        z = self.z
        Z = z + torch.nn.functional.softplus(self.data[..., 1, :], beta=10)

        return Z

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:

        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape
                )
            )
        w, W = cls.get_wW(z, Z)  # type:ignore

        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)

    @classmethod
    def get_wW(cls, z, Z):
        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape
                )
            )
        w = z
        W = _softplus_inverse(Z - z, beta=10.0)  # type:ignore

        return w, W

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor, dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(t.size(dim))
            )
        split_point = int(len_dim / 2)
        w = t.index_select(
            dim,
            torch.tensor(list(range(split_point)), dtype=torch.int64, device=t.device),
        )

        W = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)), dtype=torch.int64, device=t.device
            ),
        )
        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)


def _softplus_inverse(t: torch.Tensor, beta=1.0, threshold=20):
    below_thresh = beta * t < threshold
    res = t
    res[below_thresh] = torch.log(torch.exp(beta * t[below_thresh]) - 1.0) / beta

    return res
