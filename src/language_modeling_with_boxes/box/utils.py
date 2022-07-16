import torch
import math
from scipy import special
from torch.distributions.gumbel import Gumbel


_log1mexp_switch = math.log(0.5)


class ExpEi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        dev = input.device
        with torch.no_grad():
            x = special.exp1(input.detach().cpu()).to(dev)
            input.to(dev)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * (-torch.exp(-input) / input)
        return grad_input


class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        dev = input.device
        with torch.no_grad():
            x = special.k0(input.detach().cpu()).to(dev)
            input.to(dev)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        dev = grad_output.device
        with torch.no_grad():
            grad_input = grad_output * (-special.k1(input.detach().cpu())).to(dev)
            input.to(dev)

        return grad_input


def reparam_trick(
    mu: torch.Tensor, gumbel_beta: float, n_samples: int = 10, dist_type: str = "g_max"
) -> torch.Tensor:
    dev = mu.device
    m = Gumbel(
        torch.zeros(mu.shape[0], mu.shape[-1]).to(dev), torch.tensor([1.0]).to(dev)
    )
    samples = m.sample(torch.Size([n_samples]))
    sample = torch.mean(samples, axis=0)
    if dist_type == "g_min":
        sample = -sample

    return sample * gumbeL_beta + mu


def log1mexp(
    x: torch.Tensor, split_point=_log1mexp_switch, exp_zero_eps=1e-7
) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).

    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0

    For details, see

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
    # Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

    return Z


def log1pexp(x: torch.Tensor):
    """Computes log(1+exp(x))

    see: Page 7, eqn 10 of https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    also see: https://github.com/SurajGupta/r-source/blob/master/src/nmath/plogis.c
    """
    Z = torch.zeros_like(x)
    zone1 = x <= 18.0
    zone2 = (x > 18.0) * (x < 33.3)  # And operator using *
    zone3 = x >= 33.3
    Z[zone1] = torch.log1p(torch.exp(x[zone1]))
    Z[zone2] = x[zone2] + torch.exp(-(x[zone2]))
    Z[zone3] = x[zone3]

    return Z
