import torch
from torch.nn import functional as F
from ..box.utils import log1mexp


global use_cuda
use_cuda = torch.cuda.is_available()
device = torch.cuda.current_device() if use_cuda else "cpu"


def nll(pos, neg, **kwagrs):
    """
	The loss funtion is used for box embeddings
	Args:
	    pos = log probabiltiy for positive examples
	    neg = log probabiltiy for negaitve examples
	Output:
	    loss = - (pos + sum(log(1-exp(neg))) 
	"""
    assert (pos < 0).all(), "Log probabiltiy can not be positive"
    assert (neg < 0).all(), "Log probabiltiy can not be positive"
    return -(pos + torch.sum(log1mexp(neg), dim=1))


def nce(pos, neg, **kwagrs):
    """
	The loss function can be used for any embeddings.
	However, here we pass the unnormalised probabilities
	through sigmoid to normalised the score. Word2vec uses
	this loss function.

	Args:
	    pos: Unnormalised similarity score for positives.
	    neg: Unnormalised similarity score for negatives.
	Output:
	    loss = -(logsigmoid(pos) + sum(logsigmoid(-neg))) 
	"""
    return -(F.logsigmoid(pos) + torch.sum(F.logsigmoid(-neg), dim=1))


def max_margin(pos, neg, margin=5.0):
    """
	This is max margin loss for box embeddings.
	Here, the input scores can be un-normalised. The object here
	is to make increse the pos similarity score more than a margin
	from the negative scores. If that margin is satisfied then the
	loss is zero.

	Args:
	    pos: Unnormalised similarity(maybe log in case of Boxes) score for positives.
	    neg: Unnormalised similarity(maybe log in case of Boxes) score for negatives.
	Output:
	    loss =  - max(0, pos - mean(neg) + margin)
	"""
    # Replicate the positive score number of negative sample times
    zero = torch.tensor(0.0).to(device)
    return torch.sum(torch.max(zero, neg - pos + margin), dim=1)
