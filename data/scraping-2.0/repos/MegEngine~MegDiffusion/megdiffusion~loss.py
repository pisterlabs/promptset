"""Loss functions.

Modified from OpenAI improved/guided diffusion codebase:
https://github.com/openai/guided-diffusion/blob/master/guided_diffusion/losses.py#L328

OpenAI's code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf//utils.py
"""

import numpy as np
from megengine import Tensor
import megengine.functional as F

def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor):
    """Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 
        + F.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * F.exp(-logvar2)
    )

def discretized_gaussian_log_likelihood(x: Tensor, *, means: Tensor, log_scales: Tensor):
    """Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    Assumes data is integers [0, 255] rescaled to [-1, 1].

    Ported from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py#L116

    Args:
        x: the target images. It is assumed that this was uint8 values, rescaled to the range [-1, 1].
        means: the Gaussian mean Tensor.
        log_scales: the Gaussian log stddev Tensor.

    Retrun:
        a tensor like x of log probabilities (in nats).
    """
    def _approx_standard_normal_cdf(x: Tensor):
        """A fast approximation of the cumulative distribution function of the standard normal."""
        return 0.5 * (1.0 + F.tanh(np.sqrt(2.0 / np.pi).astype("float32") * (x + 0.044715 * F.pow(x, 3))))

    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = F.exp(-log_scales)

    # [-1, 1] Split to 255 bins
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = _approx_standard_normal_cdf(min_in)

    log_cdf_plus = F.log(F.maximum(cdf_plus, 1e-12))
    log_one_minus_cdf_min = F.log(F.maximum((1.0 - cdf_min), 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = F.where(
        x < -0.999,
        log_cdf_plus,
        F.where(x > 0.999, 
                log_one_minus_cdf_min, 
                F.log(F.maximum((cdf_delta),1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs