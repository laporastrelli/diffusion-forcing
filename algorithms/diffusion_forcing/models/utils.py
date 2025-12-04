"""
Code copied from https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import math
import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def divisible_by(numer, denom):
    return (numer % denom) == 0


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def extract_(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract(a, t, x_shape):
    """
    a: 1D tensor of length num_timesteps (e.g. sqrt_alphas_cumprod)
    t: integer tensor of any shape (e.g. [B], [B, T], [T, B], ...)
    x_shape: shape of the target tensor (e.g. x_t.shape)

    Returns: tensor of shape (*t.shape, 1, 1, ..., 1) so it can broadcast
             over x_t.
    """
    # ensure tensors & same device / dtype
    if not torch.is_tensor(a):
        a = torch.tensor(a, device=t.device, dtype=torch.float32)
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=a.device, dtype=torch.long)

    t = t.to(device=a.device, dtype=torch.long)

    # gather coefficients for all indices in t
    # a: [K], t_flat: [B*T*...], result: [B*T*...]
    out = a.gather(0, t.reshape(-1))
    out = out.view(*t.shape)  # shape == t.shape (e.g. [B, T])

    # add singleton dims so out can broadcast with x_shape
    assert len(x_shape) >= t.dim(), "t should not have more dims than x_t"
    extra_dims = len(x_shape) - t.dim()  # e.g. 5 - 2 = 3 for [B,T,C,H,W]

    return out.view(*t.shape, *([1] * extra_dims))



def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def gae_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns
