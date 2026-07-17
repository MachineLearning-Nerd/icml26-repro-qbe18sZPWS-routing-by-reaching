import math

import torch


class RewardFns:
    """Collection of static reward functions for HyperGrid."""
    @staticmethod
    def shubert(xy, height, **kwargs):
        """Symmetric Shubert function from compositional-sculpting paper.

        Uses abs() for equal mode probabilities across symmetric positions.
        """
        xy = xy / (height - 1) * 2 - 1  # scale xy from -1 to 1
        w = 2.3
        u_1 = -7.15
        u_2 = -7.15
        x_1 = (u_1 - w) + (xy[..., 0].abs() / 2.0 + 0.5) * w
        x_2 = (u_2 - w) + (xy[..., 1].abs() / 2.0 + 0.5) * w

        mn = -186.6157949555621
        mx = 210.27662470796076

        cosine_sum_1 = 0
        cosine_sum_2 = 0
        for i in range(1, 6):
            cosine_sum_1 = cosine_sum_1 + i * (x_1 * (i + 1) + i).cos()
            cosine_sum_2 = cosine_sum_2 + i * (x_2 * (i + 1) + i).cos()

        r = (cosine_sum_1 * cosine_sum_2 - mn) / (mx - mn)
        r = torch.clamp(r, min=0.0)
        return r

    @staticmethod
    def currin(xy, height, **kwargs):
        """
        Original Currin et al. (1991) function (SFU benchmark).
        f(x) = [1 - exp(-1/(2x₂))] × [2300x₁³ + 1900x₁² + 2092x₁ + 60] / [100x₁³ + 500x₁² + 4x₁ + 20]
        Input domain: x ∈ [0, 1]
        Normalized by max value (~13.77).
        """
        # grid [0, height-1] -> [0, 1]
        xy_scaled = xy / (height - 1)
        x_0 = xy_scaled[..., 0]
        x_1 = xy_scaled[..., 1]

        factor1 = 1 - torch.exp(-1 / (2 * x_1 + 1e-10))
        numer = 2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
        denom = 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20
        f = factor1 * numer / denom

        # Normalize by max value
        r = f / 13.77
        return torch.clamp(r, 0.0, 1.0)

    @staticmethod
    def branin(xy, height, **kwargs):
        """
        Original Branin-Hoo function (SFU benchmark).
        f(x) = a(x₂ - bx₁² + cx₁ - r)² + s(1-t)cos(x₁) + s
        a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π)
        Input domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
        Global minimum: 0.397887 at three locations
        Normalized to [0, 1] where higher = better (inverted).
        """
        # grid [0, height-1] -> x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
        xy_scaled = xy / (height - 1)
        x_1 = xy_scaled[..., 0] * 15 - 5   # [0,1] -> [-5, 10]
        x_2 = xy_scaled[..., 1] * 15       # [0,1] -> [0, 15]

        # Branin parameters
        a = 1
        b = 5.1 / (4 * torch.pi ** 2)
        c = 5 / torch.pi
        r = 6
        s = 10
        t = 1 / (8 * torch.pi)

        f = a * (x_2 - b * x_1 ** 2 + c * x_1 - r) ** 2 + s * (1 - t) * torch.cos(x_1) + s

        # Normalize: min=0.397887, max≈308.13
        mn, mx = 0.397887, 308.13
        r = 1 - (f - mn) / (mx - mn)  # Invert so minimum becomes maximum reward
        return torch.clamp(r, 0.0, 1.0)

    @staticmethod
    def sphere(xy, height, **kwargs):
        xy = xy / (height-1) * 2 - 1
        lo, hi = -5.12, 5.12
        lo_t = xy.new_tensor(lo)
        hi_t = xy.new_tensor(hi)
        mx = 2 * 5.12 ** 2
        # [-1,1] -> [0,1] -> [lo,hi]
        x = (xy / 2 + 0.5) * (hi_t - lo_t) + lo_t
        # Sphere
        r = (x ** 2).sum(dim=-1) / mx
        return r

    @staticmethod
    def diagonal(xy, height, **kwargs):
        """
        Diagonal sigmoid.
        Higher reward at (height-1, 0) (bottom-left), lower reward at (0, height-1) (top-right).
        Uses (x - y) instead of (x + y).
        """
        xy_scaled = xy / (height - 1) * 2 - 1  # scale to [-1, 1]
        r = ((xy_scaled[..., 0] - xy_scaled[..., 1]) * 5).sigmoid()
        r = r + 1e-5
        return r

    @staticmethod
    def circle1(xy, height, **kwargs):
        """Single Gaussian inside a circle, centered at [-h, 0]."""
        xy = xy / (height - 1) * 2 - 1  # scale to [-1, 1]
        x_1, x_2 = xy[..., 0], xy[..., 1]
        radius, h = 0.6, 0.3
        center = [-h, 0.0]

        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5
        in_mask = (dist < radius).float()
        out_mask = (dist >= radius).float()

        # Single Gaussian
        normal_std = 0.3
        density = (-0.5 * ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) / normal_std ** 2).exp()
        density = density / (2 * math.pi * normal_std ** 2)

        r = (density * 2.5 + 6.5) * in_mask + 0.1 * out_mask
        return r

    @staticmethod
    def circle2(xy, height, **kwargs):
        """Two Gaussians inside a circle, centered at [h*0.5, h*sqrt(3)/2]."""
        xy = xy / (height - 1) * 2 - 1
        x_1, x_2 = xy[..., 0], xy[..., 1]
        radius, h = 0.6, 0.3
        center = [h * 0.5, h * math.sqrt(3) / 2.0]

        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5
        in_mask = (dist < radius).float()
        out_mask = (dist >= radius).float()

        # Two Gaussians
        hm, normal_std = 0.32, 0.21
        offsets = [[-hm * math.sqrt(3) / 2, 0.5 * hm], [hm * math.sqrt(3) / 2, -0.5 * hm]]
        mixture_density = 0
        for ox, oy in offsets:
            nc = [center[0] + ox, center[1] + oy]
            density = (-0.5 * ((x_1 - nc[0]) ** 2 + (x_2 - nc[1]) ** 2) / normal_std ** 2).exp()
            mixture_density = mixture_density + density / (2 * math.pi * normal_std ** 2)
        mixture_density = mixture_density / len(offsets)

        r = (mixture_density * 3.5 + 10.5) * in_mask + 0.1 * out_mask
        return r

    @staticmethod
    def circle3(xy, height, **kwargs):
        """Three Gaussians inside a circle, centered at [h*0.5, -h*sqrt(3)/2]."""
        xy = xy / (height - 1) * 2 - 1
        x_1, x_2 = xy[..., 0], xy[..., 1]
        radius, h = 0.6, 0.3
        center = [0.5 * h, -math.sqrt(3) / 2.0 * h]

        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5
        in_mask = (dist < radius).float()
        out_mask = (dist >= radius).float()

        # Three Gaussians
        hm, normal_std = 0.32, 0.18
        offsets = [[-hm, 0.0], [0.5 * hm, math.sqrt(3) / 2 * hm], [0.5 * hm, -math.sqrt(3) / 2 * hm]]
        mixture_density = 0
        for ox, oy in offsets:
            nc = [center[0] + ox, center[1] + oy]
            density = (-0.5 * ((x_1 - nc[0]) ** 2 + (x_2 - nc[1]) ** 2) / normal_std ** 2).exp()
            mixture_density = mixture_density + density / (2 * math.pi * normal_std ** 2)
        mixture_density = mixture_density / len(offsets)

        r = (mixture_density * 2.5 + 5.5) * in_mask + 0.1 * out_mask
        return r

    @staticmethod
    def mix2(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.diagonal(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix_shubert_currin(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.currin(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix_shubert_sphere(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.sphere(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix_shubert_branin(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.branin(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix_diagonal_sphere(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.diagonal(x, height),
            RewardFns.sphere(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix3(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.diagonal(x, height),
            RewardFns.currin(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix4(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.diagonal(x, height),
            RewardFns.currin(x, height),
            RewardFns.sphere(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)

    @staticmethod
    def mix5(x, height, **kwargs):
        w_ratio = kwargs.get("w_ratio")
        dist = torch.stack((
            RewardFns.shubert(x, height),
            RewardFns.diagonal(x, height),
            RewardFns.currin(x, height),
            RewardFns.sphere(x, height),
            RewardFns.branin(x, height),
        ), dim=-1)
        return torch.sum(torch.mul(dist, w_ratio.to(dist.device)), dim=-1)


# === Factory functions for compositional rewards ===

def _normalize(p):
    """Normalize a distribution to sum to 1."""
    return p / p.sum()


def _harmonic_mean_2way(fn1, fn2):
    """Factory: p1*p2 / (p1 + p2) with normalized p1, p2."""
    def fn(x, height, **kwargs):
        p1 = _normalize(fn1(x, height))
        p2 = _normalize(fn2(x, height))
        return (p1 * p2) / (p1 + p2)
    return fn


def _contrast_2way(target_fn, other_fn):
    """Factory: target^2 / (target + other) with normalized inputs."""
    def fn(x, height, **kwargs):
        p1 = _normalize(target_fn(x, height))
        p2 = _normalize(other_fn(x, height))
        return (p1 ** 2) / (p1 + p2)
    return fn


def _harmonic_mean_3way(fn1, fn2, fn3):
    """Factory: p1*p2*p3 / (p1*p2 + p1*p3 + p2*p3) with normalized inputs."""
    def fn(x, height, **kwargs):
        p1 = _normalize(fn1(x, height))
        p2 = _normalize(fn2(x, height))
        p3 = _normalize(fn3(x, height))
        return (p1 * p2 * p3) / (p1 * p2 + p1 * p3 + p2 * p3)
    return fn


def _contrast_3way(target_fn, other1_fn, other2_fn):
    """Factory: target^4 / ((target+other1) * (target^2 + target*other2 + other1*other2))."""
    def fn(x, height, **kwargs):
        p1 = _normalize(target_fn(x, height))
        p2 = _normalize(other1_fn(x, height))
        p3 = _normalize(other2_fn(x, height))
        numerator = p1 ** 4
        denom1 = p1 + p2
        denom2 = p1 ** 2 + p1 * p3 + p2 * p3
        return numerator / (denom1 * denom2)
    return fn


# Shorthand references for base functions
_shu = RewardFns.shubert
_cur = RewardFns.currin
_sph = RewardFns.sphere
_diag = RewardFns.diagonal
_bra = RewardFns.branin
_c1 = RewardFns.circle1
_c2 = RewardFns.circle2
_c3 = RewardFns.circle3

RewardFns.registry = {
    # Base functions
    "shubert": _shu,
    "currin": _cur,
    "sphere": _sph,
    "diagonal": _diag,
    "branin": _bra,
    "circle1": _c1,
    "circle2": _c2,
    "circle3": _c3,

    # Mix functions
    "mix2": RewardFns.mix2,
    "mix3": RewardFns.mix3,
    "mix4": RewardFns.mix4,
    "mix5": RewardFns.mix5,
    "mix_shubert_currin": RewardFns.mix_shubert_currin,
    "mix_shubert_sphere": RewardFns.mix_shubert_sphere,
    "mix_shubert_branin": RewardFns.mix_shubert_branin,
    "mix_diagonal_sphere": RewardFns.mix_diagonal_sphere,

    # 2-way harmonic means
    "harmonic_mean_circle12": _harmonic_mean_2way(_c1, _c2),
    "harmonic_mean_circle23": _harmonic_mean_2way(_c2, _c3),
    "harmonic_mean_circle31": _harmonic_mean_2way(_c3, _c1),
    "harmonic_mean_shu_diag": _harmonic_mean_2way(_shu, _diag),
    "harmonic_mean_shu_sph": _harmonic_mean_2way(_shu, _sph),
    "harmonic_mean_branin_sph": _harmonic_mean_2way(_bra, _sph),

    # 2-way contrasts
    "contrast_circle12": _contrast_2way(_c1, _c2),
    "contrast_circle21": _contrast_2way(_c2, _c1),
    "contrast_circle23": _contrast_2way(_c2, _c3),
    "contrast_circle32": _contrast_2way(_c3, _c2),
    "contrast_circle13": _contrast_2way(_c1, _c3),
    "contrast_circle31": _contrast_2way(_c3, _c1),
    "contrast_shu_diag": _contrast_2way(_shu, _diag),
    "contrast_diag_shu": _contrast_2way(_diag, _shu),
    "contrast_shu_sph": _contrast_2way(_shu, _sph),
    "contrast_sph_shu": _contrast_2way(_sph, _shu),
    "contrast_branin_sph": _contrast_2way(_bra, _sph),
    "contrast_sph_branin": _contrast_2way(_sph, _bra),

    # 3-way harmonic mean
    "harmonic_mean_circle123": _harmonic_mean_3way(_c1, _c2, _c3),

    # 3-way contrasts
    "contrast_circle123": _contrast_3way(_c1, _c2, _c3),
    "contrast_circle213": _contrast_3way(_c2, _c1, _c3),
    "contrast_circle321": _contrast_3way(_c3, _c2, _c1),
}


def get_reward_fn(name, beta=1.0):
    """Get a reward function from the registry, optionally sharpened by beta.

    When beta > 1, the reward distribution becomes sharper (more peaked).
    When beta < 1, the reward distribution becomes smoother (more uniform).
    When beta == 1, the original reward function is returned unchanged.

    Args:
        name: Key in RewardFns.registry.
        beta: Exponent applied as r^beta. Default 1.0 (no sharpening).

    Returns:
        A reward function with signature fn(x, height, **kwargs) -> Tensor.
    """
    base_fn = RewardFns.registry[name]
    if beta == 1.0:
        return base_fn

    def sharpened_fn(x, height, **kwargs):
        r = base_fn(x, height, **kwargs)
        return r ** beta

    return sharpened_fn
