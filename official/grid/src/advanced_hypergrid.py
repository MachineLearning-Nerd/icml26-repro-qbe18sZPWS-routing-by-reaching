"""Advanced HyperGrid environment with custom reward functions and conditioning."""

import torch

from gfn.states import DiscreteStates
from gfn.gym import HyperGrid, ConditionalHyperGrid

from .rewards import RewardFns, get_reward_fn


class AdvancedHyperGrid(HyperGrid):
    """Extended HyperGrid environment supporting custom rewards and multi-objective conditioning.

    This class extends the base HyperGrid to support:
    - Custom reward functions (shubert, currin, branin, etc.)
    - Multi-objective reward mixing with preference weights
    - Conditioning for MO-GFN and HN-GFN training

    Args:
        ndim: Number of dimensions (default: 2)
        height: Grid height per dimension
        n_reward_fns: Number of reward functions for multi-objective
        device: Torch device ('cpu' or 'cuda')
        custom_dist: Name of reward function from RewardFns.registry
            (e.g., 'shubert', 'mix2', 'harmonic_mean')
        w_ratio: Fixed preference weights [n_reward_fns] (optional)
        min_reward: Minimum reward value to avoid numerical issues
    """

    def __init__(
        self,
        ndim,
        height,
        n_reward_fns=1,
        device='cpu',
        custom_dist=None,
        w_ratio=None,
        min_reward=1e-12,
        beta=1.0,
    ):
        self.custom_dist = custom_dist
        self.w_ratio = w_ratio
        self.beta = beta
        # for conditioning
        self.conditioning = None
        self.train_status = None
        self.n_reward_fns = n_reward_fns
        self.min_reward = min_reward
        super().__init__(
            ndim=ndim,
            height=height,
            device=device,
            calculate_partition=True,
            store_all_states=True,
        )

    def reward(self, final_states: DiscreteStates | torch.Tensor) -> torch.Tensor:
        if not self.custom_dist:
            return super().reward(final_states)

        assert isinstance(
            final_states, DiscreteStates | torch.Tensor
        ), f"final_states is {type(final_states)}"
        if isinstance(final_states, DiscreteStates):
            final_states_raw = final_states.tensor
        else:
            final_states_raw = final_states

        if self.conditioning is not None:
            w_tensor = self.conditioning[:, :self.n_reward_fns]
        elif self.w_ratio is not None:
            w_tensor = self.w_ratio.expand(final_states_raw.shape[0], -1)
        else:
            w_tensor = None

        reward_fn = get_reward_fn(self.custom_dist, beta=self.beta)
        reward = reward_fn(
            final_states_raw,
            height=self.height,
            w_ratio=w_tensor,
        )

        reward = torch.clamp(reward, min=self.min_reward)

        if isinstance(final_states, DiscreteStates):
            assert (
                reward.shape == final_states.batch_shape
            ), f"reward.shape is {reward.shape} and final_states.batch_shape is {final_states.batch_shape}"
        else:
            n_dims = len(reward.shape)
            assert (
                reward.shape == final_states.shape[:n_dims]
            ), f"reward.shape is {reward.shape} and final_states.shape is {final_states.shape}"
        return reward

    def _compute_rewards(self):
        """Compute rewards for all states. Used by true_dist and get_unormalized_true_dist."""
        assert self.all_states is not None
        assert torch.all(
            self.get_states_indices(self.all_states)
            == torch.arange(self.n_states, device=self.device)
        )
        return self.reward(self.all_states)

    @property
    def true_dist(self) -> torch.Tensor | None:
        """Returns the pmf over all states in the hypergrid."""
        if self.all_states is not None:
            self._true_dist = self._compute_rewards()
            self._true_dist /= self._true_dist.sum()
        return self._true_dist

    def get_unormalized_true_dist(self):
        if self.all_states is not None:
            self._true_dist = self._compute_rewards()
        return self._true_dist

    def set_conditioning(self, conditioning: torch.Tensor):
        """Set the conditioning for the environment."""
        self.conditioning = conditioning

    def set_train_status(self, train_status):
        self.train_status = train_status


class AdvancedConditionalHyperGrid(ConditionalHyperGrid):
    """Conditional HyperGrid for multi-objective GFlowNets (MO-GFN, HN-GFN).

    Conditions (preference vectors) are stored on States objects via
    `states.conditions` and sampled from a Dirichlet distribution automatically
    by `env.sample_conditions(batch_shape)`. `reward(states)` reads from
    `states.conditions`, so it Just Works with replay buffers (each trajectory
    carries its own condition regardless of env state).

    Args:
        ndim: Number of dimensions (default: 2)
        height: Grid height per dimension
        n_reward_fns: Number of reward functions (= condition dimension)
        device: Torch device ('cpu' or 'cuda')
        custom_dist: Name of reward function from RewardFns.registry
            (e.g., 'mix2', 'mix3')
        dirichlet_alpha: Concentration parameter for Dirichlet sampling
        min_reward: Minimum reward value to avoid numerical issues
        beta: Reward sharpening exponent
    """

    condition_dim: int

    def __init__(
        self,
        ndim=2,
        height=32,
        n_reward_fns=2,
        device='cpu',
        custom_dist=None,
        min_reward=1e-12,
        beta=1.0,
        dirichlet_alpha=1.0,
    ):
        self.custom_dist = custom_dist
        self.n_reward_fns = n_reward_fns
        self.beta = beta
        self.min_reward = min_reward
        self.dirichlet_alpha = dirichlet_alpha
        self.condition_dim = n_reward_fns  # required by ConditionalHyperGrid
        super().__init__(
            ndim=ndim,
            height=height,
            device=device,
            calculate_partition=True,
            store_all_states=True,
        )

    def sample_conditions(self, batch_shape):
        """Sample preference vectors from Dirichlet distribution."""
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        alpha = torch.full(
            batch_shape + (self.n_reward_fns,),
            self.dirichlet_alpha,
            device=self.device,
        )
        return torch.distributions.Dirichlet(alpha).sample()

    def reward(self, states: DiscreteStates) -> torch.Tensor:
        """R(x) computed from states.conditions (per-state preference weights)."""
        assert isinstance(states, DiscreteStates)
        assert states.conditions is not None, (
            "states.conditions required. Use gflownet.sample_trajectories(env, n=...) "
            "or attach conditions manually."
        )
        reward_fn = get_reward_fn(self.custom_dist, beta=self.beta)
        reward = reward_fn(
            states.tensor,
            height=self.height,
            w_ratio=states.conditions,
        )
        return torch.clamp(reward, min=self.min_reward)

    def true_dist(self, condition: torch.Tensor) -> torch.Tensor:
        """Normalized reward distribution for a given condition vector. Cached."""
        cache_key = tuple(condition.flatten().cpu().tolist())
        if not hasattr(self, '_true_dist_cache'):
            self._true_dist_cache = {}
        if cache_key not in self._true_dist_cache:
            states = self.all_states.clone()
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            states.conditions = condition.expand(self.n_states, -1)
            rewards = self.reward(states)
            self._true_dist_cache[cache_key] = rewards / rewards.sum()
        return self._true_dist_cache[cache_key]
