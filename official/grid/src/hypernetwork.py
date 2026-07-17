"""HyperNetwork-based GFlowNet (HN-GFN) implementation for HyperGrid.

This module implements HN-GFN following the original paper's architecture:
- Fixed encoder: Shared MLP that encodes states (trained via backprop)
- HyperNetwork: Generates multi-layer predictor head weights from preference vectors

Architecture matches original HN-GFN:
- Ray MLP: 3-layer MLP for preference embedding
- Weight generators for multiple layers (not just output layer)
- LeakyReLU between generated layers
- Optional logit clipping for numerical stability
"""

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gfn.preprocessors import Preprocessor
from gfn.states import DiscreteStates, States
from gfn.utils.distributions import UnsqueezedCategorical
from gfn.estimators import PolicyMixin, LogitBasedEstimator, ConditionalScalarEstimator
from gfn.utils.modules import MLP


class HyperNetwork(nn.Module):
    """HyperNetwork that generates multi-layer predictor head weights from preference vectors.

    Following the original HN-GFN paper, this generates weights for multiple layers
    (not just the output layer) to increase model capacity.

    Architecture:
        ray_mlp: preference (n_objectives,) -> hidden (ray_hidden_dim,) [3-layer MLP]
        weight_generators: hidden -> weights/biases for each layer in predictor head

    Args:
        n_objectives: Number of objectives (preference vector dimension)
        ray_hidden_dim: Hidden dimension for the ray MLP
        encoder_output_dim: Output dimension from the fixed encoder (input to generated layers)
        n_actions: Final output dimension (number of actions)
        predictor_layers: List of hidden dimensions for the generated predictor head
                         e.g., [64, 64] means: encoder_output_dim -> 64 -> 64 -> n_actions
        logit_clipping: If > 0, apply tanh-based logit clipping for numerical stability
    """

    def __init__(
        self,
        n_objectives: int,
        ray_hidden_dim: int,
        encoder_output_dim: int,
        n_actions: int,
        predictor_layers: List[int] = None,
        logit_clipping: float = 0.0,
    ):
        super().__init__()
        self.n_objectives = n_objectives
        self.ray_hidden_dim = ray_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.n_actions = n_actions
        self.logit_clipping = logit_clipping

        # Build layer dimensions: [encoder_output_dim, *predictor_layers, n_actions]
        if predictor_layers is None:
            predictor_layers = [encoder_output_dim]  # Default: one hidden layer
        self.layer_dims = [encoder_output_dim] + predictor_layers + [n_actions]
        self.n_layers = len(self.layer_dims) - 1

        # Ray MLP: preference -> hidden features (3-layer MLP as in HN-GFN paper)
        self.ray_mlp = nn.Sequential(
            nn.Linear(n_objectives, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        # Weight generators for each layer in the predictor head
        self.w_gens = nn.ModuleList()
        self.b_gens = nn.ModuleList()
        for i in range(self.n_layers):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            self.w_gens.append(nn.Linear(ray_hidden_dim, in_dim * out_dim))
            self.b_gens.append(nn.Linear(ray_hidden_dim, out_dim))

    def forward(self, preference: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictor head weights from preference vector(s).

        Args:
            preference: Preference vector of shape [batch_size, n_objectives] or [n_objectives]

        Returns:
            Dictionary with keys "fc_{i}_weights" and "fc_{i}_bias" for each layer i.
            weights shape: [batch_size, out_dim, in_dim]
            bias shape: [batch_size, out_dim]
        """
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)  # [1, n_objectives]

        # Ray MLP: preference -> hidden features
        h = self.ray_mlp(preference)  # [batch_size, ray_hidden_dim]

        # Generate weights for each layer (batched)
        batch_size = h.shape[0]
        weights_dict = {}
        for i in range(self.n_layers):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            w = self.w_gens[i](h).view(batch_size, out_dim, in_dim)
            b = self.b_gens[i](h).view(batch_size, out_dim)
            weights_dict[f"fc_{i}_weights"] = w
            weights_dict[f"fc_{i}_bias"] = b

        return weights_dict

    def apply_predictor(self, x: torch.Tensor, weights_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply the generated predictor head to input features.

        Args:
            x: Input features of shape [n_states, encoder_output_dim]
            weights_dict: Dictionary of weights from forward().
                weights shape: [batch_size, out_dim, in_dim] where batch_size
                is either 1 (shared) or n_states (per-state).

        Returns:
            Output logits of shape [n_states, n_actions]
        """
        for i in range(self.n_layers):
            w = weights_dict[f"fc_{i}_weights"]
            b = weights_dict[f"fc_{i}_bias"]
            if w.shape[0] == 1:
                # Shared weights: single preference for all states
                x = F.linear(x, w.squeeze(0), b.squeeze(0))
            else:
                # Per-state weights: batched matmul
                x = torch.einsum('bi,boi->bo', x, w) + b
            # Apply LeakyReLU between layers (not after the last layer)
            if i < self.n_layers - 1:
                x = F.leaky_relu(x)

        # Apply logit clipping if enabled
        if self.logit_clipping > 0:
            x = self.logit_clipping * torch.tanh(x)

        return x


class HNDiscretePolicyEstimator(PolicyMixin, nn.Module):
    """HyperNetwork-based Policy Estimator for discrete action spaces.

    Architecture (following original HN-GFN):
        1. Fixed Encoder: state -> state_embedding (trained via backprop)
        2. HyperNetwork: preference -> multi-layer predictor head weights
        3. Generated Predictor Head: state_embedding -> action logits (with LeakyReLU)

    Args:
        encoder: Fixed MLP encoder for state embedding
        hypernetwork: HyperNetwork instance for predictor head weight generation
        n_actions: Number of possible actions
        preprocessor: Preprocessor for converting states to tensor representations
        is_backward: Whether this is a backward policy (for pb)
    """

    def __init__(
        self,
        encoder: nn.Module,
        hypernetwork: HyperNetwork,
        n_actions: int,
        preprocessor: Preprocessor,
        is_backward: bool = False,
    ):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.hypernetwork = hypernetwork
        self.n_actions = n_actions
        self.preprocessor = preprocessor
        self.is_backward = is_backward

    def forward(
        self,
        states: States,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action logits for given states and preference.

        Args:
            states: Batch of states
            conditioning: Preference vector [n_states, n_objectives] (per-state preference)
                         or [1, n_objectives] / [n_objectives] (shared preference)

        Returns:
            Action logits of shape [n_states, n_actions]
        """
        # Preprocess states and convert to float
        x = self.preprocessor(states).float()  # [n_states, state_dim]

        # Fixed encoder: state -> embedding
        x = self.encoder(x)  # [n_states, hidden_dim]

        # Generate predictor head weights from hypernetwork
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)  # [1, n_objectives]
        weights_dict = self.hypernetwork(conditioning)

        # Apply generated predictor head (with LeakyReLU and optional logit clipping)
        logits = self.hypernetwork.apply_predictor(x, weights_dict)

        return logits

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: torch.Tensor,
        sf_bias: float = 0.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> UnsqueezedCategorical:
        """Convert module output to a probability distribution over actions.

        Uses LogitBasedEstimator's static method for numerical stability.

        Args:
            states: Batch of states
            module_output: Raw logits from forward pass
            sf_bias: Bias added to the exit action logit
            temperature: Temperature for softmax
            epsilon: Exploration probability for epsilon-greedy

        Returns:
            Categorical distribution over actions
        """
        masks = states.backward_masks if self.is_backward else states.forward_masks

        logits = LogitBasedEstimator._compute_logits_for_distribution(
            module_output,
            masks,
            sf_index=-1,
            sf_bias=sf_bias,
            temperature=temperature,
            epsilon=epsilon,
        )

        return UnsqueezedCategorical(logits=logits)

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module."""
        if self.is_backward:
            return self.n_actions - 1
        return self.n_actions


class HNScalarEstimator(ConditionalScalarEstimator):
    """HyperNetwork-based Scalar Estimator for logF estimation.

    Inherits from ConditionalScalarEstimator for compatibility with SubTBGFlowNet.

    Architecture (following original HN-GFN):
        1. Fixed Encoder: state -> state_embedding (trained via backprop)
        2. HyperNetwork: preference -> multi-layer predictor head weights
        3. Generated Predictor Head: state_embedding -> scalar (logF)

    Args:
        encoder: Fixed MLP encoder for state embedding
        hypernetwork: HyperNetwork instance for predictor head weight generation
        preprocessor: Preprocessor for converting states to tensor representations
    """

    def __init__(
        self,
        encoder: nn.Module,
        hypernetwork: HyperNetwork,
        preprocessor: Preprocessor,
    ):
        # Bypass parent __init__ and set required attributes directly
        nn.Module.__init__(self)
        self.encoder = encoder
        self.hypernetwork = hypernetwork
        self.preprocessor = preprocessor
        self.is_backward = False

    def forward(
        self,
        states: States,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar output (logF) for given states and preference.

        Args:
            states: Batch of states
            conditioning: Preference vector [n_states, n_objectives] (per-state preference)
                         or [1, n_objectives] / [n_objectives] (shared preference)

        Returns:
            Scalar outputs of shape [n_states, 1]
        """
        # Preprocess states and convert to float
        x = self.preprocessor(states).float()  # [n_states, state_dim]

        # Fixed encoder: state -> embedding
        x = self.encoder(x)  # [n_states, hidden_dim]

        # Generate predictor head weights from hypernetwork
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)
        weights_dict = self.hypernetwork(conditioning)

        # Apply generated predictor head (with LeakyReLU and optional logit clipping)
        output = self.hypernetwork.apply_predictor(x, weights_dict)

        return output

    @property
    def expected_output_dim(self) -> int:
        """Expected output dimension of the module."""
        return 1
