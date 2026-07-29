"""Actor-critic network for the PPO agent.

A small shared MLP trunk feeds two heads: a policy head producing one logit per
discrete action in :data:`Game.ACTIONS`, and a value head estimating the state
value. Illegal actions are masked out (their logits driven to ``MASK_FILL``)
before forming the categorical distribution, so the policy can only ever assign
probability to moves the game will accept.
"""
from __future__ import annotations
import math

import torch
from torch import nn
from torch.distributions import Categorical

# Large negative logit for illegal actions: e^MASK_FILL ≈ 0 probability without
# the NaNs that -inf would produce if a whole row were masked.
MASK_FILL = -1e8


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2),
                bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight init — the standard PPO initialisation."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk policy/value network with action masking."""

    def __init__(self, obs_size: int, num_actions: int,
                 hidden: int = 64, value_dim: int = 1) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden = hidden
        # value_dim=1 is the usual single state-value head. value_dim=2 is the
        # AlphaZero "twohead" value target (src/ai/alphazero.py): head 0 is the
        # dense shaped return, head 1 is P(win) — kept separate so the win
        # signal never spikes the shaped estimate near a terminal (the failure
        # of the single-head "blend" mode).
        self.value_dim = value_dim
        self.net_type = "mlp"
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_size, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        # Small policy-head gain keeps the initial policy near-uniform; unit
        # gain on the value head is the usual choice.
        self.policy_head = _layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden, value_dim), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)`` for a batch of observations.

        ``value`` is squeezed to a scalar per state when ``value_dim == 1``
        (the single-head case, unchanged); for ``value_dim > 1`` it keeps its
        trailing head axis (``[..., value_dim]``).
        """
        h = self.trunk(obs)
        value = self.value_head(h)
        if self.value_dim == 1:
            value = value.squeeze(-1)
        return self.policy_head(h), value

    def _dist(self, obs: torch.Tensor, mask: torch.Tensor
              ) -> tuple[Categorical, torch.Tensor]:
        """Masked action distribution and value for ``obs``.

        ``mask`` is a boolean tensor (``True`` = legal). Illegal logits are set
        to :data:`MASK_FILL` before building the categorical.
        """
        logits, value = self(obs)
        logits = logits.masked_fill(~mask, MASK_FILL)
        return Categorical(logits=logits), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State-value estimate(s) for ``obs`` (no policy computation)."""
        return self(obs)[1]

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score) an action and return
        ``(action, log_prob, entropy, value)``.

        When ``action`` is given its log-prob is evaluated instead of sampling
        — used during the PPO update to score the stored actions under the
        current policy.
        """
        dist, value = self._dist(obs, mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class ConvActorCritic(ActorCritic):
    """CNN policy/value net for the ``OBS_MODE == "grid"`` observation.

    The observation is a flat vector = ``GRID_CHANNELS`` board planes (``H*W``
    each) + a scalar tail; this net reshapes the planes to ``(C, H, W)``, runs a
    few full-resolution convolutions (spatial structure the flat MLP lacked),
    reduces to 2 channels, flattens, concatenates the scalar tail, and feeds the
    same two-head trunk. Reuses :class:`ActorCritic`'s masking / action helpers
    (``_dist`` / ``get_action_and_value`` / ``get_value``) by only overriding
    ``__init__`` and ``forward``.
    """

    def __init__(self, obs_size: int, num_actions: int,
                 hidden: int = 64, value_dim: int = 1) -> None:
        nn.Module.__init__(self)  # skip ActorCritic's MLP-trunk construction
        from .. import constant
        self.C = constant.GRID_CHANNELS
        self.H = constant.GRID_H
        self.W = constant.GRID_W
        self.num_scalars = constant.GRID_SCALARS
        self.grid_len = self.C * self.H * self.W
        if obs_size != self.grid_len + self.num_scalars:
            raise ValueError(
                f"obs_size {obs_size} != grid {self.grid_len} + scalars "
                f"{self.num_scalars}; OBS_MODE/GRID_* out of sync with the net")
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden = hidden
        self.value_dim = value_dim
        self.net_type = "cnn"
        self.conv = nn.Sequential(
            nn.Conv2d(self.C, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1), nn.ReLU(),  # channel reduce, keep HxW
        )
        conv_out = 2 * self.H * self.W
        self.fc = nn.Sequential(
            _layer_init(nn.Linear(conv_out + self.num_scalars, hidden)), nn.ReLU(),
            _layer_init(nn.Linear(hidden, hidden)), nn.ReLU(),
        )
        self.policy_head = _layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden, value_dim), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        grid = obs[:, :self.grid_len].reshape(-1, self.C, self.H, self.W)
        scalars = obs[:, self.grid_len:]
        h = self.conv(grid).flatten(1)
        h = self.fc(torch.cat([h, scalars], dim=1))
        logits = self.policy_head(h)
        value = self.value_head(h)
        if self.value_dim == 1:
            value = value.squeeze(-1)
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        return logits, value


def build_net(obs_size: int, num_actions: int, hidden: int = 64,
              value_dim: int = 1, net_type: str = "mlp") -> ActorCritic:
    """Construct the policy/value net of the requested ``net_type``.

    ``"mlp"`` (default) is the flat-observation :class:`ActorCritic` used by
    ``az.pt`` and deployment; ``"cnn"`` is the grid-observation
    :class:`ConvActorCritic`. Both save/load and the trainer route through here.
    """
    if net_type == "cnn":
        return ConvActorCritic(obs_size, num_actions, hidden, value_dim)
    return ActorCritic(obs_size, num_actions, hidden, value_dim)
