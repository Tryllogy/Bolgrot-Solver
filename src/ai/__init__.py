"""PPO agent that learns to play the game headlessly through the
:class:`src.game.Game` environment API (``get_observation``/``legal_actions``/
``step``).

- ``network`` — the masked actor-critic (:class:`ActorCritic`).
- ``policy``  — action selection, evaluation and (de)serialisation glue.
- ``ppo``     — the PPO training loop (rollouts, GAE, clipped update).
- ``train``   — CLI entry point (``python -m src.ai.train``).
- ``play``    — watch a trained policy with the renderer
  (``python -m src.ai.play``).
- ``mcts`` / ``alphazero`` — the planning agents (MCTS, and AlphaZero PUCT
  self-play) that actually win the full game where PPO cannot.
- ``portfolio`` — the deployable agent cascade: combine agents that fail on
  disjoint seeds to reach ~100% at ~1.4x the cost of one agent
  (``python -m src.ai.portfolio``).

Nothing here contains game logic; it only drives ``Game`` through its
environment API. Requires the optional ``ai`` dependency group (``torch``).
"""
