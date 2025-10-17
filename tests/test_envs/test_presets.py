from typing import Type

import chex
import jax
import pytest

from thants.envs import ThantsDual, ThantsQuad
from thants.types import Observations


@pytest.mark.parametrize(
    "env_type, dims",
    [
        (ThantsDual, (20, 10)),
        (ThantsQuad, (20, 20)),
    ],
)
def test_env_runs(
    key: chex.PRNGKey,
    env_type: Type[ThantsDual | ThantsQuad],
    dims: tuple[int, int],
) -> None:
    n_steps = 10
    n_agents = 9
    n_signals = 2

    env = env_type(dims=dims, n_agents=n_agents, nest_dims=(3, 3), n_signals=n_signals)

    env.max_steps = 100
    state, _ = env.reset(key)

    def step(_state, _):
        k = jax.random.split(_state.key, env.num_colonies)
        actions = [jax.random.choice(_k, 9, (n,)) for _k, n in zip(k, env.num_agents)]
        _state, timesteps = env.step(_state, actions)
        return _state, [t.observation for t in timesteps]

    final_state, obs = jax.lax.scan(step, state, None, length=n_steps)

    assert isinstance(obs, list)
    assert all([isinstance(x, Observations)] for x in obs)

    for o in obs:
        assert o.food.shape == (n_steps, n_agents, 9)
        assert o.signals.shape == (n_steps, n_agents, n_signals, 9)
        assert o.terrain.shape == (n_steps, n_agents, 9)
        assert o.ants.shape == (n_steps, n_agents, env.num_colonies, 9)
