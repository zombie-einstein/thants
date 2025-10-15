from typing import Optional

import jax.numpy as jnp
from jumanji import specs

from thants.types import Observations


def get_observation_spec(
    num_agents: int,
    num_signals: int,
    carry_capacity: float,
    num_colonies: Optional[int] = None,
) -> specs.Spec[Observations]:
    """
    Get observation spec for a given colony

    Parameters
    ----------
    num_agents
        Number of ants in the colony
    num_signals
        Number of signal channels
    carry_capacity
        Ant carrying capacity
    num_colonies
        Optional in the case of a multi-colony environment

    Returns
    -------
    Spec
        Observation specification
    """
    ants = (
        specs.BoundedArray(
            shape=(num_agents, num_colonies, 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="ants",
        )
        if num_colonies
        else specs.BoundedArray(
            shape=(num_agents, 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="ants",
        )
    )
    food = specs.BoundedArray(
        shape=(num_agents, 9),
        minimum=0.0,
        maximum=jnp.inf,
        dtype=float,
        name="food",
    )
    signals = specs.BoundedArray(
        shape=(num_agents, num_signals, 9),
        minimum=0.0,
        maximum=jnp.inf,
        dtype=float,
        name="signals",
    )
    nest = specs.BoundedArray(
        shape=(num_agents, 9),
        minimum=0.0,
        maximum=1.0,
        dtype=float,
        name="nest",
    )
    terrain = specs.BoundedArray(
        shape=(num_agents, 9),
        minimum=0.0,
        maximum=1.0,
        dtype=float,
        name="terrain",
    )
    carrying = specs.BoundedArray(
        shape=(num_agents,),
        minimum=0.0,
        maximum=carry_capacity,
        dtype=float,
        name="carrying",
    )
    return specs.Spec(
        Observations,
        "ObservationSpec",
        ants=ants,
        food=food,
        signals=signals,
        nest=nest,
        carrying=carrying,
        terrain=terrain,
    )


def get_action_spec(num_agents: int, num_signals: int) -> specs.BoundedArray:
    """
    Get action specification for a colony

    Parameters
    ----------
    num_agents
        Number of ants in the colony
    num_signals
        Number of colony signal channels

    Returns
    -------
    Spec
        Action specification
    """
    return specs.BoundedArray(
        shape=(num_agents,),
        minimum=0,
        maximum=7 + num_signals,
        dtype=int,
    )


def get_reward_spec(num_agents: int) -> specs.Array:
    """
    Get reward specification for a colony

    Parameters
    ----------
    num_agents
        Number of agents in the colony

    Returns
    -------
    Spec
        Reward specification
    """
    return specs.Array(shape=(num_agents,), dtype=float)
