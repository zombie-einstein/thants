from functools import partial
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from jumanji.viewer import MatplotlibViewer
from matplotlib import colormaps
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from thants.multi.types import State


def draw_env(state: State, colors: chex.Array) -> tuple[chex.Array, chex.Array]:
    terrain = state.terrain.astype(float)
    terrain = jnp.stack([terrain, terrain, terrain, jnp.ones_like(terrain)], axis=2)

    trans_colors = colors * jnp.array([[1.0, 1.0, 1.0, 0.5]])

    nests = [
        colony.nest[:, :, jnp.newaxis] * trans_colors[i, jnp.newaxis]
        for i, colony in enumerate(state.colonies)
    ]
    nests = jnp.sum(jnp.stack(nests, axis=0), axis=0)

    return terrain, nests


@partial(jax.jit, static_argnames="dims")
def draw_ants(
    dims: tuple[int, int], state: State, colors: chex.Array
) -> tuple[chex.Array, chex.Array]:
    ants = jnp.zeros((*dims, 4))

    for i, colony in enumerate(state.colonies):
        ants = ants.at[colony.ants.pos[:, 0], colony.ants.pos[:, 1]].set(colors[i])

    food = jnp.stack(
        [jnp.zeros(dims), jnp.ones(dims), jnp.zeros(dims), state.food], axis=2
    )

    return ants, food


class ThantsMultiColonyViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "thants",
        render_mode: str = "human",
        colony_colormap: str = "plasma",
    ) -> None:
        """
        Thants environment visualiser using Matplotlib

        Parameters
        ----------
        name
            Plot name, default ``thants``
        render_mode
            Default ``human``
        colony_colormap

        """
        self.cmap = colormaps[colony_colormap]
        super().__init__(name, render_mode)

    def _get_colony_colors(self, n: int) -> chex.Array:
        return jnp.array(self.cmap(np.linspace(0, 1, n)))

    def render(
        self, state: State, save_path: Optional[str] = None
    ) -> Optional[NDArray]:
        """Render a frame of the environment for a given state using matplotlib

        Parameters
        ----------
        state
            State object containing the current dynamics of the environment
        save_path
            Optional path to save the rendered environment image to

        Returns
        -------
        Array or None
            RGB array if the render_mode is ``rgb_array``
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self, states: Sequence[State], interval: int, save_path: Optional[str] = None
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states

        Parameters
        ----------
        states
            Sequence of ``State`` corresponding to subsequent timesteps
        interval
            Delay between frames in milliseconds, default to 200
        save_path
            The path where the animation file should be saved. If it is None,
            the plot will not be saved

        Returns
        -------
        FuncAnimation
            Animation object that can be saved as a GIF, MP4, or rendered with HTML
        """
        if not states:
            raise ValueError(f"The states argument has to be non-empty, got {states}.")

        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        dims = states[0].food.shape
        colors = self._get_colony_colors(len(states[0].colonies))

        terrain, nests = draw_env(states[0], colors)
        ants, food = draw_ants(dims, states[0], colors)

        ax.imshow(terrain, cmap="grey")
        ax.imshow(nests)
        food_img = ax.imshow(food)
        ants_img = ax.imshow(ants)

        def make_frame(state: State) -> tuple[AxesImage, AxesImage]:
            step_ants, step_food = draw_ants(dims, state, colors)
            ants_img.set_data(step_ants)
            food_img.set_data(step_food)
            return ants_img, food_img

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            blit=True,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        dims = state.food.shape
        colors = self._get_colony_colors(len(state.colonies))

        terrain, nests = draw_env(state, colors)
        ants, food = draw_ants(dims, state, colors)

        ax.imshow(terrain, cmap="grey")
        ax.imshow(nests)
        ax.imshow(food)
        ax.imshow(ants)

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: str,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = super()._get_fig_ax(
            name_suffix=name_suffix, show=show, padding=padding
        )
        return fig, ax
