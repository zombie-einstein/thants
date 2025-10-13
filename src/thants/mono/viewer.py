from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from jumanji.viewer import MatplotlibViewer
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from thants.common.types import ColorScheme
from thants.common.utils import format_plot, get_color_scheme
from thants.mono.types import State


def _draw_env(
    state: State, colors: ColorScheme
) -> tuple[chex.Array, chex.Array, chex.Array]:
    terrain = state.terrain.astype(int)
    terrain = colors.terrain.at[terrain].get()
    nest_colors = jnp.clip(colors.ants + 0.1, 0.0, 1.0)
    nest = state.colony.nest[:, :, jnp.newaxis] * nest_colors[jnp.newaxis]
    food = jnp.full((*state.food.shape, 4), colors.food)
    return terrain, nest, food


@jax.jit
def _draw_ants(state: State, colors: ColorScheme) -> chex.Array:
    dims = state.food.shape
    ants = jnp.zeros((*dims, 4))
    ants = ants.at[state.colony.ants.pos[:, 0], state.colony.ants.pos[:, 1]].set(
        colors.ants[0]
    )
    return ants


class ThantsViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "thants",
        render_mode: str = "human",
        color_sequence: str = "tab20",
    ) -> None:
        """
        Thants environment visualiser using Matplotlib

        Parameters
        ----------
        name
            Plot name, default ``thants``
        render_mode
            Default ``human``
        color_sequence
            Matplotlib colour sequence to sample from
        """
        self.color_sequence = color_sequence
        super().__init__(name, render_mode)

    def _set_figure_size(self, dims: tuple[int, int]) -> None:
        longest = max(dims[0], dims[1])
        f_dims = (10.0 * dims[1] / longest, 10.0 * dims[0] / longest)
        self.figure_size = f_dims

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
        dims = state.food.shape
        self._set_figure_size(dims)
        fig, ax = self._get_fig_ax(padding=0.01)
        fig, ax = format_plot(fig, ax, dims)
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

        env_dims = states[0].terrain.shape
        self._set_figure_size(env_dims)
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False, padding=0.01)
        fig, ax = format_plot(fig, ax, env_dims)
        plt.close(fig=fig)

        colors, ants_img, food_img = self._draw(ax, states[0])

        def make_frame(state: State) -> tuple[AxesImage, AxesImage]:
            step_ants = _draw_ants(state, colors)
            ants_img.set_data(step_ants)
            food_img.set_alpha(jnp.clip(state.food, 0.0, 1.0))
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

    def _draw(
        self, ax: plt.Axes, state: State
    ) -> tuple[ColorScheme, AxesImage, AxesImage]:
        ax.clear()
        colors = get_color_scheme(self.color_sequence, 1)

        terrain, nest, food = _draw_env(state, colors)
        ants = _draw_ants(state, colors)

        ax.imshow(terrain)
        ax.imshow(nest)

        food_img = ax.imshow(food, alpha=state.food)
        ants_img = ax.imshow(ants)

        return colors, ants_img, food_img

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
