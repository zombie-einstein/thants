from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from jumanji.viewer import MatplotlibViewer
from matplotlib import color_sequences
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from thants.types import ColorScheme, State


def format_plot(
    fig: plt.Figure, ax: plt.Axes, env_dims: tuple[float, float]
) -> tuple[plt.Figure, plt.Axes]:
    """
    Format an environment plot, remove ticks and bound to the environment dimensions.

    Parameters
    ----------
    fig
        Matplotlib figure
    ax
        Matplotlib axes
    env_dims
        Environment dimensions

    Returns
    -------
    tuple[Figure, Axes]
        Formatted figure and axes
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, env_dims[1] - 0.5)
    ax.set_ylim(-0.5, env_dims[0] - 0.5)

    return fig, ax


def get_color_scheme(color_sequence: str, n_colonies: int) -> ColorScheme:
    """
    Get a environment visualisation colour scheme from a matplotlib sequence

    Parameters
    ----------
    color_sequence
        Matplotlib color-sequence name
    n_colonies
        Number of colonies to visualise

    Returns
    -------
    ColorScheme
        Environment visualisation color-scheme
    """
    colors = color_sequences[color_sequence]
    colors = jnp.array([(*i, 1.0) for i in colors[: 3 + n_colonies]])
    return ColorScheme(terrain=colors[:2], food=colors[2], ants=colors[3:])


def _draw_env(
    state: State, colors: ColorScheme
) -> tuple[chex.Array, chex.Array, chex.Array]:
    terrain = state.terrain.astype(int)
    terrain = colors.terrain.at[terrain].get()
    nest_colors = jnp.clip(colors.ants + 0.1, 0.0, 1.0)
    empty_color = jnp.zeros((1, 4))
    nest_colors = jnp.concatenate([empty_color, nest_colors], axis=0)
    nests = nest_colors[state.colonies.nests]
    food = jnp.full((*state.food.shape, 4), colors.food)
    return terrain, nests, food


@jax.jit
def _draw_ants(state: State, colors: ColorScheme) -> chex.Array:
    dims = state.food.shape
    ants = jnp.zeros((*dims, 4))
    ants = ants.at[state.colonies.ants.pos[:, 0], state.colonies.ants.pos[:, 1]].set(
        colors.ants[state.colonies.colony_idx]
    )
    return ants


class ThantsMultiColonyViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "thants",
        render_mode: str = "human",
        color_sequence: str = "tab20",
    ) -> None:
        """
        Thants multi-colony environment visualiser using Matplotlib

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
        ax.clear()
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

        dims = states[0].food.shape
        self._set_figure_size(dims)

        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False, padding=0.01)
        fig, ax = format_plot(fig, ax, dims)
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
        colors = get_color_scheme(self.color_sequence, state.colonies.signals.shape[0])

        terrain, nests, food = _draw_env(state, colors)
        ants = _draw_ants(state, colors)

        ax.imshow(terrain)
        ax.imshow(nests)
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
