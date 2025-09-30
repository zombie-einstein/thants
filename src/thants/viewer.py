from typing import Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from jumanji.viewer import MatplotlibViewer
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from .types import State


def draw_env(dims: tuple[int, int], state: State) -> tuple[chex.Array, chex.Array]:
    frame = jnp.zeros(dims)
    ants = frame.at[state.ants.pos[:, 0], state.ants.pos[:, 1]].set(1.0)
    env = jnp.stack([ants, state.food, state.nest], axis=2)
    signals = jnp.stack(
        [state.signals[0], state.signals[1], frame, frame + 0.5], axis=2
    )
    return env, signals


class ThantsViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        env_dims: tuple[int, int],
        name: str = "thants",
        render_mode: str = "human",
    ) -> None:
        """
        Thants environment visualiser using Matplotlib

        Parameters
        ----------
        name
            Plot name, default ``thants``
        render_mode
            Default ``human``
        """
        self.env_dims = env_dims
        super().__init__(name, render_mode)

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

        env, signals = draw_env(self.env_dims, states[0])
        env_img = ax.imshow(env)
        signal_img = ax.imshow(signals)

        def make_frame(state: State) -> tuple[AxesImage, AxesImage]:
            step_env, step_signals = draw_env(self.env_dims, state)
            env_img.set_data(step_env)
            signal_img.set_data(step_signals)
            return env_img, signal_img

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
        env, signals = draw_env(self.env_dims, state)
        ax.imshow(env)
        ax.imshow(signals)

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
