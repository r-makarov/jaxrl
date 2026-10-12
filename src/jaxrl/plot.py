import matplotlib.pyplot as plt
import numpy as np

# CartPole failure limits
X_LIMIT = 2.4
THETA_LIMIT_DEG = 12.0


def plot_advantages(advantages, title="Advantages"):
    """
    Bar chart of the advantages G_t - b over one episode.

    The update makes actions with a positive advantage (blue) more likely
    and actions with a negative advantage (red) less likely.

    Parameters
    ----------
    advantages : array-like
        Advantage at each step, shape (T,).
    title : str
        Figure title.
    """
    advantages = np.asarray(advantages)
    steps = np.arange(len(advantages))
    colors = np.where(advantages >= 0, "royalblue", "crimson")

    fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
    ax.bar(steps, advantages, color=colors)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Step t", ylabel="Advantage", title=title)
    return fig


def plot_policy(states, action_probs, actions, title="Policy over one episode"):
    """
    Cart position, pole angle, P(right) and the sampled actions over one episode.

    A good policy pushes right when the pole leans right, so the pole angle
    and P(right) should move together.

    Parameters
    ----------
    states : array-like
        States visited during the episode, shape (T, 4).
    action_probs : array-like
        Action probabilities at each step, shape (T, 2).
    actions : array-like
        Sampled actions (0 = left, 1 = right), shape (T,).
    title : str
        Figure title.
    """
    states, actions = np.asarray(states), np.asarray(actions)
    steps = np.arange(len(actions))

    fig, (ax_x, ax_angle, ax_prob, ax_action) = plt.subplots(
        4, 1, figsize=(10, 7), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [2, 2, 2, 1]},
    )
    fig.suptitle(title)

    # Cart position and pole angle, with their failure limits
    ax_x.plot(steps, states[:, 0], color="black")
    ax_angle.plot(steps, np.degrees(states[:, 2]), color="black")
    for ax, limit in ((ax_x, X_LIMIT), (ax_angle, THETA_LIMIT_DEG)):
        ax.axhline(-limit, color="red", linestyle="--")
        ax.axhline(limit, color="red", linestyle="--")
    ax_x.set_ylabel("Cart position")
    ax_angle.set_ylabel("Pole angle (°)")

    # Probability of pushing right
    ax_prob.plot(steps, np.asarray(action_probs)[:, 1], color="black")
    ax_prob.axhline(0.5, color="gray", linestyle="--")
    ax_prob.set(ylim=(0, 1), ylabel="P(right)")

    # Sampled actions
    for action, name, color in ((0, "Left", "orange"), (1, "Right", "green")):
        ax_action.vlines(steps[actions == action], 0, 1, color=color, label=name)
    ax_action.set(yticks=[], ylabel="Action", xlabel="Step t")
    ax_action.legend(ncols=2, loc="lower right", bbox_to_anchor=(1, 1), frameon=False)

    return fig
