import jax
import jax.numpy as jnp

import optax
from flax import nnx


# ------------------------------------------------------------
# ROLLOUT
# ------------------------------------------------------------

@nnx.jit(static_argnames=("env", "max_steps"))
def rollout(model, env, env_params, key, max_steps):
    """
    Perform a rollout in the environment for a given episode using the current policy.

    Parameters:
    ----------
    model: nnx.Module
        The policy network.
    env: Environment
        The environment to interact with.
    env_params: dict
        Parameters for the environment.
    key: jax.random.PRNGKey
        Random key for stochastic operations.
    max_steps: int
        Maximum number of steps to take in the environment.

    Returns:
    -------
    observations: jnp.ndarray
        The observations encountered in the episode (T, state_dim).
    actions: jnp.ndarray
        The actions taken in the episode (T,).
    action_probs: jnp.ndarray
        The probabilities of possible actions (T, num_actions).
    rewards: jnp.ndarray
        The rewards received at each step (T,).
    mask: jnp.ndarray
        A boolean mask indicating which steps are valid (T,).
    """
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)

    def step_fn(carry, step_key):
        obs, env_state, done = carry
        alive = ~done
        step_key, action_key = jax.random.split(step_key)

        # Select an action based on the current observation
        action, probs = model.sample(obs, action_key)

        # Take a step in the environment
        next_obs, next_env_state, reward, step_done, _ = env.step(step_key, env_state, action, env_params)

        done = done | step_done

        next_carry = (next_obs, next_env_state, done)
        return next_carry, (obs, action, probs, reward * alive, alive)

    # lax.scan returns (final_carry, stacked_outputs)
    _, (observations, actions, action_probs, rewards, mask) = jax.lax.scan(
        step_fn, 
        (obs, env_state, jnp.array(False)),
        jax.random.split(key, max_steps)
    )

    return observations, actions, action_probs, rewards, mask


# ------------------------------------------------------------
# REINFORCE ALGORITHM
# ------------------------------------------------------------

@jax.jit
def compute_returns(rewards, gamma):
    """
    Compute the discounted returns for an episode.

    Parameters:
    ----------
    rewards: jnp.ndarray
        The rewards received at each step (T,).
    gamma: float
        The discount factor.

    Returns:
    -------
    jnp.ndarray
        The discounted returns for each step in the episodes (T,).
    """

    def step_fn(running_return, reward):
        next_return = reward + gamma * running_return
        # return new_carry, output
        return next_return, next_return
    
    _, returns = jax.lax.scan(
        step_fn,
        jnp.array(0.0),
        rewards,
        reverse=True
    )
    return returns


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

def loss_fn(model, observations, actions, advantages, mask):
    """
    Compute the REINFORCE loss.

    Parameters:
    ----------
    model: nnx.Module
        The policy network.
    observations: jnp.ndarray
        The observations encountered in the episodes (T, state_dim).
    actions: jnp.ndarray
        The actions taken in the episodes (T,).
    advantages: jnp.ndarray
        The advantages for each step in the episodes (T,).
    mask: jnp.ndarray
        The boolean mask indicating which steps are valid (T,).

    Returns:
    -------
    jnp.ndarray
        The REINFORCE loss for an episode.
    """

    # 1. Forward pass (we run it again, to get the gradients)
    logits = model(observations) # shape (T, num_actions)

    # 2. Compute the loss
    nll = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=actions) # shape (T,)

    # REINFORCE: maximize E[(G_t - b) * log pi(a|s)] i.e. minimize the negative
    # With the baseline, it stays around zero and changes with each episode,
    # so it won't fall steadily the way a supervised loss does.
    # What we want is to maximize the expected return.
    # More concise: jnp.mean(advantages * nll, where=mask)
    return jnp.sum(advantages * nll * mask) / jnp.sum(mask)


@nnx.jit
def train_step(model, optimizer, observations, actions, advantages, mask):
    """
    Perform a single training step for the REINFORCE algorithm.

    Parameters:
    ----------
    model: nnx.Module
        The policy network.
    optimizer: nnx.Optimizer
        The optimizer for updating the model parameters.
    observations: jnp.ndarray
        States visited during the episode, shape (T, state_dim).
    actions: jnp.ndarray
        Actions taken at each step, shape (T,).
    advantages: jnp.ndarray
        Advantages for each step, shape (T,).
    mask: jnp.ndarray
        The boolean mask indicating which steps are valid, shape (T,).
    """
    loss, grads = nnx.value_and_grad(loss_fn)(
        model, observations, actions, advantages, mask
    )
    optimizer.update(model, grads)

    return loss

