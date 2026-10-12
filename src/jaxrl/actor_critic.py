import jax.numpy as jnp

from flax import nnx
import optax

# ------------------------------------------------------------
# ACTOR
# ------------------------------------------------------------

def actor_loss_fn(actor, obs, action, td_error):
    """
    Policy gradient loss: advantage-weighted negative log-probability of the taken actions.

    Parameters:
    ----------
    actor: nnx.Module
        The policy network.
    obs: jnp.ndarray
        The observation.
    action: jnp.ndarray
        The taken action.
    td_error: jnp.ndarray
        The TD error for the taken action.
    """
    # 1. Forward pass
    logits = actor(obs) # shape (num_actions,)

    # 2. Compute the loss
    nll = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=action
    ) # shape ()
    loss = jnp.mean(td_error * nll)
    return loss


@nnx.jit
def actor_train_step(actor, optimizer, obs, action, td_error):
    """One gradient step on the policy"""
    loss, grads = nnx.value_and_grad(actor_loss_fn)(
        actor, obs, action, td_error
    )
    optimizer.update(actor, grads)
    return loss


#------------------------------------------------------------
# CRITIC
# ------------------------------------------------------------

def critic_loss_fn(critic, obs, td_target):
    """
    Compute the MSE loss for the critic.
    """
    value = critic(obs).squeeze(-1)  # shape ()
    loss = jnp.mean((td_target - value)**2)
    return loss


@nnx.jit
def critic_train_step(critic, optimizer, obs, td_target):
    """One gradient step on the value function"""
    loss, grads = nnx.value_and_grad(critic_loss_fn)(
        critic, obs, td_target
    )
    optimizer.update(critic, grads)
    return loss


# ------------------------------------------------------------
# TD ERROR
# ------------------------------------------------------------

@nnx.jit
def compute_td_error(critic, obs, reward, next_obs, done, gamma):
    """
    One-step TD target and TD error, from the critic's current weights.

        target = r + gamma * V(s') * (1 - done)
        error  = target - V(s)

    """
    td_target = reward + gamma * critic(next_obs).squeeze(-1) * (1 - done)
    td_error = td_target - critic(obs).squeeze(-1)
    return td_error, td_target