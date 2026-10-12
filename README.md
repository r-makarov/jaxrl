# jaxrl

Minimal implementations of policy gradient algorithms in [JAX](https://github.com/jax-ml/jax), built up from simple to more advanced. Written with [Flax NNX](https://flax.readthedocs.io/) and [Optax](https://github.com/google-deepmind/optax), with environments from [gymnax](https://github.com/RobertTLange/gymnax).

## Repository layout

Each algorithm is a short module in `src/jaxrl/` with a notebook in `notebooks/` that trains it and visualises what it learns.

```
src/jaxrl/
├── models/          # MLP, CNN
├── reinforce.py     # episode rollout, discounted returns, policy gradient loss
├── actor_critic.py  # TD error, actor and critic updates
└── plot.py          # CartPole policy and advantage plots
notebooks/           # one notebook per algorithm
```

## Algorithms

| Algorithm | Type | Notebook | Status |
|---|---|---|---|
| REINFORCE with baseline | Monte Carlo policy gradient | [REINFORCE.ipynb](notebooks/REINFORCE.ipynb) | ✅ |
| Actor-critic, separate networks | One-step online TD | [Actor_Critic.ipynb](notebooks/Actor_Critic.ipynb) | ✅ |
| Actor-critic, shared network | One-step online TD | | Planned |
| Advantage actor-critic (A2C) | --- | | Planned |
| Proximal policy optimization (PPO) | --- | | Planned |
| Soft actor-critic (SAC) | --- | | Planned |

As a supervised-learning reference, [CNN_MNIST.ipynb](notebooks/CNN_MNIST.ipynb) trains a small CNN on MNIST with a [Grain](https://github.com/google/grain) input pipeline and scikit-learn preprocessing.



