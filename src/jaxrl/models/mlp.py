import jax
from flax import nnx


class MLP(nnx.Module):

    def __init__(self, n_in, n_out, *, rngs, hidden_size=128, use_bias=True):

        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias

        self.fc1 = nnx.Linear(n_in, hidden_size, use_bias=use_bias, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, n_out, use_bias=use_bias, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample(self, x, key):
        logits = self.__call__(x)
        # Note: jax.random.categorical expects logits, not probs
        # Note: sampled action is not the same as the argmax;
        # it is sampled according to the distribution
        action = jax.random.categorical(key, logits)
        probs = nnx.softmax(logits)
        return action, probs
