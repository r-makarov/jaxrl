from flax import nnx
import jax


class CNN(nnx.Module):
    """Convolutional neural network (CNN) model for image classification."""

    def __init__(self, n_in, n_out, *, rngs):
        self.conv1 = nnx.Conv(
            in_features=n_in,
            out_features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=16,
            out_features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            rngs=rngs
        )
        self.fc1 = nnx.Linear(
            in_features=32 * 7 * 7,
            out_features=128,
            rngs=rngs
        )
        self.fc2 = nnx.Linear(
            in_features=128,
            out_features=n_out,
            rngs=rngs
        )


    def __call__(self, x: jax.Array) -> jax.Array:
        # Convolutional block 1:
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        # Convolutional block 2:
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        # Flatten the output for the fully connected layers
        x = x.reshape(x.shape[0], -1)
        # Fully connected layers:
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        return x


    # def predict(self, x: jax.Array) -> jax.Array:
    #     logits = self.__call__(x)
    #     predictions = jnp.argmax(logits, axis=1)
    #     return predictions