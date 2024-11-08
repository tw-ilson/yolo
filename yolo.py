import math
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

class CNN(nn.Module):
    act = nn.elu

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        # Layer 1
        x = self.act(
                nn.max_pool(
                    nn.Conv(64, (7,7), strides=(2,2))(x),
                    (2,2), strides=(2,2)))

        # Layer 2
        x = self.act(
                nn.max_pool(
                    nn.Conv(192, (3,3))(x),
                    (2,2), strides=(2,2)))

        # Layer 3
        x = self.act(
                nn.max_pool(
                    nn.Sequential([
                        nn.Conv(128, (1,1)),
                        nn.Conv(256, (3,3)),
                        nn.Conv(256, (1,1)),
                        nn.Conv(512, (3,3)),
                        ])(x),
                    window_shape=(2,2), strides=(2,2))
                )

        # Layer 4
        x = self.act(
                nn.max_pool(
                    nn.Sequential([
                        nn.Conv(256, (1,1)),
                        nn.Conv(512, (3,3)),
                        nn.Conv(256, (1,1)),
                        nn.Conv(512, (3,3)),
                        nn.Conv(256, (1,1)),
                        nn.Conv(512, (3,3)),
                        nn.Conv(256, (1,1)),
                        nn.Conv(512, (3,3)),
                        nn.Conv(512, (1,1)),
                        nn.Conv(1024, (3,3)),
                        ])(x),
                window_shape=(2,2), strides=(2,2))
                )

        # Layer 5
        x = self.act(
                nn.Sequential([
                    nn.Conv(512, (1,1)),
                    nn.Conv(1024, (3,3)),
                    nn.Conv(512, (1,1)),
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3), strides=(2,2)),
                    ])(x)
                )

        # Layer 6
        x = self.act(
                nn.Sequential([
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3))
                    ])(x)
                )

        # Layer 7
        x = self.act(nn.Dense(4096)(jnp.ravel(x)))

        # Output Layer
        x = jnp.reshape(nn.Dense(1470)(x), (7,7,30))
        return x


def main():
    print("Hello from yolo.py!")
    model = CNN()
    x = jnp.ones((1,448,448,3))
    model.init(jax.random.key(0), x)
    print('model initialized')

if __name__ == "__main__":
    main()
