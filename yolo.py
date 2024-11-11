import math
from types import new_class
import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import tensorflow_datasets as tfds
import safetensors
import flaxmodels
import polars as pl
from tqdm import tqdm
import cv2

def xywh2abcd(x):
    return jnp.concat((x[..., :2] + x[...,2:4]/2, x[...,:2] - x[...,2:4]/2), axis=-1)

def box_iou(box1, box2):
    # box abcd form
    lt = jnp.maximum(box1[..., :2], box2[..., :2])
    rb = jnp.minimum(box1[..., 2:4], box2[..., 2:4])
    wh = jnp.clip(rb-lt, 0)
    intersect = wh[..., 0] * wh[..., 1]
    x = jnp.prod(box1[..., :2] - box1[..., 2:4], axis=-1)
    y = jnp.prod(box2[..., :2] - box2[..., 2:4], axis=-1)
    union = x + y - intersect
    return intersect / union

def box_nms(boxes, box_pred):
    iou = box_iou(xywh2abcd(boxes), xywh2abcd(box_pred))
    responsible_onehot = jax.nn.one_hot(jnp.argmax(iou, axis=-1), iou.shape[-1])
    return responsible_onehot

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        act = nn.activation.silu

        # Layer 1
        x = act(
                nn.max_pool(
                    nn.Conv(64, (7,7), strides=(2,2))(x),
                    (2,2), strides=(2,2)))

        # Layer 2
        x = act(
                nn.max_pool(
                    nn.Conv(192, (3,3))(x),
                    (2,2), strides=(2,2)))

        # Layer 3
        x = act(
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
        x = nn.max_pool(
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

        x = nn.Sequential([
            nn.Conv(512, (1,1)),
            nn.Conv(1024, (3,3))
        ])(x)
        return x

class Head(nn.Module):
    n_class=10
    n_box=2
    region_dim=7

    def __post_init__(self) -> None:
        self.feature_dim = (5*self.n_box)+self.n_class
        self.output_dim = self.region_dim * self.region_dim * self.feature_dim
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        act = nn.activation.silu

        # Layer 5
        x = act(
                nn.Sequential([
                    nn.Conv(512, (1,1)),
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3), strides=(2,2)),
                    ])(x)
                )

        # Layer 6
        x = act(
                nn.Sequential([
                    nn.Conv(1024, (3,3)),
                    nn.Conv(1024, (3,3))
                    ])(x)
                )

        # Layer 7
        x = act(nn.Dense(4096)(jnp.reshape(x, (B, -1))))
        
        # Output Layer
        x = jnp.reshape(nn.Dense(self.output_dim)(x), (B, self.region_dim, self.region_dim, self.feature_dim))
        return x

    def yolo_loss(self, Y, Y_hat, lam_coord=5, lam_noobj=.5, epsilon=1e-6):
        B = Y_hat.shape[0]
        Y_hat = jnp.reshape(Y, (B, self.region_dim**2, self.feature_dim))
        box_actual, class_actual = Y[...,None, :4], Y[...,4:]
        box_pred, class_pred = Y_hat[...,:(5*self.n_box)].reshape((B, self.region_dim**2, self.n_box, 5)), Y_hat[...,(5*self.n_box):]
        box_pred, confidence = box_pred[..., :4], box_pred[..., 4]
        sq_err = lambda a, b: (a-b)**2
        posn_err = lambda x, x_pred, y, y_pred: sq_err(x,x_pred) + sq_err(y,y_pred)
        dim_err = lambda w, w_pred, h, h_pred: sq_err(jnp.sqrt(w+epsilon),jnp.sqrt(w_pred+epsilon)) + sq_err(jnp.sqrt(h+epsilon),jnp.sqrt(h_pred+epsilon))
        class_choice = jax.nn.one_hot(jnp.argmax(class_pred, axis=-1), self.n_class)
        predict_object = jnp.any(jnp.equal(class_choice, class_actual), axis=-1, keepdims=True)
        predictor = box_nms(box_actual, box_pred)
        l1 = lam_coord * jnp.sum(predictor * posn_err(box_pred[...,0], box_actual[..., 0], box_pred[...,1], box_actual[...,1]), axis=(1,2)) 
        l2 = lam_coord * jnp.sum(predictor * dim_err(box_pred[..., 2], box_actual[..., 2], box_pred[..., 3], box_actual[..., 3]), axis=(1,2))
        l3 = jnp.sum(predictor[...,None] * jnp.expand_dims(sq_err(class_pred, class_actual), axis=2))
        l4 = lam_noobj * jnp.sum(jnp.logical_not(predictor)[...,None] * jnp.expand_dims(sq_err(class_pred, class_actual), axis=2))
        l5  = lam_coord * jnp.sum(predict_object * sq_err(confidence, box_iou(xywh2abcd(box_pred), xywh2abcd(box_actual))))
        return l1 + l2 + l3 + l4 + l5

def train(lr=1e-1, batch_size=32):
    # backbone = CNN()
    head = Head()
    head_params = head.init(jax.random.key(0), jnp.ones((4, 7, 7, 30)))
    backbone = flaxmodels.ResNet50(output='logits', pretrained='imagenet')

    splits = {'train': 'data/train-00000-of-00001-83ef17440fd25f6f.parquet', 'validation': 'data/validation-00000-of-00001-876de533d76d48c6.parquet', 'test': 'data/test-00000-of-00001-77e7809ef7ac5cc0.parquet'}

    n_classes=10

    df = pl.read_parquet('hf://datasets/Francesco/animals-ij5d2/' + splits['train'])
    df = df.with_columns([
        # pl.col("image").struct.field("bytes").alias("image"),
        pl.col("objects").struct.field("id").explode().alias("obj_id"),
        pl.col("objects").struct.field("category").explode().alias("category"),
        pl.col("objects").struct.field("bbox").explode().alias("bbox")
        ])
    print(df)

    def preprocess(row):
        return jnp.array(cv2.imdecode(np.frombuffer(row['image']['bytes']), cv2.IMREAD_COLOR))
        # print(image.shape)
    # df = df.shuffle(seed=42).collect(batch_size=batch_size)

    # optimizer = optax.adam(lr)
    # opt_state = optimizer.init(head_params)
    #
    # epochs = 5
    # training_steps = 1000
    # for ep in range(epochs):
    #     print(f"Epoch {ep}...")
    #     pbar = tqdm(enumerate(df.to_dicts()))
    #     for step, row_dict in pbar:
    #         # image = row_dict['
    #         grad = jax.grad(head.yolo_loss)(head_params)
    #         updates, opt_state = optimizer.update(grad, opt_state, head_params)
    #         params = optax.apply_updates(head_params, updates)
    #         print('Objective function: {:.2E}'.format(head(head_params)))


def load_safetensors(model):
    tensors = {}
    with safetensors.safe_open(model, framework="flax", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def main():
    print("Hello from yolo.py!")
    train()
    
if __name__ == "__main__":
    main()
