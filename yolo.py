import math
from types import new_class
import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import safetensors
import flaxmodels
import polars as pl
from tqdm import tqdm
import cv2

# jax.profiler.start_trace("/tmp/tensorboard")

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
    iou = intersect / union
    # jax.debug.print("IOU {}", iou)
    return iou 

def box_nms(boxes, box_pred):
    iou = box_iou(xywh2abcd(boxes), xywh2abcd(box_pred))
    responsible_onehot = jax.nn.one_hot(jnp.argmax(iou, axis=-1), iou.shape[-1], dtype=jnp.bool)
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

class YoloHead(nn.Module):
    n_class:int =10
    n_box:int = 2
    region_dim:int =7

    def __post_init__(self) -> None:
        self.feature_dim = (5*self.n_box)+self.n_class
        self.output_dim = self.region_dim * self.region_dim * self.feature_dim
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]
        # B, H, W, C = x.shape
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
        x = jnp.reshape(nn.Dense(self.output_dim)(x), (B, self.region_dim*self.region_dim, self.feature_dim))
        return x

    def yolo_loss(self, Y, Y_hat, lam_coord=5, lam_noobj=.5, epsilon=1e-5):
        B = Y_hat.shape[0]
        # jax.debug.print("Y_hat isnan {}", jnp.any(jnp.isnan(Y_hat)))
        box_actual, class_actual = Y[...,None, :4], Y[...,4:]
        box_pred, class_pred = Y_hat[...,:(5*self.n_box)].reshape((B, self.region_dim**2, self.n_box, 5)), Y_hat[...,(5*self.n_box):]
        box_pred, confidence = box_pred[..., :4], box_pred[..., 4]
        sq_err = lambda a, b: (a-b)**2
        posn_err = lambda x, x_pred, y, y_pred: sq_err(x,x_pred) + sq_err(y,y_pred)
        dim_err = lambda w, w_pred, h, h_pred: sq_err(jnp.sqrt(jnp.maximum(w,epsilon)),jnp.sqrt(jnp.maximum(w_pred,epsilon))) + sq_err(jnp.sqrt(jnp.maximum(h,epsilon)),jnp.sqrt(jnp.maximum(h_pred,epsilon)))
        # class_pred_choice = jax.nn.softmax(class_pred, axis=-1)
        # jax.debug.print("softmax {}", class_pred_choice)
        class_pred_choice = jax.lax.stop_gradient(jax.nn.one_hot(jnp.argmax(class_pred, axis=-1), self.n_class))
        predict_object = jnp.any(class_pred_choice*class_actual, axis=-1, keepdims=True)
        # jax.debug.print("predict_object {}", jnp.mean(predict_object))
        predictor = jax.lax.stop_gradient(box_nms(box_actual, box_pred))
        # jax.debug.print("class_choice {}", jnp.any(jnp.isnan(class_choice)))
        # jax.debug.print("predictor_object {}", jnp.any(jnp.isnan(predict_object)))
        # jax.debug.print("predictor {}", predictor.shape)
        l1 = lam_coord * jnp.sum(predictor * posn_err(box_pred[...,0], box_actual[..., 0], box_pred[...,1], box_actual[...,1]), axis=(1,2)) 
        l2 = lam_coord * jnp.sum(predictor * dim_err(box_pred[..., 2], box_actual[..., 2], box_pred[..., 3], box_actual[..., 3]), axis=(1,2))
        l3 = jnp.sum(jnp.expand_dims(predictor, -1) * jnp.expand_dims(sq_err(class_pred, class_actual), axis=2), axis=(1,2,3))
        l4 = lam_noobj * jnp.sum((~predictor)[...,None] * jnp.expand_dims(sq_err(class_pred, class_actual), axis=2), axis=(1,2,3))
        l5  = lam_coord * jnp.sum(predict_object * sq_err(confidence, box_iou(xywh2abcd(box_pred), xywh2abcd(box_actual))), axis=(1,2))
        # jax.debug.print("coordinate loss {}", jnp.any(jnp.isnan(l1)))
        # jax.debug.print("shape loss {}", jnp.any(jnp.isnan(l2)))
        # jax.debug.print("classification loss (responsible) {}", jnp.any(jnp.isnan(l3)))
        # jax.debug.print("classification loss (other) {}", jnp.any(jnp.isnan(l4)))
        # jax.debug.print("object accuracy loss {}", jnp.any(jnp.isnan(l5)))
        return jnp.sum(l1 + l2 + l3 + l4 + l5)

def train(region_dim=7, lr=1e-1, batch_size=8, epochs=4, epoch_steps=1000):
    key = jax.random.key(seed=42)
    x_0 = jnp.ones((1, 640, 640, 3))
    #initialize resnet backbone 
    print("obtaining pretrained weights...")
    resnet = flaxmodels.ResNet18(output='logits', pretrained='imagenet')
    resnet_params = resnet.init(key, x_0)
    # print(resnet.tabulate(key, x_0, compute_flops=True))
    activations = resnet.apply(resnet_params, x_0, train=False)
    # print(activations.keys())
    # last_block_shape= activations['block4_1'].shape
    backbone = lambda image: resnet.apply(resnet_params, image, train=False)

    h_0 = jax.random.uniform(key, 1000)
    head = YoloHead(region_dim=region_dim)
    params = head.init(key, h_0)
    print(head.tabulate(key, h_0, compute_flops=True))

    print("obtaining data...")
    splits = {'train': 'data/train-00000-of-00001-83ef17440fd25f6f.parquet', 'validation': 'data/validation-00000-of-00001-876de533d76d48c6.parquet', 'test': 'data/test-00000-of-00001-77e7809ef7ac5cc0.parquet'}
    df = (pl.read_parquet('hf://datasets/Francesco/animals-ij5d2/' + splits['train'])
            .unnest("objects")
            .unnest("image")
            .explode(["id", "area", "category", "bbox"])
          )
    # print("----------------------------------------------------------------")
    # print("Dataset:")
    # print("----------------------------------------------------------------")
    # print(df)


    def prepare_batch(batch_rows):
        image_batch = []
        objects_batch = []
        for row in batch_rows.to_dicts():
            image = jnp.array(cv2.imdecode(np.frombuffer(row['bytes'], dtype=np.uint8), cv2.IMREAD_COLOR), dtype=jnp.float32) / 255
            image_batch.append(image)
            obj = jnp.tile(jnp.concat((jnp.array(row['bbox']), jax.nn.one_hot(row['category'], head.n_class))), (head.region_dim**2, 1))
            objects_batch.append(obj)
        return jnp.stack(image_batch), jnp.stack(objects_batch)

    def loss_fn(params, image, objects):
        features = jax.lax.stop_gradient(backbone(image))
        output_features = head.apply(params, features)
        loss_value = head.yolo_loss(objects, output_features)
        # jax.debug.print("loss value {}", loss_value)
        return loss_value

    @jax.jit
    def step(params, opt_state, image, labels):
        loss_grad = jax.value_and_grad(loss_fn)
        loss_value, grad = loss_grad(params, image, labels)
        print(grad)
        # updates, opt_state = optimizer.update(grad, opt_state)
        # params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
        

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    print("beginning training loop...")
    for ep in range(epochs):
        print(f"---- Epoch {ep} ----")
        pbar = tqdm(range(epoch_steps))
        for _ in pbar:
            x = df.sample(n=batch_size, shuffle=True)
            image_batch, object_batch = prepare_batch(x)
            params, opt_state, loss_value = step(params, opt_state, image_batch, object_batch)

            jax.debug.print('Loss: {}', loss_value)


def load_safetensors(model):
    tensors = {}
    with safetensors.safe_open(model, framework="flax", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)

def main():
    train()
if __name__ == "__main__":
    main()
