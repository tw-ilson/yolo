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
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    k: int
    s: int
    p: int
    c: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.c, (self.k,self.k), strides=(self.s,self.s), padding=(self.p,self.p))(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.activation.silu(x)
        return x

class SPPF(nn.Module):
    dim:int 
    k:int

    @nn.compact
    def __call__(self, x):
        h0 = ConvBlock(k=1, s=1, p=0, c=self.dim//2)(x)
        h1 = nn.max_pool(h0, (self.k,self.k), strides=(1,1), padding=((self.k//2,self.k//2),(self.k//2,self.k//2)))
        h2 = nn.max_pool(h1, (self.k,self.k), strides=(1,1), padding=((self.k//2,self.k//2),(self.k//2,self.k//2)))
        h3 = nn.max_pool(h2, (self.k,self.k), strides=(1,1), padding=((self.k//2,self.k//2),(self.k//2,self.k//2)))
        return ConvBlock(k=1,s=1,p=0, c=self.dim)(jnp.concat([h0,h1,h2,h3], axis=-1))

class BottleNeck(nn.Module):
    dim:int
    shortcut:bool

    @nn.compact
    def __call__(self,x):
        h = ConvBlock(k=3, s=1, p=1, c=self.dim)(x)
        h = ConvBlock(k=3, s=1, p=1, c=self.dim)(x)
        if self.shortcut:
            h = h+x
        return h

class C2f(nn.Module):
    shortcut:bool
    n:int
    dim:int

    @nn.compact
    def __call__(self, x):
        # B, H, W, C = x.shape
        x = ConvBlock(k=1, s=1, p=0, c=self.dim)(x)
        h0, h1 = jnp.split(x, 2, axis=-1)
        residual = [h0, h1]
        layers = [BottleNeck(shortcut=self.shortcut, dim=self.dim//2) for _ in range(self.n)]
        for i in range(self.n): 
            residual.append(layers[i](residual[-1]))
        out = ConvBlock(k=1, s=1, p=0, c=self.dim)(jnp.concat(residual, axis=-1))
        return out


class Backbone(nn.Module):
    w:float
    d:float
    r:float

    @nn.compact
    def __call__(self, x):
        h = ConvBlock(k=3,s=2,p=1,c=int(64*self.w))(x)
        h = ConvBlock(k=3,s=2,p=1,c=int(128*self.w))(h)
        h = C2f(shortcut=True, n=int(3*self.d), dim=int(128*self.w))(h)
        h = ConvBlock(k=3, s=2, p=1, c=int(256*self.w))(h)
        out1 = C2f(shortcut=True, n=int(6*self.d), dim=int(256*self.w))(h)
        h = ConvBlock(k=3, s=2, p=1, c=int(512*self.w))(out1)
        out2 = C2f(shortcut=True, n=int(6*self.d), dim=int(512*self.w))(h)
        h = ConvBlock(k=3, s=2, p=1, c=int(512*self.w*self.r))(out2)
        h = C2f(shortcut=True, n=int(3*self.d), dim=int(512*self.w*self.r))(h)
        out3 = SPPF(k=5, dim=int(512*self.w*self.r))(h)
        print("p5", out3.shape)
        return out1, out2, out3

class YoloV8Neck(nn.Module):
    w:float
    d:float
    r:float

    @nn.compact
    def __call__(self, p3, p4, p5):
        p4 = jnp.concat([p4, jax.image.resize(p5, shape=(p5.shape[0], p5.shape[1]*2, p5.shape[2]*2, p5.shape[3]), method='bilinear')], axis=-1)
        p4 = C2f(shortcut=False, n=int(3*self.d), dim=int(512*self.w))(p4)
        p3 = jnp.concat([p3, jax.image.resize(p4, shape=(p4.shape[0], p4.shape[1]*2, p4.shape[2]*2, p4.shape[3]), method='bilinear')], axis=-1)
        p3 = C2f(shortcut=False, n=int(3*self.d), dim=int(256*self.d))(p3)
        p4 = jnp.concat([p4, ConvBlock(k=3, s=2, p=1, c=int(256*self.w))(p3)], axis=-1)
        p4 = C2f(shortcut=False, n=int(3*self.d), dim=int(512*self.w))(p4)
        p5 = jnp.concat([p5, ConvBlock(k=3, s=2, p=1, c=int(512*self.w))(p4)], axis=-1)
        p5 = C2f(shortcut=False, n=int(3*self.d), dim=int(512*self.w*self.r))(p5)
        return p3, p4, p5

class DFL(nn.Module):
    c1:int

    def setup(self):
        self.cv = nn.Conv(1, strides=(1,1), use_bias=False)
        x = np.arange(self.c1)
        params = self.cv.init(jax.random.key(0), x.reshape((1, self.c1, 1, 1)))

    def __call__(self, x):
        B, C, A = x.shape
        return self.cv(jax.nn.softmax(x.reshape(B, 4, self.c1, A).transpose((2,1)), 1)).reshape((B, 4, A))


class DetectHead(nn.Module):

    def setup(
            self,
            # dim:int,
            # reg_max:int,
            nc:int,
            filters=()):
        self.ch=16
        self.nc=nc
        self.nl=len(filters)
        self.no=nc+self.ch*4
        self.stride=[8,16,32]
        c1 = max(filters[0], self.nc)
        c2 = max((filters[0]//4, self.ch*4))
        self.dfl = DFL(self.ch)
        self.cv3 = [[ConvBlock(k=3, s=1, p=1, c=c1), ConvBlock(k=3, s=1, p=1, c=c1), nn.Conv(self.nc, (1,1))] for _ in filters]
        self.cv2 = [[ConvBlock(k=3, s=1, p=1, c=c2), ConvBlock(k=3, s=1, p=1, c=c2), nn.Conv(4*self.nc, (1,1))] for _ in filters]


    # def __call__(self, x):
    #     for i in range(self.nl):
    #         x[i] = (

def main():
    d=0.33
    w=0.5
    r=2.0

    key = jax.random.key(0)
    model = Backbone(d=d, w=w, r=r)
    neck = YoloV8Neck(d=d, w=w, r=r)
    x_0 = jax.random.uniform(key, (1, 640, 640, 3))
    p3_0, p4_0, p5_0 = jax.random.uniform(key, (1, 80, 80, int(256*w))),jax.random.uniform(key, (1, 40, 40, int(512*w))),jax.random.uniform(key, (1, 20, 20, int(512*w*r)))
    backbone_params = model.init(key, x_0)
    p3_1, p4_1, p5_1 = model.apply(backbone_params, x_0)
    # print("backbone outputs: ", p3_1.shape, p4_1.shape, p5_1.shape)
    assert(all(a.shape==b.shape for a, b in zip([p3_1, p4_1, p5_1], [p3_0, p4_0, p5_0])))
    neck_params = neck.init(key, p3_0, p4_0, p5_0)


if __name__ == "__main__":
    main()
