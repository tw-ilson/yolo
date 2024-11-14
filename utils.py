
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
