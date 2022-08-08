# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch
from torch.cuda import amp

from utils.general import colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640):
    # Check YOLOv5 training batch size
    with amp.autocast():
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    """
    Automatically estimate best batch size to use `fraction` of available CUDA memory.
    It will get the mapping of the batch size to the memory usage, which is illustrated by one-order polynomial curve.
    mem = a * batch_size + b
    Then the best batch_size can be calculated by solving the equation:
    batch_size = (mem * fraction - b) / a
    It's used to leave some space for the GPU memory. So, fraction = 0.9 means that we want to leave 90% of GPU memory for the model.
    """

    prefix = colorstr('autobatch: ')
    print(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        print(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    d = str(device).upper()  # 'CUDA:0'
    t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3  # (GB)
    r = torch.cuda.memory_reserved(device) / 1024 ** 3  # (GB)
    a = torch.cuda.memory_allocated(device) / 1024 ** 3  # (GB)
    f = t - (r + a)  # free inside reserved
    print(f'{prefix}{d} {t:.3g}G total, {r:.3g}G reserved, {a:.3g}G allocated, {f:.3g}G free')

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)
    except Exception as e:
        print(f'{prefix}{e}')

    y = [x[2] for x in y if x]  # get the memory. It will eliminate the ones with no memory left.
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    print(f'{prefix}Using colorstr(batch-size {b}) for {d} {t * fraction:.3g}G/{t:.3g}G ({fraction * 100:.0f}%)')
    return b
