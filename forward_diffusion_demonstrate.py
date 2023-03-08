# forward_diffusion.demonstrate.py
# demonstrate forward diffusion process

from PIL import Image
import requests
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np

from ddpm_pytorch.forward_diffusion import *

image = Image.open('images/cat.jpg')

image_size = 128

transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into Numpy array of shape HWC, divide by 255, change pixels values from [0 - 255] to [0.0 - 1.0]
    Lambda(lambda t: (t * 2) - 1), # change pixel values from [0 - 1] to [-1 - 1]
])

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),           # rearrange dimension; CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),                                   # reversed progress of ToTensor()
])

x_start = transform(image).unsqueeze(0)
# x_start.shape == torch.Size([1, 3, 128, 128])

time_step = 40

x_noisy = q_sample(x_start, torch.tensor([time_step]))
noisy_image = reverse_transform(x_noisy.squeeze())

noisy_image.show()