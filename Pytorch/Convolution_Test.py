import numpy as np
import torch
from torch.nn import Conv2d
from PIL import Image
import matplotlib.pyplot as plt
import requests
from scipy.ndimage import zoom as resize

url_Image = "http://raw.githubusercontent.com/jmlipman/jmlipman.github.io/master/images/kumamon.jpeg"
im = Image.open(requests.get(url_Image, stream=True).raw).convert("L")

im = np.array(im)

kernel_size = 7
dilation = 3
stride = 1
padding = 5
#Define a 2D convolutions
conv_1 = Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, dilation=dilation, padding=padding)

im1 = resize(im, (0.5,0.5)).reshape(1,1,50,50)


input_image = torch.Tensor(im1)

output_image = conv_1(input_image).detach().numpy()[0,0]

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(output_image)
plt.title("conv")

plt.subplot(122)
plt.imshow(im, cmap="gray")
plt.title("origin")

plt.show()