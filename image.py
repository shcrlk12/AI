from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")

image = pipe("a castle on a floating island, fantasy art").images[0]
image.save("castle.png")
