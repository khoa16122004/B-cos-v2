# import torch

# # list all available models
# torch.hub.list('B-cos/B-cos-v2')

# # load a pretrained model
# model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)


# from PIL import Image
# import matplotlib.pyplot as plt

# # load image
# img = model.transform(Image.open('cat.png'))
# print(model.transform)  # show the transform pipeline
# print(img)
# raise
# img = img[None].requires_grad_()

# # predict and explain
# model.eval()
# expl_out = model.explain(img)
# print("Prediction:", expl_out["prediction"])  # predicted class idx
# plt.imshow(expl_out["explanation"])
# plt.show()

from bcos.models import pretrained
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def load_model(model_name: str):
    """
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - resnext50_32x4d
    - densenet121
    - densenet161
    - densenet169
    - densenet201
    - vgg11_bnu
    - convnext_tiny
    - convnext_base
    - convnext_tiny_bnu
    - convnext_base_bnu
    - densenet121_long
    - resnet50_long
    - resnet152_long
    - simple_vit_ti_patch16_224
    - simple_vit_s_patch16_224
    - simple_vit_b_patch16_224
    - simple_vit_l_patch16_224
    - vitc_ti_patch1_14
    - vitc_s_patch1_14
    - vitc_b_patch1_14
    - vitc_l_patch1_14
    - standard_simple_vit_ti_patch16_224
    - standard_simple_vit_s_patch16_224
    - standard_simple_vit_b_patch16_224
    - standard_simple_vit_l_patch16_224
    - standard_vitc_ti_patch1_14
    - standard_vitc_s_patch1_14
    - standard_vitc_b_patch1_14
    - standard_vitc_l_patch1_14
    """

    model = getattr(pretrained, model_name)(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


model = load_model("resnet18")
device = next(model.parameters()).device
img = Image.open("test_imgs/2_birds.jpg").convert("RGB")
transform_class = model.transform

print(transform_class.transforms.transforms[:-2])

raise

def add_uniform_int_noise(pil_img: Image.Image, epsilon: int = 8) -> Image.Image:
    img_np = np.asarray(pil_img).astype(np.int16)
    noise = np.random.randint(-epsilon, epsilon + 1, size=img_np.shape, dtype=np.int16)
    noisy_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_np)


img_tensor = model.transform(img).unsqueeze(0).to(device).requires_grad_(True)
results_clean = model.explain(img_tensor, 12)

noisy_img = add_uniform_int_noise(img, epsilon=8)
noisy_tensor = model.transform(noisy_img).unsqueeze(0).to(device).requires_grad_(True)
results_noisy = model.explain(noisy_tensor, 12)

clean_map = results_clean["explanation"]
noisy_map = results_noisy["explanation"]

plt.imshow(clean_map)
plt.show()
plt.imshow(noisy_map)
plt.show()









