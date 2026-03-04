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

cat_img = Image.open("cat.png").convert("RGB")
dog_img = Image.open("dog.jpg").convert("RGB")

print(model.transform)
batch = torch.stack(
    [model.transform(cat_img), model.transform(dog_img)], dim=0
).to(device).requires_grad_()

results = model.explain(batch)
logits = results.get("logits", results.get("logit"))

if logits is None:
    raise KeyError("Không thấy key 'logits'/'logit' trong kết quả explain().")

print("batch shape:", batch.shape)
print("logits shape:", logits.shape)
print("pred idx (cat, dog):", results["prediction"])
print("map shape:", results["contribution_map"].shape)
print("explanation shape:", results["explanation"].shape) # batch x w x h x 4
print("explanation dtype:", results["explanation"].dtype) # float
plt.imshow(results["explanation"][0])  # show explanation for cat
plt.show()





