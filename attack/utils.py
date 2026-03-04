from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from bcos.models import pretrained
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from bcos.models import pretrained


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

    if not hasattr(pretrained, model_name):
        raise ValueError(f"Unknown model_name '{model_name}'")

    model = getattr(pretrained, model_name)(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

def save_map_image(explained_map, output_path):
    img = np.asarray(explained_map)
    assert img.ndim == 3  # (H, W, C)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False
    )
    plt.close()

    print(f"Saved image to {output_path}")

def seed_everything(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)






