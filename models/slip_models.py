import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn


SLIP_REPO_DIR = "/home/jamesyoung0623/SLIP"
DEFAULT_CKPT_PATH = os.path.join(SLIP_REPO_DIR, "slip_base_100ep.pt")

MODEL_FACTORIES = {
    "ViT-B/16": ("SLIP_VITB16", {"ssl_mlp_dim": 4096, "ssl_emb_dim": 256}),
    "ViT-L/16": ("SLIP_VITL16", {"ssl_mlp_dim": 4096, "ssl_emb_dim": 256}),
}

_SLIP_MODELS_MODULE = None


def _load_slip_models_module():
    global _SLIP_MODELS_MODULE
    if _SLIP_MODELS_MODULE is not None:
        return _SLIP_MODELS_MODULE

    models_py = os.path.join(SLIP_REPO_DIR, "models.py")
    if not os.path.exists(models_py):
        raise FileNotFoundError(f"SLIP models.py not found: {models_py}")

    if SLIP_REPO_DIR not in sys.path:
        sys.path.insert(0, SLIP_REPO_DIR)

    # facebookresearch/SLIP expects the pre-1.0 timm registry path.
    import timm
    if not hasattr(timm.models, "registry"):
        timm.models.registry = types.SimpleNamespace(register_model=timm.models.register_model)

    spec = importlib.util.spec_from_file_location("slip_repo_models", models_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SLIP module from {models_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SLIP_MODELS_MODULE = module
    return module


class SLIPModel(nn.Module):
    def __init__(self, name, ckpt_path=DEFAULT_CKPT_PATH, num_classes=1):
        super().__init__()
        del num_classes

        if name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported SLIP model: {name}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SLIP checkpoint not found: {ckpt_path}")

        print(f"[slip] importing SLIP repo model code from {SLIP_REPO_DIR}", flush=True)
        slip_models = _load_slip_models_module()
        factory_name, factory_kwargs = MODEL_FACTORIES[name]
        print(f"[slip] building model {factory_name}", flush=True)
        self.model = getattr(slip_models, factory_name)(**factory_kwargs)

        print(f"[slip] loading checkpoint {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=True)
        self.preprocess = None
        print("[slip] checkpoint loaded", flush=True)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return features
