import os
from pathlib import Path
from typing import Optional, Union
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


def load_radimagenet_resnet50(checkpoint_path: Union[str, Path]) -> nn.Module:
    """
    Instantiates ResNet50 and loads RadImageNet pretrained weights,
    remapping sequential backbone keys to standard torchvision names.
    """
    model = models.resnet50(weights=None)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RadImageNet checkpoint not found at: {checkpoint_path}\n"
            f"Please specify a valid path via --radimagenet_path or ensure the file exists."
        )

    logging.info(f"Loading RadImageNet weights from: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint.get("net", checkpoint)))
    else:
        state_dict = checkpoint

    # Key remapping from backbone sequential indices to standard ResNet names
    key_mapping = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    }

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        for old_prefix, new_prefix in key_mapping.items():
            if k.startswith(old_prefix):
                k = k.replace(old_prefix, new_prefix, 1)
                break
        cleaned_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    logging.info(f"Loaded RadImageNet backbone. Unmatched keys: {missing_keys}")

    model.fc = nn.Identity()
    return model


class RadImageNetFeatureExtractor(nn.Module):
    """
    Feature extractor module using a ResNet50 backbone pretrained on RadImageNet.
    Outputs 2048-dimensional feature vectors for Frechet Inception Distance (FID).
    """
    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        super().__init__()

        if checkpoint_path is None:
            default_candidates = [
                os.path.expandvars("$HOME/tools/RadImageNet/RadImageNet_pytorch/ResNet50.pt"),
                os.path.expandvars("$HOME/Downloads/RadImageNet/RadImageNet_pytorch/ResNet50.pt"),
                os.path.expandvars("$HOME/data/models/RadImageNet/ResNet50.pt"),
                os.path.expandvars("$HOME/models/RadImageNet/ResNet50.pt"),
                "RadImageNet_pytorch/ResNet50.pt",
                "ResNet50.pt"
            ]
            for candidate in default_candidates:
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break

            if checkpoint_path is None:
                checkpoint_path = default_candidates[0]

        self.backbone = load_radimagenet_resnet50(checkpoint_path)
        self.backbone.eval()

        # Standard RadImageNet / ImageNet preprocessing
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.target_size = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size, antialias=True),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image tensor.
        :param x: Float tensor with values in [0, 1]. Shape (B, C, H, W) or (B, 1, H, W).
        :return: Feature embeddings of shape (B, 2048).
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Grayscale -> RGB (3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]

        x = self.transform(x)

        with torch.no_grad():
            features = self.backbone(x)

        if features.ndim > 2:
            features = torch.flatten(features, 1)

        return features
