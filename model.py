"""Sample Model class for track 1."""

import torch
from torch import nn
from torchvision.transforms import v2

# We use PatchcoreModel for an example. You can replace it with your model.
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


class Patchcore(nn.Module):

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer1", "layer2", "layer3"], 
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.transform = v2.Compose(
            [
                v2.Resize((256, 256)),
                v2.CenterCrop((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
            ],
        )

        self.model = PatchcoreModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )

    def forward(self, batch: torch.Tensor):
        batch = self.transform(batch)
        return self.model(batch)
