import torch
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.transforms.functional import to_pil_image


class CIFAR10(Dataset):
    def __init__(self) -> None:
        self.transform = T.Compose(
            [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(0.5, 0.5)]
        )

        dataset_dir = "data/cifar10"
        self.dataset = datasets.CIFAR10(
            root=dataset_dir, train=True, download=True, transform=self.transform
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        return image

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    dataset = CIFAR10()
    artifacts_dir = "artifacts/dataset"
    os.makedirs(artifacts_dir, exist_ok=True)
    for i in range(10):
        image = dataset[i]
        image_pil = to_pil_image(image * 0.5 + 0.5)
        image_pil.save(os.path.join(artifacts_dir, f"{i}.png"))
