import os

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms


class ImageWoof(ImageFolder):
    """
    Dataset class for ImageWoof dataset.
    """

    DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"

    def __init__(
        self,
        root: str,
        img_size: int,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        train=True,
    ):
        if os.path.exists(os.path.join(root, "imagewoof2-160")):
            root = os.path.join(root, "imagewoof2-160")
        elif os.path.exists(os.path.join(root, "imagewoof2-160.tgz")):
            os.system(f"tar zxvf {os.path.join(root, 'imagewoof2-160.tgz')}")
            root = os.path.join(root, "imagewoof2-160")
        else:
            download_url(self.DATASET_URL, ".")
            os.system(f"tar zxvf {os.path.join(root, 'imagewoof2-160.tgz')}")
            root = os.path.join(root, "imagewoof2-160")

        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )

        super().__init__(root, transform, target_transform, loader, is_valid_file)
