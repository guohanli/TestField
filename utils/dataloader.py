import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

crop_size = 704
train_data_root = os.path.join(os.path.dirname(__file__), "../Dataset/TrainDataset")
test_data_root = os.path.join(
    os.path.dirname(__file__), "../Dataset/TestDataset/COD10K"
)

img_transforms = transforms.Compose(
    [
        transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        transforms.Resize(size=(crop_size, crop_size), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

gt_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Resize(size=(crop_size, crop_size), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

gt_test_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

data_augmentation_transforms = transforms.Compose(
    [
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ]
)


class CODDataset(Dataset):
    def __init__(
        self,
        root,
        transforms,
        target_transforms,
        use_augmentation=True,
    ):
        image_root = os.path.join(root, "Imgs")
        gt_root = os.path.join(root, "GT")
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.use_augmentation = use_augmentation

        self.img_paths = [
            os.path.join(image_root, filename)
            for filename in os.listdir(image_root)
            if filename.endswith(".jpg")
        ]

        self.gt_paths = [
            os.path.join(gt_root, filename)
            for filename in os.listdir(gt_root)
            if filename.endswith(".png")
        ]

        self.img_paths = sorted(self.img_paths)
        self.gt_paths = sorted(self.gt_paths)

        assert len(self.img_paths) == len(
            self.gt_paths
        ), f"Number of images and ground truths must be the same, but got {len(self.img_paths)} images and {len(self.gt_paths)} ground truths."

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]

        # Load image and ground truth
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        if self.use_augmentation:
            # color jitter
            img = transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            )(img)
            img, gt = data_augmentation_transforms(img, gt)

        img = self.transforms(img)
        gt = self.target_transforms(gt)
        return {"img_path": img_path, "gt_path": gt_path, "img": img, "gt": gt}

    def __len__(self) -> int:
        return len(self.img_paths)


def get_train_dataloader(
    batch_size=16, train_data_root=train_data_root, use_augmentation=False
):
    train_dataset = CODDataset(
        root=train_data_root,
        transforms=img_transforms,
        target_transforms=gt_transforms,
        use_augmentation=use_augmentation,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_dataloader


def get_test_dataloader(test_data_root=test_data_root):
    test_dataset = CODDataset(
        root=test_data_root,
        transforms=img_transforms,
        target_transforms=gt_test_transforms,
        use_augmentation=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return test_dataloader


if __name__ == "__main__":
    dataset = CODDataset("../Dataset/TrainDataset", use_augmentation=False)
    img, gt = dataset[0]
    print(img.shape, gt.shape) 

    print(img)
    # 打印gt中不为0和1的值
    print(gt[(gt != 0) & (gt != 1)])
    print(len(gt[(gt != 0) & (gt != 1)]))
    print(len(dataset))
