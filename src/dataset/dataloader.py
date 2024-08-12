import sys
import numpy as np

sys.path.append("..")

from monai.data import (
    DataLoader,
    list_data_collate,
    DistributedSampler,
)

from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ToTensord,
    SpatialPadd,
)

from src.utils.constants import NUM_CLASS, TEMPLATE
from src.dataset.transforms import BatchWholeAndCropsd, MaskedOneHotd
from src.dataset.datasets import get_training_dataset, get_eval_dataset


def get_loader(args):
    num_class = NUM_CLASS if args.num_class == 0 else args.num_class
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "post_label"], dtype=np.float32),
            AddChanneld(keys=["image", "post_label"]),
            Orientationd(keys=["image", "post_label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "post_label"], source_key="image"),
            SpatialPadd(
                keys="image",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
                value=0,
            ),  # image non-valid region = 0
            SpatialPadd(
                keys="post_label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
                value=num_class + 1,  # mask is NUM_CLASS + 1
            ),
            BatchWholeAndCropsd(
                keys=["image", "post_label"],
                label_key="post_label",
                name_key="task",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                template=TEMPLATE,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
                mode=["bilinear", "nearest"],
                mask=num_class + 1,
            ),  # fails if any dimension of the image is smaller than the roi
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            MaskedOneHotd(
                keys=["post_label"],
                classes=num_class + 1,  # + bg
                categories=[],
            ),
            ToTensord(keys=["image", "post_label"]),
        ]
    )

    eval_transforms = Compose(
        [
            LoadImaged(keys=["image", "post_label"]),
            AddChanneld(keys=["image", "post_label"]),
            Orientationd(keys=["image", "post_label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            AsDiscreted(keys=["post_label"], to_onehot=num_class + 1),  # + bg
            ToTensord(keys=["image", "post_label"]),
        ]
    )

    # Dataset dicts (can eliminate the 2nd key and use args.phase)
    if args.phase == "train":
        train_dataset = get_training_dataset(args, train_transforms)
        train_sampler = (
            DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True)
            if args.dist
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            collate_fn=list_data_collate,
            sampler=train_sampler,
            drop_last=True,
        )
        return train_loader, train_sampler

    if args.phase == "val":
        val_dataset = get_eval_dataset(args, eval_transforms)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=list_data_collate,
        )
        return val_loader, eval_transforms

    if args.phase == "test":
        test_dataset = get_eval_dataset(args, eval_transforms)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=list_data_collate,
        )
        return test_loader, eval_transforms


if __name__ == "__main__":
    train_loader, train_sampler = get_loader()
    for index, item in enumerate(train_loader):
        print(item["image"].shape, item["label"].shape, item["task_id"])
        input()
