import os
import numpy as np
import cv2
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import pandas as pd
from pathlib import Path

from utils import create_dir

# Train, Val, Test split: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
def load_data(path: Path, split=(70, 15, 15), shuffle=True, max_items=-1):
    if not isinstance(sum(split), int):
        raise ValueError("split parameter contain ints")
    if len(split) not in (2, 3):
        raise ValueError("split parameter must contain 2 or 3 elements")
    if sum(split) != 100:
        raise ValueError("split parameter must sum to 100")

    validated_images_csv = "paths_df_validated_0_to_6500_8000_to_8499.csv"
    df = pd.read_csv(base_path / "data" / validated_images_csv)

    calc_true = df.apply(lambda x: x["valid"], axis=1)

    # get valid images
    df = df[calc_true == True]

    if shuffle:
        df = df.sample(frac=1)

    df = df.iloc[:max_items]

    inputs_img = df.iloc[:, 2]
    target_img = df.iloc[:, 3]

    inputs = inputs_img.to_numpy()
    targets = target_img.to_numpy()
    # inputs=base_path/'data'/'images'/inputs_img.to_numpy()
    # targets=base_path/'data'/'masks'/target_img.to_numpy()

    df_len = len(df)
    train_split = int(df_len * split[0] / 100)
    val_split = int(train_split + df_len * split[1] / 100) if len(split) != 2 else -1

    train_x = inputs[:train_split]
    train_y = targets[:train_split]
    val_x = inputs[train_split:val_split]
    val_y = targets[train_split:val_split]
    test_x = inputs[val_split:]
    test_y = targets[val_split:]

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def augment_data(images, masks, src_path, save_path, augment=True, size=(512, 512)):

    src_path_str = str(src_path)

    for idx, (x_file, y_file) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        x_image = cv2.imread(src_path_str + "/images/" + x_file, cv2.IMREAD_COLOR)
        y_image = cv2.imread(src_path_str + "/masks/" + y_file, cv2.IMREAD_GRAYSCALE)

        x_filename = x_file.split(".")[0]
        y_filename = y_file.split(".")[0]
        # print(x_image.shape)
        # print(y_image.shape)
        # print(y_image.max())
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x_image, mask=y_image)
            x_1 = augmented["image"]
            y_1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x_image, mask=y_image)
            x_2 = augmented["image"]
            y_2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x_image, mask=y_image)
            x_3 = augmented["image"]
            y_3 = augmented["mask"]

            X = [x_image, x_1, x_2, x_3]
            Y = [y_image, y_1, y_2, y_3]

        else:
            X = [x_image]
            Y = [y_image]

        for version, (i, m) in enumerate(zip(X, Y)):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)
            tmp_image_name = f"{x_filename}_{version}.jpeg"
            tmp_mask_name = f"{y_filename}_{version}.jpeg"

            image_path = str(save_path / "images" / tmp_image_name)
            mask_path = str(save_path / "masks" / tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(
        base_path, max_items=20
    )
    print(f"{len(train_x)=} = {len(train_y)=}")
    print(f"{len(val_x)=} = {len(val_y)=}")
    print(f"{len(test_x)=} = {len(test_y)=}")

    # directories for augmented data
    create_dir(base_path / "new_data/train/images/")
    create_dir(base_path / "new_data/train/masks/")
    create_dir(base_path / "new_data/val/images/")
    create_dir(base_path / "new_data/val/masks/")
    create_dir(base_path / "new_data/test/images/")
    create_dir(base_path / "new_data/test/masks/")

    # data augmentation
    augment_data(
        train_x,
        train_y,
        base_path / "data",
        base_path / "new_data/train",
        augment=False,
    )
    augment_data(
        val_x, val_y, base_path / "data", base_path / "new_data/val", augment=False
    )
    augment_data(
        test_x,
        test_y,
        base_path / "data",
        base_path / "new_data/test",
        augment=False,
    )
