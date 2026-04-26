import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from albumentations import (
    HorizontalFlip, RandomCrop, Rotate, RandomRotate90, Affine,
    RandomBrightnessContrast, HueSaturationValue, CoarseDropout, Compose, OneOf
)

def get_transforms():
    return Compose([
        OneOf([
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Rotate(limit=20, p=0.5)
        ], p=0.8),
        Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        RandomCrop(height=96, width=96, p=0.5),
        OneOf([
            RandomBrightnessContrast(p=0.5),
            HueSaturationValue(p=0.5)
        ], p=0.7),
        CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.3)
    ])

def augment_class_to_target(input_dir, output_dir, target_count):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, '*'))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    transform = get_transforms()
    counter = 0
    base_count = len(image_paths)

    # Copy original images
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (112, 112))
        save_path = os.path.join(output_dir, f"orig_{counter}.jpg")
        cv2.imwrite(save_path, img)
        counter += 1

    # Augmentation
    aug_needed = target_count - base_count
    if aug_needed > 0:
        with tqdm(total=aug_needed, desc=f"{os.path.basename(input_dir)}", unit="img") as pbar:
            for i in range(aug_needed):
                img_path = random.choice(image_paths)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (112, 112))
                augmented = transform(image=img)['image']

                if augmented.dtype != np.uint8:
                    augmented = (augmented * 255).astype(np.uint8)

                save_path = os.path.join(output_dir, f"aug_{i}.jpg")
                cv2.imwrite(save_path, augmented)
                pbar.update(1)

def main():
    base_path = r"path"

    input_root = os.path.join(base_path, "Original dataset")
    output_root = os.path.join(base_path, "Augmented dataset")

    os.makedirs(output_root, exist_ok=True)

    # Only 2 classes now
    class_names = ["Healthy", "PD"]

    total_target = 2000
    per_class_target = total_target // len(class_names)  

    for class_name in class_names:
        in_dir = os.path.join(input_root, class_name)
        out_dir = os.path.join(output_root, class_name)

        if not os.path.exists(in_dir):
            print(f"Missing folder: {in_dir}")
            continue

        augment_class_to_target(in_dir, out_dir, per_class_target)

if __name__ == "__main__":
    main()