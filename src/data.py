import numpy as np
import torch
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def padding_image(img, max_height, max_width):
    old_image_height, old_image_width, channels = img.shape

    new_image_width = max_width + 100
    new_image_height = max_height + 50

    scale_ratio = new_image_height / old_image_height
    resized_width = int(old_image_width * scale_ratio)

    if resized_width > new_image_width:
        scale_ratio = new_image_width / old_image_width
        resized_width = new_image_width
        new_image_height = int(old_image_height * scale_ratio)

    resized_img = cv2.resize(img, (resized_width, new_image_height), interpolation=cv2.INTER_AREA)

    color = (0, 0, 0)
    result = np.full((max_height + 50, max_width + 100, channels), color, dtype=np.uint8)

    x_center = (new_image_width - resized_width) // 2
    y_center = (max_height + 50 - new_image_height) // 2

    result[y_center:y_center + new_image_height, x_center:x_center + resized_width] = resized_img

    return result

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, max_height, max_width, transform=None, padding=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.padding=padding
        self.max_height=max_height
        self.max_width=max_width

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)

        if self.padding:
            image = padding_image(image, self.max_height, self.max_width)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        if self.transform:
            image = self.transform(image)

        return image, label


# 自定義 collate_fn
def ocr_collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)

    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = pad_sequence([torch.tensor(label, dtype=torch.long) for label in labels], batch_first=True, padding_value=0)

    return images, labels, label_lengths

def prepare_data(img_dir, img_height, img_width, normalize_mean=0.5, normalize_std=0.5, rotation_degree=10, brightness=0.1, contrast=0, saturation=0, augmentation_ratio=3, padding=False):
    image_dir = os.path.join(img_dir, "images")
    csv_dir = os.path.join(img_dir, "csv")
    
    max_height = 0
    max_width = 0
    for filename in os.listdir('./data/images'):
        image = cv2.imread(os.path.join('./data/images', filename))
    if image.shape[0] > max_height:
        max_height = image.shape[0]
    if image.shape[1] > max_width:
        max_width = image.shape[1]

    data_splits = {}
    for split in ["train", "test", "val"]:
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        df = pd.read_csv(csv_path)
        data_splits[split] = [(os.path.join(image_dir, row["image_name"]), row["label"]) for _, row in df.iterrows()]
    
    train_data, val_data, test_data = data_splits["train"], data_splits["val"], data_splits["test"]
    
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    test_paths, test_labels = zip(*test_data)

    train_paths = [x for x in train_paths]
    test_paths = [x for x in test_paths]
    val_paths = [x for x in val_paths]


    labels_ = [str(x) for x in train_labels]
    all_chars = sorted(set(''.join(labels_)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    train_labels = [str(x) for x in train_labels]
    test_labels = [str(x) for x in test_labels]
    val_labels = [str(x) for x in val_labels]

    train_labels = [[char_to_idx[x] for x in label] for label in train_labels]
    test_labels = [[char_to_idx[x] for x in label] for label in test_labels]
    val_labels = [[char_to_idx[x] for x in label] for label in val_labels]
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_height, img_width)),
        transforms.Normalize(mean=[normalize_mean], std=[normalize_std]),
        transforms.RandomAffine(degrees=rotation_degree),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    ])
    
    transform_test_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_height, img_width)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = OCRDataset(train_paths, train_labels, max_height, max_width, transform_train, padding=padding)
    val_dataset = OCRDataset(val_paths, val_labels, max_height, max_width, transform_test_val, padding=padding)
    test_dataset = OCRDataset(test_paths, test_labels, max_height, max_width, transform_test_val, padding=padding)
    
    for _ in range(augmentation_ratio - 1):
      train_dataset += OCRDataset(train_paths, train_labels, max_height, max_width, transform_train, padding=padding)

    return train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char