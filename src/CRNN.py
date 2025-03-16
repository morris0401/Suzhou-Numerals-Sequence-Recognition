import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import pandas as pd

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 確保每次運行的結果一致
    torch.backends.cudnn.benchmark = False    # 禁用高性能搜尋，保證結果穩定


max_height = 0
max_width = 0
for filename in os.listdir('./Suzhou_num/data'):
  image = cv2.imread(os.path.join('./Suzhou_num/data', filename))
  if image.shape[0] > max_height:
    max_height = image.shape[0]
  if image.shape[1] > max_width:
    max_width = image.shape[1]

print(max_height, max_width)

def padding_image(img, max_height, max_width):
    old_image_height, old_image_width, channels = img.shape

    # 設定新圖片的尺寸（避免超過範圍）
    new_image_width = max_width + 100
    new_image_height = max_height + 50

    # 計算等比例縮放後的尺寸
    scale_ratio = new_image_height / old_image_height
    resized_width = int(old_image_width * scale_ratio)

    # 如果縮放後寬度超過 new_image_width，則根據寬度縮放
    if resized_width > new_image_width:
        scale_ratio = new_image_width / old_image_width
        resized_width = new_image_width
        new_image_height = int(old_image_height * scale_ratio)

    # 縮放圖片
    resized_img = cv2.resize(img, (resized_width, new_image_height), interpolation=cv2.INTER_AREA)

    # 創建填充畫布
    color = (0, 0, 0)
    result = np.full((max_height + 50, max_width + 100, channels), color, dtype=np.uint8)

    # 計算置中偏移量
    x_center = (new_image_width - resized_width) // 2
    y_center = (max_height + 50 - new_image_height) // 2

    # 確保不超出範圍
    result[y_center:y_center + new_image_height, x_center:x_center + resized_width] = resized_img

    return result

import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)


# 自定義資料集類別
class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 讀取影像
        #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        #image = padding_image(image, max_height, max_width)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)

        return image, label


# 自定義 collate_fn
def ocr_collate_fn(batch):
    images, labels = zip(*batch)

    # 將影像轉換為張量 (batch_size, channel, height, width)
    images = torch.stack(images, dim=0)

    # 將標籤填充到相同長度
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = pad_sequence([torch.tensor(label, dtype=torch.long) for label in labels], batch_first=True, padding_value=0)

    return images, labels, label_lengths

def prepare_data(img_dir, img_height, img_width, test_size=0.1, val_size=0.2, normalize_mean=0.5, normalize_std=0.5, rotation_degree=10, brightness=0.1, contrast=0, saturation=0, augmentation_ratio=3):
    image_dir = os.path.join(img_dir, "images")
    csv_dir = os.path.join(img_dir, "csv")
    
    # 讀取 CSV 資料
    data_splits = {}
    for split in ["train", "test", "val"]:
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        df = pd.read_csv(csv_path)
        data_splits[split] = [(os.path.join(image_dir, row["image_name"]), row["label"]) for _, row in df.iterrows()]
    
    train_data, val_data, test_data = data_splits["train"], data_splits["val"], data_splits["test"]
    
    # 分離影像路徑與標籤
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    test_paths, test_labels = zip(*test_data)

    train_paths = [x for x in train_paths]
    test_paths = [x for x in test_paths]
    val_paths = [x for x in val_paths]
    print(train_paths)

    # 轉換標籤為索引
    #labels_ = [[int(y) for y in str(x)] for x in train_labels]
    labels_ = [str(x) for x in train_labels]
    all_chars = sorted(set(''.join(labels_)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}  # 0 為空白
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    label_indices = [[char_to_idx[c] for c in label] for label in labels_]

    train_labels = [str(x) for x in train_labels]
    test_labels = [str(x) for x in test_labels]
    val_labels = [str(x) for x in val_labels]

    train_labels = [[char_to_idx[x] for x in label] for label in train_labels]
    test_labels = [[char_to_idx[x] for x in label] for label in test_labels]
    val_labels = [[char_to_idx[x] for x in label] for label in val_labels]
    
    # 定義資料增強與標準化
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
    
    # 建立資料集
    train_dataset = OCRDataset(train_paths, train_labels, transform_train)
    val_dataset = OCRDataset(val_paths, val_labels, transform_test_val)
    test_dataset = OCRDataset(test_paths, test_labels, transform_test_val)
    
    for i in range(augmentation_ratio - 1):
      train_dataset += OCRDataset(train_paths, train_labels, transform_train)

    return train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char


# 資料預處理
def prepare_data_(image_dir, img_height, img_width, test_size=0.1, val_size=0.2, normalize_mean=0.5, normalize_std=0.5, rotation_degree=10, brightness=0.1, contrast=0, saturation=0, augmentation_ratio=3):
    image_paths = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            label = filename.split('_')[0]  # 檔名的數字部分
            image_paths.append(os.path.join(image_dir, filename))
            labels.append(label)

    # 轉換標籤為索引
    all_chars = sorted(set(''.join(labels)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}  # 0 為空白
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    label_indices = [[char_to_idx[c] for c in label] for label in labels]

    # 初步分割成訓練與非訓練資料
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, label_indices, test_size=test_size + val_size, random_state=42
    )

    # 再將非訓練資料分割為驗證與測試資料
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size / (test_size + val_size), random_state=42
    )

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

    train_dataset = OCRDataset(train_paths, train_labels, transform_test_val)
    val_dataset = OCRDataset(val_paths, val_labels, transform_test_val)
    test_dataset = OCRDataset(test_paths, test_labels, transform_test_val)

    #print(f"Train dataset size: {len(train_dataset)}")
    #print(f"Validation dataset size: {len(val_dataset)}")
    #print(f"Test dataset size: {len(test_dataset)}")
    #print(f"Number of classes: {len(char_to_idx)}")
    #print(f"Character to index mapping: {char_to_idx}")
    #print(f"Index to character mapping: {idx_to_char}")

    def plot_character_statistics(dataset, dataset_name):
        # 計算每個 character 的出現次數
        character_counts = Counter([char for label in dataset.labels for char in label])
        # 確保順序為 0-9
        sorted_characters = [idx_to_char[i + 1] for i in range(10)]  # idx_to_char 的索引從 1 開始
        sorted_counts = [character_counts.get(i + 1, 0) for i in range(10)]  # 將不存在的 character 設為 0
        #characters = [idx_to_char[idx] for idx in character_counts.keys()]
        #counts = character_counts.values()
        characters = sorted_characters
        counts = sorted_counts

        plt.figure(figsize=(12, 6))
        plt.bar(characters, counts)
        plt.title(f'{dataset_name} Character Frequency')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 計算每個 label 的長度頻率
        label_lengths = [len(label) for label in dataset.labels]
        length_counts = Counter(label_lengths)

        plt.figure(figsize=(12, 6))
        plt.bar(length_counts.keys(), length_counts.values())
        plt.title(f'{dataset_name} Label Length Distribution')
        plt.xlabel('Label Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    #plot_character_statistics(train_dataset, 'Train Dataset')
    #plot_character_statistics(val_dataset, 'Validation Dataset')
    #plot_character_statistics(test_dataset, 'Test Dataset')

    for i in range(augmentation_ratio - 1):
      train_dataset += OCRDataset(train_paths, train_labels, transform_train)
    #print(f"Augmented train dataset size: {len(train_dataset)}")

    return train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char



def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i in range(1):
          for images, labels, label_lengths in train_loader:
              images = images.to(device)
              labels = labels.to(device)
              label_lengths = label_lengths.to(device)

              # 預測
              outputs = model(images)
              outputs = outputs.log_softmax(2)
              seq_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

              # 計算損失
              loss = criterion(outputs, labels, seq_lengths, label_lengths)
              train_loss += loss.item()

              # 更新權重
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

        # 驗證
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels, label_lengths in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(images)
            outputs = outputs.log_softmax(2)
            seq_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels, seq_lengths, label_lengths)
            val_loss += loss.item()

    return val_loss / len(data_loader)

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]


def evaluate_test_model(model, data_loader, criterion, device, idx_to_char):
    model.eval()
    val_loss = 0.0
    total_cer = 0
    total_chars = 0  # 用於計算總字符數

    with torch.no_grad():
        for images, labels, label_lengths in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            # 模型輸出
            outputs = model(images)  # (T, N, C)
            outputs = outputs.log_softmax(2)
            seq_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            # 計算損失
            loss = criterion(outputs, labels, seq_lengths, label_lengths)
            val_loss += loss.item()

            # 解碼：取最大機率字符索引 (T, N, C) -> (T, N)
            decoded_preds = torch.argmax(outputs, dim=2).permute(1, 0)  # (N, T)

            for i, pred in enumerate(decoded_preds):
                # 去除重複字符及空白字符 (CTC 解碼)
                pred_text = []
                for j in range(len(pred)):
                    if j == 0 or pred[j] != pred[j - 1]:  # 移除連續重複字符
                        if pred[j] != 0:  # 移除空白字符
                            pred_text.append(idx_to_char[pred[j].item()])
                pred_text = ''.join(pred_text)

                # 轉換 Ground Truth (label)
                gt_text = ''.join([idx_to_char[c.item()] for c in labels[i][:label_lengths[i]]])

                # 計算 CER
                cer = levenshtein_distance(pred_text, gt_text) / max(len(gt_text), 1)
                total_cer += cer
                total_chars += len(gt_text)

    avg_loss = val_loss
    avg_cer = total_cer / 116

    #print(f"Validation Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")
    return avg_loss, avg_cer
