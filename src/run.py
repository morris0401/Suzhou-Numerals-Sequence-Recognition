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
from CRNN import prepare_data, CRNN, ocr_collate_fn, train_model, evaluate_test_model, set_random_seed, prepare_data_

set_random_seed(42)

# 調整照片的長、寬
img_height, img_width = 32, 128 # 不須更動


# 訓練時的總訓練次數 (要訓練多少次)
num_epochs = 10 # 不須更動


# 每次更新模型權重的時候，要一次使用多少筆訓練資料
batch_size = 32 # 2 的倍數，可以嘗試 2, 4, 8, 16, 32, 64。


# 更新模型權重的量級
lr = 1e-4 # 可更動


# 模型中神經網路的節點數量
map_to_seq_hidden_num = 64 # 不須更動
rnn_hidden_num = 256 # 不須更動

# 要擴增多少倍的訓練資料
augmentation_ratio = 3 # 不須更動


# 訓練資料的平均值與標準差
normalize_mean = 0.5 # 可以嘗試 0-1.5
normalize_std = 0.5 # 可以嘗試 0-1.5


# 擴增的訓練資料可以旋轉多大的角度
rotation_degree = 0 # 可以嘗試 0-20

# 擴增的訓練資料明暗度差異範圍
brightness = 0 # 可以嘗試 0-0.3


# 擴增的訓練資料對比度
contrast = 0 # 可以嘗試 0-0.3


# 擴增的訓練資料飽和度
saturation = 0 # 可以嘗試 0-0.3


# test data 佔全部資料的比例
test_size = 0.15 # 不須更動


# evaluation data 佔全部資料的比例
val_size = 0.15 # 不須更動

padding = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_dir = "./Suzhou_Numerals/data"  # 替換為你的資料夾路徑
#image_dir = "./data_original"

train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char = prepare_data(
    image_dir, img_height, img_width, test_size, val_size, normalize_mean, normalize_std, rotation_degree, brightness, contrast, saturation, augmentation_ratio, padding
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)

num_classes = len(char_to_idx) + 1  # 加上空白
model = CRNN(1, img_height, img_width, num_classes, map_to_seq_hidden=map_to_seq_hidden_num, rnn_hidden=rnn_hidden_num).to(device)

train_model(model, train_loader, val_loader, device, num_epochs, lr)

# Optional: Test evaluation
test_loss, cer = evaluate_test_model(model, test_loader, nn.CTCLoss(blank=0), device, idx_to_char)
print(f"Test Loss: {test_loss:.4f}")
print(f"CER: {cer:.4f}")

# conda activate Sushou_num
# python ./Suzhou_num/src/run.py