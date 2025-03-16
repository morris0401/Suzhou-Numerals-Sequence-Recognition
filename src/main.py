import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CRNN
from data import prepare_data, ocr_collate_fn
from train import train_model
from evaluation import evaluate_test_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a CRNN model for OCR.")
    parser.add_argument("--image_dir", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--rotation_degree", type=int, default=20, help="Rotation degree for data augmentation.")
    parser.add_argument("--brightness", type=float, default=0, help="Brightness range for augmentation.")
    parser.add_argument("--contrast", type=float, default=0, help="Contrast adjustment for augmentation.")
    parser.add_argument("--saturation", type=float, default=0, help="Saturation adjustment for augmentation.")
    parser.add_argument("--normalize_mean", type=float, default=0.5, help="Normalization mean.")
    parser.add_argument("--normalize_std", type=float, default=0.5, help="Normalization standard deviation.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    img_height, img_width = 32, 128
    map_to_seq_hidden_num = 64
    rnn_hidden_num = 256
    augmentation_ratio = 3
    padding = True

    train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char = prepare_data(
        args.image_dir, img_height, img_width, args.normalize_mean, args.normalize_std,
        args.rotation_degree, args.brightness, args.contrast, args.saturation,
        augmentation_ratio, padding
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ocr_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ocr_collate_fn)

    num_classes = len(char_to_idx) + 1
    model = CRNN(1, img_height, img_width, num_classes, map_to_seq_hidden=map_to_seq_hidden_num, rnn_hidden=rnn_hidden_num).to(device)

    train_model(model, train_loader, val_loader, device, args.num_epochs, args.lr)

    test_loss, cer = evaluate_test_model(model, test_loader, nn.CTCLoss(blank=0), device, idx_to_char)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"CER: {cer:.4f}")


if __name__ == "__main__":
    main()