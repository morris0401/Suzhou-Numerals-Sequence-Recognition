import torch

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
    total_chars = 0

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

            decoded_preds = torch.argmax(outputs, dim=2).permute(1, 0)

            for i, pred in enumerate(decoded_preds):
                pred_text = []
                for j in range(len(pred)):
                    if j == 0 or pred[j] != pred[j - 1]:
                        if pred[j] != 0:
                            pred_text.append(idx_to_char[pred[j].item()])
                pred_text = ''.join(pred_text)

                gt_text = ''.join([idx_to_char[c.item()] for c in labels[i][:label_lengths[i]]])

                cer = levenshtein_distance(pred_text, gt_text) / max(len(gt_text), 1)
                total_cer += cer
                total_chars += len(gt_text)

    val_loss = val_loss
    avg_cer = total_cer /  len(data_loader.dataset)

    return val_loss, avg_cer
