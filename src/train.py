import torch
import torch
import torch.nn as nn
import torch.optim as optim

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

    return val_loss

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

              outputs = model(images)
              outputs = outputs.log_softmax(2)
              seq_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

              loss = criterion(outputs, labels, seq_lengths, label_lengths)
              train_loss += loss.item()

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")