import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class FullSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, trail_dir, state_dir, label_dir):
        self.trail_dir = Path(trail_dir)
        self.state_dir = Path(state_dir)
        self.label_dir = Path(label_dir)
        self.sample_ids = sorted([f.stem for f in self.trail_dir.glob("*.npy")])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        trail = np.load(self.trail_dir / f"{sample_id}.npy")   # (300, 2)
        state = np.load(self.state_dir / f"{sample_id}.npy")   # (300, 4)
        label = np.load(self.label_dir / f"{sample_id}.npy")   # (4,)

        input_seq = np.concatenate([trail, state], axis=1)     # (300, 6)

        return (
            torch.tensor(input_seq, dtype=torch.float32),      # (300, 6)
            torch.tensor(label, dtype=torch.float32)           # (4,)
        )

class PendulumFullLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=4):
        super(PendulumFullLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):  # x: [batch, 300, 6]
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]           # shape: [batch, hidden_size]
        return self.fc(last_hidden)    # output: [batch, 4]

# Folder paths
state_path = "dataset/states"
trail_path = "dataset/trails"
label_path = "dataset/labels"

# Dataset
full_dataset = FullSequenceDataset(trail_path, state_path, label_path)

# Split
n = len(full_dataset)
train_size = int(0.8 * n)
val_size = int(0.1 * n)
test_size = n - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

# Dataloaders
BATCH_SIZE = 32  # Increased due to lower input size per step
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PendulumFullLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

#training
NUM_EPOCHS = 25

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs, targets = batch
        inputs = inputs.to(device)     # [B, 300, 6]
        targets = targets.to(device)   # [B, 4]

        optimizer.zero_grad()
        outputs = model(inputs)        # [B, 4]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


