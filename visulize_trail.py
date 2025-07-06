import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Path to dataset
data_dir = Path("dataset")
sequence_dir = data_dir / "trails"
label_dir = data_dir / "labels"

# Pick a random sample
idx = random.randint(0, 9999)
sequence_path = sequence_dir / f"{idx:05d}.npy"
label_path = label_dir / f"{idx:05d}.npy"

# Load data
sequence = np.load(sequence_path)  # Shape: (300, 2)
label = np.load(label_path)        # Shape: (4,)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(sequence[:, 0], sequence[:, 1], marker='o', markersize=1, linewidth=1, alpha=0.8)
plt.title(f"Pendulum Trail (Sample {idx})\nInitial State: θ1={label[0]:.2f}, ω1={label[1]:.2f}, θ2={label[2]:.2f}, ω2={label[3]:.2f}")
plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.axis('equal')
plt.grid(True)
plt.show()
