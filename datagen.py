import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

# Constants
g = 9.81
L1 = 1.5
L2 = 1.5
m1 = 1.0
m2 = 1.0
T = 5
FPS = 60
t_eval = np.linspace(0, T, T * FPS)
max_len = L1 + L2

# Output paths
OUT_DIR = Path("dataset")
(OUT_DIR / "trails").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "states").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

def equations(t, y):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δ)**2
    den2 = (L2 / L1) * den1

    dω1 = (m2 * L1 * ω1**2 * np.sin(Δ) * np.cos(Δ)
           + m2 * g * np.sin(θ2) * np.cos(Δ)
           + m2 * L2 * ω2**2 * np.sin(Δ)
           - (m1 + m2) * g * np.sin(θ1)) / den1

    dω2 = (-m2 * L2 * ω2**2 * np.sin(Δ) * np.cos(Δ)
           + (m1 + m2) * g * np.sin(θ1) * np.cos(Δ)
           - (m1 + m2) * L1 * ω1**2 * np.sin(Δ)
           - (m1 + m2) * g * np.sin(θ2)) / den2

    return [ω1, dω1, ω2, dω2]

def normalize(x, y):
    x = x / (2 * max_len) + 0.5
    y = y / (2 * max_len) + 0.5
    return np.stack([x, y], axis=1)

def generate_sample(i):
    np.random.seed(i)
    θ1 = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
    θ2 = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
    ω1 = np.random.uniform(-1, 1)
    ω2 = np.random.uniform(-1, 1)
    y0 = [θ1, ω1, θ2, ω2]

    sol = solve_ivp(equations, [0, T], y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    θ1_sol, ω1_sol, θ2_sol, ω2_sol = sol.y

    # Tip coordinates
    x1 = L1 * np.sin(θ1_sol)
    y1 = -L1 * np.cos(θ1_sol)
    x2 = x1 + L2 * np.sin(θ2_sol)
    y2 = y1 - L2 * np.cos(θ2_sol)

    # Normalize trail
    trail = normalize(x2, y2)  # (300, 2)
    state_seq = np.stack([θ1_sol, ω1_sol, θ2_sol, ω2_sol], axis=1)  # (300, 4)
    label = np.array([θ1, ω1, θ2, ω2], dtype=np.float32)  # (4,)

    # Save files
    np.save(OUT_DIR / "trails" / f"{i:05d}.npy", trail)
    np.save(OUT_DIR / "states" / f"{i:05d}.npy", state_seq)
    np.save(OUT_DIR / "labels" / f"{i:05d}.npy", label)

    return True

# Config
NUM_SAMPLES = 10000
NUM_WORKERS = min(cpu_count(), 8)

if __name__ == "__main__":
    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(generate_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))
    print("Dataset generation complete.")
