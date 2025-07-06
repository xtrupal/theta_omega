import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# System Parameters
g = 9.81
L1 = L2 = 1.5
m1 = m2 = 1.0
T = 7
fps = 60
t_eval = np.linspace(0, T, T * fps)

def equations(t, y):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δ)**2
    den2 = (L2 / L1) * den1

    dω1 = (m2 * L1 * ω1**2 * np.sin(Δ) * np.cos(Δ) +
           m2 * g * np.sin(θ2) * np.cos(Δ) +
           m2 * L2 * ω2**2 * np.sin(Δ) -
           (m1 + m2) * g * np.sin(θ1)) / den1

    dω2 = (-m2 * L2 * ω2**2 * np.sin(Δ) * np.cos(Δ) +
           (m1 + m2) * g * np.sin(θ1) * np.cos(Δ) -
           (m1 + m2) * L1 * ω1**2 * np.sin(Δ) -
           (m1 + m2) * g * np.sin(θ2)) / den2

    return [ω1, dω1, ω2, dω2]

def simulate_trail(params):
    sol = solve_ivp(equations, [0, T], params, t_eval=t_eval, rtol=1e-10, atol=1e-10)
    θ1, θ2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    return x1, y1, x2, y2


# Replace with your values

# Put the true state here
true_params = [1.3249, 0.8892, 2.3348, -0.8171]
# Put the predicted state here
pred_params = [1.3227, 0.8368, 2.3324, -0.8359]


x1_true, y1_true, x2_true, y2_true = simulate_trail(true_params)
x1_pred, y1_pred, x2_pred, y2_pred = simulate_trail(pred_params)


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
lines = []
trails = []

titles = ['Ground Truth', 'Predicted']
max_len = L1 + L2

for ax, title in zip(axs, titles):
    ax.set_xlim(-1.1 * max_len, 1.1 * max_len)
    ax.set_ylim(-1.1 * max_len, 1.1 * max_len)
    ax.set_title(title)
    ax.axis('off')
    ax.set_aspect('equal')

    line, = ax.plot([], [], 'o-', lw=2, color='black')
    trail, = ax.plot([], [], '-', lw=1, color='red', alpha=0.6)

    lines.append(line)
    trails.append(trail)

trail_x1, trail_y1 = [], []
trail_x2, trail_y2 = [], []

def update(frame):
    if frame == 0:
        trail_x1.clear()
        trail_y1.clear()
        trail_x2.clear()
        trail_y2.clear()

    # Add current point to trails
    trail_x1.append(x2_true[frame])
    trail_y1.append(y2_true[frame])
    trail_x2.append(x2_pred[frame])
    trail_y2.append(y2_pred[frame])

    # Update pendulum lines
    lines[0].set_data([0, x1_true[frame], x2_true[frame]],
                      [0, y1_true[frame], y2_true[frame]])
    lines[1].set_data([0, x1_pred[frame], x2_pred[frame]],
                      [0, y1_pred[frame], y2_pred[frame]])

    # Update trail lines
    trails[0].set_data(trail_x1, trail_y1)
    trails[1].set_data(trail_x2, trail_y2)

    return lines + trails

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=1000/fps, blit=True)
plt.tight_layout()

plt.show()
