import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import sys
import random

data_dir = r"D:\Dr.EntezarMahdi\3D-Recon\data_split_cropped"
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
print(f"Found {len(files)} volumes. Loading random one...")

vol = np.load(os.path.join(data_dir, random.choice(files)))
print(f"Shape: {vol.shape}, Range: [{vol.min():.2f}, {vol.max():.2f}], Mean: {vol.mean():.4f}")

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)
mid = vol.shape[0] // 2
im = ax.imshow(vol[mid], cmap='hot', vmin=vol.min(), vmax=vol.max())
ax.set_title(f'Slice {mid}/{vol.shape[0]-1}')
ax.axis('off')

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 'Slice', 0, vol.shape[0] - 1, valinit=mid, valstep=1)

def update(val):
    idx = int(slider.val)
    im.set_data(vol[idx])
    ax.set_title(f'Slice {idx}/{vol.shape[0]-1}')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
