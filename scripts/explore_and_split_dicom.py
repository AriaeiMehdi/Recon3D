import os
import random
import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = r"D:\Dr.EntezarMahdi\3D-Recon\data"
output_dir = r"D:\Dr.EntezarMahdi\3D-Recon\data_split"
os.makedirs(output_dir, exist_ok=True)

dcm_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dcm')])
print(f"Found {len(dcm_files)} DICOM files")

# Load first file to understand structure
sample_path = os.path.join(data_dir, dcm_files[0])
ds = pydicom.dcmread(sample_path)
print(f"\n--- Sample DICOM info ---")
print(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
print(f"Rows: {getattr(ds, 'Rows', 'N/A')}")
print(f"Columns: {getattr(ds, 'Columns', 'N/A')}")
print(f"NumberOfFrames: {getattr(ds, 'NumberOfFrames', 'N/A')}")
print(f"PixelData shape: {ds.pixel_array.shape}")
print(f"PixelData dtype: {ds.pixel_array.dtype}")

# Check a few files to confirm shape
print(f"\n--- Checking shapes ---")
for i in [0, 1, 10, 100]:
    ds = pydicom.dcmread(os.path.join(data_dir, dcm_files[i]))
    print(f"  [{i}] shape={ds.pixel_array.shape}")

# Process: reshape (512, 64, 64) -> (16, 32, 64, 64) -> 16 frames of (32, 64, 64)
print(f"\n--- Processing {len(dcm_files)} files ---")
total_volumes = 0
n_slices = 32
n_frames = 16

for i, fname in enumerate(dcm_files):
    fpath = os.path.join(data_dir, fname)
    ds = pydicom.dcmread(fpath)
    pixel_data = ds.pixel_array

    if pixel_data.shape == (512, 64, 64):
        # Reshape: (512, 64, 64) -> (16, 32, 64, 64)
        pixel_data = pixel_data.reshape(n_frames, n_slices, 64, 64)
    elif pixel_data.shape == (16, 32, 64, 64):
        pass  # Already correct shape
    else:
        print(f"  [{i+1}] WARNING: unexpected shape {pixel_data.shape}, skipping")
        continue

    for frame_idx in range(n_frames):
        volume = pixel_data[frame_idx].astype(np.float32)  # (32, 64, 64)

        # Normalize: min-max to [0, 1] per volume
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)

        base_name = fname.replace('.dcm', '')
        out_path = os.path.join(output_dir, f"{base_name}_frame{frame_idx:02d}.npy")
        np.save(out_path, volume)
        total_volumes += 1

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(dcm_files)}] processed, {total_volumes} volumes saved")

print(f"\n--- Done! Saved {total_volumes} volumes to {output_dir} ---")

# Plot a slice from a random volume
saved_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
if saved_files:
    random_file = random.choice(saved_files)
    vol = np.load(os.path.join(output_dir, random_file))
    print(f"\n--- Plotting random volume: {random_file} ---")
    print(f"Volume shape: {vol.shape}")
    print(f"Value range: [{vol.min():.2f}, {vol.max():.2f}]")

    mid_slice = vol.shape[0] // 2
    slice_2d = vol[mid_slice]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view (middle slice)
    axes[0].imshow(slice_2d, cmap='hot')
    axes[0].set_title(f'Axial (slice {mid_slice}/{vol.shape[0]})')
    axes[0].axis('off')

    # Coronal view
    mid_cor = vol.shape[1] // 2
    cor_slice = vol[:, mid_cor, :]
    axes[1].imshow(cor_slice, cmap='hot')
    axes[1].set_title(f'Coronal (slice {mid_cor}/{vol.shape[1]})')
    axes[1].axis('off')

    # Sagittal view
    mid_sag = vol.shape[2] // 2
    sag_slice = vol[:, :, mid_sag]
    axes[2].imshow(sag_slice, cmap='hot')
    axes[2].set_title(f'Sagittal (slice {mid_sag}/{vol.shape[2]})')
    axes[2].axis('off')

    plt.suptitle(f'{random_file}\nShape: {vol.shape}, Range: [{vol.min():.2f}, {vol.max():.2f}]', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(r"D:\Dr.EntezarMahdi\3D-Recon", "sample_spect_slice.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()
