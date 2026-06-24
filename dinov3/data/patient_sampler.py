"""Patient-aware infinite sampler for 3D DINOv3 SPECT training."""
import logging
import os
import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger("dinov3")


def extract_patient_id(filepath: str) -> str:
    """Extract patient ID from filename like '..._frame05.npy'."""
    basename = os.path.basename(filepath)
    # Patient ID is everything before the last _frameXX
    parts = basename.rsplit("_frame", 1)
    if len(parts) == 2:
        return parts[0]
    # Fallback: use filename without extension
    return os.path.splitext(basename)[0]


class PatientAwareInfiniteSampler(Sampler):
    """
    Infinite sampler that guarantees at most one frame per patient per batch.

    Algorithm: round-robin interleaving across patients.
    Each epoch: shuffle patients, shuffle frames within each patient,
    then emit one frame at a time from each patient in round-robin order.
    """

    def __init__(self, patient_ids, batch_size, seed=0):
        """
        Args:
            patient_ids: list of patient ID strings, one per dataset sample.
            batch_size: batch size (used for logging/validation).
            seed: base random seed.
        """
        self.patient_ids = list(patient_ids)
        self.dataset_size = len(patient_ids)
        self.batch_size = batch_size
        self.seed = seed

        # Group indices by patient ID
        self.patient_to_indices = {}
        for idx, pid in enumerate(self.patient_ids):
            if pid not in self.patient_to_indices:
                self.patient_to_indices[pid] = []
            self.patient_to_indices[pid].append(idx)

        self.n_patients = len(self.patient_to_indices)
        self.frames_per_patient = [len(v) for v in self.patient_to_indices.values()]

        # Validate
        n_unique = self.n_patients
        mean_frames = self.dataset_size / max(n_unique, 1)
        assert n_unique >= 100, f"Expected >=100 unique patients, got {n_unique}"
        assert 5 <= mean_frames <= 50, f"Expected 5-50 frames/patient, got {mean_frames:.1f}"
        assert all(pid != "" for pid in self.patient_ids), "Empty patient ID found"

        logger.info(f"Patient-aware sampling: {n_unique} unique patients, "
                     f"{mean_frames:.1f} frames/patient (mean)")

    def _generate_epoch(self, rng):
        """Generate one epoch of interleaved indices."""
        # Shuffle patient order
        patient_list = list(self.patient_to_indices.keys())
        rng.shuffle(patient_list)

        # Shuffle frames within each patient
        patient_frames = {}
        for pid in patient_list:
            frames = list(self.patient_to_indices[pid])
            rng.shuffle(frames)
            patient_frames[pid] = frames

        # Round-robin interleaving
        indices = []
        pointer = {pid: 0 for pid in patient_list}
        active_patients = list(patient_list)

        while active_patients:
            pid = active_patients.pop(0)
            if pointer[pid] < len(patient_frames[pid]):
                indices.append(patient_frames[pid][pointer[pid]])
                pointer[pid] += 1
                active_patients.append(pid)  # put back if more frames remain

        return indices

    def __iter__(self):
        epoch = 0
        while True:
            rng = np.random.default_rng(self.seed + epoch * 31337)
            indices = self._generate_epoch(rng)
            yield from indices
            epoch += 1

    def __len__(self):
        return self.dataset_size
