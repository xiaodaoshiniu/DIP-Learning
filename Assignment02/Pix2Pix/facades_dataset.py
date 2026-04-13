import os
import torch
from torch.utils.data import Dataset
import cv2


class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        self.list_file = list_file
        self.list_dir = os.path.dirname(os.path.abspath(list_file))
        with open(list_file, 'r', encoding='utf-8') as file:
            self.image_filenames = [line.strip() for line in file if line.strip()]

    def __len__(self):
        return len(self.image_filenames)

    def _resolve_path(self, img_name):
        if os.path.exists(img_name):
            return img_name

        normalized = img_name.replace('\\', os.sep).replace('/', os.sep)
        if os.path.exists(normalized):
            return normalized

        drive, tail = os.path.splitdrive(normalized)
        if drive:
            normalized = tail
        normalized = normalized.lstrip('\\/')
        if os.path.exists(normalized):
            return normalized

        parts = normalized.split(os.sep)
        if 'datasets' in parts and 'facades' in parts:
            idx = parts.index('datasets')
            candidate = os.path.join(self.list_dir, *parts[idx:])
            if os.path.exists(candidate):
                return candidate

        subset = None
        if 'train' in parts:
            subset = 'train'
        elif 'val' in parts:
            subset = 'val'

        if subset is not None:
            candidate = os.path.join(self.list_dir, 'datasets', 'facades', subset, os.path.basename(normalized))
            if os.path.exists(candidate):
                return candidate

        return img_name

    def __getitem__(self, idx):
        img_name = self._resolve_path(self.image_filenames[idx])
        img_color_semantic = cv2.imread(img_name)
        if img_color_semantic is None:
            raise FileNotFoundError(f'Failed to read dataset image: {img_name}')

        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0
        image = image * 2.0 - 1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic
