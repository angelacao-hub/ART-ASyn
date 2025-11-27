from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

def normalize_to_01(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)

# Dataset Format
#     <data_dir>
#     ├── train
#     │   ├── healthy
#     │   └── synthetic anomaly
#     │       └── <study_id>
#     │           ├── anomaly_0
#     │           ├── ...
#     │           └── mask_0
#     └── test
#     	├── healthy
#     	└── diseased

class SyntheticAnomalyDataset(Dataset):
	def __init__(self, data_dir, split):
		self.data_dir = Path(data_dir)
		self.split = split

		self.normal_paths = list((self.data_dir / split / "healthy").glob("*"))
		if split == "train":
			self.abnormal_paths = []
		else:
			self.abnormal_paths = list((self.data_dir / split / "diseased").glob("*"))

		self.n_normal = len(self.normal_paths)
		self.n_abnormal = len(self.abnormal_paths)

	def __getitem__(self, i):
		is_abnormal = i >= len(self.normal_paths)
		i = i % self.n_normal if self.n_normal != 0 and is_abnormal else i
		img_path = self.abnormal_paths[i] if is_abnormal else self.normal_paths[i]
		mask_path = Path(*img_path.parts[:-3], "ground_truth", *img_path.parts[-2:])
		study_id = img_path.stem
		suffix = img_path.suffix

		source = [self.transform_img(img_path)]
		target = source[0]
		masks = [self.transform_mask(mask_path, is_abnormal)]
		labels = [int(is_abnormal)]

		if self.split == "train":
			synthetic_anomaly_dir = self.data_dir / "train" / "synthetic anomaly" / study_id
			n_synthetic_anomalies = len(list(synthetic_anomaly_dir.glob("anomaly_*")))
			for i in range(n_synthetic_anomalies):
				img_path = synthetic_anomaly_dir / f"anomaly_{i}{suffix}"
				mask_path = synthetic_anomaly_dir / f"mask_{i}{suffix}"

				source.append(self.transform_img(img_path))
				masks.append(self.transform_mask(mask_path, 1))
				labels.append(1)
			target = source[0].repeat(n_synthetic_anomalies+1, 1, 1)
		
		return torch.cat(source), target, torch.cat(masks), torch.tensor(labels)

	@classmethod
	def transform_img(cls, path):
		img = Image.open(path).convert('L')
		transforms = T.Compose([
			T.Resize((256, 256)),
			T.ToTensor(), # C, H, W
		])
		img = transforms(img)

		return img

	@classmethod
	def transform_mask(cls, path, label):
		if not path.exists():
			return label * torch.ones((1, 256, 256))
		
		mask = Image.open(path).convert('L')
		transforms = T.Compose([
			T.Resize((256, 256)),
			T.ToTensor() # C, H, W
		])
		mask = transforms(mask)

		return mask


	def __len__(self):
		return self.n_normal + self.n_abnormal

def custom_collate(batch):
	source, target, masks, labels = zip(*batch)
	idxs = []
	for i, x in enumerate(source):
		idxs += [i] * len(x)
	return torch.cat(source).unsqueeze(1), torch.cat(target).unsqueeze(1), torch.cat(masks).unsqueeze(1), torch.cat(labels), torch.tensor(idxs)


def get_data_loader(data_dir, split, batch_size, generator=True):
	dataset = SyntheticAnomalyDataset(data_dir, split)
	loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)

	return get_generator_from_loader(loader) if generator else loader

# Keeps looping forever instead of per epoch
def get_generator_from_loader(loader):
	while True:
		yield from loader