import torch
from preprocessing.utils import normalize, get_cmap_image

@torch.no_grad()
def run_batch(model, test_data_input):
	reconstuction = model(x=test_data_input)
	error_maps = (test_data_input - reconstuction) ** 2

	abs_diff = (test_data_input - reconstuction).abs()
	output_mask = torch.where(abs_diff >= max(0.03, abs_diff.max().item() * 0.2), 1.0, 0.0)

	# if np.random.randint(2):
	# 	get_cmap_image(output_mask[0].squeeze()).save(f"temp/seg.jpg")
	# 	get_cmap_image(normalize(reconstuction[0].squeeze())).save(f"temp/reconstuction.jpg")
	# 	get_cmap_image(normalize(batch_data[0].squeeze())).save(f"temp/source.jpg")

	return output_mask, error_maps, reconstuction
