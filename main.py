import glob
import os

import numpy as np
import torch

from model.multi_directional_subspace_edit import MDSE
from model.utils import load_params, post_process_img, GEN_PARAMS
import matplotlib.pyplot as plt


OUTPUT_PATH = './results/'


@torch.no_grad()
def main():
    params = load_params()
    model = MDSE(params)
    model.load_checkpoints()
    model.subspace_model.to(params.device)
    model.subspace_model.eval()
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for fname in glob.glob('./latent_vectors/*'):
        im_name = os.path.splitext(os.path.basename(fname))[0]
        latent = np.load(fname)

        for im_idx, weights in enumerate(GEN_PARAMS[im_name]['weights']):
            directions = model.get_subspace_directions(GEN_PARAMS[im_name]['attr'])
            edit_latent = latent.copy().flatten()

            for weight_idx, weight in enumerate(weights):
                edit_latent += weight * directions[weight_idx]

            edit_tensor = torch.from_numpy(edit_latent.reshape(1, *latent.shape)).to(params.device)
            img = model.stylegan(edit_tensor, latent_space_type='WP')['image']
            img = post_process_img(img)[0]
            plt.imsave(os.path.join(OUTPUT_PATH, f'{im_name}_edit_{im_idx}.png'), img)


if __name__ == '__main__':
    main()
