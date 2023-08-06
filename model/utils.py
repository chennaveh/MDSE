import json
import numpy as np


def load_params():
    class Params:
        def __init__(self, params):
            self.__dict__.update(params)

    with open('./model/params.json', 'r') as f:
        return Params(json.load(f))


def post_process_img(imgs):
    imgs = np.clip(imgs.detach().cpu().numpy(), -1, 1) / 2 + 0.5
    imgs = imgs.transpose(0, 2, 3, 1)
    return imgs


GEN_PARAMS = {
    'img1': {
        'attr': 'gender',
        'weights': [
            [0, 0, 0, 0, 0],
            [0, -3, -3, -5, -5],
            [0, -3, 3, -5, -5],
            [3, 0, 3, -5, -5],
            [3, 3, 0, -5, -5]
        ]
    },
    'img2': {
        'attr': 'age',
        'weights': [
            [0, 0, 0, 0, 0],
            [-3, -3, 0, 0, 0],
            [0, 0, 3, 0, 10],
            [0, -6, 0, -10, 10],
            [-3, 0, 3, -10, 10]
        ]
    },
    'img3': {
        'attr': 'race',
        'weights': [
            [0, 0, 0],
            [0, 2, 2],
            [0, 2, -2],
            [-2, -2, -2],
            [4, 0, 2],
        ]
    },

}