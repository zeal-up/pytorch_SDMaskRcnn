"""Handles Config class in configurations that are specific
to the MRCNN module."""
import os,sys
sys.path.append(os.path.abspath('.'))

import math
import numpy as np

import torch

from configs.config import Config


def init_config(config_fns, cmd_args=None, is_display=False):
    """Loads configurations from YAML files, then create utilitaire
    configurations. Freeze config and display it.
    Input:
        - config_fns: list of config files, the first is the basic_config.file
        - cmd_args: some other configs recevies in cmd 
        - is_display: display all configs after initialize
    """
    Config.set_default_fn(config_fns[0])
    Config.load_default()
    for filename in config_fns[1:]:
        Config.merge(filename)

    if Config.DEVICE == 'gpu' and torch.cuda.is_available():
        Config.DEVICE = torch.device('cuda')
        Config.GPU_COUNT = torch.cuda.device_count()
    else:
        Config.DEVICE = torch.device('cpu')

    if cmd_args is not None:
        if cmd_args.batch_size:
            Config.TRAINING.BATCH_SIZE = cmd_args.batch_size
        if cmd_args.log_name:
            Config.LOG.NAME = cmd_args.log_name


    # if Config.DEVICE == torch.device('cuda'):
    #     assert Config.TRAINING.BATCH_SIZE / torch.cuda.device_count() == int(Config.TRAINING.BATCH_SIZE / torch.cuda.device_count())
    #     Config.TRAINING.IMAGES_PER_GPU = int(Config.TRAINING.BATCH_SIZE / torch.cuda.device_count())


    # Input image size
    if Config.DATASET.IMAGE.SHAPE is None:
        Config.DATASET.IMAGE.SHAPE = np.array(
            [Config.DATASET.IMAGE.MAX_DIM, Config.DATASET.IMAGE.MAX_DIM, 3])
    else:
        Config.DATASET.IMAGE.SHAPE = np.array(Config.DATASET.IMAGE.SHAPE)

    # Compute backbone size from input image size
    Config.BACKBONE.SHAPES = np.array(
        [[int(math.ceil(Config.DATASET.IMAGE.SHAPE[0] / stride)),
          int(math.ceil(Config.DATASET.IMAGE.SHAPE[1] / stride))]
         for stride in Config.BACKBONE.STRIDES])

    Config.RPN.BBOX_STD_DEV_GPU = torch.from_numpy(
        np.reshape(Config.RPN.BBOX_STD_DEV, [1, 4])
        ).float()

    # this configurations are for speeding up the training
    # clip_window just a params to constarint the anchor inside image
    # RPN.NORM is used to normalize the bbox
    height, width = Config.DATASET.IMAGE.SHAPE[:2]
    Config.RPN.CLIP_WINDOW = np.array([0, 0, height, width]).astype(np.float32)
    Config.RPN.NORM = torch.tensor(
        np.array([height, width, height, width]),
        requires_grad=False, dtype=torch.float32)

    check_config()
    Config.freeze()
    if is_display:
        Config.display()
    return Config.yaml_format()


def check_config():
    """All configuration checks must be placed here."""
    # Image size must be dividable by 2 multiple times
    h, w = Config.DATASET.IMAGE.SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be divisable by 2 at least "
                        "6 times to avoid fractions when downscaling "
                        "and upscaling. For example, use 256, 320, 384, "
                        "448, 512, ... etc. ")


if __name__ == "__main__":
    conf = init_config(['./configs/base_config.yml'])
    print(conf)