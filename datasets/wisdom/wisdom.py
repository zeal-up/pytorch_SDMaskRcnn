
import os
import sys
sys.path.append(os.path.abspath('.'))
import torch


from configs.mrcnn_config import init_config
from configs.config import Config
from configs.args import args
from libs.networks.model_component.anchors import generate_pyramid_anchors
from datasets.wisdom.wisdomDataset import ImageDataset
from datasets.data_generator import DataGenerator



if __name__ == '__main__':

    # Configurations
    # conf_fns = ['./configs/base_config.yml', './datasets/wisdom/wisdomConfig.yml']
    conf_fns = ['./configs/base_config.yml', './datasets/wisdom/wisdomInference.yml']
    init_config(conf_fns)
    ## basic config and special config such as dataset
    dataset_train = ImageDataset('train')
    anchors = generate_pyramid_anchors(
        Config.RPN.ANCHOR.SCALES,
        Config.RPN.ANCHOR.RATIOS,
        Config.BACKBONE.SHAPES,
        Config.BACKBONE.STRIDES,
        Config.RPN.ANCHOR.STRIDE,
        Config.TRAINING.BATCH_SIZE
    ).to(Config.DEVICE)
    train_set = DataGenerator(dataset_train, augmentation=None,
                            anchors=anchors[0])
    print(anchors[0].shape)
    for step, inputs in enumerate(train_set):
        print('train set load',len(train_set))
        print('train inputs', '\nimage:', inputs[0].shape, '\nimage_metas:', inputs[1],\
        '\nrpn_match:', inputs[2].shape, '\nrpn_bbox.shape: ',inputs[3].shape,\
        '\ngt_class_ids:', inputs[4][:20], '\ngt_boxes：', inputs[5].shape,'\ngt_mask.shape:',inputs[6].shape)
        if step == 1:
            break
    
    # train_generator = torch.utils.data.DataLoader(
    #         train_set, shuffle=True, batch_size=Config.TRAINING.BATCH_SIZE,
    #         num_workers=4)
    # print(len(train_generator),'train_generator')
    
    
