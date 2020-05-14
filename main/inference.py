import os,sys 
sys.path.append(os.path.abspath('.'))
from os.path import join as pjoin
import numpy as np 
import pickle
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import matplotlib

from configs.args import args
from configs.mrcnn_config import init_config
from configs.config import Config

from libs.utils import dataset_utils as utils
from libs.networks.network_pipeline import MaskRCNN
from libs.networks.model_component.anchors import generate_pyramid_anchors

from libs.utils.bar_utils import create_bar
from libs.utils.logging_creater import creat_logger
from libs.utils.exceptions import NoBoxHasPositiveArea, NoBoxToKeep

from datasets.wisdom.wisdomDataset import ImageDataset
from datasets.tless.tlessDataset import TlessDataset
from datasets.data_generator import DataGenerator

from libs.inference_funs.coco_benchmark import coco_benchmark

from libs.visualize.visualize import display_instances

args.extra_config_fns = './datasets/wisdom/wisdomInference.yml'
if not isinstance(args.extra_config_fns, (list, tuple)):
    args.extra_config_fns = [args.extra_config_fns]
config_fns = [args.base_config_fn]
config_fns.extend(args.extra_config_fns)
init_config(config_fns, args, is_display=True)


# test config
args.resume = './log/wisdom_2020-05-08-08-17-04/checkpoints/69.pth'
dataset_length = 1000
##################################################################

# create dataset 
if args.dataset_type:
    inf_set = TlessDataset('inference', argument=False)
else:
    inf_set = ImageDataset('val')

print('Length of inference dataset', len(inf_set))

##################################################################
# create network and load checkpoint 
net = MaskRCNN(backbone=Config.BACKBONE.NAME)
net = net.to(Config.DEVICE)

if not args.resume:
    assert False, 'Input checkpoint path for inference'

net.load_state_dict(torch.load(args.resume)['net'])
print('Load checkpoint from:{}'.format(args.resume))
net.inf() # set inference mode, will also set as eval mode 
#################################################################
# set logging path and some config
inf_length = len(inf_set)
bar = create_bar('Infering', dataset_length)

log_dir = pjoin(
    os.path.dirname(os.path.dirname(args.resume)),
    'eval_results'
)
log_file = pjoin(log_dir, 'result.log')
logger = creat_logger(log_file, log_name='result-logger', console_handler=True, file_handler=True)

mask_out_dir = pjoin(log_dir, 'mask_inf')
mask_gt_tpath = pjoin(mask_out_dir, 'gt_masks', 'image_{im_id:06d}.npy')
mask_pred_tpath = pjoin(mask_out_dir, 'pred_masks', 'image_{im_id:06d}.npy')
visualize_tpath = pjoin(mask_out_dir, 'visualize', 'image_{im_id:06d}.png')
meta_file = pjoin(mask_out_dir, 'metas.pkl')
if not os.path.exists(os.path.dirname(mask_gt_tpath)):
    os.makedirs(os.path.dirname(mask_gt_tpath))
    os.makedirs(os.path.dirname(mask_pred_tpath))
    os.makedirs(os.path.dirname(visualize_tpath))
################################################################
# ap,ar = coco_benchmark(os.path.dirname(mask_pred_tpath), os.path.dirname(mask_gt_tpath))
# inference
bar.start()
with torch.no_grad():
    all_metas = {}
    for i,image_id in enumerate(inf_set.image_ids[:dataset_length]):
        image_name = inf_set.image_info[image_id]['id']
        image = inf_set.load_image(image_id)
        molded_image, image_metas = utils.mold_image(image)
        molded_image = molded_image / float(Config.DATASET.IMAGE.NORM_VALUE)
        molded_image = torch.tensor(molded_image, dtype=torch.float32)
        molded_image = molded_image.permute(2,0,1).unsqueeze(0).contiguous()
        molded_image = molded_image.to(Config.DEVICE)
        try:
            detections, mrcnn_masks = net(molded_image) # set as inference mode
        except (NoBoxHasPositiveArea, NoBoxToKeep) as e:
            logger.warning('Image:{} cannot detection any box'.format(image_name))
            continue
        mrcnn_masks = mrcnn_masks.permute(0, 2, 3, 1) # num_pred x h x w x num_classes
        # return DetectionOutput(final_rois, final_class_ids, final_scores, final_masks)
        result = utils.unmold_detections(detections, mrcnn_masks, image_metas)
        result = result.cpu()
        pred_masks = result.masks.detach().cpu().numpy()
        mask_sum = pred_masks.sum((0,1))
        if mask_sum.sum() == 0:
            logger.warning("Image:{} preded mask is all zero".format(image_name))
            continue
        pred_masks = pred_masks[:,:,mask_sum!=0]
        gt_masks, _ = inf_set.load_mask(image_id) # h x w x nb
        if gt_masks.shape == (0,):
            logger.warning("Image:{} has no mask".format(image_name))
            continue
        mask_sum = gt_masks.sum((0,1))
        if mask_sum.sum() == 0:
            logger.warning("Image:{} gt mask is all zero".format(image_name))
            continue
        gt_masks = gt_masks[:,:,mask_sum!=0]
        # print(result.masks.shape) # 384 x 512 x 9
        # print(gt_masks.shape) # 384 x 5126 x 6

        # save the transpose so it's (n, h, w) instead of (h, w, n)
        
        np.save(mask_gt_tpath.format(im_id=image_id), gt_masks.transpose(2,1,0))
        np.save(mask_pred_tpath.format(im_id=image_id), pred_masks.transpose(2,1,0))
        all_metas[image_id] = image_metas.to_numpy()

        # display the image and mask 
        rgb = inf_set.load_rgb_image(image_id)
        fig = display_instances(
            rgb, result.rois, result.masks, result.class_ids,
            ['background', 'foreground'], result.scores,
            show_bbox=True, show_mask_pixels=True,
            title="Predictions for image:{}".format(image_id)
        )
        fig.savefig(visualize_tpath.format(im_id=image_id))
        matplotlib.pyplot.close()
        ##################################
        bar.update(i)
    bar.finish()
    print('Ouput finish, begin computing the precision')
    with open(meta_file, 'wb') as f:
        pickle.dump(all_metas, f)
    
    ap,ar = coco_benchmark(os.path.dirname(mask_pred_tpath), os.path.dirname(mask_gt_tpath))
    logger.info('The AP is {} | The AR is {}'.format(ap, ar))
    
    

    


        




    


