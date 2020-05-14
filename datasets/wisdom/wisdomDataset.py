
import os,sys 
sys.path.append(os.path.abspath('.'))

import numpy as np
import skimage
from skimage.io import imread
from configs.config import Config

from datasets.dataset_handler import DatasetHandler


"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools 
and image resizing for networks.
Directory structure must be as follows:
$base_path/
    test_indices.npy
    train_indices.npy
    images/ (Train/Test Images here)
        image_000000.png
        image_000001.png
        ...
    segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(DatasetHandler):
    def __init__(self, subset):
        assert Config.DATASET.DATA_ROOT != "", "You must provide the path to a dataset!"

        self.dataset_name = Config.DATASET.NAME
        self.base_path = Config.DATASET.DATA_ROOT
        self.mask = Config.DATASET.MASK_PATH
        self.image = Config.DATASET.IMAGE_PATH
        self.train_indice_file = os.path.join(self.base_path, Config.DATASET.TRAIN_INDICES)
        self.val_indice_file = os.path.join(self.base_path, Config.DATASET.VAL_INDICES)
        
        self._channels = Config.DATASET.IMAGE.CHANNEL_COUNT
        super().__init__()
        self.load(subset)
        self.prepare()
        

    def load(self, subset):
        # Load the indices for imset.
        if subset == 'train':
            split_file = self.train_indice_file
        else:
            split_file = self.val_indice_file

        self.image_id = np.load(split_file)
        
        self.add_class(self.dataset_name, 1, 'fg')

        for i in self.image_id:
            if 'numpy' in self.image:
                p = os.path.join(self.base_path, self.image,
                                'image_{:06d}.npy'.format(i))
            else:
                p = os.path.join(self.base_path, self.image,
                                'image_{:06d}.png'.format(i))
            self.add_image(self.dataset_name, image_id=i, path=p)



    def load_image(self, image_id):
        # loads image from path
        if 'numpy' in self.image:
            image = np.load(self.image_info[image_id]['path']).squeeze()
        else:
            image = skimage.io.imread(self.image_info[image_id]['path'])

        if self._channels < 4 and image.shape[-1] == 4 and image.ndim == 3:
            image = image[...,:3]
        if self._channels == 1 and image.ndim == 2:
            image = image[:,:,np.newaxis]
        elif self._channels == 1 and image.ndim == 3:
            image = image[:,:,0,np.newaxis]
        elif self._channels == 3 and image.ndim == 3 and image.shape[-1] == 1:
            image = skimage.color.gray2rgb(image)
        elif self._channels == 4 and image.shape[-1] == 3:
            concat_image = np.concatenate([image, image[:,:,0:1]], axis=2)
            assert concat_image.shape == (image.shape[0], image.shape[1], image.shape[2] + 1), concat_image.shape
            image = concat_image
        return image

    def load_rgb_image(self, image_id):
        '''
        load rgb image, only used in inference time
        '''
        image = skimage.io.imread(self.image_info[image_id]['path'].replace('depth_ims', 'color_ims'))
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == self.dataset_name:
            return info["path"] + "-{:d}".format(info["flip"])
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        # loads mask from path
        info = self.image_info[image_id]
        _image_id = info['id']
        Is = []
        file_name = os.path.join(self.base_path, self.mask,
          'image_{:06d}.png'.format(_image_id))       

        all_masks = skimage.io.imread(file_name)

        for i in np.arange(1,np.max(all_masks)+1):
            I = all_masks == i # We ignore the background, so the first instance is 0-indexed.
            if np.any(I):
                I = I[:,:,np.newaxis]
                Is.append(I)
        if len(Is) > 0:
            mask = np.concatenate(Is, 2)
        else:
            mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)


        class_ids = np.array([1 for _ in range(mask.shape[2])])
        return mask, class_ids.astype(np.int32)

    @property
    def indices(self):
        return self.image_id


if __name__=='__main__':
    from configs.mrcnn_config import init_config
    # config_fns = ['./configs/base_config.yml', './datasets/wisdom/wisdomConfig.yml']
    config_fns = ['./configs/base_config.yml', './datasets/wisdom/wisdomInference.yml']
    init_config(config_fns)
    dataset_train = ImageDataset('val')
    print(len(dataset_train))
    image = dataset_train.load_image(0)
    mask, class_id = dataset_train.load_mask(0)
    mask, class_id = dataset_train.load_mask(5)
    mask, class_id = dataset_train.load_mask(90)

    # dataset_test = ImageDataset(config)
    # dataset_test.load(config['dataset']['val_indices'], augment=True)
    # dataset_test.prepare()
