import numpy as np 

class Tensorboard_logger(object):
    def __init__(self, writter, loss_name):
        '''
        Input:
            - writter: a SummaryWritter object
            - loss_name: the display title of each loss 
        '''
        self.writter = writter
        self.loss_name = loss_name
        self.set_training_log([1] * len(loss_name))
        self.set_val_log([1] * len(loss_name))
        self.train_prefix = 'train/'
        self.val_prefix = 'test/'

    def set_training_log(self, indicator):
        assert len(indicator) == len(self.loss_name),\
            'The len(indicator) must match the receive name list'
        self.train_log_ind = np.array(indicator, dtype=np.uint8)

    def set_val_log(self, indicator):
        assert len(indicator) == len(self.loss_name),\
            'The len(indicator) must match the receive name list'
        self.val_log_ind = np.array(indicator, dtype=np.uint8)

    def log_train(self, *args, step=None):
        self._log_general(*args, mode='train', step=step)

    def log_val(self, *args, step=None):
        self._log_general(*args, mode='val', step=step)
        

    def _log_general(self, *args, mode, step):
        assert mode in ['train', 'val']
        log_ind = {'train':self.train_log_ind, 'val':self.val_log_ind}[mode]
        prefix = {'train':self.train_prefix, 'val':self.val_prefix}[mode]
        assert len(args) == sum(log_ind)
        assert step is not None

        for i, (name,log) in enumerate(zip(self.loss_name, log_ind)):
            if not log:
                continue
            name = prefix+name
            # print('log')
            self.writter.add_scalar(name, args[i], step)

class Maskrcnn_logger(Tensorboard_logger):
    def __init__(self, writter):
        loss_name = ['total', 'rpn_class', 'rpn_box', 'mrcnn_class', 'mrcnn_bbox', 'mrcnn_mask']
        super().__init__(writter, loss_name)

    def log_train(self, Losses, step):
        super().log_train(Losses.total.item(),\
                          Losses.rpn_class.item(),\
                          Losses.rpn_bbox.item(),\
                          Losses.mrcnn_class.item(),\
                          Losses.mrcnn_bbox.item(),\
                          Losses.mrcnn_mask.item(), step=step)

    def log_val(self, Losses, step):
        super().log_val(Losses.total.item(),\
                          Losses.rpn_class.item(),\
                          Losses.rpn_bbox.item(),\
                          Losses.mrcnn_class.item(),\
                          Losses.mrcnn_bbox.item(),\
                          Losses.mrcnn_mask.item(), step=step)
    



if __name__ == "__main__":
    logger = Tensorboard_logger(None, ['loss1', 'loss2', 'loss3'])
    logger._log_general(1,2,2, mode='train', step=1)
