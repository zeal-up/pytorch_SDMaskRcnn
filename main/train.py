import os,sys 
sys.path.append(os.path.abspath('.'))
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in [1])
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
from datetime import datetime

from configs.args import args
from configs.mrcnn_config import init_config
from configs.config import Config


# import networks and loss function
from libs.networks.network_pipeline import MaskRCNN
from libs.networks.network_wrapper import Network_Wrapper
from libs.networks.model_component.anchors import generate_pyramid_anchors
from libs.networks.Losses.losses import Losses

from libs.utils.AverageMeter import AverageMeter
from libs.utils.tensorboard_writter import Maskrcnn_logger

# import dataset 
from datasets.wisdom.wisdomDataset import ImageDataset
from datasets.data_generator import DataGenerator

from tensorboardX import SummaryWriter

# set random seed
np.random.seed(1)
torch.manual_seed(1)
torch.random.manual_seed(1)
torch.cuda.random.manual_seed_all(1)

# initialize the configuration
if not isinstance(args.extra_config_fns, (list, tuple)):
    args.extra_config_fns = [args.extra_config_fns]
config_fns = [args.base_config_fn]
config_fns.extend(args.extra_config_fns)
init_config(config_fns, args, is_display=True)



# specify the log path and checkpoints location
# It will first create a /log dir in the project root. 
# All experiments' training information will be saved in ./log 
# Each experiment has it's unique name as log_name+TIMESTAMP.
# checkpoints will be saved in ./log/log_name_TIMESTAMP/checkpoints
TIMESTAMP = "_{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
log_dir = os.path.join('./log', Config.LOG.NAME)
log_dir += TIMESTAMP
writer = SummaryWriter(log_dir)
mrcnn_writer = Maskrcnn_logger(writer)
checkpoint_dir = os.path.join(log_dir, 'checkpoints')


# define log variance
total_loss = AverageMeter()



if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def train_epoch(net, optimizer, dataloader, epoch):
    net.train()
    size = len(dataloader)
    for batch, data in enumerate(dataloader):

        # forward to compute loss 
        # In multi-gpu training, the losses is the loss gathered from different divice
        losses = net(data) 
        loss = Losses(*[lo.mean() for lo in losses])

        loss.total.backward()

        # clip gradient, copy from maskrcnn
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        # record loss

        # log every args.log_interval batch
        if batch % Config.LOG.INTERVAL == 0:
            step = epoch + batch/size
            print('Epoch:{:03d} Batch{:04d}/{:05d} | loss:{:<6.4f}'.format(epoch, batch, size, loss.total.item()))
            mrcnn_writer.log_train(loss, step)

        

def val(net, dataloader, epoch):
    net.eval()
    total_loss = Losses()
    batches = len(dataloader)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):

            # forward to compute loss 
            # In multi-gpu training, the losses is the loss gathered from different divice
            losses = net(data) 
            loss = Losses(*[lo.mean() for lo in losses])
            total_loss = total_loss + loss

        total_loss = total_loss / float(batches)
        step = epoch + 1 # align the traing curve
        mrcnn_writer.log_val(total_loss, step)

        print('Val_Epoch:{:03d} | loss:{:<6.4f}'.format(epoch, total_loss.total.item()))





def main():
    # record the args in tensorboard
    writer.add_text('config', str(Config.yaml_format()), 0)

    train_handler = ImageDataset('train')
    val_handler = ImageDataset('val')
    
    anchors = generate_pyramid_anchors(
        Config.RPN.ANCHOR.SCALES,
        Config.RPN.ANCHOR.RATIOS,
        Config.BACKBONE.SHAPES,
        Config.BACKBONE.STRIDES,
        Config.RPN.ANCHOR.STRIDE,
        Config.TRAINING.BATCH_SIZE
    )[0]
    
    train_set = DataGenerator(train_handler, anchors=anchors)
    val_set = DataGenerator(val_handler, anchors=anchors)

    print('Length of train dataset', len(train_set))
    print('Length of val dataset', len(val_set))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=Config.TRAINING.BATCH_SIZE,
        num_workers=Config.TRAINING.WORKERS,
        shuffle=True,
        drop_last=True
    )

    if Config.TRAINING.VAL_SAMPLE_LENGTH:
        sampler = torch.utils.data.RandomSampler(val_set, replacement=True, num_samples=Config.TRAINING.VAL_SAMPLE_LENGTH)
    else:
        sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=Config.TRAINING.BATCH_SIZE,
        num_workers=Config.TRAINING.WORKERS,
        sampler=sampler,
        shuffle=False,
        drop_last=False
    )

    net_wrapper = Network_Wrapper(backbone=Config.BACKBONE.NAME)

    if Config.GPU_COUNT is not None and Config.GPU_COUNT > 1:
        net_wrapper = torch.nn.DataParallel(net_wrapper)
    net_wrapper = net_wrapper.to(Config.DEVICE)

    if Config.TRAINING.OPTIM == 'sgd':
        trainables_wo_bn = [param for name, param in net_wrapper.named_parameters()
                            if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in net_wrapper.named_parameters()
                              if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn,
             'weight_decay': Config.TRAINING.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=Config.TRAINING.LR, momentum=Config.TRAINING.MOMENTUM)

    elif Config.TRAINING.OPTIM == 'adam':
        optimizer = torch.optim.Adam(
            net_wrapper.parameters(), 
            lr=Config.TRAINING.LR, 
            weight_decay=Config.TRAINING.WEIGHT_DECAY)
    else:
        assert False, 'invalid optimizer'

    begin_epoch = 0
    if args.resume:
        state = torch.load(args.resume)
        try:
            net_wrapper.module.load_state_dict(state['net'])
        except:
            net_wrapper.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optim'])
        begin_epoch = state['epoch'] + 1


    lr_schedular = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.TRAINING.LR_DECAY_EPOCH,
        gamma=Config.TRAINING.LR_DECAY_RATE,
        last_epoch=-1 if begin_epoch==0 else begin_epoch
    )

    for epoch in range(begin_epoch, Config.TRAINING.EPOCHS):
        if args.eval_first and epoch == begin_epoch:
            print('first eval')
            val(net_wrapper, val_loader, epoch)
            print('first train')
        train_epoch(net_wrapper, optimizer, train_loader, epoch)

        if (epoch+1) % Config.TRAINING.VALIDATION_INTERVAL == 0:
            val(net_wrapper, val_loader, epoch) # imbedding to train()

        if (epoch+1) % Config.TRAINING.SAVE_INTERVAL == 0:
            state = {}
            state['net'] = net_wrapper.module.state_dict()
            state['optim'] = optimizer.state_dict()
            state['epoch'] = epoch
            torch.save(state, checkpoint_dir+'/{}.pth'.format(epoch))
            
        lr_schedular.step()



if __name__ == '__main__':
    main()