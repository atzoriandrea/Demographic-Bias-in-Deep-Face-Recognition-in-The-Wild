import os
import torchmetrics
import sys
import shutil
import argparse
import logging as logger
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import gc

# import intel_pytorch_extension as ipex
sys.path.append('../')
sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset
from models_definitions.backbone.backbone_def import BackboneFactory
from models_definitions.head.head_def import HeadFactory
import os

try:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:21")
except:
    pass

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        label = label.type(torch.int64)
        pred = self.head.forward(feat, label)
        return pred


def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf):
    """
    Train one epoch by traditional training.
    """
    metric = torchmetrics.Accuracy().to(conf.device)
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        elif conf.head_type == 'MagFace':
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            loss = criterion(outputs, labels) + loss_g
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)

        argMs = torch.argmax(outputs, 1)
        acc = metric(argMs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])

        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            acc_avg = metric.compute()
            logger.info('Epoch %d, iter %d/%d, lr %f, accuracy %f, loss %f' %
                        (cur_epoch, batch_idx, len(data_loader), lr, acc_avg, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if (batch_idx + 1) % conf.save_freq == 0 and conf.saveall:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
    if (not conf.saveall and cur_epoch == conf.epoches - 1) or conf.saveall:
        saved_name = 'Epoch_%d.pt' % cur_epoch
        state = {'state_dict': model.module.state_dict(),
                 'epoch': cur_epoch, 'batch_id': batch_idx}
        # torch.save(state, os.path.join(conf.out_dir, saved_name))
        torch.save(model, os.path.join(conf.out_dir, saved_name))
        logger.info('Save checkpoint %s to disk...' % saved_name)
    return acc.cpu().detach().numpy(), loss.cpu().detach().numpy()


def get_test_set_acc(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf):
    temp_acc = torch.tensor([0], device=conf.device, dtype=torch.float)
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        elif conf.head_type == 'MagFace':
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            loss = criterion(outputs, labels) + loss_g
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.shape[0])
        argMs = torch.argmax(outputs, 1)
        temp_acc += torch.sum(torch.eq(labels, argMs)) / data_loader.batch_size
        # print(batch_idx)
        if (batch_idx + 1) == len(data_loader):
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            acc_avg = temp_acc / batch_idx
            logger.info('TEST SET: Epoch %d, iter %d/%d, lr %f, accuracy %f, loss %f' %
                        (cur_epoch, batch_idx, len(data_loader), lr, acc_avg, loss_avg))
    return temp_acc.cpu().detach().numpy(), loss.cpu().detach().numpy()


def train(conf, logpath=""):
    """Total training procedure.
    """
    data_loader_train = DataLoader(ImageDataset(conf.data_root_train, conf.train_file),
                                   conf.batch_size, shuffle=True)
    if conf.cuda:
        conf.device = torch.device('cuda:0')
    # elif conf.intel:
    # conf.device = ipex.DEVICE
    else:
        conf.device = torch.device("cpu")
    print(conf.device)
    if conf.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    model = FaceModel(backbone_factory, head_factory)
    ori_epoch = 0
    if conf.resume:
        # ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
        # state_dict = torch.load(args.pretrain_model)['state_dict']
        # model.load_state_dict(state_dict)
        model = torch.load(args.pretrain_model)
        ori_epoch = args.resume_from_epoch
    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=conf.lr)
    # milestones = [x > ori_epoch for x in conf.milestones]
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=conf.milestones, gamma=0.1)
    loss_meter_train, loss_meter_test = AverageMeter(), AverageMeter()
    model.train()
    train_set_data = []
    for epoch in range(ori_epoch, conf.epoches):
        train_set_data.append(train_one_epoch(data_loader_train, model, optimizer,
                                              criterion, epoch, loss_meter_train, conf))
        # test_set_data.append(get_test_set_acc(data_loader_test, model, optimizer,criterion, epoch, loss_meter_test, conf))
        # lr_schedule.step()
    np.save(os.path.join(conf.log_dir, "train_results.npy"), np.asarray(train_set_data))
    # np.save("test_results.npy", np.asarray(test_set_data))


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root_train", type=str,
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      help="The training file path.")
    '''
    conf.add_argument("--data_root_test", type=str,
                      help="The root folder of test set.")
    conf.add_argument("--test_file",type=str,
                      help="The test file path.")
    '''
    conf.add_argument("--backbone_type", type=str,
                      help="Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type=str,
                      help="the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type=str,
                      help="mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type=str,
                      help="the path of head_conf.yaml.")
    conf.add_argument('--lr', type=float, default=0.1,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str,
                      help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=9,
                      help='The training epoches.')
    conf.add_argument('--step', type=str, default='2,5,7',
                      help='Step for lr.')
    conf.add_argument('--print_freq', type=int, default=10,
                      help='The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=10,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=128,
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for sgd.')
    conf.add_argument('--log_dir', type=str, default='log',
                      help='The directory to save log.log')
    conf.add_argument('--cuda', default=False,
                      help='The directory to save log.log')
    conf.add_argument('--intel', default=False,
                      help='The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type=str,
                      help='The directory to save tensorboardx logs')
    conf.add_argument('--saveall', type=str, default=False,
                      help='Save all trained models')
    conf.add_argument('--pretrain_model', type=str, default='mv_epoch_8.pt',
                      help='The path of pretrained model')
    conf.add_argument('--resume', '-r', action='store_true', default=False,
                      help='Whether to resume from a checkpoint.')
    conf.add_argument('--resume_from_epoch', type=int, default=1,
                      help='Previous epoch.')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer

    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
