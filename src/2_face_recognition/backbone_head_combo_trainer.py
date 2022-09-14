import argparse
import shutil
import sys
import os

from tensorboardX import SummaryWriter

sys.path.append('../')
from training.train import train


def get_conf(conf, bbt, bcf, hf, hcf, lr, out, ep, step, bs, ):
    # conf.data_root_train=drt
    # conf.train_file=tf
    conf.backbone_type = bbt
    conf.backbone_conf_file = os.path.abspath(bcf)
    conf.head_type = hf
    conf.head_conf_file = os.path.abspath(hcf)
    conf.lr = lr
    conf.out_dir = out
    conf.epoches = ep
    conf.step = step
    conf.print_freq = 500
    conf.save_freq = 3000
    conf.batch_size = bs
    conf.momentum = 0.9
    conf.log_dir = out
    conf.cuda = True
    conf.intel = False
    conf.resume = False
    conf.saveall = False
    conf.tensorboardx_logdir ='mv-hrnet'
    conf.milestones = [int(num) for num in args.step.split(',')]
    return conf


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Path to the model')
        parser.add_argument('--data_root_train', metavar='path', required=True,
                            help='path to dataset basepath')
        parser.add_argument('--train_file', metavar='path', required=True,
                            help='path to train file')
        parser.add_argument('--outfolder', metavar='path', required=True,
                            help='path to train file')
        args = parser.parse_args()

        backbone_conf_file = '../training_mode/backbone_conf.yaml'
        head_conf_file = "head_conf_diveface.yaml"
        backbones = ["HRNet", "AttentionNet", "ResNeSt", "RepVGG"]
        heads = ["AdaM-Softmax", "MagFace", "MV-Softmax", "NPCFace"]
        epochs = 80
        lr = 0.1
        step = '5,25,68'
        batch_size = 128
        cuda = True
        for b in backbones:
            for h in heads:
                try:
                    out_dir = os.path.join(args.outfolder, "+".join([b, h]))
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    conf = get_conf(args, b, backbone_conf_file, h, head_conf_file, lr, out_dir, epochs, step, batch_size)
                    if not os.path.exists(conf.log_dir):
                        os.makedirs(conf.log_dir)
                    tensorboardx_logdir = os.path.join(conf.log_dir, conf.tensorboardx_logdir)
                    if os.path.exists(tensorboardx_logdir):
                        shutil.rmtree(tensorboardx_logdir)
                    writer = SummaryWriter(log_dir=tensorboardx_logdir)
                    args.writer = writer
                    train(conf)

                except Exception as e2:
                    print(e2)

    except Exception as e:
        print(e)
        sys.exit(1)
