import os
import sys
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms
import torch.optim as optim
from data_split.MyDataset import MyDataSet
from tqdm import tqdm
import math
import random
import warnings
import json
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from data_split.data_split2 import data_set_split
import os
from model_v2 import MobileNetV2
os.environ["CUDA_VISIBLE_DEVICES"]='1'

# 参考 https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--root_datasets",default="/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/class_B01_B03/",help='Dataset directory path')
parser.add_argument("--target_data_folder",default="/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/class_B01_B03_output_2/",help='Dataset directory path')
parser.add_argument("--weights_folder",default="/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/weights/weights_B01_B03_2/",help='Dataset directory path')
parser.add_argument('--resume', default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/weights/weights_B01_B03/model-300_of_1000-0.004918756429105997-0.6492027334851936.pth',
                    type = str,help='Directory for saving checkpoint models')

parser.add_argument('--balance_data', action='store_true',help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--net', default="MobileNetV2",help="The network architecture, it can be MobileNetV1, MobileNetV2,or MobileNetV3.")
parser.add_argument('--lr', '--learning-rate', default= 0.001, type=float, # 5e-4  0.001
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
# Train params
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=1000, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--use_cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lrf', type=float, default=0.001) # 0.01
parser.add_argument('--seed', default=10, type=int,help='seed for initializing training. ')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



# 参数分别是 训练loader, net, criterion(loss),optimizer,device,debug_steps,epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()

# @torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():   # 可以跟上面的@torch.no_grad()  替换掉
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            pred = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num


def adjust_learning_rate(args):
    lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    return lf

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # 加了这句话，会让种子失效
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        '''设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，
        为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。'''
        logging.info("Use Cuda.")
    logging.info(args) # 打印你的args
    tb_writer = SummaryWriter()
    if args.net == 'MobileNetV2':
        create_net = MobileNetV2(num_classes = 2).to(device) # My的模型  -----------------------------------------
        if args.resume is not None:
            # path="你的 .pt文件路径"
            # model = "你的网络"
            checkpoint = torch.load(args.resume, map_location=device)
            create_net.load_state_dict(checkpoint)
            print("=========================we are using pretrained model for train======================")
            
    transform = {
        "train": transforms.Compose([
            transforms.Resize([800,400]),
            transforms.ToTensor(),
            transforms.Normalize([0.1440268, 0.09442947, 0.040968522], [0.16813725, 0.11243858, 0.04965655])
        ]),
        "val": transforms.Compose([
            transforms.Resize([800,400]),
            transforms.ToTensor(),
            transforms.Normalize([0.1440268, 0.09442947, 0.040968522], [0.16813725, 0.11243858, 0.04965655])
        ])}

    train_images_path, train_images_labels, val_images_path, val_images_labels = data_set_split(args.root_datasets,args.target_data_folder)

    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_labels, transform=transform["train"])
    # print("train_dataset",train_dataset)
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_labels, transform=transform["val"])
    #class2json(train_dataset)  # 在分割数据的时候已经生成json文件了
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=val_dataset.collate_fn)
    pg = [p for p in create_net.parameters() if p.requires_grad]
    '''
    调整学习率：是基于在每一个epoch调整，同时学习率的调整要在优化器参数之后更新。
    "即先优化器更新，在学习率更新"
    '''
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    lf = adjust_learning_rate(args)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(args.num_epochs):
        # train
        mean_loss = train_one_epoch(model=create_net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=create_net,
                       data_loader=val_loader,
                       device=device)
        logging.info("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if (epoch+1) % 5 == 0:
            torch.save(create_net.state_dict(), args.weights_folder +"/model-{}_of_{}-{}-{}.pth".format(epoch+1,args.num_epochs,mean_loss,acc))

if __name__ == '__main__':
    main()



