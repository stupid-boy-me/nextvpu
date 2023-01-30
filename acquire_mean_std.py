from __future__ import print_function, division
import torch
import torch.utils.data
import torchvision
import torchvision.datasets as Datasat
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    dataset = Datasat.ImageFolder(root="/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20230103_class_2_B03_F03/class_C02_F03/", transform=torchvision.transforms.ToTensor())
    print(dataset)
    print(getStat(dataset))
