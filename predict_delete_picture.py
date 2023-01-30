from email.policy import strict
import os
import json
import sys
import logging
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_v2 import MobileNetV2
import argparse 
from tqdm import tqdm
import cv2
import shutil

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221210_dog_cat/',type=str,help='type of dataset')
parser.add_argument("--weights_path",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/weights_dog_cat/model-45_of_1000-0.03839210420846939-0.9607843137254902.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221209_all/json/class.json',type=str,help='json_path')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trf():
    data_transform = transforms.Compose(
        [transforms.Resize([300,300]),
        transforms.ToTensor(),
        transforms.Normalize([0.541211, 0.51455045, 0.45166057], [0.23086329, 0.23280795, 0.23413937])])
    return data_transform



def main():
    correct_num = 0
    logging.info("================>>开始预测<<================")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.json_path, "r") as f:
        class_indict = json.load(f)
    # 加载图像
    for class_ in os.listdir(args.test_dir):
        # 每一个类别的路径
        class_path = os.path.join(args.test_dir,class_)
        all_num = len(os.listdir(class_path))
        for filename in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path,filename)
            if not os.path.exists(image_path):
                logging.info("image_path is not exists!!!")
            else:
                image = Image.open(image_path)
                image = trf()(image)
                image = torch.unsqueeze(image, dim=0)
                # 1.model 第一次放入GPU
                create_model = MobileNetV2(num_classes=2).to(device)
                # 2. model权重文件
                assert os.path.exists(args.weights_path), "file: '{}' dose not exist.".format(args.weights_path)
                # 3.权重文件放入模型中
                create_model.load_state_dict(torch.load(args.weights_path, map_location=device))
                create_model.eval()
                with torch.no_grad():  # 第二次   数据需要放入GPU
                    output = torch.squeeze(create_model(image.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy() # # predict_cla 3
                    predict_class = class_indict[str(predict_cla)] # 预测的类别,要跟实际的类别进行一个比较
                    if predict_class != class_:
                        '''在这个地方，不打算去预测准确率，去将不符合条件的照片筛选出来'''
                        correct_num += 1
                        print("all_num is {},准确的类别是：{},但这张照片{}不符合条件！".format(all_num,class_,image_path))
                        os.remove(image_path)
                        print("不符合条件的照片{}已删除,已删除的数量是{}!".format(image_path,correct_num))
        
        correct_num = 0
    logging.info("================>>结束预测！<<================")

if __name__ == '__main__':
    main()
