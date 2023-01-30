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
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20230103_class_2_B03_F03/class_C02_F03_output/test/',type=str,help='type of dataset')
parser.add_argument("--weights_path",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20230103_class_2_B03_F03/weights/model-125_of_1000-0.03512895479798317-0.9169642857142857.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default='/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20230103_class_2_B03_F03/class.json',type=str,help='json_path')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trf():
    data_transform = transforms.Compose(
        [transforms.Resize([800,400]),
        transforms.ToTensor(),
        transforms.Normalize([0.23214373, 0.1577399, 0.059987396], [0.2582842, 0.17773658, 0.06860409])])
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
                    if predict_class == class_:
                        correct_num += 1

        logging.info("类别{}的准确率是{}".format(class_, correct_num / all_num))
        correct_num = 0
    logging.info("================>>结束预测！<<================")


if __name__ == '__main__':
    main()
