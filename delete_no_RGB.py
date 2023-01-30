#coding=gbk
import os
import shutil
from PIL import Image
img_path = '/algdata01/yiguo.huang/yanye/test6mobilenet/Data/data20221205_class/data_C03_F03/data_C03_F03/F03/'
filenames = os.listdir(img_path)
for filename in filenames:
    image_path = os.path.join(img_path,filename)
    fp = open(image_path, 'rb')
    image = Image.open(fp)
    fp.close()
    if image.mode != 'RGB':
        os.remove(image_path)
        print('11')
