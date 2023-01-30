# coding:utf-8
import os
from PIL import Image
from tqdm import tqdm

# bmp 转换为png
def bmpToJpg(file_path,src_dir):
    for fileName in tqdm(os.listdir(file_path)):
        # print('--fileName--', fileName)  # 看下当前文件夹内的文件名字
        # print(fileName)
        newFileName = fileName[0:fileName.find(".")] + ".png"  # 改后缀
        # print('--newFileName--', newFileName)
        im = Image.open(file_path + "/" + fileName)
        im.save(src_dir + "/" + newFileName)  # 保存到当前文件夹内


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "del " + file_path + "/*." + imageFormat
    os.system(command)


def main():
    file_path = r"E:\HYG\20221220\f03"
    src_dir =   r'E:\HYG\20221220\F03_png'
    bmpToJpg(file_path,src_dir)
    # deleteImages(file_path, "bmp")


if __name__ == '__main__':
    main()
