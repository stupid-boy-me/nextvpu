# -*- coding:utf-8 -*-

import os
import random

class ImageRename():
    def __init__(self):
        self.path = r'输入文件夹的路径'
        print(self.path)

    def rename(self):
        filelist = os.listdir(self.path)
        random.shuffle(filelist)
        print(filelist)
        total_num = len(filelist)

        i = 1  # 图片开始名称

        for item in filelist:
            # print item
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i), '0>6s') + '.png')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
