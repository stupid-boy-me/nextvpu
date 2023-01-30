import os
from tqdm import tqdm
pth_path = "/algdata01/yiguo.huang/yanye/test6mobilenet/weights/weights_qingza_non_qingza_new//"
best = 0.0
for pth_name in os.listdir(pth_path):
    acc_split = pth_name.split('-')[3].split('.')
    acc = float(acc_split[0]+"."+acc_split[1])
    print(acc)
    if acc >= best:
        best = acc

for pth_name1 in os.listdir(pth_path):
    if float(pth_name1.split('-')[3].split('.')[0]+"."+pth_name1.split('-')[3].split('.')[1]) == best:
        print(os.path.join(pth_path,pth_name1))
