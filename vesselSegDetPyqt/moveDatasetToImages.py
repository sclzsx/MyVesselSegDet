from pathlib import Path
import os

data_dir = 'D:/0_projects/热力图显示识别/MyVesselSegDet/datasets/eyepacs_binary_rgb_tiny/train'
save_dir = './images'

cnt = 0
for path in Path(data_dir).rglob('*.png'):
    label = path.parent.name
    os.rename(str(path), save_dir + '/' + str(label) + '_' + str(cnt) + '.png')
    cnt += 1
