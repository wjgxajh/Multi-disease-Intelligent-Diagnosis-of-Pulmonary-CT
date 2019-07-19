import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from tqdm import tqdm
#%matplotlib inline
label_dict = {}
label_dict[1]  = 'nodule'
label_dict[5]  = 'stripe'
label_dict[31] = 'artery'
label_dict[32] = 'lymph'

def load_itk(file_name, file_path):
    '''
    modified from https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
    '''
    
    # Reads the image using SimpleITK
    file = os.path.join(file_path, file_name + '.mhd')
    itkimage = sitk.ReadImage(file)

    # 首先将图像转换为numpy数组，然后按z、y、x的顺序无序排列尺寸以获取轴
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # 读取CT扫描的原点，将用于将坐标从世界转换为体素，反之亦然。
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # 沿每个尺寸读取间距
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def getConfigData(seriesuid, anns_all, file_path,
              clipmin=-1000, clipmax=600):

    seriesuid = str(seriesuid)
    ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy()  # 标注当中的数据
    ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)

    if ann_df.shape[0] == 0:
        print('no annoatation')
        del ct
        return None, None, None, ann_df

    boxes = []
    num = (ann_df.coordZ - origin[0]) / spacing[0]
    for num in tqdm(range(ann_df.shape[0])):
        ann = ann_df.values[num]
        boxes.append([(ann[3] - origin[0])/spacing[0], ann[1] - ann[4] / 2, ann[2] - ann[5] / 2, ann[1] + ann[4] / 2,ann[2] + ann[5] / 2, ann[7]])

    # coordinate transform: world to voxel
    # 世界、体素坐标的转换：
    # 体素坐标 = （标注中的坐标-mhd中的原点坐标）/mhd中的间距
    # 体素直径 = 标注中的直径/mhd中的间距
    ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
    ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
    ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]

    ann_df.diameterX = ann_df.diameterX / spacing[2]
    ann_df.diameterY = ann_df.diameterY / spacing[1]
    ann_df.diameterZ = ann_df.diameterZ / spacing[0]
    ann_df['labelstr'] = ann_df.label.apply(lambda x: label_dict[x])
    return ct, origin, spacing, boxes

def plot_scan(seriesuid, anns_all, file_path, plot_path='./visualization/', 
              clipmin=-1000, clipmax=600, only_df=False, return_ct=False):
    '''
    input:
    seriesuid: specify the scan plotted.
    anns_all:  the annotation provided (Dataframe).
    file_path: the path of the data.
    plot_path: the path of the visualization, default: make a subdirectory under the current dir.   
    clip_min:  the lower boundary which is used for clipping the CT valued for the lung window.
    clip_max:  the upper boundary which is used for clipping the CT valued for the lung window.
    only_df:   if True, only return the dataframe , and do not plot.
    return_ct: if True, return the dataframe with the ct array.
    
    return:
    ann_df:    return the annotation dataframe according to the seriesuid
    
    Mediastinum window: clipmin=-150, clipmax=250
    '''
    seriesuid = str(seriesuid)
    ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy() #标注当中的数据
    ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)
    ct_clip = ct.clip(min=clipmin, max=clipmax)
    
    # coordinate transform: world to voxel
    #世界、体素坐标的转换：
    # 体素坐标 = （标注中的坐标-mhd中的原点坐标）/mhd中的间距
    # 体素直径 = 标注中的直径/mhd中的间距
    ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
    ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
    ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]

    ann_df.diameterX = ann_df.diameterX / spacing[2]
    ann_df.diameterY = ann_df.diameterY / spacing[1]
    ann_df.diameterZ = ann_df.diameterZ / spacing[0]
    ann_df['labelstr'] = ann_df.label.apply(lambda x:label_dict[x])
    
    if ann_df.shape[0] == 0:
        print('no annoatation')
        del ct
        return ann_df
    
    # plot phase
    if not os.path.exists(plot_path): os.mkdir(plot_path)
    scan_plot_path = os.path.join(plot_path, seriesuid)
    if not os.path.exists(scan_plot_path): os.mkdir(scan_plot_path)

    for num in tqdm(range(ct_clip.shape[0])):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.imshow(ct_clip[num], cmap=plt.cm.gray)

        for _, ann in ann_df.iterrows():
            x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
            color = 'r'
            if num > z - d/2 and num < z + d / 2:
                ax.add_artist(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, color=color))
                text = label_dict[ann.label]
                ax.add_artist(plt.Text(x - w / 2, y - h / 2, text, size='x-large', color=color))

        title = (3 - len(str(num))) * '0' + str(num)
        ax.set_title(title)
        ax.axis('off')
        plt.savefig(os.path.join(scan_plot_path, title), dpi=100)
        plt.close()
    
    del ct
    return ann_df

seriesuid = '688588'
file_path = os.path.abspath('data/train')
anns_path = os.path.abspath('data/chestCT_round1_annotation.csv')

anns_all = pd.read_csv(anns_path)
plot_scan(seriesuid, anns_all, file_path)