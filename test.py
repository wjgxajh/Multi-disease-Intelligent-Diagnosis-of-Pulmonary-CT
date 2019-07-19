import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('myspace/')
from utils import build_ct_scans, build_annotations, patch_generator

#%matplotlib inline
# obtain the data details with the path saved mhd
thick_df= build_ct_scans('./data/mhd/')
thick_df['unify'] = thick_df.z_spacing.apply(lambda x: round(x) == 5)

# obtain the annotation details with the csv file and the data detail
annotation = build_annotations('chestCT_round1_annotation.csv', thick_df)

#-----------------------------------------------------------------------------------------------------------
# only preprocess the data with annotations and z spacing is 5 mm 
thick_df['label_num'] = [annotation.query('seriesuid == "%s"' % x).shape[0] for x in thick_df.index]
thick_train_df = thick_df.query('unify == True & label_num > 0').copy()
thick_train_df

# the config used to generate 3D labels
# the parameters is setted without careful adjustment, particular the anchors.
config = {}

config['phase'] = 'train'
config['load_base_path'] = './data/npy/'

config['stride'] = 2
config['anchors'] = [[1,5,5], [1,10,10], [2,20,20], [3,30,30], [5,50,50]]
config['num_neg'] = 2000
config['ann_label_list'] = [1,5,31,32]

config['th_neg'] = 0.02
config['th_pos_val'] = 1
config['th_pos_train'] = 0.2

config

#-------------------------------------------------------------------------------------------------------------
# get the 3D patch 
generator = patch_generator(thick_train_df.sample(frac=1), annotation, config)
seriesuid, image, label, ann, ann_patch, ann_df = next(generator)
print(seriesuid, image.shape, label.shape)
# 0 for nodule, 1 for stripe, 2 for artery, 3 for lymph, 4 for negative anchors
print((label[:,:,:,:,0] == 1).sum(), (label[:,:,:,:,1] == 1).sum(), (label[:,:,:,:,2] == 1).sum(), 
      (label[:,:,:,:,3] == 1).sum(), (label[:,:,:,:,4] == 1).sum())

# the annotation in the patch
ann_patch

#可视化初始化变量 ------------------------------------------------------
input_size = [16, 128, 128]
stride = config['stride']
anchors = np.asarray(config['anchors'])

# compute the output size
output_size = []
for i in range(3):
    assert(input_size[i] % stride == 0)
    output_size.append(input_size[i] // stride)

# compute the anchors location
offset = (stride - 1) / 2
oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)


#可视化正样本的锚点 ------------------------------------------------------
z_loc, y_loc, x_loc, a_loc, _ = np.where(label[:,:,:,:,:4] == 1)

z_center, y_center, x_center, a_center = oz[z_loc], oh[y_loc], ow[x_loc], anchors[a_loc]
z_anchor, y_anchor, x_anchor = a_center.T
fig, axes = plt.subplots(4,4,figsize=(16,16))

count = 0
for axe in axes:
    for ax in axe:
        ax.imshow(image[count], cmap=plt.cm.gray)
        
        ann_series = ann.squeeze()
        
        for coordx, coordy, coordz, diax, diay, diaz in zip(ann_patch.coordX, ann_patch.coordY, ann_patch.coordZ, 
                                                            ann_patch.diameterX, 
                                                            ann_patch.diameterY, 
                                                            ann_patch.diameterZ):
            # print(coordx, coordy, coordz, diax, diay, diaz)
            if coordz - diaz / 2 < count < coordz + diaz / 2:
                ax.add_artist(plt.Rectangle((coordx - diax / 2, coordy - diay / 2), diax, diay,
                                            fill=False, color='r'))
                    
                    
        for coordx, coordy, coordz, diax, diay, diaz in zip(x_center, y_center, z_center, x_anchor,y_anchor,z_anchor):
            # print(coordz - diaz / 2, count, coordz + diaz / 2)
            if coordz - diaz / 2 <= count <= coordz + diaz / 2:
                ax.add_artist(plt.Rectangle((coordx - diax / 2, coordy - diay / 2), diax, diay,
                                            fill=False, color='y'))
        ax.set_axis_off()
        ax.set_title(count)
        count += 1

#可视化负的锚点 ------------------------------------------------------
neg_vis_num = 300

z_loc, y_loc, x_loc, a_loc = np.where(label[:,:,:,:,4] == 1)
z_center, y_center, x_center, a_center = oz[z_loc], oh[y_loc], ow[x_loc], anchors[a_loc]
z_anchor, y_anchor, x_anchor = a_center.T

sample_index = random.sample(range(len(z_anchor)), neg_vis_num)
z_center, y_center, x_center = z_center[sample_index], y_center[sample_index], x_center[sample_index]
z_anchor, y_anchor, x_anchor = z_anchor[sample_index], y_anchor[sample_index], x_anchor[sample_index]
fig, axes = plt.subplots(4,4,figsize=(16,16))
count = 0
for axe in axes:
    for ax in axe:
        ax.imshow(image[count], cmap=plt.cm.gray)
        ann_series = ann.squeeze()
        for coordx, coordy, coordz, diax, diay, diaz in zip(ann_patch.coordX, ann_patch.coordY, ann_patch.coordZ, 
                                                            ann_patch.diameterX, 
                                                            ann_patch.diameterY, 
                                                            ann_patch.diameterZ):
            # print(coordx, coordy, coordz, diax, diay, diaz)
            if coordz - diaz / 2 < count < coordz + diaz / 2:
                ax.add_artist(plt.Rectangle((coordx - diax / 2, coordy - diay / 2), diax, diay,
                                            fill=False, color='r'))
                    
                    
        for coordx, coordy, coordz, diax, diay, diaz in zip(x_center, y_center, z_center, x_anchor,y_anchor,z_anchor):
            # print(coordz - diaz / 2, count, coordz + diaz / 2)
            if coordz - diaz / 2 <= count <= coordz + diaz / 2:
                ax.add_artist(plt.Rectangle((coordx - diax / 2, coordy - diay / 2), diax, diay,
                                            fill=False, color='y'))
                    
                    
        ax.set_axis_off()
        ax.set_title(count)
        
        count += 1
