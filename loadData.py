import SimpleITK as sitk
import numpy as np
import os
import tensorflow as tf
import csv
import matplotlib as  plt
tf.app.flags.DEFINE_string('test_data_path', 'data/ICPR2018/', '')
FLAGS = tf.app.flags

#加载数据
def loadTrainData(path):
    files = []
    dir = os.walk(path)
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('mhd'):
                itkimage = sitk.ReadImage(os.path.join(parent,filename))
                ct_scan = sitk.GetArrayFromImage(itkimage)
                files.append(ct_scan)
    # image = files[0]
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    print('Find {} images'.format(len(files)))
    return files


#加载标签数据
def loadCsvData(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        csvData = []
        for x in range(len(dataset) - 1):
            csvData.append(dataset[x])
        return csvData