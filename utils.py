import os
import numpy as np
import pandas as pd
from label_mapping import LabelMapping


def build_ct_scans(base_path):
    images = os.listdir(base_path)
    print('file number: ', len(images))
    images = [x[:-4] for x in images if x.endswith('.mhd')]
    print('scan number: ', len(images))

    transformmatrix = []
    x_dim, y_dim, z_dim = [], [], []
    x_offset, y_offset, z_offset = [], [], []
    x_spacing, y_spacing, z_spacing = [], [], []

    for name in images:
        path = os.path.join(base_path, name + '.mhd')

        with open(path) as f:
            content = f.readlines()


        if not content[-3].startswith('DimSize'): print('warn')
        x_dim.append((int(content[-3].split()[-3])))
        y_dim.append((int(content[-3].split()[-2])))
        z_dim.append((int(content[-3].split()[-1])))

        if not content[-5].startswith('ElementSpacing'): print('warn')
        x_spacing.append((float(content[-5].split()[-3])))
        y_spacing.append((float(content[-5].split()[-2])))
        z_spacing.append(round(float(content[-5].split()[-1]), 2))

        if not content[-8].startswith('Offset'): print('warn')
        x_offset.append((float(content[-8].split()[-3])))
        y_offset.append((float(content[-8].split()[-2])))
        z_offset.append((float(content[-8].split()[-1])))

        if not content[-9].startswith('TransformMatrix'): print('warn')
        transformmatrix.append((float(content[-9].split()[-1])))


    df = pd.DataFrame(x_spacing, columns=['x_spacing'], index=images)
    df['y_spacing'] = y_spacing
    df['z_spacing'] = z_spacing
    df['x_dim'] = x_dim
    df['y_dim'] = y_dim
    df['z_dim'] = z_dim
    df['x_offset'] = x_offset
    df['y_offset'] = y_offset
    df['z_offset'] = z_offset
    df['transformmatrix'] = transformmatrix
    
    return df



def build_annotations(annotation_path, thick_df):
    annotation = pd.read_csv(annotation_path)
    annotation = annotation.loc[annotation.seriesuid.isin(thick_df.index)]

    x_spacing = annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'x_spacing'])
    y_spacing = annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'y_spacing'])
    z_spacing = annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'z_spacing'])

    annotation.coordX = (annotation.coordX - 
                         annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'x_offset'])) / x_spacing
    annotation.coordY = (annotation.coordY - 
                         annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'y_offset'])) / y_spacing
    annotation.coordZ = (annotation.coordZ - 
                         annotation.seriesuid.apply(lambda x: thick_df.loc[str(x),'z_offset'])) / z_spacing

    annotation.diameterX = annotation.diameterX / x_spacing
    annotation.diameterY = annotation.diameterY / y_spacing
    annotation.diameterZ = annotation.diameterZ / z_spacing
    
    return annotation


def make_patch_complete(start, end, shape):
    if start < 0:
        end = end - start
        start = 0
    if end > shape:
        start = start - (end - shape)
        end = shape
    return start, end


def patch_generator(df, annotation_df, config, zyx=[16, 128, 128]):

    count = 0
    z, y ,x = zyx
    max_iters = len(df)
    labelmapping = LabelMapping(config)
    
    while True:
        seriesuid = df.iloc[count].name
        
        load_path = os.path.join(config['load_base_path'], seriesuid + '.npy')
        ct = np.load(load_path)
        ann_df = annotation_df.query('seriesuid == %s' % seriesuid).copy()

        ann = ann_df.sample(1)
        coordX, coordY, coordZ = ann.coordX, ann.coordY, ann.coordZ

        x_start, x_end = int(coordX) - x // 2, int(coordX) + x // 2
        y_start, y_end = int(coordY) - y // 2, int(coordY) + y // 2
        z_start, z_end = int(coordZ) - z // 2, int(coordZ) + z // 2

        x_start, x_end = make_patch_complete(x_start, x_end, ct.shape[2])
        y_start, y_end = make_patch_complete(y_start, y_end, ct.shape[1])
        z_start, z_end = make_patch_complete(z_start, z_end, ct.shape[0])
        
        # image
        image = ct[z_start:z_end, y_start:y_end, x_start:x_end].clip(-1024, 1024) / 1024
        
        # annotation concert
        ann.coordX = ann.coordX - x_start
        ann.coordY = ann.coordY - y_start
        ann.coordZ = ann.coordZ - z_start
        
        ann_df.coordX = ann_df.coordX - x_start
        ann_df.coordY = ann_df.coordY - y_start
        ann_df.coordZ = ann_df.coordZ - z_start        
                
        ann_patch = ann_df.query('0 < coordX < %d & 0 < coordY < %d & 0 < coordZ < %d' % (image.shape[2], 
                                                                                          image.shape[1], 
                                                                                          image.shape[0]))
        
        del ct
        
        # label
        label = labelmapping(input_size=zyx, ann_patch=ann_patch)

        yield seriesuid, image, label, ann, ann_patch, ann_df
        
        count += 1
        if count == max_iters:
            df = df.sample(frac=1)
            count = 0