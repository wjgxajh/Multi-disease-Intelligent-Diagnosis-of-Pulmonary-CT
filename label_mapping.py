'''
modified from https://raw.githubusercontent.com/lfz/DSB2017/master/training/detector/data.py
'''

import random
import numpy as np

class LabelMapping(object):
    '''
    genrate the label according to the annotation.
    '''
    def __init__(self, config):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = config['phase']
        self.ann_label_list = config['ann_label_list']
        if self.phase == 'train':
            self.th_pos = config['th_pos_train']
        elif self.phase == 'val':
            self.th_pos = config['th_pos_val']

            
    def __call__(self, input_size, ann_patch, Debug=False):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        ann_label_list = self.ann_label_list
        
        bboxes = np.array(ann_patch.loc[:, ['coordZ', 'coordY', 'coordX', 'diameterZ', 'diameterY', 'diameterX']])
        
        
        # compute the output size
        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] // stride)
        

        # initialize the label, 11 include 5 one-hot encoding.
        label = -1 * np.ones(output_size + [len(anchors), 11], np.float32)
        offset = (stride - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)


        # filter the anchors with iou > th_neg.
        for bbox, ann_label in zip(bboxes, ann_patch.label.tolist()):
            for i, anchor in enumerate(anchors):
                # print(i, ann_label, anchor, bbox)
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 4] = 0
        
        # reverse the above anchors to generate the negative anchors.
        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 4] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, :5] = 0
            label[neg_z, neg_h, neg_w, neg_a, 4] = 1
            
            
        # genearate the positive anchor, if numbers > 1, random choose 1.
        for bbox, ann_label in zip(bboxes, ann_patch.label.tolist()):

            iz, ih, iw, ia = [], [], [], []
            for i, anchor in enumerate(anchors):
                if Debug: print(i, ann_label, anchor, bbox)
                iiz, iih, iiw = select_samples(bbox, anchor, th_pos, oz, oh, ow)
                if Debug: print(iiz.shape)
                iz.append(iiz)
                ih.append(iih)
                iw.append(iiw)
                ia.append(i * np.ones((len(iiz),), np.int64))
            iz = np.concatenate(iz, 0)
            ih = np.concatenate(ih, 0)
            iw = np.concatenate(iw, 0)
            ia = np.concatenate(ia, 0)
            if Debug: print(iz.shape, ih.shape, iw.shape, ia.shape)

            flag = True 
            if len(iz) == 0:
                pos = []
                for i in range(3):
                    pos.append(max(0, int(np.round((bbox[i] - offset) / stride))))
                idx = np.argmin(np.abs(np.log(bbox[3] / anchors[:, 0])) + 
                                np.abs(np.log(bbox[4] / anchors[:, 1])) + 
                                np.abs(np.log(bbox[5] / anchors[:, 2])))
                pos.append(idx)
                flag = False
            else:
                idx = random.sample(range(len(iz)), 1)[0]
                pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
            dz = (bbox[0] - oz[pos[0]]) / anchors[pos[3], 0]
            dh = (bbox[1] - oh[pos[1]]) / anchors[pos[3], 1]
            dw = (bbox[2] - ow[pos[2]]) / anchors[pos[3], 2]
            diaz = np.log(bbox[3] / anchors[pos[3], 0])
            diah = np.log(bbox[4] / anchors[pos[3], 1])
            diaw = np.log(bbox[5] / anchors[pos[3], 2])
            one_hot_label = list(np.eye(5)[ann_label_list.index(ann_label)])
            label[pos[0], pos[1], pos[2], pos[3], :] = one_hot_label + [dz, dh, dw, diaz, diah, diaw]

        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, diameter = bbox
    diaz, diah, diaw = diameter
    ancz, anch, ancw = anchor

    anchor = np.array(anchor)
    diameter = np.array(diameter)

    max_overlap = [min(diaz, ancz), min(diah, anch), min(diaw, ancw)]
    min_overlap = [np.prod(anchor) + np.prod(diameter) - np.prod(max_overlap)] * 3
    min_overlap[0] = min_overlap[0] * th / max_overlap[1] / max_overlap[2]
    min_overlap[1] = min_overlap[1] * th / max_overlap[0] / max_overlap[2]
    min_overlap[2] = min_overlap[2] * th / max_overlap[0] / max_overlap[1]

    if (np.array(min_overlap) > np.array(max_overlap)).any():
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(diaz - ancz) - (max_overlap[0] - min_overlap[0])
        e = z + 0.5 * np.abs(diaz - ancz) + (max_overlap[0] - min_overlap[0])
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(diah - anch) - (max_overlap[1] - min_overlap[1])
        e = h + 0.5 * np.abs(diah - anch) + (max_overlap[1] - min_overlap[1])
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(diaw - ancw) - (max_overlap[2] - min_overlap[2])
        e = w + 0.5 * np.abs(diaw - ancw) + (max_overlap[2] - min_overlap[2])
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)

    r0 = anchor / 2
    s0 = centers - r0
    e0 = centers + r0

    r1 = diameter / 2
    s1 = bbox[:3] - r1
    s1 = s1.reshape((1, -1))
    e1 = bbox[:3] + r1
    e1 = e1.reshape((1, -1))

    overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

    intersection = np.prod(overlap, axis=1)
    union = np.prod(anchor) + np.prod(diameter) - intersection

    iou = intersection / union

    mask = iou >= th

    iz = iz[mask]
    ih = ih[mask]
    iw = iw[mask]
    return iz, ih, iw