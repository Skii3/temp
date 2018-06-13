from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

file = 'train_val'
filesave = file + '_result/'
if os.path.exists("../data/"+filesave):
    shutil.rmtree("../data/"+filesave)
os.makedirs("../data/"+filesave)

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f



def draw_boxes(img,image_name,boxes,scale,color=None):
    base_name = image_name.split('/')[-1]
    tag = 0
    with open('../data/'+ filesave + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if color == None:
                tag = 1
                if box[8] >= 0.9:
                    color = (0, 255, 0)
                elif box[8] >= 0.8:
                    color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            if tag == 1:
                line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
                f.write(line)
    if tag == 1:
        img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("../data/"+filesave, base_name), img)
    return img

def iou(boxPred,boxLabel):
    ''' input: [N1,4] [N2,4]
        output: [N1,?]
    '''
    len_pred = len(boxPred)
    len_label = len(boxLabel)
    ratio = []
    for i in range(0,len_pred,1):
        box1 = boxPred[i]
        x1 = np.repeat(box1[0],len_label,axis=0)
        y1 = np.repeat(box1[1],len_label,axis=0)
        width1 = np.repeat(box1[2] - box1[0],len_label,axis=0)
        height1 = np.repeat(box1[3] - box1[1],len_label,axis=0)

        x2 = boxLabel[:,0]
        y2 = boxLabel[:, 1]
        width2 = boxLabel[:,2] - x2
        height2 = boxLabel[:,3] - y2

        endx = np.max([x1+width1,x2+width2],axis=0)
        startx = np.min([x1,x2],axis=0)
        width = width1 + width2 - (endx - startx)

        endy = np.max([y1 + height1, y2 + height2], axis=0)
        starty = np.min([y1, y2], axis=0)
        height = height1 + height2 - (endy - starty)

        Area = np.multiply(width,height)
        Area1 = np.multiply(width1,height1)
        Area2 = np.multiply(width2,height2)

        ratio1 = Area / (Area1+Area2-Area)
        ratio1[width <= 0] = 0
        ratio1[height <= 0] = 0
        ratio.append(ratio1)
    return ratio

def MAP(boxPred,boxLabel):
    ratio = iou(boxPred,boxLabel)
    ratio_choose = np.max(ratio,axis=0)
    thresh = 0.6
    true_positive = np.sum(ratio_choose > thresh)
    precesion = true_positive / float(len(boxPred))
    return precesion,len(boxPred)


def ctpn(sess, net, image_name,boxlabel):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)

    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    img = draw_boxes(img, image_name, boxes, scale,None)
    boxlabel2 = np.transpose(np.array(
        [boxlabel[:, 0], boxlabel[:, 1], boxlabel[:, 2], boxlabel[:, 1], boxlabel[:, 0], boxlabel[:, 3], boxlabel[:, 2],
         boxlabel[:, 3], np.ones(len(boxlabel))]))
    draw_boxes(img, image_name, boxlabel2, 1, (0, 0, 0))
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    boxes = boxes / scale
    return boxes


def read_label(im_txt):
    boxes_label = []
    temp_label = []
    tag_last = -1
    count = 0
    file_name = []
    if len(im_txt) > 1:
        im_txt = im_txt[1]
        print('Choose %s'%im_txt)
    else:
        im_txt = im_txt[0]

    f = open(im_txt)
    line = f.readline()
    while line:
        line = line.split(' ')
        tag_this = float(line[0])
        if count == 0:
            tag_last = tag_this
            file_name.append("../data/"+file+'/'+line[1])
        if tag_this != tag_last:
            file_name.append("../data/"+file+'/'+line[1])
            boxes_label.append(temp_label)
            temp_label = []
            temp_label.append(np.array(list(map(float, [w for w in line[2:6]]))))
            tag_last = tag_this
        else:
            temp_label.append(np.array(list(map(float,[w for w in line[2:6]]))))
        count = count + 1
        line = f.readline()
    boxes_label.append(temp_label)
    return boxes_label, file_name


if __name__ == '__main__':

    cfg_from_file('./text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state('../'+cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)
    #im_names = glob.glob(os.path.join(cfg.DATA_DIR, file, '*.png')) + \
    #           glob.glob(os.path.join(cfg.DATA_DIR, file, '*.jpg'))
    im_txt = glob.glob(os.path.join(cfg.DATA_DIR, file, '*.txt'))
    boxes_label,file_name = read_label(im_txt)
    idx = 0
    numSample = 0.0
    map = 0.0
    num = 0
    for im_name in file_name:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        boxes_pred = ctpn(sess, net, im_name, np.array(boxes_label[idx]))
        boxes_pred = np.array(boxes_pred)
        boxes_pred = boxes_pred[:,[0,1,6,7]]
        if len(boxes_pred) != 0:
            map1,numSample1 = MAP(boxes_pred,np.array(boxes_label[idx]))
            print('Map for %s: %f' % (im_name,map1))
            map = (map * numSample + map1 * numSample1) / (numSample + numSample1)
            numSample = numSample + numSample1
            num = num + 1
        idx = idx + 1

    print('Map for all val image: %f, num: %f' % (map,num))

