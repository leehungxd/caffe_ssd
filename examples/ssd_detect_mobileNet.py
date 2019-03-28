import numpy as np  
import sys,os  
import cv2
caffe_root = '/caffe/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe
import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from myTools import myTools
mytool = myTools()
from tqdm import tqdm

caffe.set_device(0)
caffe.set_mode_gpu()

net_file= '/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/MobileNetSSD_deploy.prototxt'
caffe_model='/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_7000.caffemodel'
test_dir = "/caffe/test_data/JPEGImages"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

# load datasets labels
# labelmap_file = '/caffe/train_data/anngic/labelmap_anngic.prototxt'
labelmap_file = '/caffe/train_data/bdd100k/labelmap_bdd100k.prototxt'
# labelmap_file = '/caffe/caffe-ssd/data/VOC0712/labelmap_voc.prototxt'

file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def parse_outputs(detections, thr):
    # Parse the outputs.
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= thr]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    return top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax

def draw_bounding_boxes(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax):

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):

        xmin = int(round(top_xmin[i] * image.shape[1] + 10**-10))
        ymin = int(round(top_ymin[i] * image.shape[0] + 10**-10))
        xmax = int(round(top_xmax[i] * image.shape[1] + 10**-10))
        ymax = int(round(top_ymax[i] * image.shape[0] + 10**-10))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    image = origimg[:,:,(2,1,0)]
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img

    '''original use opencv'''
    # out = net.forward()
    # box, conf, cls = postprocess(origimg, out)
    #
    # for i in range(len(box)):
    #     if conf[i] > 0.1:
    #         p1 = (box[i][0], box[i][1])
    #         p2 = (box[i][2], box[i][3])
    #         cv2.rectangle(origimg, p1, p2, (0,255,0), 2)
    #         p3 = (max(p1[0], 15), max(p1[1], 15))
    #         title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
    #         cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow("SSD", origimg)
    # k = cv2.waitKey(0) & 0xff
    # # Exit if ESC pressed
    # if k == 27: return False

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = \
        parse_outputs(detections, 0.3)

    draw_bounding_boxes(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax)

    plt.show()

    return True


if __name__ == '__main__':

    testFolder = '/caffe/test_data/Annotations'
    detectResults = []
    for i in range(1):

        threshold = 0.2
        print('The threshold is %f.' % threshold)
        TPFPFN = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        files = mytool.GetFileList(testFolder, 'xml')
        for fileName in tqdm(files):
            print()

    for f in os.listdir(test_dir):
        print(test_dir + "/" + f)
        if detect(test_dir + "/" + f) == False:
           break
