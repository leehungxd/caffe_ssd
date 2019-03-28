import numpy as np  
import sys,os  
import cv2
caffe_root = '/caffe/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= '/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/MobileNetSSD_deploy.prototxt'
caffe_model='/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_10000.caffemodel'
test_dir = "/caffe/test_data/JPEGImages"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

# CLASSES = ('background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ( "none_of_the_above", "traffic light", "traffic sign", "car",
            "person", "bus", "truck", "rider", "bike", "motor", "train")


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
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        if conf[i] > 0.1:
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(origimg, p1, p2, (0,255,0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True


for f in os.listdir(test_dir):
    print(test_dir + "/" + f)
    if detect(test_dir + "/" + f) == False:
       break
