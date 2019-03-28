import sys, os
os.chdir('/caffe/caffe-ssd/')
sys.path.insert(0, 'python')
import caffe, cv2, random, datetime, time, gc
import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from myTools import myTools
mytool = myTools()
from tqdm import tqdm

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = '/caffe/caffe-ssd/models/MobileNet/anngic/SSD_300x300/' \
            'MobileNetSSD_deploy.prototxt'
model_weights = '/caffe/caffe-ssd/models/MobileNet/anngic/SSD_300x300/' \
                'Mobile_anngic_SSD_300x300_iter_8000.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
def trainsformerSet(modelType):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    if modelType == 'MobileNet':
        transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))  # mean pixel
    elif modelType == 'VGGNet':
        transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    return transformer

def labelMapSet(dataSets):
    # load datasets labels
    if dataSets == 'anngic':
        labelmap_file = '/caffe/train_data/anngic/labelmap_anngic.prototxt'
    elif dataSets == 'bdd100k':
        labelmap_file = '/caffe/train_data/bdd100k/labelmap_bdd100k.prototxt'
    elif dataSets == 'voc':
        labelmap_file = '/caffe/caffe-ssd/data/VOC0712/labelmap_voc.prototxt'

    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    return labelmap

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

def parse_outputs(detections, thr, labelmap):
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

def draw_bounding_boxes_opencv(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax):

    color_map = {'car': (255, 0, 0), 'person': (0, 255, 0), 'bus': (0, 0, 255),
                 'truck':(156, 102, 31), "traffic light":(255,99,71),
                 "traffic sign":(0,255,255), "rider":(8,46,84),
                 "bike":(118,128,105), "motor":(255,215,0), "train":(160,32,240)}

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = color_map[label_name]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, display_txt, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

def outPutToBBAndConf(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax):

    detections = []
    for i in range(len(top_conf)):
        detection = []
        # detection.append(top_conf[i])
        detection.append(top_labels[i])

        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))

        detection.append([xmin, ymin, xmax, ymax])

        detections.append(detection)
    return detections

if __name__ == '__main__':

    # set net to batch size of 1
    image_resize = 300
    net.blobs['data'].reshape(1, 3, image_resize, image_resize)

    inputModel = 'image'
    dataSets = 'anngic'
    modelType = 'MobileNet'
    staticMAP = False
    transformer = trainsformerSet(modelType)
    labelmap = labelMapSet(dataSets)

    if inputModel == 'image':
        testFolder = '/caffe/test_data/Annotations'
        detectResults = []
        for i in range(1):

            threshold = 0.3
            print('The threshold is %f.' % threshold)
            TPFPFN = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            files = mytool.GetFileList(testFolder, 'xml')
            for fileName in tqdm(files):

                startTime = time.clock()#datetime.datetime.now()
                imgName = fileName.replace('.xml', '.jpg')
                imgPath = imgName.replace('Annotations', 'JPEGImages')

                image = caffe.io.load_image(imgPath)
                plt.imshow(image)

                transformed_image = transformer.preprocess('data', image)

                if modelType == 'MobileNet':
                    transformed_image *= 0.007843

                net.blobs['data'].data[...] = transformed_image

                # Forward pass.
                detections = net.forward()['detection_out']

                # Parse the outputs.
                top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = \
                    parse_outputs(detections, threshold, labelmap)
                del detections, transformed_image, imgName, imgPath
                gc.collect()

                if staticMAP == True:

                    xmlPath = os.path.join(testFolder, fileName)
                    detectionResults = outPutToBBAndConf(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax)
                    groundTurthes = mytool.ParseObjectLocationXml(xmlPath)
                    carTP, carFP, carFN, riderTP, riderFP, riderFN, personTP, personFP, personFN = \
                        mytool.StatisticAP(detectionResults, groundTurthes)
                    del top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax, \
                        detectionResults, groundTurthes
                    gc.collect()
                    # TPFPFN[0] += carTP
                    # TPFPFN[1] += carFP
                    # TPFPFN[2] += carFN
                    # TPFPFN[3] += riderTP
                    # TPFPFN[4] += riderFP
                    # TPFPFN[5] += riderFN
                    # TPFPFN[6] += personTP
                    # TPFPFN[7] += personFP
                    # TPFPFN[8] += personFN
                    TPFPFN += [carTP, carFP, carFN, riderTP, riderFP, riderFN, personTP, personFP, personFN]
                    del carTP, carFP, carFN, riderTP, riderFP, riderFN, personTP, personFP, personFN
                    gc.collect()
                else:
                    # Draw bounding boxes of all objects
                    draw_bounding_boxes(image, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax)

                endTime = time.clock()#datetime.datetime.now()

                plt.show()
            carP, carR, riderP, riderR, personP, personR, map = mytool.StatisticMapPreciseRecall(TPFPFN)
            detectResult = ('threshold = %f,carP = %f, carR = %f, riderP = %f, riderR = %f, personP = %f, personR = %f, MAP = %f'
                  %(threshold, carP, carR, riderP, riderR, personP, personR, map))

            detectResults.append(detectResult)
            del carP, carR, riderP, riderR, personP, personR, map, TPFPFN, files, detectResult
            gc.collect()

        if staticMAP == True:
            algorithm = 'SSD'
            RPMapFile = '/caffe/test_data/PRMap_%s_%s.txt'%(modelType, algorithm)
            PRMapFile = open(RPMapFile, 'a')
            for item in detectResults:
                PRMapFile.write(item + '\n')
            PRMapFile.close()

    elif inputModel == 'video':
        testFile = '/caffe/test_data/rider.avi'
        capture = cv2.VideoCapture(testFile)
        frameCount = 1

        if (capture.isOpened() == False):
            print('Error opening video stream or file!!!')

        while(capture.isOpened()):
            ret, frame = capture.read()
            if ret == True:

                img = frame
                if modelType == 'MobileNet':
                    transformed_image = cv2.resize(frame, (300, 300))
                    transformed_image = transformed_image - 127.5
                    if modelType == 'MobileNet':
                        transformed_image *= 0.007843

                    img = transformed_image.astype(np.float32)
                    img = img.transpose((2, 0, 1))
                    img = np.expand_dims(img, axis=0)
                elif modelType == 'VGGNet':
                    img = transformer.preprocess('data', frame)
                net.blobs['data'].data[...] = img

                # Forward pass.
                detections = net.forward()['detection_out']

                # Parse the outputs.
                top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = \
                    parse_outputs(detections, 0.2, labelmap)

                # Draw bounding boxes of all objects
                draw_bounding_boxes_opencv(frame, top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            cv2.imshow('video detection', frame)

        capture.release()