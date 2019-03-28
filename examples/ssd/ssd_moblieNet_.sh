cd /caffe/caffe-ssd
./build/tools/caffe train \
--solver="/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/solver.prototxt" \
--gpu 0 2>&1 | tee /caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/MobileNet_BDD100K_SSD_300x300.log
# --snapshot="/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_4000.solverstate" \
# --weights="/caffe/caffe-ssd/models/MobileNet/mobilenet_iter_73000.caffemodel" \
