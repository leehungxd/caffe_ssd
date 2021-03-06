cd /caffe/caffe-ssd
./build/tools/caffe train \
--solver="/caffe/caffe-ssd/models/MobileNet/anngic/SSD_300x300/solver.prototxt" \
--weights="/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_7000.caffemodel" \
--gpu 0 2>&1 | tee /caffe/caffe-ssd/models/MobileNet/anngic/SSD_300x300/MobileNet_anngic_SSD_300x300.log
# --snapshot="/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_4000.solverstate" \

