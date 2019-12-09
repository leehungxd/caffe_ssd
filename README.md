# caffe
在caffe_ssd的基础上，通过替换基础网络，实现了mobileNetV1_ssd,mobileNetV2_ssd,shuffleNetV1_ssd,shuffleNetV2_ssd等轻量级目标检测网络；
支持特殊层：
  depthwise_conv(conv_dw)；
  focal_loss;
  axpy;
  shuffle_channel;
