#!usr/bin/env python
# coding:utf-8
import sys
import os
os.chdir('/caffe/caffe-ssd/')
sys.path.insert(0, 'python')
import shutil
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import multiprocessing
import time
import collections
import signal
import caffe
import argparse


def prune_layer(prototxt_old, caffemodel, layer_name, layer_name_next, prune_ratio):
    # 剪枝，并且保存新的caffemodel
    net = caffe.Net(prototxt_old, caffemodel, caffe.TEST)
    W = net.params[layer_name][0].data
    W_next = net.params[layer_name_next][0].data
    output_num_1 = net.params[layer_name][0].data.shape[0]
    output_num_2 = net.params[layer_name][0].data.shape[1]
    prune_num = np.int(output_num_1 * prune_ratio)  # 需要砍的通道数
    L1_norm_list = np.zeros(output_num_1)
    for i in range(output_num_1):
        for j in range(output_num_2):
            L1_norm_list[i] += np.linalg.norm(net.params[layer_name][0].data[i][j], ord=1)
    sorted_index = np.argsort(L1_norm_list)
    prune_list = sorted_index[0:prune_num]
    _index = np.sort(prune_list)
    reverse_index = _index[::-1]
    for t in range(len(reverse_index)):
        W = np.delete(W, (reverse_index[t]), axis=0)
        W_next = np.delete(W_next, (reverse_index[t]), axis=1)
    print '需要砍掉的通道数为：{}'.format(str(prune_num))

    print L1_norm_list[sorted_index]
    plt.hist(L1_norm_list)
    plt.show()
    return W, W_next


def prune_layer_2(prototxt_old, caffemodel, layer_name, layer_name_next,
                  layer_name_loc, layer_name_conf, layer_name_norm, prune_ratio):
    # 剪枝，并且保存新的caffemodel
    net = caffe.Net(prototxt_old, caffemodel, caffe.TEST)
    W = net.params[layer_name][0].data
    W_next = net.params[layer_name_next][0].data
    #     W_next = []
    #     W_next_norm = net.params[layer_name_norm][0].data
    W_next_norm = []
    W_next_loc = net.params[layer_name_loc][0].data
    W_next_conf = net.params[layer_name_conf][0].data
    output_num_1 = net.params[layer_name][0].data.shape[0]
    output_num_2 = net.params[layer_name][0].data.shape[1]
    prune_num = np.int(output_num_1 * prune_ratio)  # 需要砍的通道数
    L1_norm_list = np.zeros(output_num_1)
    for i in range(output_num_1):
        for j in range(output_num_2):
            L1_norm_list[i] += np.linalg.norm(net.params[layer_name][0].data[i][j], ord=1)
    sorted_index = np.argsort(L1_norm_list)
    prune_list = sorted_index[0:prune_num]
    _index = np.sort(prune_list)
    reverse_index = _index[::-1]
    for t in range(len(reverse_index)):
        W = np.delete(W, (reverse_index[t]), axis=0)
        W_next = np.delete(W_next, (reverse_index[t]), axis=1)
        #         W_next_norm = np.delete(W_next_norm,(reverse_index[t]),axis=0)
        W_next_loc = np.delete(W_next_loc, (reverse_index[t]), axis=1)
        W_next_conf = np.delete(W_next_conf, (reverse_index[t]), axis=1)
    print '需要砍掉的通道数为：{}'.format(str(prune_num))

    print L1_norm_list[sorted_index]
    plt.hist(L1_norm_list)
    plt.show()
    return W, W_next, W_next_loc, W_next_conf, W_next_norm

def prune_conf_loc_layer(prototxt_old, caffemodel, layers, prune_ratio):
    # 剪枝，并且保存新的caffemodel
    net = caffe.Net(prototxt_old, caffemodel, caffe.TEST)
    w_all =[]
    for layer_name in layers:
        W = net.params[layer_name][0].data

        output_num_1 = net.params[layer_name][0].data.shape[0]
        output_num_2 = net.params[layer_name][0].data.shape[1]
        prune_num = np.int(output_num_1 * prune_ratio)  # 需要砍的通道数
        L1_norm_list = np.zeros(output_num_1)
        for i in range(output_num_1):
            for j in range(output_num_2):
                L1_norm_list[i] += np.linalg.norm(net.params[layer_name][0].data[i][j], ord=1)
        sorted_index = np.argsort(L1_norm_list)
        prune_list = sorted_index[0:prune_num]
        _index = np.sort(prune_list)
        reverse_index = _index[::-1]
        for t in range(len(reverse_index)):
            W = np.delete(W, (reverse_index[t]), axis=0)
        print '需要砍掉的通道数为：{}'.format(str(prune_num))

        print L1_norm_list[sorted_index]
        # plt.hist(L1_norm_list)
        # plt.show()
        w_all.append(W)
    return w_all

if __name__ == '__main__':

    prune = 'last'

    if prune == 'middle':
        prototxt = '/caffe/caffe-ssd/models/VGGNet/BDD100K/SSD_300x300/deploy.prototxt'  # 'vgg_deploy.prototxt'
        caffemodel = '/caffe/caffe-ssd/models/VGGNet/BDD100K/SSD_300x300/bdd_small_objects_remove_models/VGG_BDD100K_SSD_300x300_iter_10000.caffemodel'  # 'VGG16_SSD300_Car_V97.caffemodel'
        layer1 = 'conv4_3_norm_mbox_conf'
        layer2 = 'conv4_3_norm_mbox_conf_perm'
        ratio = 0.64
        layernew1 = 'conv4_3_norm_mbox_conf0'
        layernew2 = 'conv4_3_norm_mbox_conf_perm0'
        prototxtnew = '/caffe/caffe-ssd/models/VGGNet/BDD100K/SSD_300x300/deploy4_3.prototxt'  # 'vgg_deploy_new.prototxt'
        caffemodelout = '/caffe/caffe-ssd/models/VGGNet/BDD100K/SSD_300x300/bdd_small_objects_remove_models/VGG_BDD100K_SSD_300x300_iter_10000_4_3.caffemodel'  # 'VGG16_SSD300_Car_V98.caffemodel'

        W, W_next = prune_layer(prototxt, caffemodel, layer1, layer2, ratio)
        print '请根据需要砍掉的通道数，更新prototxt（更新待砍层和下一层的层名以及待砍层output通道数），修改完后按任意键继续进程'
        # raw_input("Press Enter to continue ...")

        net = caffe.Net(prototxtnew, caffemodel, caffe.TEST)
        net.params[layernew1][0].data[...] = W
        net.params[layernew2][0].data[...] = W_next
        net.save(caffemodelout)
    elif prune == 'last':
        prototxt = '/caffe/caffe-ssd/models/MobileNet/mobilenet_iter_73000.prototxt'  # 'vgg_deploy.prototxt'
        caffemodel = '/caffe/caffe-ssd/models/MobileNet/mobilenet_iter_73000.caffemodel'  # 'VGG16_SSD300_Car_V97.caffemodel'

        layers_new = ['conv11_mbox_conf','conv13_mbox_conf','conv14_2_mbox_conf','conv15_2_mbox_conf','conv16_2_mbox_conf','conv17_2_mbox_conf']
        ratio = 0.48

        layers = ['conv11_mbox_conf0','conv13_mbox_conf0','conv14_2_mbox_conf0','conv15_2_mbox_conf0','conv16_2_mbox_conf0','conv17_2_mbox_conf0']
        prototxtnew = '/caffe/caffe-ssd/models/MobileNet/mobilenet_iter_73000_0.prototxt'  # 'vgg_deploy_new.prototxt'
        caffemodelout = '/caffe/caffe-ssd/models/MobileNet/mobilenet_iter_73000_0.caffemodel'  # 'VGG16_SSD300_Car_V98.caffemodel'

        W1 = prune_conf_loc_layer(prototxt, caffemodel, layers, ratio)

        # print '请根据需要砍掉的通道数，更新prototxt（更新待砍层和下一层的层名以及待砍层output通道数），修改完后按任意键继续进程'
        # raw_input("Press Enter to continue ...")

        net = caffe.Net(prototxtnew, caffemodel, caffe.TEST)
        for i in range(len(layers_new)):
            net.params[layers_new[i]][0].data[...] = W1[i]


        net.save(caffemodelout)