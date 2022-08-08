# coding:utf-8

import os
import random
import argparse
import shutil


def try1():
    test_percent = 1.0  # 训练集和验证集所占比例。 这里没有划分测试集
    train_percent = 0.9  # 训练集所占比例，可自己进行调整
    xmlfilepath = r"d:\download\NEU-DET\TRAIN\ANNOTATIONS"
    txtsavepath = r"d:\download\NEU-DET\TRAIN\ANNOTATIONS_txt"
    total_xml = os.listdir(xmlfilepath)
    if os.path.exists(txtsavepath):
        shutil.rmtree(txtsavepath)
    os.makedirs(txtsavepath)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * test_percent)
    tr = int(num * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_trainval = open(txtsavepath + '/trainval.txt', 'w')
    file_test = open(txtsavepath + '/test.txt', 'w')
    file_train = open(txtsavepath + '/train.txt', 'w')
    file_val = open(txtsavepath + '/val.txt', 'w')

    for i in list_index:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


import xml.etree.ElementTree as ET

import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_name, txt_folder_path, xml_folder_path):
    classes = ['crazing', 'patches', 'inclusion', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # in_file = open('./indata/' + image_name[:-3] + 'xml')  # xml文件路径
    out_txt_path = os.path.join(txt_folder_path, image_name[:-3] + 'txt')
    xml_path = os.path.join(xml_folder_path, image_name[:-3] + 'xml')
    out_file_handle = open(out_txt_path, 'a')
    xml_file_handle = open(xml_path, 'r')

    xml_text = xml_file_handle.read()
    root = ET.fromstring(xml_text)
    xml_file_handle.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        # save two decimal places
        bb = [round(bb[0] * 100) / 100, round(bb[1] * 100) / 100, round(bb[2] * 100) / 100, round(bb[3] * 100) / 100]

        out_file_handle.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    out_file_handle.close()


def xml2txt():
    ##########nhuk#################################### param setting
    classes = ['crazing', 'patches', 'inclusion', 'pitted_surface', 'rolled-in_scale', 'scratches']
    dataset_base_path = r"datasets/neu_det"
    ##########nhuk####################################
    txt_folder = os.path.join(dataset_base_path, "ANNOTATIONS_txt")
    anno_folder = os.path.join(dataset_base_path, "ANNOTATIONS")
    image_folder = os.path.join(dataset_base_path, "IMAGES")
    if os.path.exists(txt_folder):
        shutil.rmtree(txt_folder)
    os.makedirs(txt_folder)

    for image_path in os.listdir(image_folder):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name, txt_folder, anno_folder)


def train_test_split():
    ##########nhuk#################################### param setting
    dataset_base_path = r"datasets/neu_det"
    ##########nhuk####################################
    anno_folder = os.path.join(dataset_base_path, "ANNOTATIONS")
    image_folder = os.path.join(dataset_base_path, "IMAGES")
    split_info_folder = os.path.join(dataset_base_path, "split_info")

    trainval_percent = 0.9  # trainval:test = 0.8:0.2
    train_percent = 0.9  # train:val = 0.7:0.3
    total_xml = os.listdir(anno_folder)
    txtsavepath = split_info_folder

    if os.path.exists(txtsavepath):
        shutil.rmtree(txtsavepath)
    os.makedirs(txtsavepath)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval_list = random.sample(list_index, tv)
    train_list = random.sample(trainval_list, tr)

    file_trainval = open(txtsavepath + '/trainval.txt', 'w')
    file_test = open(txtsavepath + '/test.txt', 'w')
    file_train = open(txtsavepath + '/train.txt', 'w')
    file_val = open(txtsavepath + '/val.txt', 'w')

    for i in list_index:
        name = total_xml[i][:-4] + '\n'
        if i in trainval_list:
            file_trainval.write(name)
            if i in train_list:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


if __name__ == '__main__':
    train_test_split()
