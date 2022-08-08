# coding:utf-8

import os
import random
import argparse
import shutil

import xml.etree.ElementTree as ET

import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob


def check_folders(*folders):
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


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


def convert_annotation(self, image_name, txt_folder_path, xml_folder_path):
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


def convert_whole():
    txt_folder_path = r'd:\download\NEU-DET\TRAIN\ANNOTATIONS_txt'
    xml_folder_path = r'd:\download\NEU-DET\TRAIN\ANNOTATIONS'
    copy_files(xml_folder_path, txt_folder_path)
    for file in os.listdir(txt_folder_path):
        convert_annotation(file, txt_folder_path, xml_folder_path)


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


def train_val_test_split():
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


def copy_files(src_folder, dst_folder):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)
    for file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, file), dst_folder)


def alloc_data():
    original_dataset_path = r"datasets/neu_det"
    original_images_path = os.path.join(original_dataset_path, "IMAGES")
    original_annos_path = os.path.join(original_dataset_path, "ANNOTATIONS_txt")

    split_dataset_folder = r"datasets/neu_det_split"
    train_folder = os.path.join(split_dataset_folder, "train")
    train_images_folder = os.path.join(train_folder, "images")
    train_labels_folder = os.path.join(train_folder, "labels")
    val_folder = os.path.join(split_dataset_folder, "val")
    val_images_folder = os.path.join(val_folder, "images")
    val_labels_folder = os.path.join(val_folder, "labels")
    test_folder = os.path.join(split_dataset_folder, "test")
    test_images_folder = os.path.join(test_folder, "images")
    test_labels_folder = os.path.join(test_folder, "labels")

    split_info_folder = os.path.join(original_dataset_path, "split_info")
    train_txt = os.path.join(split_info_folder, "train.txt")
    val_txt = os.path.join(split_info_folder, "val.txt")
    test_txt = os.path.join(split_info_folder, "test.txt")

    check_folders(train_images_folder, train_labels_folder, val_images_folder, val_labels_folder, test_images_folder, test_labels_folder)

    def read_and_copy(txt_file, img_dst_folder, label_dst_folder, src_img_folder=original_images_path, src_label_folder=original_annos_path):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                src_img_file = os.path.join(src_img_folder, line + '.jpg')
                src_label_file = os.path.join(src_label_folder, line + '.txt')
                dst_img_file = os.path.join(img_dst_folder, line + '.jpg')
                dst_label_file = os.path.join(label_dst_folder, line + '.txt')
                shutil.copy(src_img_file, dst_img_file)
                shutil.copy(src_label_file, dst_label_file)

    read_and_copy(train_txt, train_images_folder, train_labels_folder)
    read_and_copy(val_txt, val_images_folder, val_labels_folder)
    read_and_copy(test_txt, test_images_folder, test_labels_folder)

    assert len(os.listdir(train_images_folder)) == len(os.listdir(train_labels_folder))
    assert len(os.listdir(val_images_folder)) == len(os.listdir(val_labels_folder))
    assert len(os.listdir(test_images_folder)) == len(os.listdir(test_labels_folder))
    assert len(os.listdir(original_images_path)) == len(os.listdir(train_images_folder)) + len(os.listdir(val_images_folder)) + len(os.listdir(test_images_folder))


def split_train_val_xml():
    ##########nhuk#################################### param setting
    original_path = 'datasets/neu_det'
    target_path = 'datasets/neu_det_train_val'

    # original_path = 'd:/download/NEU-DET/neu_det'
    # target_path = 'd:/download/NEU-DET/neu_det_train_val_xml'

    file_ext = '.xml'
    train_val_ratio = 0.9

    ##########nhuk####################################
    if file_ext == '.xml':
        original_anno_path = os.path.join(original_path, 'ANNOTATIONS')
    else:
        original_anno_path = os.path.join(original_path, 'ANNOTATIONS_txt')
    original_image_path = os.path.join(original_path, 'IMAGES')
    # base_path = 'datasets/neu_det_train_val_xml/'
    train_base_path = os.path.join(target_path, 'train')
    val_base_path = os.path.join(target_path, 'val')

    train_data_path = os.path.join(train_base_path, 'images')
    train_anno_path = os.path.join(train_base_path, 'labels')

    val_data_path = os.path.join(val_base_path, 'images')
    val_anno_path = os.path.join(val_base_path, 'labels')

    check_folders(train_data_path, val_data_path, train_anno_path, val_anno_path)
    all_path = os.listdir(original_image_path)
    train_num = int(len(all_path) * train_val_ratio)
    train_path_list = random.sample(all_path, train_num)
    val_path_list = list(set(all_path) - set(train_path_list))
    for path in train_path_list:
        shutil.copy(os.path.join(original_image_path, path), train_data_path)
        xml_path = path[:-4] + file_ext
        shutil.copy(os.path.join(original_anno_path, xml_path), train_anno_path)
    for path in val_path_list:
        shutil.copy(os.path.join(original_image_path, path), val_data_path)
        xml_path = path[:-4] + file_ext
        shutil.copy(os.path.join(original_anno_path, xml_path), val_anno_path)


class convert_to_voc_2017:

    def __init__(self):

        self.target_path = r'datasets\neu_det_train_val'

        self.train_base_path = os.path.join(self.target_path, 'train')
        self.val_base_path = os.path.join(self.target_path, 'val')

        self.train_data_path = os.path.join(self.train_base_path, 'images')
        self.train_anno_path = os.path.join(self.train_base_path, 'labels')

        self.val_data_path = os.path.join(self.val_base_path, 'images')
        self.val_anno_path = os.path.join(self.val_base_path, 'labels')

        self.train_whole_txt = os.path.join(self.target_path, 'train_whole.txt')
        self.val_whole_txt = os.path.join(self.target_path, 'val_whole.txt')

        if os.path.exists(self.train_whole_txt):
            os.remove(self.train_whole_txt)
        if os.path.exists(self.val_whole_txt):
            os.remove(self.val_whole_txt)

    def convert(self, size, box):
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

    def convert_annotation(self, image_name, image_folder_path, xml_folder_path, whole_txt_path):
        classes = ['crazing', 'patches', 'inclusion', 'pitted_surface', 'rolled-in_scale', 'scratches']
        # in_file = open('./indata/' + image_name[:-3] + 'xml')  # xml文件路径
        target_txt = open(whole_txt_path, "a")
        target_txt.write(os.path.join(image_folder_path, image_name) + " ")

        xml_path = os.path.join(xml_folder_path, image_name[:-3] + 'xml')
        xml_file_handle = open(xml_path, 'r')

        xml_text = xml_file_handle.read()
        root = ET.fromstring(xml_text)
        xml_file_handle.close()

        for obj in root.iter('object'):
            cls = obj.find('name').text

            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
            #      float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))

            target_txt.write(str(cls_id) + " " + " ".join(map(str, b)) + ' ')

        target_txt.write("\n")

        target_txt.close()

    def convert_whole(self):
        def convert_one_folder(data_one_folder, anno_one_folder, whole_txt_path):
            image_list = os.listdir(data_one_folder)
            for image_name in image_list:
                self.convert_annotation(image_name, data_one_folder, anno_one_folder, whole_txt_path)

        convert_one_folder(self.train_data_path, self.train_anno_path, self.train_whole_txt)
        convert_one_folder(self.val_data_path, self.val_anno_path, self.val_whole_txt)


if __name__ == '__main__':
    split_train_val_xml()
    ctv = convert_to_voc_2017()
    ctv.convert_whole()

    # ##########nhuk#################################### train_val_split
    # split_train_val_xml()
    # ##########nhuk####################################

    # ##########nhuk####################################  train_val_test_split
    # xml2txt()
    # train_val_test_split()
    # alloc_data()
    # ##########nhuk####################################
