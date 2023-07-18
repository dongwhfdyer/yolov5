# read txt file, strip all the spaces, and split the string into a list
from imutils import build_montages
import os
import json
import xml.etree.ElementTree as ET
import os
import random
import re
import shutil
import shutil

import numpy as np
from cv2 import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import pandas as pd


def restrip_space(many_space_string):
    return re.sub('\s+', ' ', many_space_string)


def read_file(filename):
    data_dict = {}
    with open(filename) as f:
        readlines = f.readlines()
        header = restrip_space(readlines[0]).split(" ")
        print(header)
        for index, line in enumerate(readlines[1:]):
            line = line.strip()
            line = re.sub("\s+", " ", line)
            line = line.split(" ")
            if data_dict.get(header[0]):
                data_dict[line[0]].append(line[1:])
            else:
                data_dict[line[0]] = [line[1:]]
        one_line = readlines[0].strip()
        strip_space_one_line = re.sub('\s+', ' ', one_line).split(" ")


def read_file_v2(filename):
    data_dict = {}
    with open(filename) as f:
        readlines = f.readlines()
        for index, line in enumerate(readlines[1:]):
            line = line.strip()
            line = restrip_space(line)

        one_line = readlines[0].strip()
        strip_space_one_line = re.sub('\s+', ' ', one_line).split(" ")


def read_file_and_resave(filename1, filename2):
    new_lines = ""
    with open(filename1) as f:
        readlines = f.readlines()
        for index, line in enumerate(readlines[1:-1]):
            line = line.strip()
            line = re.sub("\s+", " ", line)
            new_lines += line + "\n"
    with open(filename2, "w") as f:
        f.write(new_lines)


def read_one_table_return_one_dataframe(filename):
    data = pd.read_csv(filename, sep=" ", header=None)
    # delete two column at one time
    data.drop(data.columns[1:3, ], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    data.header = ["name", 'presion', "recall", "mAP"]
    # move first row to the last row
    data.loc[len(data)] = data.iloc[0]
    data.drop(data.index[0], inplace=True)
    # add header to data
    data.columns = data.header
    return data


def process_two_table_using_panda(derain_data_frame, rainy_data_frame):
    # insert the first column of rainy data frame to the second column of derain data frame

    new_data_frame = pd.DataFrame(columns=["name", "rain_presion", "derain_presion", "rain_recall", "derain_recall", "rain_mAP", "derain_mAP"])
    new_data_frame["name"] = rainy_data_frame["name"]
    new_data_frame["rain_presion"] = rainy_data_frame["presion"]
    new_data_frame["derain_presion"] = derain_data_frame["presion"]
    new_data_frame["rain_recall"] = rainy_data_frame["recall"]
    new_data_frame["derain_recall"] = derain_data_frame["recall"]
    new_data_frame["rain_mAP"] = rainy_data_frame["mAP"]
    new_data_frame["derain_mAP"] = derain_data_frame["mAP"]

    print("hello!")
    print("hello!")
    return new_data_frame


def log2newlog2two_table2one_table2excel():
    # read_file_and_resave(r"D:\ANewspace\code\yolov5_new\runs\val\exp6\log.txt", r"D:\ANewspace\code\yolov5_new\runs\val\exp6\new_log.txt")
    # read_file_and_resave(r"D:\ANewspace\code\yolov5_new\runs\val\exp8\log.txt", r"D:\ANewspace\code\yolov5_new\runs\val\exp8\new_log.txt")
    derain_data_frame = read_one_table_return_one_dataframe(r"D:\ANewspace\code\yolov5_new\runs\val\exp8\new_log.txt")
    rainy_data_frame = read_one_table_return_one_dataframe(r"D:\ANewspace\code\yolov5_new\runs\val\exp6\new_log.txt")
    one_table = process_two_table_using_panda(derain_data_frame, rainy_data_frame)
    # four decimal place and save to excel using panda

    excel_save_path = r"D:\ANewspace\code\yolov5_new\runs\val\rainyVSderain_mAP.xlsx"

    writer = pd.ExcelWriter(excel_save_path)
    one_table.to_excel(writer, 'sheet1', index=False)
    writer.save()
    print("cool")
    # one_table.to_excel(r"D:\ANewspace\code\yolov5_new\runs\val\derainVSrainy_mAP.xlsx", float_format="%.4f")


def random_pick_images():
    folder_path = r"datasets/neu_det/IMAGES"
    new_folder = r"datasets/neu_det_random"
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
    os.makedirs(new_folder)
    file_list = os.listdir(folder_path)
    random_files = random.choices(file_list, k=30)
    for file in random_files:
        shutil.copy(folder_path + "/" + file, new_folder)


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


def check_folder_exist(*folders_path):
    for folder_path in folders_path:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def pick_crazing_images_to_folder():
    original_data_folder = r"datasets/neu_det/IMAGES/"
    original_label_folder = r"datasets/neu_det/ANNOTATIONS_txt/"
    crazing_data_path = r"datasets/neu_det_crazing/"
    crzaing_train_data_path = os.path.join(crazing_data_path, "train")
    crzaing_val_data_path = os.path.join(crazing_data_path, "val")
    train_label_folder = os.path.join(crzaing_train_data_path, "labels")
    train_image_folder = os.path.join(crzaing_train_data_path, "images")
    val_label_folder = os.path.join(crzaing_val_data_path, "labels")
    val_image_folder = os.path.join(crzaing_val_data_path, "images")
    check_folder_exist(train_label_folder, train_image_folder, val_label_folder, val_image_folder)

    train_val_ratio = 0.9  # train:val = 0.9:0.1
    total_data_list = os.listdir(original_data_folder)
    #  filter the images with "crazing" in the name
    total_data_list = [data for data in total_data_list if "crazing" in data or "rolled-in_scale" in data]
    num = len(total_data_list)
    list_index = range(num)
    tv = int(num * train_val_ratio)
    train_list = random.sample(list_index, tv)
    val_list = [i for i in list_index if i not in train_list]
    for i in list_index:
        if i in train_list:
            shutil.copy(os.path.join(original_data_folder, total_data_list[i]), train_image_folder)
            shutil.copy(os.path.join(original_label_folder, total_data_list[i].replace(".jpg", ".txt")), train_label_folder)
        else:
            shutil.copy(os.path.join(original_data_folder, total_data_list[i]), val_image_folder)
            shutil.copy(os.path.join(original_label_folder, total_data_list[i].replace(".jpg", ".txt")), val_label_folder)


def change_index():
    crazing_data_path = r"datasets/neu_det_crazing/"
    crzaing_train_data_path = os.path.join(crazing_data_path, "train")
    crzaing_val_data_path = os.path.join(crazing_data_path, "val")
    train_label_folder = os.path.join(crzaing_train_data_path, "labels")
    val_label_folder = os.path.join(crzaing_val_data_path, "labels")
    classes_map = {"crazing": 0, "rolled-in_scale": 1}

    def process_one_folder(folder_path):
        label_paths = os.listdir(folder_path)
        for label_path in label_paths:
            path = os.path.join(folder_path, label_path)
            changed_line = ""
            with open(path, 'r') as f:
                lines = f.readlines()
                if "crazing" in label_path:
                    label_id = 0
                elif "rolled-in_scale" in label_path:
                    label_id = 1
                for line in lines:
                    changed_line += str(label_id) + line[1:]
            with open(path, 'w') as f:
                f.write(changed_line)

    process_one_folder(train_label_folder)
    process_one_folder(val_label_folder)


def create_crazing_rolled_dataset():
    pick_crazing_images_to_folder()
    change_index()


def decode_json(json_folder_path, json_name, txt_folder):
    name2id = {"7": 7, "6": 6}

    def convert(img_size, box):
        dw = 1. / (img_size[0])
        dh = 1. / (img_size[1])
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    txt_name = os.path.join(txt_folder, json_name.replace(".json", ".txt"))
    # txt_name = 'C:\\Users\\86189\\Desktop\\' + json_name[0:-5] + '.txt'
    # 存放txt的绝对路径
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:

        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')



def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def copy_folder_everything(src, dst, if_leave_root=False):
    """
    copy src's content to dst(not src itself)
    copy the folder and all its content
    :param src: the source path
    :param dst: the destination path
    :return:
    """

    create_folders(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


class tackle_position_X_shape_datasets:
    """
    YOLO datasets preprocess class.
    It can tackle with dataset with xml format. It can convert xml data to txt data.
    """

    def __init__(self, dataset_path):
        self.dataset_folder = dataset_path
        self.xml_folder = os.path.join(self.dataset_folder, "xml_labels")
        self.img_folder = os.path.join(self.dataset_folder, "images")
        self.labels_folder = os.path.join(self.dataset_folder, "labels")
        self.draw_folder = os.path.join(self.dataset_folder, "draw")
        self.cropped_folder = os.path.join(self.dataset_folder, "cropped")
        self.classes = ['lab']

        ##########nhuk#################################### divide the dataset into train and test
        self.train_folder = os.path.join(self.dataset_folder, "train")
        self.train_img_folder = os.path.join(self.train_folder, "images")
        self.train_label_folder = os.path.join(self.train_folder, "labels")
        self.val_folder = os.path.join(self.dataset_folder, "val")
        self.val_img_folder = os.path.join(self.val_folder, "images")
        self.val_label_folder = os.path.join(self.val_folder, "labels")
        ##########nhuk####################################

    def rename_data_and_sort(self):
        """
        rename images and labels and sort them by name
        :return:
        """
        img_list = os.listdir(self.img_folder)
        img_list.sort()
        for i in range(len(img_list)):
            os.rename(os.path.join(self.img_folder, img_list[i]), os.path.join(self.img_folder, str(i).zfill(4) + ".jpg"))
            os.rename(os.path.join(self.xml_folder, img_list[i].split(".")[0] + ".xml"), os.path.join(self.xml_folder, str(i).zfill(4) + ".xml"))

    def draw_bndbox_on_datasets(self):
        delete_folders(self.draw_folder)
        create_folders(self.draw_folder)
        for img_name in os.listdir(self.img_folder):
            img_path = os.path.join(self.img_folder, img_name)
            xml_path = os.path.join(self.xml_folder, img_name.replace(".jpg", ".xml"))
            img_draw_name = img_name.replace(".jpg", "_draw.jpg")
            self.one_img_draw_bndbox(img_path, xml_path, self.draw_folder, img_draw_name)

    def one_img_draw_bndbox(self, img_path, xml_path, out_path, img_draw_name):
        '''

        :param img_path: str, 原图像的地址
        :param xml_path: str, xml文件的地址
        :param out_path: 保存图像的地址
        :param img_draw_name:保存图像的名称
        :return: None
        '''

        image = cv2.imread(img_path)
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xml_path)
        collection = DOMTree.documentElement
        # print(collection)

        '''
        获得左上角坐标(xmin, ymin)，右下角坐标(xmax, ymax)
        x.firstchild.data:获取元素第一个子节点的数据；
        x.childNodes[0]：:获取元素第一个子节点;
        x.childNodes[0].nodeValue.:也是获取元素第一个子节点值的意思
        '''
        # 示例：提取图片名称、宽、高
        # filename = collection.getElementsByTagName('path')[0].firstChild.data
        width = collection.getElementsByTagName('width')[0].firstChild.data
        height = collection.getElementsByTagName('height')[0].childNodes[0].nodeValue
        # print(filename)
        # print(width)
        # print(height)

        object_elements = collection.getElementsByTagName('item')
        for object_element in object_elements:
            # 获得类别名称
            object_name = object_element.getElementsByTagName('name')[0].firstChild.data
            # print('object name: ', object_name)
            # 获得第一个 bndbox，一个object下只有一个bndbox，第一个就是，他的下标是0
            bndbox_element = object_element.getElementsByTagName('bndbox')[0]
            xmin = bndbox_element.getElementsByTagName('xmin')[0].firstChild.data
            ymin = bndbox_element.getElementsByTagName('ymin')[0].firstChild.data
            xmax = bndbox_element.getElementsByTagName('xmax')[0].firstChild.data
            ymax = bndbox_element.getElementsByTagName('ymax')[0].firstChild.data
            # get truncated value

            # 用红框把图像中的人脸框出,红色 (0, 0, 255)。
            '''
            import cv2
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    (xmin,ymin) -----------
               |          |
               |          |
               |          |
               ------------(xmax,ymax)
            '''
            try:
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                color = (0, 255, 0)
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                # cv2.putText()参数依次是：图像，文字内容，坐标(左上角坐标) ，字体，大小，颜色，字体厚度
                # 用黄色字体在图像中写出类别名称，黄色 (0, 255, 255)
                image = cv2.putText(image, object_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            except:
                print("data invalid!")

        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(out_path, img_draw_name), image)
        return

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

    def convert_annotation(self, image_name, txt_folder_path, xml_folder_path):
        # in_file = open('./indata/' + image_name[:-3] + 'xml')  # ml文件路径

        out_txt_path = os.path.join(txt_folder_path, image_name[:-3] + 'txt')
        xml_path = os.path.join(xml_folder_path, image_name[:-3] + 'xml')
        out_file_handle = open(out_txt_path, 'a', encoding='utf-8')
        xml_file_handle = open(xml_path, 'r', encoding='utf-8')

        xml_text = xml_file_handle.read()
        root = ET.fromstring(xml_text)
        xml_file_handle.close()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('item'):
            cls = obj.find('name').text

            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            # save two decimal places
            bb = [round(bb[0] * 100) / 100, round(bb[1] * 100) / 100, round(bb[2] * 100) / 100, round(bb[3] * 100) / 100]

            out_file_handle.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file_handle.close()

    # def convert_whole(self):
    #     txt_folder_path = r'd:\download\NEU-DET\TRAIN\ANNOTATIONS_txt'
    #     xml_folder_path = r'd:\download\NEU-DET\TRAIN\ANNOTATIONS'
    #     copy_files(xml_folder_path, txt_folder_path)
    #     for file in os.listdir(txt_folder_path):
    #         self.convert_annotation(file, txt_folder_path, xml_folder_path)

    def xml2txt(self):
        delete_folders(self.labels_folder)
        create_folders(self.labels_folder)

        for image_path in os.listdir(self.img_folder):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径
            image_name = image_path.split('\\')[-1]
            self.convert_annotation(image_name, self.labels_folder, self.xml_folder)

    def train_val_split(self, trainval_percent=0.9):
        """
        :param trainval_percent: train: val = 9 : 1
        :return: None
        """
        delete_folders(self.train_folder, self.val_folder)
        create_folders(self.train_img_folder, self.train_label_folder, self.val_img_folder, self.val_label_folder)
        total_num = len(os.listdir(self.img_folder))
        list_index = range(total_num)
        tv = int(total_num * trainval_percent)
        train_index = random.sample(list_index, tv)
        val_index = [i for i in list_index if i not in train_index]
        for index in train_index:
            shutil.copy(os.path.join(self.img_folder, os.listdir(self.img_folder)[index]), self.train_img_folder)
            shutil.copy(os.path.join(self.labels_folder, os.listdir(self.labels_folder)[index]), self.train_label_folder)
        for index in val_index:
            shutil.copy(os.path.join(self.img_folder, os.listdir(self.img_folder)[index]), self.val_img_folder)
            shutil.copy(os.path.join(self.labels_folder, os.listdir(self.labels_folder)[index]), self.val_label_folder)

    def crop_img_on_dataset(self):
        delete_folders(self.cropped_folder)
        create_folders(self.cropped_folder)
        for img_name in os.listdir(self.img_folder):
            img_path = os.path.join(self.img_folder, img_name)
            xml_path = os.path.join(self.xml_folder, img_name.replace(".jpg", ".xml"))
            img_cropped_name = img_name.replace(".jpg", "_cropped.jpg")
            self.crop_img_based_on_xml(img_path, xml_path, self.cropped_folder, img_cropped_name)

    def crop_img_based_on_xml(self, img_path, xml_path, out_path, img_draw_name):

        image = cv2.imread(img_path)
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(xml_path)
        collection = DOMTree.documentElement

        object_elements = collection.getElementsByTagName('item')
        obj_ind = 0
        for object_element in object_elements:
            # 获得类别名称
            object_name = object_element.getElementsByTagName('name')[0].firstChild.data
            # print('object name: ', object_name)
            # 获得第一个 bndbox，一个object下只有一个bndbox，第一个就是，他的下标是0
            bndbox_element = object_element.getElementsByTagName('bndbox')[0]
            xmin = bndbox_element.getElementsByTagName('xmin')[0].firstChild.data
            ymin = bndbox_element.getElementsByTagName('ymin')[0].firstChild.data
            xmax = bndbox_element.getElementsByTagName('xmax')[0].firstChild.data
            ymax = bndbox_element.getElementsByTagName('ymax')[0].firstChild.data

            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            try:
                croped_image = image[ymin:ymax, xmin:xmax]
                each_cropped_object_name = img_draw_name.replace(".jpg", "_" + object_name + "_" + str(obj_ind) + ".jpg")
                cv2.imwrite(os.path.join(out_path, each_cropped_object_name), croped_image)
                obj_ind += 1
            except Exception as e:
                print("######################################## error")
                print(e)
                print(each_cropped_object_name)


class cropped_X_shaped_dataset:
    def __init__(self):
        self.data_path = r"datasets\shibie\cropped"
        self.result_path = r"datasets\shibie\result"
        self.out_path = r"datasets\shibie\processed"

    def random_sample(self, sample_size):
        img_list = os.listdir(self.data_path)
        random.shuffle(img_list)
        return img_list[sample_size:]

    def extract_whole_dataset(self):
        delete_folders(self.out_path, self.result_path)
        create_folders(self.out_path, self.result_path)
        for img_name in self.random_sample(20):  # 主题新颖 应用广阔 企划书
            img_path = os.path.join(self.data_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (960, 960))

            transform_seq = [self.process_img,
                             # self.dilate_img,
                             # self.dilate_img,
                             self.RGB2Black,
                             self.dilate_img,
                             self.dilate_img,
                             # self.erode_img,
                             # self.erode_img,
                             self.erode_img,
                             ]

            result, img_proc = self.transform_guy(transform_seq, img)

            ##########nhuk#################################### save img
            img_save_name = img_name.replace(".jpg", "_proc.jpg")
            # text_content = " ".join([proc.__name__ for proc in transform_seq])
            # cv2.addText(img_proc, text_content, (10, 10),"Times")
            result_save_name = img_name.replace(".jpg", "_result.jpg")
            cv2.imwrite(os.path.join(self.out_path, img_save_name), img_proc)

            result = cv2.resize(result, (960, 960))
            # draw center point
            # cv2.circle(result, (int(result.shape[1] / 2), int(result.shape[0] / 2)), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.result_path, result_save_name), result)
            ##########nhuk####################################

    def process_img(self, img: np.ndarray) -> np.ndarray:

        result = img.copy()

        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        high_V = 245
        low_V = 0
        high_S = 255
        low_S = 30
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, low_S, low_V])
        upper1 = np.array([40, high_S, high_V])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([130, low_S, low_V])
        upper2 = np.array([179, high_S, high_V])

        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        full_mask = lower_mask + upper_mask

        img_ = cv2.bitwise_and(result, result, mask=full_mask)

        return img_

    def concat_img(self, *img_lists):
        """

        :param img_lists: [img_1, img_2, img_3, ...] img_1: np.ndarray
        :return:
        """
        cat_one_cv2 = img_lists[0]
        if len(cat_one_cv2.shape) == 2:
            cat_one_cv2 = cv2.cvtColor(cat_one_cv2, cv2.COLOR_GRAY2BGR)
        cat_one_shape = cat_one_cv2.shape
        for i in range(1, len(img_lists)):
            cat_other_one_cv2 = img_lists[i]
            if len(cat_other_one_cv2.shape) == 2:
                cat_other_one_cv2 = cv2.cvtColor(cat_other_one_cv2, cv2.COLOR_GRAY2BGR)

            cat_other_one_cv2 = cv2.resize(cat_other_one_cv2, (cat_one_shape[1], cat_one_shape[0]))
            # cat_one_cv2 = np.concatenate((cat_one_cv2, cat_other_one_cv2), axis=1)
            cat_one_cv2 = np.hstack((cat_one_cv2, cat_other_one_cv2))
        return cat_one_cv2

    def dilate_img(self, img):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def erode_img(self, img):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def RGB2Black(self, img, thresh=80):
        thresh = 10
        # assign blue channel to zeros
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]
        return img

    def get_hsv_value(self, data_path):
        def getpos(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
                print(HSV[y, x])

        for img_name in os.listdir(data_path):
            img_path = os.path.join(data_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (960, 960))
            HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            cv2.imshow("imageHSV", HSV)
            cv2.imshow('image', image)
            # cv2.setMouseCallback("imageHSV", getpos)
            # cv2.waitKey(0)

            cv2.setMouseCallback("image", getpos)
            cv2.waitKey(0)

    def transform_guy(self, transform_seq, img):
        transformed_list = [img]
        for transform in transform_seq:
            img = transform(img)
            transformed_list.append(img)

        concat_img = self.concat_img(*transformed_list)
        return transformed_list[-1], concat_img

    def line_test(self):
        data_path = self.result_path
        for img_name in os.listdir(data_path):
            img_path = os.path.join(data_path, img_name)
            img = cv2.imread(img_path)
            edged = cv2.Canny(img, 30, 150)
            # show it
            cv2.imshow("image", edged)
            cv2.waitKey(0)


def whole_process():
    tttacle = tackle_position_X_shape_datasets(new_dataset_path)
    tttacle.rename_data_and_sort()
    tttacle.draw_bndbox_on_datasets()
    # after inspect the data with naked eye, we can see that the bndbox is not correct, so we need to fix it.
    tttacle.xml2txt()


if __name__ == '__main__':
    # tac_cropped = cropped_X_shaped_dataset()
    # tac_cropped.extract_whole_dataset()
    # tac_cropped.line_test()
    # tac_cropped.get_hsv_value(r"d:\ANewspace\code\yolov5_new\datasets\shibie\wrong_label")
    # line_detection()
    # get_hsv_value()
    ##########nhuk#################################### X_shape_dataset
    # dataset_path = "datasets\shibie"
    # delete_folders("datasets\shibie")
    # dataset_path = "d:\download\shibie\shibie"
    new_dataset_path = "datasets\shibie"
    # copy_folder_everything(dataset_path, new_dataset_path)
    tttacle = tackle_position_X_shape_datasets(new_dataset_path)
    # tttacle.train_val_split()
    # tttacle.xml2txt()
    # tttacle.crop_img_on_dataset()
    # tttacle.draw_bndbox_on_datasets()
    # tttacle.rename_data_and_sort()
    ##########nhuk####################################
