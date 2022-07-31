# read txt file, strip all the spaces, and split the string into a list
import os
import random
import re
import shutil

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

if __name__ == '__main__':
    create_crazing_rolled_dataset()
