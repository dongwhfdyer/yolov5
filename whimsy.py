# read txt file, strip all the spaces, and split the string into a list
import re

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



if __name__ == '__main__':
    log2newlog2two_table2one_table2excel()
