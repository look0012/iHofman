import csv
from tqdm import tqdm
import os

def ReadMyCsv1(SaveList, fileName):
    """读取CSV文件，并将其存储在SaveList中"""
    with open(fileName, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in tqdm(csv_reader, desc=f"Reading {fileName}"):
            SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    """读取CSV文件并尝试将数据转换为浮点数格式"""
    with open(fileName, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in tqdm(csv_reader, desc=f"Reading {fileName}"):
            counter = 1
            while counter < len(row):
                try:
                    row[counter] = float(row[counter])
                except ValueError:
                    row[counter] = 0.0
                counter += 1
            SaveList.append(row)
    return


def StorFile(data, fileName):
    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    file_path = os.path.join(result_folder, fileName)

    with open(file_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    return