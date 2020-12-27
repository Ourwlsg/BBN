import os
import glob

import sys

sys.path.insert(0, "..")
import random
import numpy as np
import os
import shutil
import pandas as pd


def locationByLabel():
    DIR_IMAGES = r'/workspace/data/cassava1920/train_images/train_images/'
    FILE_CSV = r'/workspace/data/cassava1920/train.csv'

    DF = pd.read_csv(FILE_CSV)
    # print(df['image_id'].groupby(df['label']))
    for df in DF.groupby('label'):
        print(df[1]["label"].iloc[0], len(df[1]["label"]))
        categoryDir = f'/workspace/BBN/cassava/data/train/{df[1]["label"].iloc[0]}'
        os.makedirs(categoryDir, exist_ok=True)
        for image_id in df[1]["image_id"]:
            shutil.copyfile(f"{DIR_IMAGES}/{image_id}", f"{categoryDir}/{image_id}")


if __name__ == '__main__':

    # locationByLabel()
    K_FOLD = 5
    DIR_CV = '/workspace/BBN/cassava/data/new_cv20/'
    traindata_path = '/workspace/BBN/cassava/data/train'
    # traindata_path = '/home/zhucc/kaggle/pytorch_classification/data/train'

    # FILE_CSV = r'/workspace/data/cassava1920/train.csv'
    # FILE_CSV = r'/workspace/data/cassava1920/train.csv'
    FILE_CSV = r'/workspace/data/cassava/new_train.csv'
    # FILE_CSV = r'/workspace/data/cassava/train.csv'

    dataframe = pd.read_csv(FILE_CSV)
    for label in os.listdir(traindata_path):
        print(label)
        random.seed(2020)
        # img_list = glob.glob(os.path.join(traindata_path, label, '*.jpg'))
        img_list = [os.path.join(traindata_path, label, img_id) for img_id in
                    dataframe[dataframe["label"] == int(label)]["image_id"]]
        random.shuffle(img_list)
        val_list = []
        train_list = []
        for k in range(0, K_FOLD):
            txtpath = DIR_CV + 'fold_' + str(k)
            os.makedirs(txtpath, exist_ok=True)
            val_list = img_list[(len(img_list) // K_FOLD) * k:(len(img_list) // K_FOLD) * (k + 1)]
            train_list = [image for image in img_list if image not in val_list]
            with open(txtpath + '/train.txt', 'a')as fr:
                for img in train_list:
                    # print(img + ' ' + str(label))
                    fr.write(img + ' ' + str(label))
                    fr.write('\n')

            with open(txtpath + '/val.txt', 'a')as fv:
                for img in val_list:
                    # print(img + ' ' + str(label))
                    fv.write(img + ' ' + str(label))
                    fv.write('\n')
