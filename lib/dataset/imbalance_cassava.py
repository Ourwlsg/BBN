# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.
import argparse
import ast

import torch
import torchvision
import torchvision.transforms as transforms
from lib.config.mydefault import _C as cfg, update_config
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
import cv2
import albumentations as A

mean, std = [0.430316, 0.496727, 0.313513], [0.238024, 0.240754, 0.228859]


def get_train_transform_strong(mean=mean, std=std, size=0):
    train_transform = A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.RandomCrop(width=size, height=size),
        # A.Transpose(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.1),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(p=0.3),
            A.IAAEmboss(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ], p=0.1),
        A.GaussianBlur(p=0.2),
        A.RandomGamma(p=0.1),
        A.CoarseDropout(p=0.5),
        # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(p=0.1),
        # A.ToTensor(),
        A.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_train_transform(mean=mean, std=std, size=0):
    train_transform = A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(width=size, height=size),
        # A.CoarseDropout(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.1),
        A.ShiftScaleRotate(p=0.3),
        A.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_test_transform(mean=mean, std=std, size=0):
    return A.Compose([
        A.Resize(width=int(size * (256 / 224)), height=int(size * (256 / 224)), interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(width=size, height=size),
        # A.ToTensor(),
        A.Normalize(mean=mean, std=std),
    ])


class IMBALANCECASSAVA(Dataset):
    cls_num = 5

    # train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    # valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    def __init__(self, label_file, mode, cfg, imb_type='exp'):

        with open(label_file, 'r') as f:
            # label_file的格式, (label_file image_label)
            dataAndtarget = np.array(list(map(lambda line: line.strip().split(' '), f)))
            self.data = dataAndtarget[:, 0]
            self.targets = [int(target) for target in dataAndtarget[:, 1]]
        train = True if mode == "train" else False
        self.cfg = cfg
        self.train = train
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        rand_number = cfg.DATASET.IMBALANCECASSAVA.RANDOM_SEED
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            # imb_factor = self.cfg.DATASET.IMBALANCECASSAVA.RATIO
            # img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            # self.gen_imbalanced_data(img_num_list)
            self.transform = get_train_transform(size=cfg.INPUT_SIZE)
        else:
            self.transform = get_test_transform(size=cfg.INPUT_SIZE)
        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    # cls_num=5, imb_type='exp', imb_factor=0.02
    # def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    #     img_max = len(self.data) / cls_num
    #     img_num_per_cls = []
    #     if imb_type == 'exp':
    #         for cls_idx in range(cls_num):
    #             num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
    #             img_num_per_cls.append(int(num))
    #     elif imb_type == 'step':
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max))
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max * imb_factor))
    #     else:
    #         img_num_per_cls.extend([int(img_max)] * cls_num)
    #     return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img_path, target = self.data[index], self.targets[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = dict()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img_path, sample_label = self.data[sample_index], self.targets[sample_index]
            sample_img_row = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)
            if sample_img_row is None:
                print(sample_img_path)
            sample_img_row = cv2.cvtColor(sample_img_row, cv2.COLOR_BGR2RGB)
            sample_img = self.transform(image=sample_img_row)['image']
            meta['sample_image'] = torch.from_numpy(sample_img).permute(2, 0, 1)
            meta['sample_label'] = sample_label

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return torch.from_numpy(img).permute(2, 0, 1), target, meta

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    # def gen_imbalanced_data(self, img_num_per_cls):
    #     new_data = []
    #     new_targets = []
    #     targets_np = np.array(self.targets, dtype=np.int64)
    #     classes = np.unique(targets_np)
    #     # np.random.shuffle(classes)
    #     self.num_per_cls_dict = dict()
    #     for the_class, the_img_num in zip(classes, img_num_per_cls):
    #         self.num_per_cls_dict[the_class] = the_img_num
    #         idx = np.where(targets_np == the_class)[0]
    #         np.random.shuffle(idx)
    #         selec_idx = idx[:the_img_num]
    #         new_data.append(self.data[selec_idx, ...])
    #         new_targets.extend([the_class, ] * the_img_num)
    #     new_data = np.vstack(new_data)
    #     self.data = new_data
    #     self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.data)


def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN cassava")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../../configs/cassava.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type=ast.literal_eval,
        dest='auto_resume',
        required=False,
        default=True,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_label_file = "/home/zhucc/kaggle/pytorch_classification/data/cv/fold_0/train.txt"
    val_label_file = "/home/zhucc/kaggle/pytorch_classification/data/cv/fold_0/val.txt"

    update_config(cfg, args)
    train_set = IMBALANCECASSAVA(train_label_file, 'train', cfg)
    print(train_set.get_annotations())
    print(train_set.get_num_classes())
    print(train_set.class_weight)
    # data = list(train_set.class_dict.values())
    # print(len(np.array(li)) for li in data)

    # val_set = IMBALANCECASSAVA(val_label_file, 'val', cfg)
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    # val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=False, num_workers=2)
    #
    # train_loader = iter(train_dataloader)
    # train_data, train_label, train_meta = next(train_loader)
    # print(train_data.shape, train_label, train_meta['sample_image'].shape, train_meta['sample_label'])
    # print("================================")
    # val_loader = iter(val_dataloader)
    # val_data, val_label, val_meta = next(val_loader)
    # print(val_data.shape, val_label, val_meta)

