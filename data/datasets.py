# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import threading

import torch
import cv2
import scipy.io as scio
from PIL import Image
from torch.utils.data.dataset import T_co

import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py
from tqdm import tqdm
import time
from torch.utils.data import Dataset


class DataSetBase(Dataset):
    txt_feat_len = 0

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        pass


class WIKI(DataSetBase):
    wiki_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    wiki_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    initialized = False
    lock = threading.Lock()
    train_label = None
    train_img_name = None
    train_txt = None
    test_label = None
    test_img_name = None
    test_txt = None
    @classmethod
    def init(cls,data_path_config):
        with cls.lock:
            if not cls.initialized:
                print("Initialization code here")
                cls.initialized = True
                label_set = scio.loadmat(data_path_config['labelDir'])
                cls.test_txt = np.array(label_set['T_te'], dtype=np.float32)
                cls.train_txt = np.array(label_set['T_tr'], dtype=np.float32)

                cls.test_label = []
                with open(data_path_config['testLabel'], 'r') as f:
                    total_lines = len(f.readlines())
                    f.seek(0)  # Reset the file pointer back to the beginning of the file
                    for line in tqdm(f.readlines(), total=total_lines, desc="Processing testLabel"):
                        cls.test_label.extend([int(line.split()[-1]) - 1])

                cls.test_img_name = []
                with open(data_path_config['testLabel'], 'r') as f:
                    total_lines = len(f.readlines())
                    f.seek(0)  # Reset the file pointer back to the beginning of the file
                    for line in tqdm(f.readlines(), total=total_lines, desc="Processing test_img_name"):
                        cls.test_img_name.extend([line.split()[1]])

                cls.train_label = []
                with open(data_path_config['trainLabel'], 'r') as f:
                    total_lines = len(f.readlines())
                    f.seek(0)  # Reset the file pointer back to the beginning of the file
                    for line in tqdm(f.readlines(), total=total_lines, desc="Processing trainLabel"):
                        cls.train_label.extend([int(line.split()[-1]) - 1])

                cls.train_img_name = []
                with open(data_path_config['trainLabel'], 'r') as f:
                    total_lines = len(f.readlines())
                    f.seek(0)  # Reset the file pointer back to the beginning of the file
                    for line in tqdm(f.readlines(), total=total_lines, desc="Processing train_img_name"):
                        cls.train_img_name.extend([line.split()[1]])

                txt_feat_len = cls.train_txt.shape[1]
                DataSetBase.txt_feat_len = txt_feat_len
    def __init__(self, transform=None, target_transform=None, train=True, data_path_config=None):
        super().__init__()
        WIKI.init(data_path_config)
        if train:
            self.label = self.__class__.train_label
            self.img_name = self.__class__.train_img_name
            self.txt = self.__class__.train_txt
        else:
            self.label = self.__class__.test_label
            self.img_name = self.__class__.test_img_name
            self.txt = self.__class__.test_txt
        self.transform = transform
        self.target_transform = target_transform

        self.f_name = ['art', 'biology', 'geography', 'history', 'literature', 'media', 'music', 'royalty', 'sport',
                       'warfare']
        self.data_dir = data_path_config['dataDir']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        path = self.data_dir + '/' + self.f_name[self.label[index]] + '/' + self.img_name[index] + '.jpg'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        target = self.label[index]
        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        return len(self.label)


class MIRFlickr(DataSetBase):
    mir_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mir_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    initialized = False
    lock = threading.Lock()
    indexTest = None
    indexDatabase = None
    indexTrain = None
    label_set = None
    txt_set = None
    @classmethod
    def init(cls,data_path_config):
        with cls.lock:
            if not cls.initialized:
                print("Initialization code here")
                cls.initialized = True
                cls.label_set = scio.loadmat(data_path_config['labelDir'])
                cls.label_set = np.array(cls.label_set['LAll'], dtype=np.float32)
                cls.txt_set = scio.loadmat(data_path_config['txtDir'])
                cls.txt_set = np.array(cls.txt_set['YAll'], dtype=np.float32)
                database_index = np.arange(cls.label_set.shape[0])
                first = True
                for label in tqdm(range(cls.label_set.shape[1]), desc=f'初始化'):
                    index = np.where(cls.label_set[:, label] == 1)[0]

                    N = index.shape[0]
                    perm = np.random.permutation(N)
                    index = index[perm]
                    if first:
                        test_index = index[:160]
                        train_index = index[160:160 + 400]
                        first = False
                    else:
                        ind = np.array([i for i in list(index) if i not in (list(train_index) + list(test_index))])
                        test_index = np.concatenate((test_index, ind[:80]))
                        train_index = np.concatenate((train_index, ind[80:80 + 200]))

                #database_index = np.array([i for i in list(range(self.label_set.shape[0])) if i not in list(test_index)])
                #database_index = train_index

                if train_index.shape[0] < 5000:
                    pick = np.array([i for i in list(database_index) if i not in list(train_index)])
                    N = pick.shape[0]
                    perm = np.random.permutation(N)
                    pick = pick[perm]
                    res = 5000 - train_index.shape[0]
                    train_index = np.concatenate((train_index, pick[:res]))

                cls.indexTest = test_index
                cls.indexDatabase = database_index
                cls.indexTrain = train_index

                DataSetBase.txt_feat_len = cls.txt_set.shape[1]

    def __init__(self, transform=None, target_transform=None, train=True, database=False, data_path_config=None):
        super().__init__()
        MIRFlickr.init(data_path_config)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.train_labels = self.label_set[self.indexTrain]
            self.train_index = self.indexTrain
            self.txt = self.txt_set[self.indexTrain]
        elif database:
            self.train_labels = self.label_set[self.indexDatabase]
            self.train_index = self.indexDatabase
            self.txt = self.txt_set[self.indexDatabase]
        else:
            self.train_labels = self.label_set[self.indexTest]
            self.train_index = self.indexTest
            self.txt = self.txt_set[self.indexTest]

        self.img_dir = data_path_config['imgDir']

    def __getitem__(self, index):
        mir_flickr = h5py.File(self.img_dir, 'r', libver='latest', swmr=True)
        img, target = mir_flickr['IAll'][self.train_index[index]], self.train_labels[index]
        img = Image.fromarray(np.transpose(img, (2, 1, 0)))
        mir_flickr.close()

        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)


class NUSWIDE(DataSetBase):
    nus_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    nus_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    initialized = False
    lock = threading.Lock()
    indexTest = None
    indexDatabase = None
    indexTrain = None
    label_set = None
    txt_set = None
    @classmethod
    def init(cls,data_path_config):
        with cls.lock:
            if not cls.initialized:
                print("Initialization code here")
                cls.initialized = True
                cls.label_set = scio.loadmat(data_path_config['labelDir'])
                cls.label_set = np.array(cls.label_set['LAll'], dtype=np.float32)
                txt_file = h5py.File(data_path_config['txtDir'])
                cls.txt_set = np.array(txt_file['YAll']).transpose()
                txt_file.close()

                first = True

                for label in tqdm(range(cls.label_set.shape[1]), desc=f'初始化'):
                    index = np.where(cls.label_set[:, label] == 1)[0]

                    N = index.shape[0]
                    perm = np.random.permutation(N)
                    index = index[perm]

                    if first:
                        test_index = index[:200]
                        train_index = index[200:700]
                        first = False
                    else:
                        ind = np.array([i for i in list(index) if i not in (list(train_index) + list(test_index))])
                        test_index = np.concatenate((test_index, ind[:200]))
                        train_index = np.concatenate((train_index, ind[200:700]))

                database_index = np.array([i for i in list(range(cls.label_set.shape[0])) if i not in list(test_index)])

                cls.indexTest = test_index
                cls.indexDatabase = database_index
                cls.indexTrain = train_index

                txt_feat_len = cls.txt_set.shape[1]
                DataSetBase.txt_feat_len = txt_feat_len
    def __init__(self, transform=None, target_transform=None, train=True, database=False, data_path_config=None):
        super().__init__()
        NUSWIDE.init(data_path_config)
        if train:
            self.train_labels = self.__class__.label_set[self.__class__.indexTrain]
            self.train_index = self.__class__.indexTrain
            self.txt = self.__class__.txt_set[self.__class__.indexTrain]
        elif database:
            self.train_labels = self.__class__.label_set[self.__class__.indexDatabase]
            self.train_index = self.__class__.indexDatabase
            self.txt = self.__class__.txt_set[self.__class__.indexDatabase]
        else:
            self.train_labels = self.__class__.label_set[self.__class__.indexTest]
            self.train_index = self.__class__.indexTest
            self.txt = self.__class__.txt_set[self.__class__.indexTest]

        self.transform = transform
        self.target_transform = target_transform

        self.img_dir = data_path_config['imgDir']

    def __getitem__(self, index):
        nuswide = h5py.File(self.img_dir, 'r', libver='latest', swmr=True)
        img, target = nuswide['IAll'][self.train_index[index]], self.train_labels[index]
        img = Image.fromarray(np.transpose(img, (2, 1, 0)))
        nuswide.close()

        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)


def get_dataset(dataset_name, data_path):
    if dataset_name == "wiki":
        train_dataset = WIKI(data_path_config=data_path, train=True,
                             transform=WIKI.wiki_train_transform)
        test_dataset = WIKI(data_path_config=data_path, train=False,
                            transform=WIKI.wiki_test_transform)
        database_dataset = WIKI(data_path_config=data_path, train=True,
                                transform=WIKI.wiki_test_transform)

    if dataset_name == "mirFlickr25k":
        train_dataset = MIRFlickr(train=True, transform=MIRFlickr.mir_train_transform,
                                  data_path_config=data_path)
        test_dataset = MIRFlickr(train=False, database=False,
                                 transform=MIRFlickr.mir_test_transform,
                                 data_path_config=data_path)
        database_dataset = MIRFlickr(train=False, database=True,
                                     transform=MIRFlickr.mir_test_transform,
                                     data_path_config=data_path)

    if dataset_name == "nusWide":
        train_dataset = NUSWIDE(train=True, transform=NUSWIDE.nus_train_transform,
                                data_path_config=data_path)
        test_dataset = NUSWIDE(train=False, database=False,
                               transform=NUSWIDE.nus_test_transform,
                               data_path_config=data_path)
        database_dataset = NUSWIDE(train=False, database=True,
                                   transform=NUSWIDE.nus_test_transform,
                                   data_path_config=data_path)
    return train_dataset, test_dataset, database_dataset
