# -*- coding: utf-8 -*-
import random
import shutil
import os
import torch
import numpy as np
import torch.utils.data as udata
from src.utils import *
import glob, math, shutil, h5py
import scipy.io as sio
from torchvision.transforms import ToTensor
from tqdm import tqdm

import h5py

def prepare_data_seis_recon(data_path, output_fname,
                            patch_size: tuple,
                            stride: tuple,
                            misstype='random',
                            remove_zero=True,
                            start_train=0,
                            start_val=0,
                            key='',
                            train_num=10000,
                            val_num=1000,
                            norm='log1p',
                            compression=None,
                            **kwargs):


    print("Input path: ", data_path)


    full_dataset = glob.glob(os.path.join(data_path, '*.mat'))
    patches = []
    for file_path in full_dataset:
        arr = load_data_auto(file_path, key=key)
        match norm:
            case 'absmax':
                arr = arr / np.max(np.abs(arr))
            case 'log1p':
                arr = norm_by_log(arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
            case 'TPNN':
                arr, _, _ = norm_by_arcsinh_std(arr)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
            case 'none':
                pass

        tmp_patches = Ar2Patch_3d(arr, *patch_size, *stride, remove_zero)
        patches.extend(tmp_patches)

    total_num = train_num + val_num
    print(f'Total patches: {len(patches)}')
    # 随机选取total_num个patch
    if len(patches) <= total_num:
        raise ValueError("Number of patches not enough. Please increase the size of the entire data, or reduce "
                         "the required number of patches.")

    print(f"Randomly sampling {total_num} patches from {len(patches)} patches")
    patches = random.sample(patches, total_num)

    # 划分数据集
    train_patches = patches[:train_num]
    val_patches = patches[train_num:train_num + val_num]

    print("Removing Traces of Training Patches")
    miss_train_patches = []
    for patch in train_patches:
        match misstype:
            case 'random':
                miss_patch, _ = remove_traces_3d(patch, 0.4, 0.6, 'r', 0.)

        miss_train_patches.append(miss_patch)

    print("Removing Traces of Validation Patches")
    miss_val_patches = []
    for patch in val_patches:
        match misstype:
            case 'random':
                miss_patch, _ = remove_traces_3d(patch, 0.5, 0.7, 'r', 0.)

        miss_val_patches.append(miss_patch)

    if start_train != 0 or start_val != 0:
        try:
            print(f"{output_fname} already exists. "
                  f"Appending {train_num} training patches and {val_num} validation patches to {output_fname}.")
            h5_file = h5py.File(output_fname, "a")
            train_label = h5_file["train/label"]
            train_feature = h5_file["train/feature"]
            val_label = h5_file["val/label"]
            val_feature = h5_file["val/feature"]

            train_label.resize(start_train + train_num, axis=0)
            train_feature.resize(start_train + train_num, axis=0)
            val_label.resize(start_val + val_num, axis=0)
            val_feature.resize(start_val + val_num, axis=0)

            train_label[start_train:] = np.asarray(train_patches, dtype=np.float32)
            train_feature[start_train:] = np.asarray(miss_train_patches, dtype=np.float32)
            val_label[start_val:] = np.asarray(val_patches, dtype=np.float32)
            val_feature[start_val:] = np.asarray(miss_val_patches, dtype=np.float32)
        except Exception:
            print(f"{output_fname} corrupted. Please manually delete it and regenerate the dataset.")

    else:
        print(f"{output_fname} doesn't exist. Creating dataset from scratch.")
        h5_file = h5py.File(output_fname, "w")
        train_label = h5_file.create_dataset(
            "train/label", data=np.asarray(train_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        train_feature = h5_file.create_dataset(
            "train/feature", data=np.asarray(miss_train_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        val_label = h5_file.create_dataset(
            "val/label", data=np.asarray(val_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        val_feature = h5_file.create_dataset(
            "val/feature", data=np.asarray(miss_val_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)

    # 输出数据集信息
    print('\nTraining set: %d patches' % train_num)
    print('Validation set: %d patches' % val_num)
    print('\nTotal Training set: %d patches' % len(train_label))
    print('Total Validation set: %d patches' % len(val_label))
    print(f"Dataset saved to {output_fname}")
    print("Keys:", list(h5_file.keys()))
    h5_file.close()

def prepare_data_seis_random_denoising(data_path, output_fname,
                            patch_size: tuple,
                            stride: tuple,
                            noisy_snr=3.,
                            remove_zero=True,
                            start_train=0,
                            start_val=0,
                            key='',
                            train_num=10000,
                            val_num=1000,
                            norm='log1p',
                            compression=None,
                            **kwargs):


    print("Input path: ", data_path)


    full_dataset = glob.glob(os.path.join(data_path, '*.mat'))
    patches = []
    noisy_patches = []
    for file_path in full_dataset:
        arr = load_data_auto(file_path, key=key)
        noisy_arr, _ = add_gaussian_noise(arr, noisy_snr)
        print(f"SNR of the noisy data: {array_SNR(noisy_arr, arr)}")
        arr_max = np.max(np.abs(arr))
        match norm:
            case 'absmax':
                arr = arr / arr_max
                noisy_arr = noisy_arr / arr_max
            case 'log1p':
                arr = norm_by_log(arr)
                noisy_arr = norm_by_log(noisy_arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
                noisy_arr, _ = norm_by_log_max(noisy_arr)
            case 'TPNN':
                arr, arr_mean, arr_std = norm_by_arcsinh_std(arr)
                noisy_arr, _, _ = norm_by_arcsinh_std(noisy_arr, arr_mean, arr_std)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
                noisy_arr = norm_by_arcsinh(noisy_arr)
            case 'none':
                pass

        tmp_patches = Ar2Patch_3d(arr, *patch_size, *stride, remove_zero)
        tmp_noisy_patches = Ar2Patch_3d(noisy_arr, *patch_size, *stride, remove_zero)
        patches.extend(tmp_patches)
        noisy_patches.extend(tmp_noisy_patches)

    total_num = train_num + val_num
    print(f'Total patches: {len(patches)}')
    # 随机选取total_num个patch
    if len(patches) <= total_num:
        raise ValueError("Number of patches not enough. Please increase the size of the entire data, or reduce "
                         "the required number of patches.")

    print(f"Randomly sampling {total_num} patches from {len(patches)} patches")

    indices = random.sample(range(len(patches)), total_num)

    patches = [patches[i] for i in indices]
    noisy_patches = [noisy_patches[i] for i in indices]

    # 划分数据集
    train_patches = patches[:train_num]
    noisy_train_patches = noisy_patches[:train_num]
    val_patches = patches[train_num:train_num + val_num]
    noisy_val_patches = noisy_patches[train_num:train_num + val_num]


    if start_train != 0 or start_val != 0:
        try:
            print(f"{output_fname} already exists. "
                  f"Appending {train_num} training patches and {val_num} validation patches to {output_fname}.")
            h5_file = h5py.File(output_fname, "a")
            train_label = h5_file["train/label"]
            train_feature = h5_file["train/feature"]
            val_label = h5_file["val/label"]
            val_feature = h5_file["val/feature"]

            train_label.resize(start_train + train_num, axis=0)
            train_feature.resize(start_train + train_num, axis=0)
            val_label.resize(start_val + val_num, axis=0)
            val_feature.resize(start_val + val_num, axis=0)

            train_label[start_train:] = np.asarray(train_patches, dtype=np.float32)
            train_feature[start_train:] = np.asarray(noisy_train_patches, dtype=np.float32)
            val_label[start_val:] = np.asarray(val_patches, dtype=np.float32)
            val_feature[start_val:] = np.asarray(noisy_val_patches, dtype=np.float32)
        except Exception:
            print(f"{output_fname} corrupted. Please manually delete it and regenerate the dataset.")

    else:
        print(f"{output_fname} doesn't exist. Creating dataset from scratch.")
        h5_file = h5py.File(output_fname, "w")
        train_label = h5_file.create_dataset(
            "train/label", data=np.asarray(train_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        train_feature = h5_file.create_dataset(
            "train/feature", data=np.asarray(noisy_train_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        val_label = h5_file.create_dataset(
            "val/label", data=np.asarray(val_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)
        val_feature = h5_file.create_dataset(
            "val/feature", data=np.asarray(noisy_val_patches, dtype=np.float32),
            maxshape=(None, *patch_size), compression=compression)

    # 输出数据集信息
    print('\nTraining set: %d patches' % train_num)
    print('Validation set: %d patches' % val_num)
    print('\nTotal Training set: %d patches' % len(train_label))
    print('Total Validation set: %d patches' % len(val_label))
    print(f"Dataset saved to {output_fname}")
    print("Keys:", list(h5_file.keys()))
    h5_file.close()



class MyDataset_3d(udata.Dataset):
    '''

    '''
    def __init__(self, dataset_fname='./data', istrain=True
                 ):
        super(MyDataset_3d, self).__init__()
        self.h5_file = h5py.File(dataset_fname, 'r')
        if istrain:
            self.labels = self.h5_file['train/label']
            self.features = self.h5_file['train/feature']
        else:
            self.labels = self.h5_file['val/label']
            self.features = self.h5_file['val/feature']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        # numpy -> torch
        feature = torch.from_numpy(np.array(feature)).float()
        label = torch.from_numpy(np.array(label)).float()
        #
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return feature, label



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='recon_training')
    parser.add_argument("--task", type=str, default="interpolation",
                        help="The task you want to generate dataset for.  'interpolation' or 'denoising'")
    parser.add_argument("--net", type=str, default="LMGM",
                        help="The network you want to generate dataset for.   'LMGM', 'FAT', or 'ANN'")
    parser.add_argument("--tnum", type=int, default=6000,
                        help="The number of samples you want to generate for training.")
    parser.add_argument("--vnum", type=int, default=1500,
                        help="The number of samples you want to generate for validation.")
    parser.add_argument("--norm", type=str, default="absmax",
                        help="The normalization you want to apply to your data.  'absmax', 'TPNN', or 'none'")
    parser.add_argument("--df", type=str, default="45shots",
                        help="The dataset foldername.")
    opt = parser.parse_args()


    start_train = 0
    start_val = 0
    train_num = opt.tnum
    val_num = opt.vnum

    match opt.net:
        case "LMGM" | "FAT":
            shape = (32, 32, 8)
            stride = (32, 32, 8)
            if opt.df == "parihaka":
                stride = (20, 20, 6)

        case "ANN":
            shape = (32, 32, 32)
            stride = (20, 20, 20)
            if opt.df == "parihaka":
                stride = (14, 14, 14)
        case _:
            shape = (32, 32, 8)
            stride = (32, 32, 8)



    norm = opt.norm
    dname = opt.df

    shapename = f"{shape[0]}" if shape[0] == shape[1] == shape[2] else f"{shape[0]}{shape[1]}{shape[2]}"
    input_path = f'./data/{dname}/train/'
    n_patches = round(train_num / 1000., 1) if train_num % 1000 != 0 \
        else round(train_num / 1000)
    output_fname = f"./dataset/3d/{dname}_{shapename}_{str(n_patches)}k_{norm}_{opt.task}.h5"

    if start_train == 0 and start_val == 0:
        try:
            os.remove(output_fname)
        except FileNotFoundError:
            pass
        os.makedirs("./dataset/3d/", exist_ok=True)

    match opt.task:
        case "interpolation":
            prepare_data_seis_recon(input_path, output_fname, shape, stride,
                                    remove_zero=False, misstype='random',
                                    key='', train_num=train_num, val_num=val_num,
                                    start_train=start_train, start_val=start_val,
                                    norm=norm, compression=None )

        case "denoising":
            prepare_data_seis_random_denoising(input_path, output_fname, shape, stride,
                                    remove_zero=False, noisy_snr=3.,
                                    key='', train_num=train_num, val_num=val_num,
                                    start_train=start_train, start_val=start_val,
                                    norm=norm, compression=None)
        case _:
            pass

    ds = MyDataset_3d(dataset_fname=output_fname, istrain=True)
    dl = udata.DataLoader(ds, batch_size=1, shuffle=True)

    feature, label = next(iter(dl))
    # print(feature)
    # print(label)
    print(feature.shape)
    print(label.shape)
