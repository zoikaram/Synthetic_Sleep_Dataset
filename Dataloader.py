import einops
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
import numpy as np
import csv
import torch
import einops
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
import numpy as np
import csv
import torch
import os
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
import copy
from scipy import signal as sg
import random
import pickle
import h5py
from tqdm import tqdm
from scipy.io import loadmat
from tqdm.notebook import tqdm
import multiprocessing
from joblib import Parallel, delayed
import scipy
import sys
import psutil
#import tensorflow as tf
import copy
import matplotlib.pyplot as plt


def _init_fn(worker_id):
    """
    This function is fed into the dataloaders to have deterministic shuffle.
    :param worker_id:
    :return:
    """
    np.random.seed(15 + worker_id)



class Sleep_Dataset(Dataset):

    def __init__(self, config, file_dirs, num_views):
        super()
        self.dataset = file_dirs
        self.config = config
        self.num_views = num_views
        self.seq_views = self.config.seq_views
        self.keep_view = self.config.keep_view
        self.inner_overlap = self.config.inner_overlap
        self.normalize = self.config.normalize

        # This assert is used in case we want the label from the center one (seq to one)
        # assert seq_length[0] >0 and seq_length[0]%2 !=0, "Outer sequence length must be a positive odd integer"

        self.outer_seq_length = self.config.seq_length[0]
        self.inner_seq_length = self.config.seq_length[1]
        self._get_len()
        self._get_cumulatives()

    def _get_len(self):
        self.dataset_true_length = int(np.array([int(g) for g in self.dataset[1]]).sum() / self.outer_seq_length)

    def _get_cumulatives(self):
        self.cumulative_lengths = [0]
        for i in range(len(self.dataset[0])):
            self.cumulative_lengths.append(int(self.dataset[1, i]) + self.cumulative_lengths[i])

    def get_normalized_values(self):
        metrics = {"mean": np.zeros([129, ]), "mean_sq": np.zeros([129]), "sum": 0, "count_labels": np.zeros([5])}
        for filename in self.dataset[0]:
            with h5py.File(filename, 'r') as f:

                if metrics["sum"] == 0:
                    metrics["mean"] += np.array(f["X2"]).mean(axis=1).mean(axis=1)
                    X2_squared = np.square(np.array(f["X2"]))
                    meanXsquared_i = X2_squared.mean(axis=1).mean(axis=1)
                    metrics["mean_sq"] += meanXsquared_i
                    metrics["sum"] += f["X2"].shape[-1] * f["X2"].shape[-2]
                else:
                    meanX_i = np.array(f["X2"]).mean(axis=1).mean(axis=1)
                    X2_squared = np.square(np.array(f["X2"]))
                    meanXsquared_i = X2_squared.mean(axis=1).mean(axis=1)
                    Ni = f["X2"].shape[-1] * f["X2"].shape[-2]
                    metrics["mean"] = (metrics["mean"] * metrics["sum"] + meanX_i * Ni) / (metrics["sum"] + Ni)
                    metrics["mean_sq"] = (metrics["mean_sq"] * metrics["sum"] + meanXsquared_i * Ni) / (
                                metrics["sum"] + Ni)
                    metrics["sum"] += Ni
                for l in f["label"]:
                    metrics["count_labels"][int(l)] += 1
        varX = -np.multiply(metrics["mean"], metrics["mean"]) + metrics["mean_sq"]
        metrics["std"] = np.sqrt(varX * metrics["sum"] / (metrics["sum"] - 1))
        return metrics

    def load_mat(self, file, remain, data_idx, mod):

        f = h5py.File(file[0], 'r', swmr=True)  # swmr is to run multiple models in parallel

        # Find the end of the (inner) batch
        if remain + data_idx > int(file[1]):
            end_idx = int(file[1])
        else:
            end_idx = data_idx + remain

        # Normalize data
        if "stft" in mod:
            # X2 contains (freq_bins x time_bins x time_windows)
            signal = f["X2"][:, :, data_idx:end_idx]
            signal = np.expand_dims(signal, axis=1)
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):

                signal = einops.rearrange(signal, "freq channels time inner -> inner time channels freq")
                signal = (signal - self.mean[mod]) / self.std[mod]
                signal = einops.rearrange(signal, "inner time channels freq -> inner channels freq time")
            else:
                signal = einops.rearrange(signal, "freq channels time inner -> inner channels freq time")

        elif "time" in mod:
            # X1 contains (time_values x time_windows)
            signal = f["X1"][:, data_idx:end_idx]
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                signal = einops.rearrange(signal, "time inner -> inner time")
                signal = (signal - self.mean[mod]) / self.std[mod]
            else:
                signal = einops.rearrange(signal, "time inner -> inner time")

        label = f["label"][0, data_idx:end_idx]
        # Inits are used to show whether there is continuity with the previous and after windows (1 0 -> beggining, 0 1 -> end, 1 1 -> cut)
        init = np.zeros([end_idx - data_idx])
        if data_idx == 0:
            init[0] = 1
        elif data_idx + self.outer_seq_length == int(file[1]):
            init[-1] = 1

        # Remain gives us whether we should open another file to fill in the (inner) batch or not
        remain_2 = remain - (end_idx - data_idx)

        # Reset data_idx if we need to open another file
        if remain + data_idx > int(file[1]):
            data_idx = 0

        # Turn arrays into Tensors
        init = torch.from_numpy(init)
        img = torch.from_numpy(signal)
        label = torch.from_numpy(label) - 1  # adjustment

        return img, label, init, remain_2, data_idx

    def set_mean_std(self, mean, std):
        '''
        Set mean and std for this dataset. Each of them is supposed to be a dict with
        '''
        self.mean = mean
        self.std = std

    def __getitem__(self, index):

        index = index * self.outer_seq_length
        data_idx = index % self.dataset_true_length

        # Based on the cumulative array find if we need one or two files.
        # Every file contains more than the minimum inner seq length, so we do not support 3 or more files on a single inner batch
        two_files = False
        for sum_i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[sum_i + 1] > index and self.cumulative_lengths[sum_i] <= index:
                file_idx = sum_i
                data_idx = index - self.cumulative_lengths[sum_i]
                if self.cumulative_lengths[sum_i + 1] - index < self.outer_seq_length and self.cumulative_lengths[
                    sum_i + 1] - index != 0:
                    two_files = True
                break

        # get the filenames
        filenames = []
        for view_i in range(0, 2 * self.num_views, 2):
            filenames.append([self.dataset[view_i][file_idx], self.dataset[view_i + 1][file_idx]])
            if two_files:
                filenames.append([self.dataset[view_i][file_idx + 1], self.dataset[view_i + 1][file_idx]])

        images = []
        step_file = 2 if two_files else 1
        prev_label = torch.Tensor([])
        for seq_files in range(0, len(filenames), step_file):
            remain = self.outer_seq_length

            # Get modality name -> {}_{}.format(type, name) (ex. stft_eeg)
            mod = self.config.data_view_dir[int(seq_files / step_file)][1][0] + "_" + \
                  self.config.data_view_dir[int(seq_files / step_file)][1][1]

# change load.mat n aepistrfei tin simulated img
            img, label, init, remain, data_idx_2 = self.load_mat(filenames[seq_files], remain, data_idx, mod)
            if step_file == 2:
                img_2, label_2, init_2, remain, _ = self.load_mat(filenames[seq_files + 1], remain, data_idx_2, mod)
                img = torch.cat([img, img_2], dim=0)
                if remain != 0:
                    raise Warning("Remain between two files is not satisfied!")
                init = torch.cat([init, init_2], dim=0).squeeze()
                label = torch.cat([label, label_2], dim=0).squeeze()
            if len(prev_label) != 0 and not torch.eq(prev_label, label).any():
                raise Warning("Our modalities do not have the same labels")
            prev_label = label
            # remove nan values
            img[img != img] = -20.0
            images.append(img)

        # Return also the ids of the data to assure that dataloader is running smoothly
        ids = torch.arange(data_idx * self.outer_seq_length, data_idx * self.outer_seq_length + self.outer_seq_length)

        output = copy.deepcopy(images)
        # Reshape img to create the inner windows
        if self.inner_seq_length != 0:
            for i, img in enumerate(images):
                if self.keep_view[i] == 1:
                    img_shape = list(img.shape)
                    assert img_shape[
                               -1] % self.inner_seq_length == 0, "Quants of time in each view/modality must be divisable by the inner sequence length"
                    dim = 1 if self.outer_seq_length > 1 else 0
                    start_index = 0
                    windows = []

                    window_samples = int(img.shape[-1] / self.inner_seq_length)
                    inner_oversample = int(window_samples * self.inner_overlap[i])
                    assert inner_oversample != 0, "Overlapping in the inner sequence length is not possible"
                    # TODO: For some reason we dont take the last 1.5 secs or the 30 samples. Investigate that.
                    while (start_index + window_samples < img.shape[-1] + 1):
                        if len(img.shape) == 3:
                            current_window = img[:, :, start_index:start_index + window_samples]
                        elif len(img.shape) == 4:
                            current_window = img[:, :, :, start_index:start_index + window_samples]
                        elif len(img.shape) == 5:
                            current_window = img[:, :, :, :, start_index:start_index + window_samples]
                        start_index = int(start_index + inner_oversample)
                        windows.append(current_window.unsqueeze(dim=dim))
                    windows = torch.cat(windows, dim=dim)
                    if self.seq_views[i]:
                        output.append(windows)
                    else:
                        output[i] = windows

        return output, label.long(), init, ids

    def __len__(self):
        return self.dataset_true_length

    def preload_data(self):
        data_len = self.__len__()
        g_output, g_labels, g_init, g_ids = [], [], [], []
        pbar = tqdm(range(data_len), desc="Pre-loading validation data", leave=False)
        pre_loaded_idx = data_len
        for i in pbar:
            output, label, init, ids = self.__getitem__(index=i)
            g_output.append(output)
            g_labels.append(label)
            g_init.append(init)
            g_ids.append(ids)
            output_size = sys.getsizeof(g_output)
            labels_size = sys.getsizeof(g_labels)
            init_size = sys.getsizeof(g_init)
            ids_size = sys.getsizeof(g_ids)

            total_size = psutil.virtual_memory().percent

            # total_size = output_size + labels_size + init_size + ids_size

            pbar.set_description(
                "Pre-loading validation data {0:d} / {1:d}  RAM is {2:.1f}%".format(i, data_len, total_size))
            pbar.refresh()
            if total_size > self.config.byte_limits:
                pre_loaded_idx = i
                break
        total_size = output_size + labels_size + init_size + ids_size

        print("Cashed are {0:.0f} Gb and {1:.3f} Mb".format(total_size // (10 ** 9),
                                                            (total_size % (10 ** 9)) / (10 ** 6)))

        return g_output, g_labels, g_init, g_ids, pre_loaded_idx

    def choose_specific_patient(self, patient_num):
        changed_dirs = []
        for i in range(len(self.dataset)): changed_dirs.append([])
        for i in range(len(self.dataset[0])):
            if "patient_{}".format(f'{patient_num:02}') in self.dataset[0][i] or "n{}".format(f'{patient_num:04}') in \
                    self.dataset[0][i]:
                for j in range(len(self.dataset)):
                    changed_dirs[j].append(self.dataset[j][i])
        self.dataset = np.array(changed_dirs)
        self._get_len()

    def print_statistics_per_patient(self):
        for patient in range(len(self.dataset[0])):
            f = h5py.File(self.dataset[0][patient], 'r')
            labels = f["label"]
            c, counts = np.unique(labels[:, 0], return_counts=True)
            s = "File: {} has {} windows with labels ".format(self.dataset[0][patient], len(labels))
            for i in range(len(c)):
                s += "{}-{} ".format(c[i], f'{counts[i]:04}')
            print(s)

    def transform_images(self, images, num):
        aug_method = getattr(self.tf, self.aug[str(num)]["method"])
        # aug_method = globals()[self.aug[num]["method"]]
        for i in range(len(images)):
            # print("{}_{}".format(i,num))
            images[i] = aug_method(images[i], self.aug[str(num)], num)
        return images


class SleepDataLoader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test = self._get_datasets()
        if self.config.normalize and not self.config.load_ongoing:

            # calculate_metrics
            if not self.config.calculate_metrics:
                mean, std = self.load_metrics()
            else:
                mean, std = self.calculate_mean_std(sleep_dataset_train.dataset)

            sleep_dataset_train.set_mean_std(mean, std)
            sleep_dataset_val.set_mean_std(mean, std)
            sleep_dataset_test.set_mean_std(mean, std)
            self.metrics = {"mean": mean, "std": std}

        print(self.config.seq_length)
        if self.config.seq_length[0] > 1:
            shuffle_training_data = False
        elif hasattr(self.config, "shuffle_train"):
            shuffle_training_data = self.config.shuffle_train
        else:
            shuffle_training_data = True

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train, batch_size=self.config.batch_size,
                                                        shuffle=shuffle_training_data,
                                                        num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory,
                                                        worker_init_fn=_init_fn)
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val, batch_size=self.config.test_batch_size,
                                                        shuffle=False, num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test, batch_size=self.config.test_batch_size,
                                                       shuffle=False, num_workers=self.config.data_loader_workers,
                                                       pin_memory=self.config.pin_memory)
        if self.config.print_statistics and not self.config.load_ongoing:
            self._statistics_mat()
        else:
            # equal w
            self.weights = np.ones(self.config.num_classes)
            print("Class weights are ", end="")
            print(self.weights)

    def load_metrics(self):

        print("Loading metrics from {}".format(self.config.metrics_dir))
        metrics_file = open(self.config.metrics_dir, "rb")
        self.metrics = pickle.load(metrics_file)

        mean, std = {}, {}

        for i, f in enumerate(self.config.data_view_dir):
            mod = f[1][0] + "_" + f[1][1]
            mean[mod] = self.metrics["train"]["mean_{}".format(f[1][0])][f[1][1]]
            std[mod] = self.metrics["train"]["std_{}".format(f[1][0])][f[1][1]]

        return mean, std

    def load_metrics_ongoing(self, metrics):
        mean = metrics["mean"]
        std = metrics["std"]
        self.metrics = metrics
        self.train_loader.dataset.set_mean_std(mean, std)
        self.valid_loader.dataset.set_mean_std(mean, std)
        self.test_loader.dataset.set_mean_std(mean, std)

    def _gather_metrics(self, metrics_scramble):
        metrics = {}
        metrics["mean"], metrics["mean_sq"], metrics["std"], metrics["sum"] = {}, {}, {}, {}
        for i in range(len(self.config.data_view_dir)):
            mod = self.config.data_view_dir[i][1][0] + "_" + self.config.data_view_dir[i][1][1]
            metrics["mean"][mod], metrics["mean_sq"][mod], metrics["std"][mod], metrics["sum"][
                mod] = None, None, None, 0

        for i in range(len(self.config.data_view_dir)):
            mod = self.config.data_view_dir[i][1][0] + "_" + self.config.data_view_dir[i][1][1]
            for metrics_p in metrics_scramble:
                if metrics_p == []:
                    continue
                if metrics["sum"][mod] == 0:
                    metrics["mean"][mod] = metrics_p["mean"][mod]
                    metrics["mean_sq"][mod] = metrics_p["mean_sq"][mod]
                else:
                    metrics["mean"][mod] = (metrics["mean"][mod] * metrics["sum"][mod] + metrics_p["mean"][mod] *
                                            metrics_p["sum"][mod]) / (
                                                   metrics_p["sum"][mod] + metrics["sum"][mod])
                    metrics["mean_sq"][mod] = (metrics["mean_sq"][mod] * metrics["sum"][mod] + metrics_p["mean_sq"][
                        mod] * metrics_p["sum"][mod]) / (
                                                      metrics_p["sum"][mod] + metrics["sum"][mod])
                metrics["sum"][mod] += metrics_p["sum"][mod]
            varX = -np.multiply(metrics["mean"][mod], metrics["mean"][mod]) + metrics["mean_sq"][mod]
            metrics["std"][mod] = np.sqrt((varX * metrics["sum"][mod]) / (metrics["sum"][mod] - 1))

        return metrics

    def _parallel_file_calculate_mean_std(self, dataset, file_idx):
        metrics = {}
        metrics["mean"], metrics["mean_sq"], metrics["std"], metrics["sum"] = {}, {}, {}, {}
        for i in range(len(self.config.data_view_dir)):
            mod = self.config.data_view_dir[i][1][0] + "_" + self.config.data_view_dir[i][1][1]
            metrics["mean"][mod], metrics["mean_sq"][mod], metrics["std"][mod], metrics["sum"][
                mod] = None, None, None, 0

        for view_i in range(0, 2 * len(self.config.data_view_dir), 2):
            f = h5py.File(dataset[view_i][file_idx], 'r')
            mod = self.config.data_view_dir[int(view_i / 2)][1][0] + "_" + \
                  self.config.data_view_dir[int(view_i / 2)][1][1]
            if self.config.data_view_dir[int(view_i / 2)][1][0] == "time":
                data = np.array(f["X1"])
                file_mean = data.mean()
                file_mean_sq = np.square(data).mean()
                file_length = data.shape[0] * data.shape[1]

            elif self.config.data_view_dir[int(view_i / 2)][1][0] == "stft":
                data = np.array(f["X2"])
                file_mean = data.mean(axis=(1, 2))
                file_mean_sq = np.square(data).mean(axis=(1, 2))
                file_length = data.shape[1] * data.shape[2]

            metrics["mean"][mod] = file_mean
            metrics["mean_sq"][mod] = file_mean_sq
            metrics["sum"][mod] += file_length

        return metrics

    def _save_metrics(self, metrics):
        try:
            metrics_file = open(self.config.metrics_dir, "wb")
            pickle.dump(metrics, metrics_file)
            metrics_file.close()
            print("Metrics saved!")
        except:
            print("Error on saving metrics")

    def calculate_mean_std(self, dataset):

        num_cores = 6
        #         metrics_scramble = Parallel(n_jobs=num_cores)(delayed(self._parallel_file_calculate_mean_std)(dataset, file_idx) for file_idx in tqdm(range(len(dataset[0])), "Mean std calculations"))

        for file_idx in tqdm(range(len(dataset[0])), "Mean std calculations"):
            m = self._parallel_file_calculate_mean_std(dataset, file_idx)
        metrics = self._gather_metrics(metrics_scramble)
        if self.config.save_metrics:
            self._save_metrics(metrics)

        return metrics["mean"], metrics["std"]

    def _get_datasets(self):
        views_train = [self.config.data_roots + "/" + i[0] for i in self.config.data_view_dir]

        dirs_train_whole, train_len = self._read_dirs_mat(views_train)

        # Generate a test and val set from the training data
        dirs_train, dirs_val, dirs_test = self._split_data_mat(dirs_train_whole, self.config.val_split_rate)
        if not hasattr(self.config, "seq_legth"):
            self.config.seq_legth = [1, 0]

        valid_dataset = Sleep_Dataset(self.config, dirs_val, train_len)
        train_dataset = Sleep_Dataset(self.config, dirs_train, train_len)
        test_dataset = Sleep_Dataset(self.config, dirs_test, train_len)

        return train_dataset, valid_dataset, test_dataset

    def _read_dirs_mat(self, view_dirs):

        dataset = []
        for i, view_dir in enumerate(view_dirs):
            with open(view_dir) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                datafile_names = []
                len_windows = []
                for row in csv_reader:
                    dt = row[0].split("-")
                    datafile_names.append(dt[0])
                    len_windows.append(int(dt[1]))
            dataset.append(datafile_names)
            dataset.append(len_windows)
        num_views = len(view_dirs)
        return np.array(dataset), num_views

    def _unique_dataloader(self, dataset):
        label_description = "label" if self.config.huy_data else "labels"
        counts = np.zeros(self.config.num_classes)
        # Calculate weights for unbalanced classes
        total_classes = np.arange(self.config.num_classes)
        for file in tqdm(dataset[0], "Counting"):
            f = h5py.File(file, "r")["label"]
            labels = np.array(f).squeeze()
            if len(labels.shape) > 1:
                labels = labels[:, 0]
            classes, c = np.unique(labels, return_counts=True)

            # This is in case we only keep the patients that have all labels
            if len(c) < self.config.num_classes:
                # print(file)
                continue
            for i, cl in enumerate(classes):
                counts[int(cl - 1)] += c[int(i)]

        return total_classes, counts

    def _statistics_mat(self):

        total = []
        classes, counts = self._unique_dataloader(self.train_loader.dataset.dataset)
        total.append(["Training", classes, counts])
        v_classes, v_counts = self._unique_dataloader(self.valid_loader.dataset.dataset)
        total.append(["Validation", v_classes, v_counts])
        if self.config.use_test_set:
            t_classes, t_counts = self._unique_dataloader(self.test_loader.dataset.dataset)
            total.append(["Test", t_classes, t_counts])

        for label, cl, c in total:
            s = "In {} set we got ".format(label)
            for i in range(len(counts)):
                s = s + "Label {} : {} ".format(cl[i], int(c[i]))
            print(s)
        temperature = 0
        self.weights = counts.sum() / (counts + temperature)
        norm = np.linalg.norm(self.weights)
        self.weights = self.weights / norm
        print(self.weights)

    def _split_data_mat(self, dirs_train_whole, split_rate):
        dirs_test = np.array([])
        if (self.config.split_method == "patients_huy"):
            print("We are splitting dataset by huy splits")
            dirs_train, dirs_val, dirs_test = self._split_patients_huy(dirs_train_whole, 0)
        else:
            raise ValueError("No splitting method named {} exists.".format(self.config.split_method))
        return dirs_train, dirs_val, dirs_test

    def _split_patients_huy(self, dirs_train_whole, fold):

        f = loadmat(self.config.folds_file)
        f["train_sub"] = f["train_sub"].squeeze() - 1
        f["eval_sub"] = f["eval_sub"].squeeze() - 1
        f["test_sub"] = f["test_sub"].squeeze() - 1

        train_idx, val_idx, test_idx = [], [], []

        num_difference, prev = 0, -1
        for index, file_name in enumerate(dirs_train_whole[0]):
            patient_num = int(file_name.split("/")[-1][1:5])
            num_difference += patient_num - prev - 1
            prev = patient_num
            if patient_num - num_difference in f["train_sub"]:
                train_idx.append(index)
            elif patient_num - num_difference in f["eval_sub"]:
                val_idx.append(index)
            elif patient_num - num_difference in f["test_sub"]:
                test_idx.append(index)
            else:
                raise Warning("Splitting is not going well, some patient has no house. \n {}".format(file_name))
        return np.array(dirs_train_whole[:, train_idx]), np.array(dirs_train_whole[:, val_idx]), np.array(
            dirs_train_whole[:, test_idx])


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



config = {
    "seed": 28,
    "batch_size": 32,
    "test_batch_size": 32,
    "num_classes": 5,
    "data_roots": "/esat/biomeddata/SHHS_Dataset/no_backup",
    #"data_roots": "/esat/biomeddata/data_drop/r0779205/thesis",
    "data_view_dir": [
        [
            "/patient_mat_list.txt", ["stft", "eeg"] #patient_mat_list_500.txt
        ]
    ],
    "split_method": "patients_huy",
    "folds_file": "/esat/biomeddata/SHHS_Dataset/no_backup/data_split_eval.mat",
    "validation": True,
    "use_test_set": True,
    "seq_length": [
        21,
        0
    ],
    "shuffle_train": True,
    "seq_views": [
        False
    ],
    "keep_view": [
        1,
        0
    ],
    "inner_overlap": [
        0.5,
        0.5
    ],
    "num_modalities": 1,
    "dataloader_class": "SleepDataLoader",
    "normalize": True,
    "calculate_metrics": False,
    "save_metrics": False,
    "metrics_dir": "/esat/biomeddata/SHHS_Dataset/no_backup/metrics_eeg_eog_emg.pkl",
    "print_statistics": False,
    "data_loader_workers": 8,
    "pin_memory": True,
    "async_loading": True,
    "tdqm_disable": True,
}


config = dotdict(config)
dataloader = SleepDataLoader(config)
a = next(iter(dataloader.train_loader))



print("Data[0] shape is ", end="")
print(list(a[0][0].shape))
print("Labels shape is ", end="")
print(list(a[1].shape))
print("Inits shape is ", end="")
print(list(a[2].shape))
print("Idx shape is ", end="")
print(list(a[3].shape))

print(
    "\n32 is the batch size, 21 is the inner batch (single datapoints that are taken sequential) \nand the rest are the values")
import matplotlib.pyplot as plt

#print(a[1][22][0])

eeg = a[0][0][22][0]
t = np.arange(0, eeg.shape[-1])
f = np.arange(0, eeg.shape[-2])

# for signal in eeg:
signal = eeg[0]
plt.figure()
plt.title("REM (Real Data)")
plt.xlabel("Time bins")
plt.ylabel("Freq bins")
plt.pcolormesh(t, f, signal, vmin=signal.min(), vmax=signal.max(), shading='gouraud')
plt.colorbar()
plt.clim(-4,4)
plt.savefig('/esat/biomeddata/data_drop/r0779205/thesis/stft.png')
