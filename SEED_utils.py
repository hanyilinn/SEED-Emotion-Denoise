import scipy.io as scio
import numpy as np
import os
import torch
import torch.nn.functional as F


def get_labels(label_path):
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(label_list):
    look_up_table = {-1: [1, 0, 0],
                     0: [0, 1, 0],
                     1: [0, 0, 1]}
    label_onehot = [np.asarray(look_up_table[label]) for label in label_list]
    return label_onehot


def get_frequency_band_idx(frequency_band):
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]


def build_extracted_features_dataset(folder_path, feature_name, frequency_band):
    frequency_idx = get_frequency_band_idx(frequency_band)
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    feature_vector_dict = {}
    label_dict = {}
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_features_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                     verify_compressed_data_integrity=False)
                    subject_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for trials in range(1, 16):
                        feature_vector_list = []
                        label_list = []
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T
                        feature_vector_list.extend(_ for _ in cur_feature)
                        for _ in range(len(cur_feature)):
                            label_list.append(labels[trials - 1])
                        feature_vector_trial_dict[str(trials)] = feature_vector_list
                        label_trial_dict[str(trials)] = label_list
                    feature_vector_dict[subject_name] = feature_vector_trial_dict
                    label_dict[subject_name] = label_trial_dict
                else:
                    continue
    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict


def build_preprocessed_eeg_dataset_CNN(folder_path):
    feature_vector_dict = {}
    label_dict = {}
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    try:
        all_mat_file = os.walk(folder_path) 
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_trials_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                   verify_compressed_data_integrity=False)
                    experiment_name = file_name.split('.')[0] # 去掉文件扩展名后的部分 eg: 1_20131030.mat -> 1_20131030
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for key in all_trials_dict.keys(): # eg: djc_eeg1
                        if 'eeg' not in key:
                            continue
                        feature_vector_list = []
                        label_list = []
                        cur_trial = all_trials_dict[key]
                        length = len(cur_trial[0])
                        pos = 0
                        while pos + 800 <= length:
                            feature_vector_list.append(np.asarray(cur_trial[:, pos:pos + 800]))
                            raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                            label_list.append(raw_label)
                            pos += 800
                        trial = key.split('_')[1][3:]
                        feature_vector_trial_dict[trial] = np.asarray(feature_vector_list)
                        label_trial_dict[trial] = np.asarray(label_2_onehot(label_list))

                    feature_vector_dict[experiment_name] = feature_vector_trial_dict
                    label_dict[experiment_name] = label_trial_dict
                else:
                    continue

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict


def subject_independent_data_split(feature_vector_dict, label_dict, test_subject_set):
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    for experiment in feature_vector_dict.keys():
        subject = experiment.split('_')[0]
        for trial in feature_vector_dict[experiment].keys():
            if subject in test_subject_set:
                test_feature.extend(feature_vector_dict[experiment][trial])
                test_label.extend(label_dict[experiment][trial])
            else:
                train_feature.extend(feature_vector_dict[experiment][trial])
                train_label.extend(label_dict[experiment][trial])
    return train_feature, train_label, test_feature, test_label


class RawEEGDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list, desire_shape):
        self.feature_list = feature_list
        self.label_list = label_list
        self.desire_shape = desire_shape

    def __getitem__(self, index):
        self.feature_list[index] = self.feature_list[index].reshape(self.desire_shape)
        feature = F.normalize(torch.from_numpy(self.feature_list[index]).float(), p=2, dim=2)
        label = torch.from_numpy(self.label_list[index]).long()
        label = torch.argmax(label)
        return feature, label

    def __len__(self):
        return len(self.label_list)

