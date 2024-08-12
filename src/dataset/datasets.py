from monai.transforms import apply_transform

import os
import sys

import numpy as np

sys.path.append("..")
from src.utils.utils import get_key
from src.utils.constants import NUM_CLASS, TASK_NAMES, TEMPLATE, REVERSE_TASK_NAMES

from monai.data import Dataset, CacheDataset


class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey

    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img["name"])
            self.data_dic[key].append(img)

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f"the dataset {key} has no data"
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return (
            apply_transform(self.transform, data_i)
            if self.transform is not None
            else data_i
        )

    def __getitem__(self, index):
        # the index generated outside is only used to select the dataset
        # the corresponding data in each dataset is selelcted by the
        # np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_status()

    def data_status(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img["name"])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f"the dataset {key} has no data"
            self.data_num.append(item)

        self.datasetlen = len(self.datasetkey)

    def index_uniform(self, index):
        # the index generated outside is only used to select the dataset
        # the corresponding data in each dataset is selelcted by the
        # np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)


class UniformClassesDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey

    def dataset_split(self, data, datasetkey):
        data_classes = dict(
            zip(np.arange(1, NUM_CLASS + 1), [[] for _ in range(NUM_CLASS)])
        )

        # dict of ALL the classes and the datasets where they are present
        for key in datasetkey:
            dataset_classes = TEMPLATE[key]
            for cl in dataset_classes:
                data_classes[cl].append(key)

        # eliminate classes for which we have no images (maybe didnt load all datasets)
        for key in list(data_classes):
            if len(data_classes[key]) == 0:
                print(f"Category {key} has no data")
                del data_classes[key]

        # get image indexes per dataset
        dataset_idxs = dict(zip(datasetkey, [[] for _ in range(len(datasetkey))]))
        for idx, datum in enumerate(data):
            d_key = REVERSE_TASK_NAMES[datum["task"]]
            dataset_idxs[d_key].append(idx)

        # dict of the image indexes per class
        self.data_num = dict(
            zip(list(data_classes.keys()), [[] for _ in range(len(data_classes))])
        )
        for key, item in data_classes.items():
            for i in item:
                self.data_num[key].extend(dataset_idxs[i])

        self.datasetlen = len(self.data_num)

    def index_uniform(self, index):
        # the index generated outside is only used to select the dataset
        # the corresponding data in each dataset is selelcted by the
        # np.random.randint function
        set_index = list(self.data_num.keys())[index % self.datasetlen]
        data_index = np.random.choice(self.data_num[set_index], size=1)[0]
        return data_index, set_index

    def _transform(self, index: int, class_id: None):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        data_i["class_id"] = class_id
        return (
            apply_transform(self.transform, data_i)
            if self.transform is not None
            else data_i
        )

    def __getitem__(self, index):
        post_index, cls_index = self.index_uniform(index)
        return self._transform(post_index, cls_index)


class UniformClassesCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey

    def dataset_split(self, data, datasetkey):
        data_classes = dict(
            zip(np.arange(1, NUM_CLASS + 1), [[] for _ in range(NUM_CLASS)])
        )

        # dict of ALL the classes and the datasets where they are present
        for key in datasetkey:
            dataset_classes = TEMPLATE[key]
            for cl in dataset_classes:
                data_classes[cl].append(key)

        # eliminate classes for which we have no images (maybe didnt load all datasets)
        for key in list(data_classes):
            if len(data_classes[key]) == 0:
                print(f"Category {key} has no data")
                del data_classes[key]

        # get image indexes per dataset
        dataset_idxs = dict(zip(datasetkey, [[] for _ in range(len(datasetkey))]))
        for idx, datum in enumerate(data):
            d_key = REVERSE_TASK_NAMES[datum["task"]]
            dataset_idxs[d_key].append(idx)

        # dict of the image indexes per class
        self.data_num = dict(
            zip(list(data_classes.keys()), [[] for _ in range(len(data_classes))])
        )
        for key, item in data_classes.items():
            for i in item:
                self.data_num[key].extend(dataset_idxs[i])

        self.datasetlen = len(self.data_num)

    def index_uniform(self, index):
        set_index = list(self.data_num.keys())[index % self.datasetlen]
        data_index = np.random.choice(self.data_num[set_index], size=1)[0]
        return data_index, set_index

    def _transform(self, index: int, class_id: None):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        data_i["class_id"] = class_id
        return (
            apply_transform(self.transform, data_i)
            if self.transform is not None
            else data_i
        )

    def __getitem__(self, index):
        post_index, cls_index = self.index_uniform(index)
        return self._transform(post_index, cls_index)


def get_dataset_list(args, is_train=True):
    images = []
    train_post_lbl = []
    names = []
    task_names = []

    for task_id in args.datasetkey:
        task = TASK_NAMES[task_id]

        file_name = f"{task}_{args.phase}.txt"

        dataset_list = os.path.join(args.data_txt_path, file_name)
        label_folder = "post_labels"

        for line in open(dataset_list):
            name = line.strip().split()[1]
            images.append(args.data_root_path + line.strip().split()[0])
            # labels.append(args.data_root_path + line.strip().split()[1])
            train_post_lbl.append(
                args.data_root_path + name.replace("labels", label_folder)
            )
            names.append(name)
            task_names.append(task)
    data_dicts = [
        {"image": image, "post_label": post_label, "name": name, "task": task}
        for image, post_label, name, task in zip(
            images, train_post_lbl, names, task_names
        )
    ]
    print("dataset len {}".format(len(data_dicts)))
    return data_dicts


def get_training_dataset(args, transforms):
    data_dicts = get_dataset_list(args)
    if args.cache_dataset:
        if args.uniform_sample:
            dataset = UniformClassesCacheDataset(
                data=data_dicts,
                transform=transforms,
                cache_rate=args.cache_rate,
                datasetkey=args.datasetkey,
            )
        else:
            dataset = CacheDataset(
                data=data_dicts,
                transform=transforms,
                cache_rate=args.cache_rate,
            )
    else:
        if args.uniform_sample:
            dataset = UniformClassesDataset(
                data=data_dicts,
                transform=transforms,
                datasetkey=args.datasetkey,
            )
        else:
            dataset = Dataset(data=data_dicts, transform=transforms)
    return dataset


def get_eval_dataset(args, transforms):
    data_dicts = get_dataset_list(args, is_train=False)
    if args.end_test != 0:
        data_dicts = data_dicts[args.init_test : args.end_test]
    else:
        data_dicts = data_dicts[args.init_test :]
    print("Evaluating {} images".format(len(data_dicts)))

    dataset = Dataset(data=data_dicts, transform=transforms)
    return dataset
