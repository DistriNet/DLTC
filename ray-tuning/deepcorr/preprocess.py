import pickle
import yaml
from random import shuffle
import numpy as np
import os
import h5py
from tqdm.notebook import tqdm


class Preprocess:
    """
    pos: paired flows
    neg: unpaired flows
    """

    def __init__(self, config_yml="config.yml"):
        super().__init__()
        with open(config_yml) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config_yml = config_yml
        self.flow_size = self.config["data"]["flow_size"]
        self.dataset_path = self.config["data"]["dataset_path"]
        self.h5_path = self.config["data"]["h5_path"]
        self.pickle_path = self.config["data"]["pickle_path"]
        print("Pickle path: ", self.pickle_path, "; h5 path: ", self.h5_path)
        self.n_neg_per_pos = self.config["data"]["n_neg_per_pos"]
        self.mod = 1 + self.n_neg_per_pos  # number of flows generated from each sample
        self.ratio_train = self.config["data"]["ratio_train"]
        return

    def get_dataset(self):
        """
        Get the reorganized dataset saved in .pkl format
        """
        try:
            with open(self.pickle_path, "rb") as f:
                dataset = pickle.load(f)
        except:
            print("No dataset?!")
            dataset = []
            for name in tqdm(
                os.listdir(self.dataset_path), desc=f"Generating {self.pickle_path}"
            ):
                if (
                    name.split("_")[-1]
                    == self.pickle_path.split("/")[-1].split(".")[0] + ".pickle"
                ):
                    with open(os.path.join(self.dataset_path, name), "rb") as f:
                        dataset += pickle.load(f)
            with open(self.pickle_path, "wb") as f:
                pickle.dump(dataset, f)
        return dataset

    def get_xy(self, dataset):
        """
        dataset: list
        return: x: ndarray, for each dataset sample there are 1 positive(paired) flow + some negative(unpaired) flows.
                            each input data is in shape of [8 * flow_size].
                y: ndarray
        """
        n_pos = self.n_pos
        n_flows = self.n_flows
        flow_size = self.flow_size
        mod = self.mod
        x = np.zeros((n_flows, 8, flow_size, 1))
        y = np.zeros((n_flows))
        for i in tqdm(range(n_pos), desc="Generating x, y"):
            index = mod * i

            up_here_time = dataset[i]["here"][0]["<-"][:flow_size]
            up_here_time += [0] * (flow_size - len(up_here_time))
            down_there_time = dataset[i]["there"][0]["->"][:flow_size]
            down_there_time += [0] * (flow_size - len(down_there_time))
            up_there_time = dataset[i]["there"][0]["<-"][:flow_size]
            up_there_time += [0] * (flow_size - len(up_there_time))
            down_here_time = dataset[i]["here"][0]["->"][:flow_size]
            down_here_time += [0] * (flow_size - len(down_here_time))

            up_here_size = dataset[i]["here"][1]["<-"][:flow_size]
            up_here_size += [0] * (flow_size - len(up_here_size))
            down_there_size = dataset[i]["there"][1]["->"][:flow_size]
            down_there_size += [0] * (flow_size - len(down_there_size))
            up_there_size = dataset[i]["there"][1]["<-"][:flow_size]
            up_there_size += [0] * (flow_size - len(up_there_size))
            down_here_size = dataset[i]["here"][1]["->"][:flow_size]
            down_here_size += [0] * (flow_size - len(down_here_size))

            x[index, 0, :, 0] = np.array(up_here_time) * 1000.0
            x[index, 1, :, 0] = np.array(down_there_time) * 1000.0
            x[index, 2, :, 0] = np.array(up_there_time) * 1000.0
            x[index, 3, :, 0] = np.array(down_here_time) * 1000.0

            x[index, 4, :, 0] = np.array(up_here_size) / 1000.0
            x[index, 5, :, 0] = np.array(down_there_size) / 1000.0
            x[index, 6, :, 0] = np.array(up_there_size) / 1000.0
            x[index, 7, :, 0] = np.array(down_here_size) / 1000.0

            y[index] = 1

            indices = list(range(n_pos))
            unpaired = indices[:i] + indices[i + 1 :]
            shuffle(unpaired)
            for j in range(self.n_neg_per_pos):
                index = mod * i + j + 1

                up_here_time = dataset[unpaired[j]]["here"][0]["<-"][:flow_size]
                up_here_time += [0] * (flow_size - len(up_here_time))
                down_there_time = dataset[i]["there"][0]["->"][:flow_size]
                down_there_time += [0] * (flow_size - len(down_there_time))
                up_there_time = dataset[i]["there"][0]["<-"][:flow_size]
                up_there_time += [0] * (flow_size - len(up_there_time))
                down_here_time = dataset[unpaired[j]]["here"][0]["->"][:flow_size]
                down_here_time += [0] * (flow_size - len(down_here_time))

                up_here_size = dataset[unpaired[j]]["here"][1]["<-"][:flow_size]
                up_here_size += [0] * (flow_size - len(up_here_size))
                down_there_size = dataset[i]["there"][1]["->"][:flow_size]
                down_there_size += [0] * (flow_size - len(down_there_size))
                up_there_size = dataset[i]["there"][1]["<-"][:flow_size]
                up_there_size += [0] * (flow_size - len(up_there_size))
                down_here_size = dataset[unpaired[j]]["here"][1]["->"][:flow_size]
                down_here_size += [0] * (flow_size - len(down_here_size))

                x[index, 0, :, 0] = np.array(up_here_time) * 1000.0
                x[index, 1, :, 0] = np.array(down_there_time) * 1000.0
                x[index, 2, :, 0] = np.array(up_there_time) * 1000.0
                x[index, 3, :, 0] = np.array(down_here_time) * 1000.0

                x[index, 4, :, 0] = np.array(up_here_size) / 1000.0
                x[index, 5, :, 0] = np.array(down_there_size) / 1000.0
                x[index, 6, :, 0] = np.array(up_there_size) / 1000.0
                x[index, 7, :, 0] = np.array(down_here_size) / 1000.0
                y[index] = 0
        return x, y

    def get_indices(self):
        """
        Return indices for training and testing in the x, y matrix.
        """
        indices = list(range(self.n_pos))
        n_train = int(self.n_pos * self.ratio_train)
        shuffle(indices)
        train_indices = []
        for i in indices[:n_train]:
            train_indices += list(
                range(i * self.mod, i * self.mod + self.n_neg_per_pos + 1)
            )
        test_indices = []
        for i in indices[n_train:]:
            test_indices += list(
                range(i * self.mod, i * self.mod + self.n_neg_per_pos + 1)
            )
        return train_indices, test_indices

    def gen_h5(self, overwrite=False):
        if os.path.exists(self.h5_path) and not overwrite:
            print(f"{self.h5_path} exists already!")
            return
        dataset = self.get_dataset()
        self.n_pos = len(dataset)
        self.n_flows = self.n_pos * self.mod
        print(self.n_flows)
        print(
            f"Generating h5 file: {self.n_flows} pairs - with {self.n_pos} associated and {(self.mod-1)*self.n_pos} non-associated ({100*self.n_pos/self.n_flows}\% pos)"
        )
        x, y = self.get_xy(dataset)
        train_indices, test_indices = self.get_indices()

        with h5py.File(self.h5_path, "w") as h5f:
            g = h5f.create_group("data")
            g.create_dataset("x", data=x)
            g.create_dataset("y", data=y)
            g = h5f.create_group("indices")
            g.create_dataset("train", data=train_indices)
            g.create_dataset("test", data=test_indices)
        return

    def gen_crossval_indices(self, overwrite=False):
        """
        To generate k-fold-cross-validation indices.
        {'train_0':[],'val_0':[],'train_1':[],'val_1':[],...} is saved as .pkl
        """
        crossval_indices_path = self.config["data"]["crossval_indices_path"]
        if os.path.exists(crossval_indices_path) and not overwrite:
            print(f"{crossval_indices_path} exists already.")
            return
        with h5py.File(self.h5_path, "r") as f:
            ids = list(f["indices"]["train"])
        n_ids = len(ids)
        shuffle(ids)
        n_fold = self.config["data"]["n_fold"]
        res = {}
        if n_fold <= 1:
            val_ratio = 0.05
            res["train_0"] = ids[int(val_ratio * n_ids) :]
            res["val_0"] = ids[: int(val_ratio * n_ids)]
        else:
            for i in range(n_fold):
                left = int(i / n_fold * n_ids)
                right = int((i + 1) / n_fold * n_ids)
                res["train_{}".format(i)] = ids[:left] + ids[right:]
                res["val_{}".format(i)] = ids[left:right]
        for i in res.values():
            shuffle(i)
        with open(crossval_indices_path, "wb") as f:
            pickle.dump(res, f)
        return

    def main_run(self):
        self.gen_h5()
        self.gen_crossval_indices()
        return


if __name__ == "__main__":
    p = Preprocess()
    p.main_run()
