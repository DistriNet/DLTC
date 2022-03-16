import os

import numpy as np
from .pipeline import Dataset
from .helper import print_red, calc_param_size
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from .model import DeepCorrCNN
from sklearn.metrics import classification_report, average_precision_score

class Base:
    def __init__(self, config, cv_i=0, pred_only=False, h5_path=None, debug=False):
        """
        Base class for training and prediction
        conf: config.yml path
        cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
        pred_only: if True, only used for prediction process.
        h5_path: if None, use default .h5 file in config.yml, otherwise, use the given path.
        """
        self.config = config
        self.debug = debug
        self.cv_i = cv_i
        self.pred_only = pred_only
        self.h5_path = h5_path
        self._init_log()
        self._init_device()
        self._init_dataset()

    def _init_log(self):
        log_path = self.config["data"]["log_path"]["tf"]
        self.train_log = os.path.join(log_path, "train")
        try:
            os.makedirs(self.train_log)
        except FileExistsError:
            pass

    def _init_device(self):
        seed = self.config["data"]["seed"]
        np.random.seed(seed)
        tf.random.set_seed(seed)

        try:
            gpu_check = tf.config.list_physical_devices("GPU")
        except AttributeError:
            gpu_check = tf.test.is_gpu_available()
        except IndexError:
            gpu_check = False
        if not (gpu_check):
            print_red("We are running on CPU!")
        return

    def _init_dataset(self):
        dataset = Dataset(
            cf=self.config,
            cv_i=self.cv_i,
            test_only=self.pred_only,
            h5_path=self.h5_path,
        )
        if not self.pred_only:
            self.train_generator = dataset.train_generator
            self.val_generator = dataset.val_generator
        self.test_generator = dataset.test_generator
        return


class Server(Base):
    def __init__(self, config, cv_i=0, new_lr=False, pred_only=False, h5_path=None):
        """
        conf: config.yml path
        cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
        new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
        pred_only: if True, only used for prediction process.
        h5_path: if None, use default .h5 file in config.yml, otherwise, use the given path.
        """
        super().__init__(config=config, cv_i=cv_i, pred_only=pred_only, h5_path=h5_path)
        self.pred_only = pred_only
        self._init_model()

    def _init_model(self):
        self.model = DeepCorrCNN(
            conv_filters=self.config["conv_layers"],
            kernel_sizes=self.config["kernel_sizes"],
            max_pool_sizes=self.config["max_pool_sizes"],
            strides=self.config["strides"],
            dense_layers=self.config["dense_layers"],
            drop_p=self.config["drop_p"],
        )
        self.model(
            np.random.rand(1, 8, 300, 1).astype("float32"), training=not self.pred_only
        )

        self.loss = lambda props, y_truth: tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=props, labels=y_truth, pos_weight=self.config["pos_weight"]
            )
        )
        if not self.pred_only:
            self.optimizer = Adam(self.config["lr"])

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def train_epoch(self, epoch):
        train_loss, train_props, train_y = self.single_epoch(self.train_generator)
        val_loss, val_props, val_y = self.single_epoch(
            self.val_generator, training=False, desc="Val"
        )

        report = self._calculate_metrics(train_props, train_y, val_props, val_y)
        report["training_iteration"] = epoch
        report["train_loss"] = train_loss
        report["val_loss"] = val_loss

        return report

    def _calculate_metrics(self, train_props, train_y, val_props, val_y, threshold=0.5):
        """
        train_props: Model output for train set
        train_y: Labels of train set.
        val_props: Model output for validation set.
        val_y: Labels for validation set.
        threshold: Given threshold to compute a prediction in classification setting.
        """
        train_report = classification_report(
            train_y, train_props > threshold, output_dict=True
        )
        val_report = classification_report(val_y, val_props > threshold, output_dict=True)
        train_report = {k.replace(" ", "_"): v for k, v in train_report.items()}
        val_report = {k.replace(" ", "_"): v for k, v in val_report.items()}

        av_prec = average_precision_score(val_y, val_props)
        val_report["av_prec"] = av_prec

        return {"train": train_report, "val": val_report}

    @tf.function
    def step(self, x, y_truth):
        """
        x: Output of the model for the mini-batch.
        y_truth: ground truth labels for the mini-batch.
        """
        x = x
        y_truth = y_truth
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            props = tf.reshape(
                props, [-1]
            )  # specific for tf.nn.sigmoid_cross_entropy_with_logits
            loss = self.loss(props, y_truth)
            props = tf.sigmoid(props)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return props, loss

    def single_epoch(self, generator, training=True, desc="Training"):
        """
        generator: instance of class Generator.
        training: if True, for training; otherwise, for val or testing.
        desc: desc str uesd for tqdm.
        """
        n_steps = generator.spe
        total_loss = 0
        saved_props = []
        saved_y_truth = []

        for step, (x, y_truth) in enumerate(generator.epoch()):
            x = tf.constant(x.astype("float32"))
            y_truth = tf.constant(y_truth.astype("float32"))
            if training:
                props, loss = self.step(x, y_truth)
            else:
                props = self.model(x, training=False)
                # specific for tf.nn.sigmoid_cross_entropy_with_logits
                props = tf.reshape(props, [-1])
                loss = self.loss(props, y_truth)
                props = tf.sigmoid(props)
            total_loss += loss.numpy()
            saved_props = np.hstack([saved_props, props.numpy()])
            saved_y_truth = np.hstack([saved_y_truth, y_truth.numpy()])

        avg_loss = round(total_loss / (n_steps), 3)

        return avg_loss, saved_props, saved_y_truth

    def predict(self):
        _, props, y_truth = self.single_epoch(
            self.test_generator, training=False, desc="Test", save_props=True
        )
        return props, y_truth
