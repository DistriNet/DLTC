import pdb
import yaml
import os
import time
import sys
import numpy as np
import pipeline
from helper import print_red, calc_param_size, ReduceLROnPlateau
from tqdm.notebook import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle
from utils import accuracy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import DeepCorrCNN
import shutil
from sklearn.metrics import average_precision_score, f1_score


class Base:
    def __init__(self, cf='config.yml', fold=-1, run=-1, pred_only=False, init_data=True, h5_path=None):
        '''
        Base class for training and prediction
        cf: config.yml path
        fold: Which test fold in the cross validation. 
        run: Which seed.
        pred_only: if True, only used for prediction process.
        h5_path: if None, use default .h5 file in config.yml, otherwise, use the given path.
        '''
        self.cf = cf
        self.fold = fold
        self.run = run
        self.pred_only = pred_only
        self.h5_path =h5_path
        self._init_config()
        self._init_log()
        self._init_device(self.run)
        if init_data:
            self._init_dataset()
        
    def _init_config(self):
        with open(self.cf) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.debug = self.config["FLAG_DEBUG"]
        return
    
    def _init_log(self):
        log_path = self.config['data']['log_path']['tf']
        self.train_log = os.path.join(log_path, 'train')
        try:
            os.makedirs(self.train_log)
        except FileExistsError:
            pass

    def _init_device(self, seed=-1):
        if seed <= 0:
            seed = self.config['data']['seed']
        print("Seed ", seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        try: 
            gpu_check = tf.config.list_physical_devices('GPU')
#             tf.config.experimental.set_memory_growth(gpu_check[0], True)
        except AttributeError: 
            gpu_check = tf.test.is_gpu_available()
        except IndexError:
            gpu_check = False
        if not (gpu_check):
            print_red('We are running on CPU!')
        return
    
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, fold=self.fold, 
                                   test_only=self.pred_only, 
                                   h5_path=self.h5_path) 
        if not self.pred_only:
            self.train_generator = dataset.train_generator
            self.val_generator = dataset.val_generator
        self.test_generator = dataset.test_generator
        return

    
class Server(Base):
    def __init__(self, cf='config.yml', fold=-1, run=-1, new_lr=False, pred_only=False, init_data=True, h5_path=None, load_path=None, tdc=False):
        '''
        cf: config.yml path
        fold: Which test fold in the cross validation.
        run: Which seed.
        new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
        pred_only: if True, only used for prediction process.
        h5_path: if None, use default .h5 file in config.yml, otherwise, use the given path.
        tdc: type of saved model weights, defines the function to use for restoring the weights.
        '''
        super().__init__(cf=cf, fold=fold, run=run, pred_only=pred_only, init_data=init_data, h5_path=h5_path)
        self.pred_only = pred_only
        self.weighted_loss = self.config['train']['weighted_loss']
        self._init_model()
        self.check_resume(new_lr=new_lr, load_path=load_path, tdc=tdc)
    
    def _init_model(self):
        self.model = DeepCorrCNN(
            conv_filters=self.config["train"]["conv_filters"],
            max_pool_sizes=self.config["train"]["max_pool_sizes"],
            strides=self.config["train"]["strides"],
            dense_layers=self.config["train"]["dense_layers"],
            drop_p=self.config["train"]["drop_p"],
        )
        self.model(
            np.random.rand(1, 8, 300, 1).astype("float32"), training=not self.pred_only
        )
        print(
            "Param size = {:.3f} MB".format(
                calc_param_size(self.model.trainable_variables)
            )
        )
        
        if self.weighted_loss:
            self.loss = lambda props, y_truth: tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=props, labels=y_truth,
                                                         pos_weight=int(self.config['train']['pos_weight']))
            )
        else:
            self.loss = lambda props, y_truth: tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=props, labels=y_truth)
        )
        if not self.pred_only:
            self.optimizer = Adam(self.config['train']["lr"])
            self.scheduler = ReduceLROnPlateau(self.optimizer, framework="tf")
        #print(self.model.convs.summary())
        print(self.model.dense.summary())

        self.n_fold = self.config['data']['n_fold']
        self.mode = self.config['train']['mode']
        self.save_checkpoints = self.mode == "train" # != "tune"
     
        self.patience = 0
        if "patience" in self.config['train']:
            self.patience = self.config['train']['patience']
            print("Early stopping enabled, patience ", self.patience) 
        
        #print("Saving checkpoints: ", self.save_checkpoints)
        
        
    def check_resume(self, new_lr=False, load_path=None, tdc=False):
        '''
        Restore saved model, parameters and statistics.
        new_lr: change learning rate.
        load_path: saved model.
        tdc: type of saved model weights, defines the function to use for restoring the weights.
        '''
        print("Restore for prediction only: ", self.pred_only)
        if self.pred_only:
            if load_path:
                self.model = tf.keras.models.load_model(load_path)
            else:
                # If fold is 0 and higher, cross-validation evaluation
                if self.fold >= 0:
                    if self.run >= 0:
                        load_path = os.path.join(self.train_log, self.config['train']['train_best']
                                                 + str(self.fold) + '-' + str(self.run))
                        print(load_path)
                    else:
                        load_path = os.path.join(self.train_log, self.config['train']['train_best'] + str(self.fold))
                        print(load_path)
                else:    
                    load_path = os.path.join(self.train_log, self.config['train']['train_best'])
                    print(load_path)
            
            if tdc:
                self.model = tf.keras.models.load_model(load_path)
            else:
                self.model.load_weights(load_path)
            print("Reloaded best weights from ", load_path)
            return
        
        if self.save_checkpoints:
            checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            
            if self.fold < 0:
                # Stats updates after each epoch
                self.last_aux = os.path.join(self.train_log, self.config['train']['last_aux'])
                # Save best weights
                self.train_best = os.path.join(self.train_log, self.config['train']['train_best'])
                # Stats updated only after best epoch
                self.best_aux = os.path.join(self.train_log, self.config['train']['best_aux'])
                # Uncomment to save last weights too
                #self.train_last = os.path.join(self.train_log, self.config['train']['train_last'])
            else:
                # Check if seed is specified:
                if self.run >= 0:
                    self.last_aux = os.path.join(self.train_log, self.config['train']['last_aux'].replace(".pkl",
                                                     str(self.fold) + '-' + str(self.run) + ".pkl"))
                    self.train_best = os.path.join(self.train_log, self.config['train']['train_best']
                                                   + str(self.fold) + '-' + str(self.run))
                    self.best_aux = os.path.join(self.train_log, self.config['train']['best_aux'].replace(".pkl",
                                                     str(self.fold) + '-' + str(self.run) + ".pkl"))
                else:
                    self.last_aux = os.path.join(self.train_log, self.config['train']['last_aux'].replace(".pkl",
                                                     str(self.fold) +".pkl"))
                    self.train_best = os.path.join(self.train_log, self.config['train']['train_best']
                                                   + str(self.fold))
                    self.best_aux = os.path.join(self.train_log, self.config['train']['best_aux'].replace(".pkl",
                                                     str(self.fold) + ".pkl"))
            print("Best model is saved here: ", self.train_best)
            
            '''
            if os.path.exists(self.last_aux):
                checkpoint.restore(self.manager.latest_checkpoint)
                with open(self.last_aux, 'rb') as f:
                    state_dicts = pickle.load(f)
                self.epoch = state_dicts['epoch'] + 1
                self.history = state_dicts['history']
                if new_lr:
                    del(self.optimizer) # I'm not sure if this is necessary
                    self.optimizer = Adam(self.config['train']['lr']) 
                else:
                    self.scheduler.load_state_dict(state_dicts['scheduler'])
                self.best_val_loss = state_dicts['best_loss']
                self.best_epoch = state_dicts['best_epoch']
            else:
                self.epoch = 0
                self.best_epoch = 0
                self.history = defaultdict(list)
                self.best_val_loss = float('inf')
            '''
            
        self.epoch = 0
        self.best_loss_epoch = 0
        self.best_av_prec_epoch = 0
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_av_prec = 0

            
    def train(self, early_cond="best av prec"):
        '''
        Train the model. If self.save_checkpoints is true, update checkpoint at every epoch, save best weights.
        early_cond: which metric is used as a condition for early stopping: "best loss" or "best av prec".
        Patience value for early stopping is set in the config file.
        '''
        n_epochs = self.config['train']['epochs']
        for epoch in range(n_epochs):
            is_best = False
            # Train epoch
            loss = self.single_epoch(self.train_generator)
            # Validation
            val_loss, val_props, val_y = self.single_epoch(self.val_generator, 
                                                      training = False, save_props = True,
                                                      desc = 'Val')
            
            # Compute AP (Average Precision) on validation data
            val_av_prec = average_precision_score(val_y, val_props)
            
            print(f"AP[{self.epoch}]={round(val_av_prec, 4)}, val loss {val_loss}; ", end='')
            
            self.scheduler.step(val_loss)
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_av_prec'].append(val_av_prec)
            
            # Update best loss, check if early stopping should be done based on no improvement on the validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_loss_epoch = epoch
                if early_cond == "best loss":
                    is_best = True
            elif early_cond == "best loss":
                # Early stopping
                if self.epoch - self.best_loss_epoch >= self.patience:
                    print("\nEarly stopping at epoch {}: no better loss after {} epochs.".format(self.epoch, self.patience))
                    break
                else:
                    print("(X)", end='')
            
            # Update best AP, check if early stopping should be done based on no improvement on the validation AP
            if val_av_prec > self.best_val_av_prec:
                self.best_val_av_prec = val_av_prec
                self.best_av_prec_epoch = epoch
                if early_cond == "best av prec":
                    is_best = True
            elif early_cond == "best av prec":
                # Early stopping TODO
                if self.epoch - self.best_av_prec_epoch >= self.patience:
                    print("\nEarly stopping at epoch {}: no better AP after {} epochs.".format(self.epoch, self.patience))
                    break
                else:
                    print("(X)", end='')
                
            if self.save_checkpoints:
                # Save what the current epoch ends up with.
                # SAVE LAST:
                # self.manager.save()
                state_dicts = {
                    'epoch': self.epoch,
                    'history': self.history,
                    'scheduler': self.scheduler.state_dict(),
                    'best_loss': self.best_val_loss,
                    'best_av_prec': self.best_val_av_prec,
                    'best_loss_epoch': self.best_loss_epoch,
                    'best_av_prec_epoch': self.best_av_prec_epoch
                }
                with open(self.last_aux, 'wb') as f:
                    pickle.dump(state_dicts, f)

                if is_best:
                    self.model.save_weights(self.train_best)
                    shutil.copy(self.last_aux, self.best_aux)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                to_continue = input(f'OOPS: self.epoch > n_epochs: {self.epoch}>{n_epochs}. Continue?')
                if to_continue != 'y':
                    break
        return
    
    @tf.function
    def step(self, x, y_truth):
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            props = tf.reshape(props, [-1]) # specific for tf.nn.sigmoid_cross_entropy_with_logits
            loss = self.loss(props, y_truth)
#             loss += tf.add_n(self.model.losses) # l2 regularization
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return props, loss
 

    def single_epoch(self, generator, training=True, desc='Training', save_props=False):
        '''
        One epoch over data. Returns loss.
        generator: instance of class Generator.
        training: if True, for training; otherwise, for val or testing.
        desc: desc str uesd for tqdm.
        save_props: if True, also return actual predictions and ground truth labels
        '''
        n_steps = generator.spe
        sum_loss = 0
        if save_props:
            saved_props = []
            saved_y_truth = []
        with tqdm(generator.epoch(), total = n_steps,
                  desc = f'{desc} | Epoch {self.epoch}' if not self.pred_only else desc) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
            #for step, (x, y_truth) in enumerate(generator.epoch()):
                x = tf.constant(x.astype('float32'))
                y_truth = tf.constant(y_truth.astype('float32'))
                if training:
                    props, loss = self.step(x, y_truth)
                else:
                    props = self.model(x, training=False)
                    props = tf.reshape(props, [-1]) # specific for tf.nn.sigmoid_cross_entropy_with_logits
                    loss = self.loss(props, y_truth)
                sum_loss += loss.numpy()
                if save_props:
                    saved_props = np.hstack([saved_props, props.numpy()])
                    saved_y_truth = np.hstack([saved_y_truth, y_truth.numpy()])

                #postfix = OrderedDict()
                #postfix['Loss'] = round(sum_loss/(step+1), 3)
                #pbar.set_postfix(postfix)

                if self.debug and step >= 1:
                    break
                    
        avg_loss = round(sum_loss / n_steps, 3)
        
        if save_props:
            return avg_loss, saved_props, saved_y_truth
        else:
            return avg_loss
    
    
    def predict(self, load_path=None, tdc=False, test_data=None, save_props=True):
        '''
        Test the model on given data. Returns AP, predictions and ground truth labels.
        generator: instance of class Generator.
        load_path: saved model.
        tdc: type of saved model weights, defines the function to use for restoring the weights.
        save_props: if True, also return actual predictions and ground truth labels
        '''
        if self.fold >= 0:
            if self.run >= 0:
                load_path = os.path.join(self.train_log, self.config['train']['train_best']
                                         + str(self.fold) + '-' + str(self.run))
                print(load_path)
            else:
                load_path = os.path.join(self.train_log, self.config['train']['train_best'] + str(self.fold))
                print(load_path)
        else:
            load_path = os.path.join(self.train_log, self.config['train']['train_best'])
            
        if tdc:
            self.model = tf.keras.models.load_model(save_path)
        else:
            self.model.load_weights(save_path)
            
        print("Reloaded best weights from ", save_path)
        
        if test_data == "train":
            gen = self.train_generator
        elif test_data == "val":
            gen = self.val_generator
        else:
            # test
            gen = self.test_generator
            
        loss, props, y_truth = self.single_epoch(gen, 
                                                 training = False, 
                                                 desc = 'Test',
                                                 save_props = save_props)
        
        av_prec = average_precision_score(y_truth, props)
        print(f'Testing result: loss = {loss}, AP = {av_prec}')
        return av_prec, props, y_truth
    
    
if __name__ == '__main__':
    t = Training()
    t.train()
    t.testing()