import pdb
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Layer, Dense, Conv2D, MaxPool2D, Dropout, Flatten,
                                     GlobalAveragePooling2D, ZeroPadding2D)
import tensorflow as tf
import numpy as np

import yaml

class DeepCorrCNN(Model):
    def __init__(
        self,
        conv_filters,
        max_pool_sizes,
        strides,
        dense_layers,
        drop_p,
        debug=False,
    ):

        '''
        conv_filters: filters for the first two conv layers
        dense_layers: units for the last dense layers
        drop_p: dropout rate
        '''
        super().__init__(self)
        self.debug = debug
        # self.convs = Sequential(
        #     [
        #         Conv2D(conv_filters[0], [2, 20], strides=2, activation="relu"),
        #         MaxPool2D([1, 5]),
        #         Conv2D(conv_filters[1], [4, 10], strides=2, activation="relu"),
        #         MaxPool2D([1, 3]),
        #     ]
        # )

        
        layers = [x
            for (conv_filter, (kernel_win, kernel_size)), (stride1, stride2), maxpool_size in zip(
                    conv_filters, strides, max_pool_sizes
            )
            for x in (
                Conv2D(conv_filter, (kernel_win, kernel_size), strides=(stride1, stride2), activation="relu"),
                MaxPool2D((1, maxpool_size)),
            )]
        
        #print(layers)
        
        self.convs = Sequential(layers)

        self.flatten = Flatten()
        self.dense = Sequential()
        for i, units in enumerate(dense_layers):
            self.dense.add(
                Dense(units, activation=("relu" if i < len(dense_layers) - 1 else None))
            )
            if i < len(dense_layers) - 2:
                self.dense.add(Dropout(drop_p))
        self.dense.add(Dense(1, activation=None))

            
    def call(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        y = self.dense(x)
        if self.debug:
            self.convs.summary()
            self.dense.summary()
        return y
    
    
    def get_activations(self, x, name, debug=False):
        y = self.convs.get_layer(name)(x)
        if debug:
            self.convs.summary()
            self.dense.summary()
        return y
        
        
      
        
        
       

    