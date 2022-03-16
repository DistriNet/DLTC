from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Dropout,
    Flatten,
)


class DeepCorrCNN(Model):
    def __init__(
        self,
        conv_filters,
        kernel_sizes,
        max_pool_sizes,
        strides,
        dense_layers,
        drop_p,
        debug=False,
    ):
        """
        conv_filters: filters for the first two conv layers
        dense_layers: units for the last dense layers
        kernel_sizes: dimensions of the convolution filters
        strides: offset of the pooling window each step.
        max_pool_sizes: size of the max pooling window
        dense_layers: dense layers for the sequential model
        drop_p: dropout rate
        """
        super().__init__(self)
        self.debug = debug

        self.convs = Sequential(
            [
                x
                for conv_filter, kernel_size, maxpool_size, stride in zip(
                    conv_filters, kernel_sizes, max_pool_sizes, strides
                )
                for x in (
                    Conv2D(
                        conv_filter,
                        kernel_size,
                        strides=(stride, 1),
                        activation="relu",
                    ),
                    MaxPool2D((1, maxpool_size)),
                )
            ]
        )

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
