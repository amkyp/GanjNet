# models/gcn_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from spektral.data import Graph, Dataset
from spektral.data.loaders import SingleLoader
from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class GCNConv(Conv):
    """
    Custom Graph Convolutional Layer.
    """
    def __init__(
        self,
        channels,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.channels = channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs):
        x, a = inputs
        if isinstance(a, tf.SparseTensor):
            a = tf.sparse.to_dense(a)
        output = K.dot(x, self.kernel)
        output = ops.modal_dot(a, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = self.activation(output)
        return output

    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)

class CustomGraphDataset(Dataset):
    """
    Custom Dataset class for Spektral.
    """
    def __init__(self, adj_matrix, feature_matrix, label_vector, mask, **kwargs):
        self.adj_matrix = adj_matrix
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector
        self.mask = mask
        super().__init__(**kwargs)

    def read(self):
        return [Graph(x=self.feature_matrix, a=self.adj_matrix, y=self.label_vector, mask=self.mask)]

    def get_y(self):
        return self.label_vector

    def get_mask(self):
        return self.mask

class GCNModel(tf.keras.Model):
    """
    Graph Convolutional Network model with embedding extraction.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = GCNConv(16, activation='relu')
        self.conv2 = GCNConv(16, activation='relu')
        self.conv3 = GCNConv(num_classes, activation=None)  # No activation for logits

    def call(self, inputs, return_embeddings=False):
        x, a = inputs
        h1 = self.conv1([x, a])
        h2 = self.conv2([h1, a])
        embeddings = h2  # Save embeddings after the second layer
        logits = self.conv3([h2, a])
        if return_embeddings:
            return logits, embeddings
        return logits

    def get_embeddings(self, inputs):
        """
        Extracts embeddings from the model.
        """
        _, embeddings = self.call(inputs, return_embeddings=True)
        return embeddings.numpy()

def train_model(train_dataset, val_dataset, num_classes, word, epochs=1000, lr=0.001):
    """
    Trains the GCN model and extracts embeddings.
    """
    model = GCNModel(num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), weighted_metrics=['accuracy'])

    train_loader = SingleLoader(train_dataset, sample_weights=train_dataset[0].mask)
    val_loader = SingleLoader(val_dataset, sample_weights=val_dataset[0].mask)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=15,
        monitor='val_loss',
        restore_best_weights=True
    )

    history = model.fit(
        train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=epochs,
        callbacks=[early_stopping],
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch, verbose=1,
    )

    # Extract node embeddings after training
    embeddings = model.get_embeddings([train_dataset[0].x, train_dataset[0].a])

    return model, history, embeddings

def save_model(model, word, output_dir):
    """
    Saves the trained model.
    """
    model.save(os.path.join(output_dir, f"{word}_model.keras"))
    print(f"Model for word '{word}' saved to {output_dir}")
