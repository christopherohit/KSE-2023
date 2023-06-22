import os 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAvgPool2D, Dense, Flatten, Conv2D, Lambda, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import schedules, Adam, SGD, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def model_configuration():
    
    width, height, channel = 11024, 1024, 32
    batch_size = 128
    num_class = 7
    verbose = 1
    n = 3
    init_fm_dims = 16
    shorcut_type = 'identity'