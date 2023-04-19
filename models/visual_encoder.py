
import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv3D, BatchNormalization, Bidirectional, LSTM, Reshape

import numpy as np

import tensorflow as tf

class Visual_Encoder():
    def __init__(self):
      super().__init__()
      
      self.model = Sequential(
         Conv3D(3, kernel_size=(5,5,5), strides=(1,2,2), activation='relu', padding='same', input_shape=(40,96,96,3)),
         BatchNormalization(),
         Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'),
         BatchNormalization(),
         Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'),
         BatchNormalization(),
         Conv3D(64, kernel_size=(3,3,3), strides=(1,2,2), activation='relu', padding='same'),
         BatchNormalization(),
         Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'),
         BatchNormalization(),
         Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'),
         BatchNormalization(),
         Conv3D(512, kernel_size=(3,3,3), strides=(1,1,1), activation='relu', padding='same'),
         BatchNormalization(),
         Flatten(),
         Reshape(target_shape=(40,512)),
         Bidirectional(LSTM(256, return_sequences=True))
      )

    def forward(self, x):
      x = self.model(x)
      return x
      
    

