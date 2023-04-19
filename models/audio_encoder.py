import numpy as np

import os
import numpy as np
from tensorflow.keras.layers import Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Reshape

data_dir ="C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\dataset"

# Set the image dimensions
img_height = 160
img_width = 160

# Get the list of file names in the data directory
file_list = os.listdir(data_dir)

# Initialize an empty array to hold the mel-spectrogram data
data = np.zeros((len(file_list), img_height, img_width))

class Audio_Encoder():
    def __init__(self):
      super().__init__()
      
      self.model = Sequential(
         Conv2D(32, (3, 3), activation='relu', padding='same'),
         MaxPooling2D((2, 2), padding='same'),
         Conv2D(64, (3, 3), activation='relu', padding='same'),
         MaxPooling2D((2, 2), padding='same'),
         Conv2D(128, (3, 3), activation='relu', padding='same'),
         MaxPooling2D((2, 2), padding='same'),
         Flatten(),
         Dense(128, activation='relu'),
         Dense(img_height * img_width, activation='linear'),
         Reshape((img_height, img_width, 1))
      )

    def forward(self, x):
      encoded_features = self.model(x)
      return encoded_features