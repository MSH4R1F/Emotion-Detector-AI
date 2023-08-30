import numpy as np
# Numpy needed to perform  mathematical operations on arrays
import pandas as pd
# pandas needed to perform advanced data manipulaion


import matplotlib.pyplot as plt
# pyplot is needed for interactive plots and simple cases of programmatic plot generation


from keras.layers import Flatten, Dense


# Importing the rest of the Keras modules
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy





# Working with pre trained model 

# The input shape has the standard dimensions for an image
# Mobile Net returns a Keras image classification model,

base_model = MobileNet( input_shape=(224,224,3), include_top= False )


# To prevent retraining of the model every layer trainability is set to False
for layer in base_model.layers:
    layer.trainable = False

# We are using this to flatten the matrix to a one-dimensiona array
x = Flatten()(base_model.output)

# Activate the softmax layer and give the density of 7 layers so there will be 7 layers.

x = Dense(units=7 , activation='softmax' )(x)

#Creating our model
#Model groups layers into an object with training and inference features.
model = Model(base_model.input, x)

# Configures the model for training.
model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )


# In order to make the most of our few training examples, we will "augment" them via a number of random 
# transformations, so that our model would never see twice the exact same picture. This helps prevent overfitting and 
# helps the model generalize better.
train_datagen = ImageDataGenerator(
     zoom_range = 0.2, # randomly zooming inside pictures
     shear_range = 0.2, # randomly applying shearing transformations
     horizontal_flip=True, #andomly flipping half of the images horizontally
     rescale = 1./255 # a value by which we will multiply the data before any other processing
)


# read the images from our folders containing images.
train_data = train_datagen.flow_from_directory(directory= "train", 
                                               target_size=(224,224), #size of your input images, every image will be resized to this size.
                                               batch_size=32, #No. of images to be yielded from the generator per batch.
                                  )


# Testing :    "train_data.class_indices"




# Prepare our test data
val_datagen = ImageDataGenerator(rescale = 1./255 )

# Importing from our test images
val_data = val_datagen.flow_from_directory(directory= "test", 
                                           target_size=(224,224), 
                                           batch_size=32,
                                  )



# to visualize the images in the traing data denerator 

train_img , label = train_data.next()


# Testing : 
# function when called will prot the images 
def plotImages(img_arr, label):
  """
  input  :- images array 
  output :- plots the images 
  """
  count = 0
  for im, l in zip(img_arr,label) :
    plt.imshow(im)
    plt.title(im.shape)
    plt.axis = False
    plt.show()
    
    count += 1
    if count == 10:
      break

# function call to plot the images 
plotImages(train_img, label)