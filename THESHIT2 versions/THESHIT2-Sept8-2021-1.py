# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Transfer Learning Thesis

# https://thedatafrog.com/en/articles/image-recognition-transfer-learning/

# ## Imports

# +
# openCV to concatenate
import cv2
import numpy as np

# for image name
import random

# define and move to dataset directory
import os
# the directory where the data is stored
directory = "C:/Users/user/Desktop/Dataset"
os.chdir(directory)

# duplicate combined image to other directory
import shutil

#for stopping code if not the right amount of images
import sys

# import the needed packages
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np

# data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#k-fold cross validation
import pandas as pd                                     #store all images and their class/label
from sklearn.model_selection import StratifiedKFold     
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split    #split the dataset to train-test
import tensorflow as tf                                 #for callbacks
# -

# ## Crop Image to Reduce Background

# We shall crop the image in order to reduce the background so that the egg in the image would be more focused on. The cropping procedure will not affect the size diference of the eggs from one another.

# https://stackoverflow.com/questions/61927877/how-to-crop-opencv-image-from-center

# +
tadirect = directory + "/TwoAngleEggs"
#crop width and height
width = 3600
height = 3600

for x in os.listdir(tadirect):
    directory2 = ""
    directory2 = tadirect + "/" + x     #images to be cropped
    #print(directory2)
    
    curdir = os.listdir(directory2)
    
    for z in curdir:
        tempim1 = directory2 + "/" + z
        
        im1 = cv2.imread(tempim1)
        
        center = (im1.shape[0] / 2, im1.shape[1] / 2)
        #print(center)
        x = center[0] - width/2
        y = center[1] - height/2

        #print(x)
        #print(y)
        
        #slicing to perform cropping
        print("cropping " + z + " in dimensions (" + str(width) + "," + str(height) + ")")
        crop_img = im1[int(y):int(y+width), int(x):int(x+height)]
        
        #save and replace the image
        cv2.imwrite(tempim1, crop_img)
        print("crop finish " + z)

print("\nAll finished")
# -

# ## Concatenate two angles of egg

# two images of an egg are taken, one at its front and one at its back.

# +
#"""
tadirect = directory + "/TwoAngleEggs"

#needed for k-fold validation later on
image = []

#check if there is the right amount of images (1 for each angle of egg)
for w in os.listdir(tadirect):
    directory2 = ""
    directory2 = tadirect + "/" + w
    curdir = os.listdir(directory2)

    #number of files in directory
    filecnt = len([name for name in curdir])
    
    if filecnt%2 == 1:
        print("ERROR (" + w + "): Incorrect number of images.")
        sys.exit()

for x in os.listdir(tadirect):
    directory2 = ""
    directory2 = tadirect + "/" + x                            #where the two images of two angles of egg is stored
    directory3 = directory  + "/CombinedTwoAngleEggs/" + x     #where to put the combined image of the angles
    directory4 = directory  + "/ToBeUsed/" + x                 #where combined and data augmented of the image are stored
    #print(directory2)
    
    curdir = os.listdir(directory2)
    #print(curdir)
    
    #number of files in directory
    filecnt = len([name for name in curdir])
    
    for y in range(filecnt):
        if y%2 == 0:
            #print(directory2 + "/" + curdir[y])
            tempim1 = directory2 + "/" + curdir[y]
            im1 = cv2.imread(tempim1)
            #im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            
            #print(directory2 + "/" + curdir[y+1])
            tempim2 = directory2 + "/" + curdir[y+1]
            im2 = cv2.imread(tempim2)
            
            # vertically concatenates images 
            # of same width 
            
            #resize for ResNet50
            width = 224
            height = 224

            dim = (width,height)
            
            im_v = cv2.vconcat([cv2.resize(im1, dim), cv2.resize(im2, dim)])
            #im_v = cv2.vconcat(im1,im2)
            
            # show the output image
            #cv2.imshow("image",im_v)
            #save image
            
            #create folder to store concatenated images if not existing
            newpath = directory3 + "2"
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            """    
            newpath2 = directory4 + "3"
            if not os.path.exists(newpath2):
                os.makedirs(newpath2)
            """
            
            combine_img = newpath + "/IMG_" + str(random.randint(0,99999999)) + "_" + str(random.randint(0,999999)) + ".jpg"
            
            cv2.imwrite(combine_img,cv2.resize(im_v, dim))
            #shutil.copy2(combine_img, newpath2)
            
            if x == "Balut-penoy":
                lbl = "0"
            elif x == "Salted egg":
                lbl = "1"
            else:
                lbl = "2"
            
            image.append([combine_img,lbl])
            
            print(curdir[y] + " and " + curdir[y+1] + " combined.")

print("All finished")
#"""
           
"""
im1 = cv2.imread("C:/Users/user/Desktop/Dataset/Balut-penoy/IMG_20210902_081744.jpg")
im2 = cv2.imread("C:/Users/user/Desktop/Dataset/Balut-penoy/IMG_20210902_081801.jpg")
#im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

width = 224
height = 224

dim = (width,height)

im_v = cv2.vconcat([cv2.resize(im1, dim), cv2.resize(im2, dim)])

cv2.imshow("wew",im_v)
cv2.waitKey()

#create a new image from the images
#cv2.imwrite('C:/Users/user/Desktop/Dataset/Balut-penoy/Lolo3.jpg', im_v)
"""

print()

# +
#for x in image:
#    print(x)
# -

df = pd.DataFrame(image,columns=['filename','label'])
df

print(df.isna().any())

# ## Split Data to train and test set

## Split the data to train and test
X = df[['filename']]
y = df[['label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Save the split sets to train and test set folders.

# +
#folder to store the train and test set folders
fortraintest = directory + "/ToBeUsed"
if not os.path.exists(fortraintest):
    os.makedirs(fortraintest)

#train set folder
tr = fortraintest + "/train_set"
if not os.path.exists(tr):
    os.makedirs(tr)
else:
    #remove previous folder and create new one
    shutil.rmtree(tr)
    os.makedirs(tr)
    
#test set folder
te = fortraintest + "/test_set"
if not os.path.exists(te):
    os.makedirs(te)
else:
    #remove previous folder and create new one
    shutil.rmtree(te)
    os.makedirs(te)

#copying the images to the train set folder
for index, row in X_train.iterrows():
    trcpy = shutil.copy2(row['filename'], tr)
    #X_train.loc[(X_train['filename'] == row['filename']),'filename']=trcpy
    #X_train.loc[:,'filename'].replace(row['filename'],trcpy, inplace=True)
    #print(row['filename'])
    #print(trcpy)
    #print(X_train.loc[X_train['filename']==row['filename'],'filename'])
    X_train.loc[X_train['filename']==row['filename'],'filename']=trcpy
    #print(X_train.loc[X_train['filename']==row['filename'],'filename'])

#copying the images to the test set folder
for index, row in X_test.iterrows():
    tecpy = shutil.copy2(row['filename'], te)
    #X_test.loc[(X_test['filename'] == row['filename']),'filename']=tecpy
    #X_test.loc[:,'filename'].replace(row['filename'],tecpy, inplace=True)
    X_test.loc[X_test['filename']==row['filename'],'filename']=tecpy
# -

df2 = pd.concat([X_train,y_train],axis=1)
df2

# ## K-fold validation on train set

# https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538

# +
Y = df2[['label']]

kf = KFold(n_splits = 5)
                         
skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True)
# -

# for data augmentation

# +
resnet50 = keras.applications.resnet50

imgdatagen = ImageDataGenerator(
        preprocessing_function = resnet50.preprocess_input,
        horizontal_flip = True,
        vertical_flip = True,
    )


# -

# to save best model per fold

def get_model_name(k):
    return 'model_'+str(k)+'.h5'


# plot_history

def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.legend(['train_accuracy','val_accuracy'])
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.legend(['train_loss','val_loss'])
    
    plt.show() 


# getting the folds and creating the data generators

"""
#if exist, create new one for another set of models with new parameters

#directory where models per set parameters are stored
gen_dir = "C:/Users/user/Desktop"

save_cur = ""  #check if there is any saved_models folders

for x in os.listdir(gen_dir):
    if "saved_models" in x:
        print(x)
        save_cur = x
        
if save_cur == "":
    save_dir = gen_dir + "/saved_models1"
    #print(save_dir)
    os.makedirs(save_dir)
else:
    save_dir = gen_dir + "/saved_models" + str(int(save_cur[-1]) + 1)
    #print(save_dir)
    os.makedirs(save_dir)
"""

# +
#initialize parameters
VALIDATION_ACCURACY = []                          #store all validation_accuracy
VALIDATION_LOSS = []                              #store all validation_loss
image_dir = tr                                    #where the train set images are stored
batch_size = 32                                   #batch size
fold_var = 1                                      #number of fold of the model to be called back
n = len(df2)                                      #number of data in train set
num_epochs = 100                                  #number of epochs
dim = (224,224)                                   #required dimensions for process_unit (ResNet50)


#===========if exist, create new one for another set of models with new parameters======================

#directory where models per set parameters are stored
gen_dir = "C:/Users/user/Desktop"

save_cur = ""  #check if there is any saved_models folders

for x in os.listdir(gen_dir):
    if "saved_models" in x:
        #print(x)
        save_cur = x

#save_dir is where the created models using the current set of parameters are added        
        
if save_cur == "":
    save_dir = gen_dir + "/saved_models1"
    #print(save_dir)
    os.makedirs(save_dir)
else:
    save_dir = gen_dir + "/saved_models" + str(int(save_cur[-1]) + 1)
    #print(save_dir)
    os.makedirs(save_dir)

#=========================================================================================================    
    
for train_index, val_index in kf.split(np.zeros(n),Y):
    training_data = df2.iloc[train_index]
    validation_data = df2.iloc[val_index]

    train_data_generator = imgdatagen.flow_from_dataframe(training_data, directory = image_dir,
                               x_col = "filename", y_col = "label",
                              class_mode = "categorical", target_size = dim, batch_size = batch_size,
                                shuffle = True)
    valid_data_generator  = imgdatagen.flow_from_dataframe(validation_data, directory = image_dir,
                            x_col = "filename", y_col = "label",
                            class_mode = "categorical", target_size = dim, batch_size = batch_size,
                                shuffle = True)

    # CREATE NEW MODEL
    conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    for layer in conv_model.layers:
        layer.trainable = False
    x = keras.layers.Flatten()(conv_model.output)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    #x = keras.layers.Dense(100, activation='relu')(x)

    x = keras.layers.Dropout(0.2)(x) #to further reduce overfitting

    predictions = keras.layers.Dense(3, activation='softmax')(x)
    full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
    
    # COMPILE NEW MODEL
    full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.0001),
                  metrics=['accuracy']) #categorical_crossentropy cause it is more than 2 classes

    # CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+"/"+get_model_name(fold_var), 
                            monitor='val_accuracy', verbose=1, 
                            save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
   
    print("Fold number " + str(fold_var))

    # FIT THE MODEL
    history = full_model.fit(train_data_generator,
                epochs=num_epochs,
                callbacks=callbacks_list,
                validation_data=valid_data_generator)
    
    #PLOT HISTORY
    plot_history(history, yrange=(0.9,1))

    # LOAD BEST MODEL to evaluate the performance of the model
    full_model.load_weights(save_dir+"/model_"+str(fold_var)+".h5")
    
    results = full_model.evaluate(valid_data_generator)
    results = dict(zip(full_model.metrics_names,results))

    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])

    tf.keras.backend.clear_session()

    fold_var += 1    
# -

print("Validation Accuracy of All Folds:")
print(VALIDATION_ACCURACY)
print("Validation Loss of All Folds:")
print(VALIDATION_LOSS)

# ## Data Augmentation

# We will be applying data augmentation in order to make the model more robust and accurate.

# +
batch_size = 1

def generators(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = 0.3,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ('Salted egg2','Balut-penoy2','Table egg2'),
        batch_size = batch_size,
        subset = 'training',
        seed = 42
    )

    val_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ('Salted egg2','Balut-penoy2','Table egg2'),
        batch_size = batch_size,
        subset = 'validation',
        seed = 42
    )
    
    return train_dataset, val_dataset


# -

# ## Plot accuracy and loss as a function of the epoch

# The functon below will graph the training and validation accuracy, as well as the training and validation loss of the training and testing of the model

def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    
    plt.show() 


# ## Split the data to Train and Test Set

# split the dataset to train and test set with a preprocessing function of resnet50.preprocess_input for ResNet50 will be used.

resnet50 = keras.applications.resnet50
directory2 = directory + "/CombinedTwoAngleEggs"
os.chdir(directory2)
train_dataset, val_dataset = generators((224,224), preprocessing=resnet50.preprocess_input)

print(train_dataset[0][1])

# ## Add Custom Classifier

# remove the classifier from the ResNet50 architecture and add a new one that will give the desired output (balut/penoy, salted egg, table egg)

# +
conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in conv_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
#x = keras.layers.Dense(100, activation='relu')(x)

x = keras.layers.Dropout(0.2)(x) #to further reduce overfitting

predictions = keras.layers.Dense(3, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()
# -

# ## Train the Model

# After that, we will train the model.

# +
#compile the model
full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.0001),
                  metrics=['accuracy']) #categorical_crossentropy cause it is more than 2 classes   
    
history = full_model.fit_generator(
    train_dataset, 
    validation_data = val_dataset,
    workers=10,
    epochs=100,
) meron ata teka
# -

plot_history(history, yrange=(0.9,1))

# ## Train the model using K-fold Cross Validation

# Another way to train the model is to use K-fold Cross Validation. We will set the number of folds to 5.

# +
batch_size = 10

def generators2(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        vertical_flip = True,
    )

    height, width = shape

    augmented = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ('Salted egg2','Balut-penoy2','Table egg2'),
        batch_size = batch_size,
        seed = 42
    )
    
    return augmented


# -

print(os.getcwd())

# +
#augmented = generators2((224,224),preprocessing=resnet50.preprocess_input)
# -

print(train_dataset[0])

# +
#compile the model
full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.0001),
                  metrics=['accuracy']) #categorical_crossentropy cause it is more than 2 classes

# define 10-fold cross validation test harness
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []

#print(kfold.split(augmented))

for train, test in kfold.split(train_dataset):
    #compile the model
    full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.0001),
                  metrics=['accuracy']) #categorical_crossentropy cause it is more than 2 classes
    
    print("TRAIN:\n" + str(train))
    print("TEST:\n" + str(test))
    
    #print(augmented[train])
    
    history = full_model.fit_generator(
        train, 
        validation_data = test,
        workers=10,
        epochs=1,
    )
    
    #plot_history(history, yrange=(0.9,1))
# -

print(train_dataset[0])



# +
x = [1,2,3,4,5,6,7,8,9,10,11,12]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(x):
    print (train_index, test_index)
# -

import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 1, 1, 0])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
print(skf)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ## Save and Load The Model

# https://www.tensorflow.org/guide/keras/save_and_serialize

# we will save the trained model and load it for model prediction/evaluation.

# +
#save model to same directory as the notebook
full_model.save('duckEggQual_model')

#load the model
reconstructed_model = keras.models.load_model('duckEggQual_model')
# -

# ## Model Evaluation

# we will now evaluate the model by predicting a selected image of its category.

# +
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

img_path = 'C:\Users\user\Desktop\Dataset\CombinedTwoAngleEggs\Balut-penoy2'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(full_model.predict(x))
plt.imshow(img)
