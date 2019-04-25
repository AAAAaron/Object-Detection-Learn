# utf-8
import os
import numpy as np
import tensorflow as tf
import random
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split

# import argparse
from keras.callbacks import ModelCheckpoint
# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--weights", required=True, help="path toweights directory")
# args = vars(ap.parse_args())

HEIGHT,WIDTH,CHANNELS=224,224,3


def read_image(file_path, preprocess,height=HEIGHT,width=WIDTH):
    img2 = image.load_img(file_path, target_size=(height, width))
    #img2 = cv2.imread(file_path)
    #img2= cv2.resize(img2,(HEIGHT,WIDTH),interpolation=cv2.INTER_NEAREST)
    
    # image.save_img(os.path.join('./data/cunc',os.path.basename(file_path)),img)
    # image.save_img(os.path.join('./data/cv',os.path.basename(file_path)),img2)
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)

    if preprocess:
        x = preprocess_input(x)
    return x

pic_num=300        
def read_and_process_image(data_dir,width=WIDTH, height=HEIGHT, channels=CHANNELS, preprocess=True):
    
    train_classes= [data_dir +  i for i in os.listdir(data_dir) ]
    class_label=[i for i in os.listdir(data_dir)]
    with open('./name.csv','w') as f:
        for clsname in class_label:
            f.writelines(clsname)
            f.writelines('\r\n')
    train_images = []
    for train_class in train_classes:
        newlist=[]
        dirnames=[]
        for i in os.listdir(train_class):
            dirnames.append(i)
        if len(dirnames)<=pic_num:
            continue
        chooseindex =np.random.randint(0,len(dirnames),size=(1,pic_num))
        for i in chooseindex[0]:
            newlist.append(train_class+'/'+dirnames[i])
        train_images= train_images + newlist
    
    random.shuffle(train_images)
    
    def prep_data(images, proprocess):
        count = len(images)
        data = np.ndarray((count, height, width, channels), dtype = np.float32)
        labels=[]
        for i, image_file in enumerate(images):
            image = read_image(image_file, preprocess,height,width)
            data[i] = image
            labels.append(class_label.index(image_file.split('/')[-2]))
        return data,labels
    
    
    X ,labels= prep_data(train_images, preprocess)
    
    assert X.shape[0] == len(labels)
    
    print("Train shape: {}".format(X.shape))
    
    return X, labels

def vgg16_model(num_classes,input_shape= (HEIGHT,WIDTH,CHANNELS)):
    vgg16 = VGG16(include_top=False, weights='imagenet',input_shape=input_shape)
    
    for layer in vgg16.layers:
        layer.trainable = False
    last = vgg16.output
    x = Flatten()(last)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=vgg16.input, outputs=x)
    
    return model

if __name__ == "__main__":
    HEIGHT,WIDTH,CHANNELS=224,224,3
    n_cls=134
    # read img
    X, Y = read_and_process_image('/home/aaron/project/data/')
    plt.figure(figsize=(25,10))
    sns.countplot(Y)
    plt.savefig('./countplot.png',dpi=500)
    plt.close()
    Y = np_utils.to_categorical(Y)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # one-hot
    
    # y_test = np_utils.to_categorical(y_test)
    model_vgg16 = vgg16_model(num_classes=n_cls)
    # model_vgg16.summary()

    model_vgg16.compile(loss='categorical_crossentropy',optimizer = Adam(0.0001), metrics = ['accuracy'])
    
    
    checkpoint = ModelCheckpoint('./model/model-134.h5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    history = model_vgg16.fit(X,Y, validation_split=0.33,shuffle=True,epochs=50,batch_size=64,verbose=True,callbacks=callbacks)
    # score = model_vgg16.evaluate(X_test, y_test, verbose=1)
    # print("Large CNN Error: %.2f%%" %(100-score[1]*100))
