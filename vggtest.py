# coding=utf-8
from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
# from keras.backend import softmax
import numpy as np
import os
from sklearn.model_selection import train_test_split
import vggtrain

def get_images(data_file_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(data_file_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


resultpath='./model/model-134.h5'
model2 = load_model(resultpath)

def main_predict(img_path):
    # test data
    X=vggtrain.read_image(img_path,True)
    # the first way of load mode
    # model2 = load_model(resultpath)
    # model2.compile(loss=categorical_crossentropy,
    #               optimizer=Adam(0.0001), metrics=['accuracy'])

    # test_loss, test_acc = model2.evaluate(X, Y, verbose=0)
    # print('Test loss:', test_loss)
    # print('Test accuracy:', test_acc)
    y = model2.predict(X)  
    cls_name=np.argmax(y[0])
    cls_prob=np.max(y[0])
    # print(y)
    # print("----------predicct is: {},pro is {}".format(cls_name,cls_prob))
    return cls_name,cls_prob,y

if __name__ == "__main__":
    names=[]
    with open('./vgg/Cls2Name.csv') as f:
        for line in f.readlines():
            names.append(line[:-2])   #这样会多一个\r\n 
    for img in get_images('./data/test/'):
        print (img)
        cls_name ,cls_prob,all_prob=main_predict(img)
        print ('the img {} is predicted as {} ({}) '.format(img,names[cls_name],cls_prob))
        with open(img.replace(img[-4:],'.csv'),'w') as f:
            f.writelines('{},{},{}'.format(cls_name,names[cls_name],cls_prob))
            f.writelines('\r\n')
            for index,item in enumerate(all_prob[0]):
                f.writelines('{},{},{}\r\n'.format(index,names[index],item))

