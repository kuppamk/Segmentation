import os
import cv2
import glob
import numpy as np
from data_parameters import color_dict, N
from model_parameters import SIZE

def generate_train_test_val_paths(images_home_dir):
    train_path = os.path.join(images_home_dir,'train').replace("\\","/")
    val_path = os.path.join(images_home_dir,'val').replace("\\","/")
    test_path = os.path.join(images_home_dir,'test').replace("\\","/")
    return train_path, val_path, test_path

def generate_list_of_paths(train_path):
    train = []
    for subdir in os.listdir(train_path):
        sub_list = list(map(lambda x: x.replace("\\","/"),glob.glob(os.path.join(train_path,subdir,'*.png'))))
        train.extend(sub_list)
    return train

def label_json(x, ltrain_path):
    name = x.split('/',4)[-1].replace('leftImg8bit.png','gtFine_polygons.json')
    return os.path.join(ltrain_path,name).replace("\\","/")

def label_paths(x):
    lst = x.split('/')
    lst[-1] = lst[-1].replace('leftImg8bit','gtFine_color')
    lst[-4] = lst[-4].replace('leftImg8bit','gtFine')
    return '/'.join(lst)

def read_label(trn_img, shape=(1024, 2048)):
    lbl_path = label_paths(trn_img)
    lbl = cv2.imread(lbl_path)
    lbl = cv2.resize(lbl,shape)
    return lbl

def labeling_img(lbl):
    label_img = np.zeros((lbl.shape[0],lbl.shape[1],N),dtype=np.uint8)
    for i,color in color_dict.items():
        channel1 = (lbl[:,:,0]==color[0])
        channel2 = (lbl[:,:,1]==color[1])
        channel3 = (lbl[:,:,2]==color[2])
        label_img[:,:,i] = (channel1 & channel2 & channel3)
    #label_img = label_img.reshape(shape[0]*shape[1],-1)
    return label_img

def label2img(predict):
    """
    This will take logitcs of the model and return the image like colors
    """
    lbl = np.argmax(predict,axis=-1).reshape(SIZE)
    new_image = np.zeros((lbl.shape[0],lbl.shape[1],3),np.uint8)
    for color in color_dict:
        a = (lbl == color).astype(np.uint8)
        new_image[:,:,0] += a*color_dict[color][0]
        new_image[:,:,1] += a*color_dict[color][1]
        new_image[:,:,2] += a*color_dict[color][2]
    return new_image


