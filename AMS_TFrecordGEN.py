# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:56:24 2017

@author: vgdev
"""

# This script converts our training data from bitmap images in labeled folders to TFRecord file(s)

import cv2
import numpy as np
import os
import tensorflow as tf 
import sys
from random import shuffle

# TRAIN, VALIDATION, TEST 
#PORTIONS = [0.75,0.20,0.05]
PORTIONS = [1.0, 0, 0]
# IMAGE DIM (WIDTH, HEIGHT)/home/paperspace/ams/model/ams_train/
#RESIZE_DIM = (32,32)

TRAINING_IMAGES_FOLDER_PATH = '/home/paperspace/ams/Training_Data' 

TRAIN_TFR_NAME = TRAINING_IMAGES_FOLDER_PATH + '/AMP_train.tfrecords' 
VALID_TFR_NAME = TRAINING_IMAGES_FOLDER_PATH + '/AMP_validation.tfrecords'
TEST_TFR_NAME =  TRAINING_IMAGES_FOLDER_PATH + '/AMP_test.tfrecords'


# Takes a pathname and returns properly resized image 
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # don't resize here, instead, crop or pad to maintain sizing info
    #img = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    return img
    
# returns a list of tuples: [(filename, label)]
def get_filenames(): 
    if (os.path.isfile(TRAINING_IMAGES_FOLDER_PATH + '/label_map.txt')): 
        os.remove(TRAINING_IMAGES_FOLDER_PATH + '/label_map.txt')
    
    f = open(TRAINING_IMAGES_FOLDER_PATH + '/label_map.txt', 'w')
    
    images = []
    label_names = os.listdir(TRAINING_IMAGES_FOLDER_PATH)
    
    label = 0 
    for lab_name in label_names: 
        if (os.path.isdir(TRAINING_IMAGES_FOLDER_PATH+'/'+lab_name)):
            image_names = os.listdir(TRAINING_IMAGES_FOLDER_PATH+'/'+lab_name)
            f.write(lab_name + ' ' + str(label) + '\n')
            for img_name in image_names: 
                images.append( (lab_name + '/' + img_name, label) ) 
            label += 1
        
    f.close()
    return images
 
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
    
        
# takes the TFR file name, list of tuples representing files and labels and a string distinguishing the portion it represents: train, val, test
def create_TFRecord(TF_name, file_list, portion): 
    # Open a TFRecord 
    writer = tf.python_io.TFRecordWriter(TF_name)
    
    count = 0
    for img_path, label in file_list: 
        # load image 
        count+=1
        img = load_image(TRAINING_IMAGES_FOLDER_PATH + '/' + img_path)

        # create feature 
        feature = {'label': _int64_feature(label) , 
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                   'height':_int64_feature(img.shape[1]),
                   'width':_int64_feature(img.shape[0])}
                   

        # create example protocol buffer 
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # serialize to string and write to file  
        writer.write(example.SerializeToString())
        
    # close writer 
    writer.close()
    # flush sys.stdout
    sys.stdout.flush()
    print(portion + ' :: count = ' + str(count))

# for use in ams classification 
def convert(img_dir='/home/vgdev/local/spyder_working_directory/wafer_objects/new_data_obj_imgs'): 
    path = img_dir + '/amsTFrecord/'
    name = 'ams_tmp.tfrecords'
    
    if (os.path.exists(path)):
        if (os.path.exists(path + '/' + name)): 
            os.remove(path + '/' + name)
    else: 
        os.makedirs(path)
    
    writer = tf.python_io.TFRecordWriter(path + '/' + name)
    id_list = []
    filenames = os.listdir(img_dir)
    count = 0 
    
    for file in filenames: 

        if (file[-4:] == '.bmp'):
            count +=1 
            _id = int(file[0:-4])
            img = load_image(img_dir + '/' + file)
            id_list.append(_id)
        
            feature = {'label': _int64_feature(-1) , 
                       'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'height':_int64_feature(img.shape[1]),
                       'width':_int64_feature(img.shape[0]),
                       'id':_int64_feature(_id)}
                      
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            writer.write(example.SerializeToString())
        
    print(str(count) + ' images saved in ams_tmp.tfrecords')
    return id_list
 
def shuffle_and_separate(file_list): 
    shuffle(file_list)
    
    train = file_list[0:int(PORTIONS[0]*len(file_list))]
    val = file_list[int(PORTIONS[0]*len(file_list)):int(sum(PORTIONS[0:2])*len(file_list))]
    test = file_list[int(sum(PORTIONS[0:2])*len(file_list)):]
    
    return train, val, test 

def main(): 
    print('beginning data conversion: .bmp/folders -> TFRecord ' )
    
    train_set, val_set, test_set = shuffle_and_separate(get_filenames())
    create_TFRecord(TRAIN_TFR_NAME, train_set, 'train')
    create_TFRecord(VALID_TFR_NAME, val_set, 'eval')
    create_TFRecord(TEST_TFR_NAME, test_set, 'test')
    
    print('Complete.')