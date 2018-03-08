# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 07:59:44 2017

@author: evans
"""

import AMS_classifier as amp 
import tensorflow as tf
import os
import numpy as np
import smtplib
from datetime import date
import shutil
import tempfile 

SAVED_MODEL_PATH = 'AMS_CNN_TFMODELS/AMP_model_4' # default is "/tmp/AMP_model"

new_objects_dir_path="new_data_obj_imgs"

#label_map_path (str) is the path to a .txt file that specifies what each integer label corresponds to
label_map_path = 'Training_Data/label_map.txt'

CLASSIFIED_OBJ_DIR = "classified_objects"


'''
INPUT(S): 
    user (str) username to use for email notification 
    bias (list<int>) labels for the expected object types. If None, then no bias. 
OUTPUT(S): 
    None
SUMMARY: 
    Loads a trained CNN model and performs data prediction. Bias probabilities by provided bias value 
'''
def classify_new_data(user=None, bias=None): 
    data, img_names  = prepare_data()
    
    # Create the Estimator
    AMP_classifier = tf.estimator.Estimator(
    model_fn=amp.cnn_model_fn, model_dir= SAVED_MODEL_PATH)
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":data}, 
        y=None, 
        num_epochs=1, 
        shuffle=False) 
    
    class_count = []    
    predictions = AMP_classifier.predict(input_fn=predict_input_fn)
    
    for p in predictions: 
        if (bias is None): 
            class_count.append(p['classes'])
        else: 
            class_count.append(get_biased_class(p['probabilities'],bias))
        
    msg = print_class_count(class_count)
    
    try: 
        sort_objects(class_count, img_names)
    except: 
        print('failed to sort images into classified objects directory.')
    
    try: 
        email_results(msg, user)
    except: 
        print('email notification failed')

'''
INPUT(S): 
    prob (list<float>) list of probabilites of each class type 
    bias (list<int>) list of class values of expected object types
OUTPUT(S): 
    clas (int) the 
SUMMARY: 
    returns the indice of prob that is also in bias and has the greatest value. e.g. chooses the most likely
    class in the expected object types 
'''
def get_biased_class(prob, bias): 
    
    # create dict with ALLOWED indices: label->prob
    p_dic = dict()
    for i,p in enumerate(prob):
        if (i in bias): 
            p_dic[i]=p 
            
    max_i = -1
    max_p = -1
    for i in p_dic.keys(): 
        if (p_dic[i] > max_p): 
            max_p = p_dic[i]
            max_i = i 
            
    return max_i 
    
    

'''
INPUT(S): 
    class_count (list<int>)
    img_names (list<string>)
OUTPUT(S): 
    None 
SUMMARY: 
    sorts the new object images into their corresponding folder according to the CNN prediction
'''
def sort_objects(class_count, img_names, label_map = '/home/vgdev/Desktop/Training_Data/label_map.txt'):
    global new_objects_dir_path
    global CLASSIFIED_OBJ_DIR
    labels = get_label_names(label_map)
    
    # store labels in dict for faster retrieval
    lab_dict = {}
    for label, val in labels: 
        lab_dict[val] = label
    
    #clear directory 
    delete_dir(CLASSIFIED_OBJ_DIR)
    # zip labels and img names and iterate, sort into each folder, creating folder if nonexistant 

    for clas, name in zip(class_count, img_names): 
        label = lab_dict[clas]
        target_dir = CLASSIFIED_OBJ_DIR + '/' + label
        obj_path = new_objects_dir_path + '/' + name
        
        if (not os.path.exists(target_dir)): 
            os.makedirs(target_dir)
        
        if (os.path.exists(obj_path)):
            shutil.copy(obj_path, target_dir + '/' + name)
        else: 
            print('failed to move an object image file to classified directory')
        
    print('succuessfully copied object images to classified objects directory')
    
'''
INPUT(S): 
    path (str) the path to directory to be deleted 
OUTPUT(S): 
    None
SUMMARY: 
    Deletes all contents of given directory 
'''         
def delete_dir(path): 
    if (os.path.exists(path)):
        # `tempfile.mktemp` Returns an absolute pathname of a file that 
        # did not exist at the time the call is made. We pass
        # dir=os.path.dirname(dir_name) here to ensure we will move
        # to the same filesystem. Otherwise, shutil.copy2 will be used
        # internally and the problem remains.
        tmp = tempfile.mktemp(dir=os.path.dirname(path))
        # Rename the dir.
        shutil.move(path, tmp)
        # And delete it.
        shutil.rmtree(tmp)
        
'''
INPUT(S): 
    None
OUTPUT(S): 
    imgs (np.array) with shape [n,784] where n is the number of images to classify. Images are flattened to 1-d array. 
SUMMARY: 
    loads each image into memory from dir_path and appends it to imgs array 
'''
def prepare_data(): 
    global new_objects_dir_path
    img_names = os.listdir(new_objects_dir_path)
    imgs = []
    for name in img_names: 
        # we will need to add tf.image.per_image_standardization here to test the new classifier
        img = amp.load_image(new_objects_dir_path+'/'+name)
        imgs.append(img)
        
    return np.array(imgs), img_names

'''
INPUT(S): 
    class_count (list<int>) a list class predictions, list indice matches input data indice 
OUTPUT(S): 
    msg_str (str) is the results message  
SUMMARY: 
    Retrieves label associations and creates a results message string
'''
def print_class_count(class_count): 
    msg_str = ''
    total_count = 0
    labels = get_label_names()
    for label, val in labels: 
        count = 0 
        for i in class_count: 
            if (i == val): 
                count += 1 
                total_count += 1 
        lab_str = label + " - " + str (count)
        msg_str += lab_str + '\n'
    msg_str += 'total count : ' + str(total_count)
    print(msg_str)
    return(msg_str)

'''
INPUT(S): 
    None
OUTPUT(S):  
    labels (list<(str, int)>) returns a list of tuples representing (label_name, label_int)
SUMMARY: 
    retrieves label name and matches it to the proper label value 
'''      
def get_label_names(path = '/home/vgdev/Desktop/Training_Data/label_map.txt'):
    f = open(path, 'r')
    cont = f.readlines()
    labels = []
    for line in cont: 
        line = line.rstrip() 
        if (line[-2] == ' '): # 2 digit number
            val = int(line[-1])
            label = line[0:-2]
        else: # 3 digit number
            val = int(line[-2:])
            label = line[0:-3] 
        labels.append( (label, val) )
    return labels 
        
    
'''
INPUT(S): 
    msg (str) classification results message
    username (str) the email id of recipient. Must be visiongate employee with valid email address, usually username = lastname
OUTPUT(S):  
    None 
SUMMARY: 
    Sends a results email to provided valid visiongate user
'''
def email_results(msg, username): 
    if (username is not None): 
        smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
        smtpObj.ehlo()
        
        smtpObj.starttls()
        
        smtpObj.login('visiongate.internal.use@gmail.com', ' 12visiongate!@')
        
        try: 
            msg = str(date.today()) + '\n\n' + msg
        except: 
            print('email time stamp failed')
        
        #  sendmail() will not send a message body if the string includes ':'
        smtpObj.sendmail('visiongate.internal.use@gmail.com', username + '@visiongate3d.com', 'Subject: AMP RESULTS - DO NOT REPLY\n' + msg + '\n\n')
        
        smtpObj.quit()
        
        print('email notification succuessful')

#classify_new_data(bias=[3,6,7,9])
