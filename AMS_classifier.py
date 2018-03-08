"""Convolutional Neural Network Estimator, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
import os 
import sys
import time

    # TRAIN, VALIDATION, TEST  
PORTIONS = [0.65,0.30,0.05]

# IMAGE DIM (WIDTH, HEIGHT)
RESIZE_DIM = (28,28)

TRAINING_IMAGES_FOLDER_PATH = '/home/vgdev/Desktop/Training_Data' 

TRAIN_TFR_NAME = TRAINING_IMAGES_FOLDER_PATH + '/AMP_train.tfrecords' 
VALID_TFR_NAME = TRAINING_IMAGES_FOLDER_PATH + '/AMP_validation.tfrecords'
TEST_TFR_NAME =  TRAINING_IMAGES_FOLDER_PATH + '/AMP_test.tfrecords'


tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_data(training_dir): 
    if (input('artificially increase data set size?(y/N) ') == 'y'):
        transforms = True 
    else: 
        transforms = False
        
    tic = time.time()
    data = np.array([])
    labels = np.array([])

    files = get_filenames(training_dir)
    count = 0
    leng = len(files)
    print('# of files to load: ' + str(leng))
    for i, (path, label) in enumerate(files): 
        try:
            if(i%100==0): 
                sys.stdout.write("Loading Images: %d%%    \r" %((i/leng)*100))
                sys.stdout.flush()
            
            img = tf.image.per_image_standardization(load_image(training_dir + '/' + path))
            
            # these transformations take quite a while, consider saving them as seperate images? 
            if (transforms): 
                img_LR = tf.image.flip_left_right(img)
                img_UD = tf.image.flip_up_down(img)
                #img_cont_h = tf.image.adjust_contrast(img, 1.25)
                #img_cont_l = tf.image.adjust_contrast(img, 0.75)
                #img_bright_h = tf.image.adjust_brightness(img, 15)
                #img_bright_l = tf.image.adjust_brightness(img, -15)
                #imgs = [img_LR,img_UD,img_cont_h,img_cont_l,img_bright_h,img_bright_l, img]
                
                imgs = np.array([img_LR, img_UD, img])
                #imgs = list(map(lambda x: tf.Session().run(tf.reshape(x,[784,1])), imgs))
                
                if (i ==0): 
                    data=imgs
                    labels = np.array([label]*len(imgs))
                else: 
                    np.concatenate(data, imgs)
                    np.concatenate(labels,np.array([label]*len(imgs)))
            else: 
                np.concatenate(labels, label)
                np.concatenate(data, img)
        except: 
            count+=1
            raise
    
    #print(data[0].shape)
    #data = np.array(data)
    #labels = np.array(labels)
    print()
    print(data.shape)
    print('failed to load ' + str(count) + 'object images')
    print('time to load images : ' + str(time.time() - tic))
    return (data, labels)

def shuffle_and_separate(data,labels): 
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)    
    
    train = data[0:int(PORTIONS[0]*len(data))]
    train_labels = labels[0:int(PORTIONS[0]*len(data))]
    val = data[int(PORTIONS[0]*len(data)):int(sum(PORTIONS[0:1])*len(data))]
    val_labels = labels[int(PORTIONS[0]*len(data)):int(sum(PORTIONS[0:1])*len(data))]
    test = data[int(sum(PORTIONS[0:1])*len(data)):]
    test_labels = labels[int(sum(PORTIONS[0:1])*len(data)):]
    
    return train, val, test , train_labels, val_labels, test_labels 
    
# Takes a pathname and returns properly resized image 
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.flatten()
    #img = np.expand_dims(img.astype(np.float32),2)
    return img
    
def get_filenames(path): 
    if (os.path.isfile(path + '/label_map.txt')): 
        os.remove(path + '/label_map.txt')
    
    f = open(path + '/label_map.txt', 'w')
    
    images = []
    label_names = os.listdir(path)
    
    label = 0 
    for lab_name in label_names: 
        if (os.path.isdir(path+'/'+lab_name)):
            image_names = os.listdir(path+'/'+lab_name)
            f.write(lab_name + ' ' + str(label) + '\n')
            for img_name in image_names: 
                images.append( (lab_name + '/' + img_name, label) ) 
            label += 1
        
    f.close()
    return images

def main(unused_argv):
    global TRAINING_IMAGES_FOLDER_PATH
    training = False
    in1 = input('Train (Y/n) if n, testing. ')
    if (in1 is 'Y'): 
        training = True
    
    in2 = input('name and path to save model? (leave blank for /tmp/AMP_model) : ' )
    model_name = '/tmp/AMP_model'
    if (in2.strip(' ') != ''): 
        model_name = in2
    
    in3 = input('training data directory? (leave blank for ~/Desktop/Training\ Data) : ')
    if (in3.strip(' ') != ''): 
        train_path = in3
    else :
        train_path = TRAINING_IMAGES_FOLDER_PATH
    
    # Load training and eval data
    data, labels = get_data(train_path)
    train, val, test , train_labels, val_labels, test_labels = shuffle_and_separate( data, labels)
    
    train_data = train  # Returns np.array
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data = val  # Returns np.array
    eval_labels = np.asarray(val_labels, dtype=np.int32)
    test_data = test 
    test_labels = np.asarray(test_labels, dtype=np.int32)
    
      # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=model_name)
    
      # Set up logging for predictions
      # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

    if (training):
    
          # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
        
        mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
        
          # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
    
    else: 
        if (len(test_data) > 0): 
            # test data set predicitions 
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":test_data}, 
            y=test_labels, 
            num_epochs=1, 
            shuffle=False)     
            
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":test_data}, 
            y=None, 
            num_epochs=1, 
            shuffle=False) 
            
            test_results = mnist_classifier.evaluate(input_fn=test_input_fn)
            print(test_results)   
            
            print('creating predictor\n\n') 
            
            predictions = mnist_classifier.predict(input_fn=predict_input_fn)
            right = 0
            wrong = 0
            for l,p in zip(test_labels, predictions): 
                #print(p['classes'])
                #print(str(l) + ' ' + str(p))
                if (l == p['classes']): 
                    right += 1
                else: 
                    wrong += 1 
                    
            print('%right: ' + str(right/(right + wrong)))
            print('total in test set: ' + str(right + wrong))
            
            #print('\nlabels\n')
            #print(test_labels)
        else: 
            print('ERROR: NO OBJECTS RECOGNIZED FROM SEGMENTATION')
        
        
        


if __name__ == "__main__":
    tf.app.run()
