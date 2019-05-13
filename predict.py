import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from matplotlib import  pyplot  as  plt
import matplotlib.image as pli

image_size=64

num_channels=3
images=[]

test_path='E:/cats dogs/cats'
i=0

path = os.path.join(test_path,'*g')
files = glob.glob(path)
for fl in files:
    
   
    image=cv2.imread(fl)
    plt.imshow(image)
    
   
    image1=cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image1)
    images=np.array(images, dtype=np.uint8)
    images=images.astype('float32')
    images=np.multiply(images, 1.0/255.0) 
    
    x_batch=images.reshape(1, image_size,image_size,num_channels)
    
    sess=tf.Session()
    saver=tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-5500.meta')
    saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-5500')
    
    graph=tf.get_default_graph()
    
    y_pred = graph.get_tensor_by_name('softmax/y_pred:0')
    
    
    x = graph.get_tensor_by_name('inputs/x:0')
    y_true = graph.get_tensor_by_name('inputs/y_true:0') 
    y_test_images=np.zeros((1, 2)) 
    
    
    feed_dict_testing={x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    
    res_label = ['dog','cat']
    final_class=res_label[result.argmax()]
    
    plt.title("This   is   a   {}".format(final_class))
    plt.show()
    plt.pause(3)
    
    print(final_class)
    if  final_class=='cat':
        i=i+1
    images=[]

            
print(i)

"""
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf


image_size=64
num_channels=3
images=[]

test_path='E:/cats dogs/dogs'
i=0

path = os.path.join(test_path,'*g')
files = glob.glob(path)
for fl in files:
    
   
    image=cv2.imread(fl)
    cv2.imshow("cats and dogs",image)
    cv2.waitKey(3500)
    cv2.destroyAllWindows()
    image1=cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image1)
    images=np.array(images, dtype=np.uint8)
    images=images.astype('float32')
    images=np.multiply(images, 1.0/255.0) 
    
    x_batch=images.reshape(1, image_size,image_size,num_channels)
    
    sess=tf.Session()
    saver=tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-5500.meta')
    saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-5500')
    
    graph=tf.get_default_graph()
    
    y_pred = graph.get_tensor_by_name('softmax/y_pred:0')
    
    
    x = graph.get_tensor_by_name('inputs/x:0')
    y_true = graph.get_tensor_by_name('inputs/y_true:0') 
    y_test_images=np.zeros((1, 2)) 
    
    
    feed_dict_testing={x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    
    res_label = ['dog','cat']
    final_class=res_label[result.argmax()]
    
    print(final_class)
    if  final_class=='dog':
        i=i+1
    images=[]

            
print(i)
"""      
        
       

"""
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

image_size=64
num_channels=3
images=[]

path='E:/cats dogs/dog.1016.jpg'
image=cv2.imread(path)
image=cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images=np.array(images, dtype=np.uint8)
images=images.astype('float32')
images=np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch=images.reshape(1, image_size,image_size,num_channels)

sess=tf.Session()
saver=tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-900.meta')
saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-900')

graph=tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name('softmax/y_pred:0')

## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name('inputs/x:0')
y_true = graph.get_tensor_by_name('inputs/y_true:0') 
y_test_images=np.zeros((1, 2)) 

### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing={x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
# dog [1 0]
res_label = ['dog','cat']
print(res_label[result.argmax()])
"""


