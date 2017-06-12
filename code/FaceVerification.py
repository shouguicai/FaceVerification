""" Face Verification by CAICAI.
    Ref: https://github.com/davidsandberg/facenet
"""
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import ndimage

def FaceVerification(X1,X2):
    """
      Args:
      x1: picture 1.
      x2: picture 2.
      size of x1,x2 need to be bigger than [160*160*3]
    Returns:
      whether x1,x2 belong to the same person
      1 - is same 
      0 - not same
    """
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            num_images = 2
            image_size = 160
            images = np.zeros((num_images, image_size, image_size, 3))
            #resize
            img1 = ndimage.interpolation.zoom(X1, (image_size/X1.shape[0], image_size/X1.shape[1], 1.0))
            img2 = ndimage.interpolation.zoom(X2, (image_size/X2.shape[0], image_size/X2.shape[1], 1.0))
            # prewhiten
            images[0,:,:,:] = facenet.prewhiten(img1)
            images[1,:,:,:] = facenet.prewhiten(img2)

            # add path
            model='./models/20170512-110547.pb'
            svm_classifier_filename ='./models/my_classifier.pkl'

            # Load the model
            facenet.load_model(model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            emb_array = np.zeros((num_images, embedding_size))

            #print(embeddings.shape)
            #print(emb_array.shape)
            #print(images.shape)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            
            mode = 'FNN'

            if (mode=='SVM'):
                classifier_filename_exp = os.path.expanduser(svm_classifier_filename)
                #print(embeddings.shape)
                #print(emb_array.shape)

                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                embeddings1 = emb_array[0::2]  # x[startAt:endBefore:skip]
                embeddings2 = emb_array[1::2]

                prediction1 = model.predict_proba(embeddings1)
                best_class_indices1 = np.argmax(prediction1, axis=1)
                best_class_probabilities1 = prediction1[np.arange(len(best_class_indices1)), best_class_indices1]

                prediction2 = model.predict_proba(embeddings2)
                best_class_indices2 = np.argmax(prediction2, axis=1)
                best_class_probabilities2 = prediction2[np.arange(len(best_class_indices2)), best_class_indices2]

                Y = np.equal(best_class_indices1, best_class_indices2)
            elif (mode=='FNN'):
                #load meta graph and restore weights
                saver = tf.train.import_meta_graph('./models/sparse_models/20170612-204222/my-model-fc-sc0-3400.meta')
                saver.restore(sess,tf.train.latest_checkpoint('./models/sparse_models/20170612-204222/'))
 
                # Get input and output tensors
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("input/x-input:0")
                keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
                y_hat = graph.get_tensor_by_name("output/activation:0")
                feed_dict = { x:emb_array,keep_prob:1.0 }
                predictions = sess.run(y_hat, feed_dict=feed_dict)
                best_class_indices1 = np.where(predictions[0]==np.max(predictions[0]))[0][0]
                best_class_indices2 = np.where(predictions[1]==np.max(predictions[1]))[0][0]
                Y = np.equal(best_class_indices1,best_class_indices2)

    return int(Y)