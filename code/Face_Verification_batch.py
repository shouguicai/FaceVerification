""" Face Verification by CAICAI.
    Ref: https://github.com/davidsandberg/facenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.validate_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.validate_dir), pairs, args.validate_file_ext)
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            image_size = args.image_size
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on Validate images')
            batch_size = args.validate_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            #print('%d %d %d'%(nrof_images,batch_size,nrof_batches))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[start_index:end_index,:] = emb

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            embeddings1 = emb_array[0::2]
            embeddings2 = emb_array[1::2]

            #print(embeddings1.shape)

            predictions1 = model.predict_proba(embeddings1)
            best_class_indices1 = np.argmax(predictions1, axis=1)
            best_class_probabilities1 = predictions1[np.arange(len(best_class_indices1)), best_class_indices1]

            predictions2 = model.predict_proba(embeddings2)
            best_class_indices2 = np.argmax(predictions2, axis=1)
            best_class_probabilities2 = predictions2[np.arange(len(best_class_indices2)), best_class_indices2]

            predict_issame = np.equal(best_class_indices1, best_class_indices2)
            true_accept = np.sum(np.logical_and(predict_issame, actual_issame))       
            true_reject = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
            # tp: predict same , actual same ; true answer
            # tn: predict diff , actual diff ; true answer

            num_pairs = min(len(actual_issame), embeddings1.shape[0])
            n_same = np.sum(actual_issame)
            n_diff = np.sum(np.logical_not(actual_issame))
            accuracy = float(true_accept+true_reject)/num_pairs
            accuracy_accepct = float(true_accept) / float(n_same)
            accuracy_reject = float(true_reject) / float(n_diff)
            print('Accuracy(total,accept,reject): %.3f %.3f %.3f' % (accuracy,accuracy_accepct,accuracy_reject))
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', 
        default='./models/20170512-110547.pb')
    parser.add_argument('--classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' , type=str, 
        default='./models/my_classifier.pkl')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)

    # Parameters for validation 
    parser.add_argument('--validate_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='./datasets/my_dataset/labels.txt')
    parser.add_argument('--validate_file_ext', type=str,
        help='The file extension for the Validate dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--validate_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='./datasets/my_dataset/test/')
    parser.add_argument('--validate_batch_size', type=int,
        help='Number of images to process in a batch in the Validate set.', default=100)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
