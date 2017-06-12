""" Face Verification by CAICAI.
    Ref: https://github.com/davidsandberg/facenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import facenet
import os
import sys
import time
import math
import pickle
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            dataset = facenet.get_dataset(args.data_dir)       
                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            start_time = time.time()
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            duration = time.time() - start_time
            print('Calculating features Time %.3f' % duration)  

            '''
            # save emb_array and labels
            scipy.io.savemat(args.save_path,{'emb_array':emb_array,'labels':labels})
            print('Embeddings data saved')
            '''
            
            mode = 'FNN'

            if (mode=='SVM'):

                classifier_filename_exp = os.path.expanduser(args.classifier_filename)

                start_time = time.time()
                    
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                duration = time.time() - start_time
                print('classifier Time %.3f' % duration)      

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)

            elif (mode=='FNN'):

                start_time = time.time()
                #load meta graph and restore weights
                saver = tf.train.import_meta_graph('./models/sparse_models/20170612-204222/my-model-fc-sc0-3400.meta')
                saver.restore(sess,tf.train.latest_checkpoint('./models/sparse_models/20170612-204222/'))
 
                # Get input and output tensors
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("input/x-input:0")
                keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
                y_hat = graph.get_tensor_by_name("output/activation:0")
                feed_dict = { x:emb_array,keep_prob:1.0 }
                predictions = tf.argmax(y_hat, 1)
                best_class_indices = sess.run(predictions, feed_dict=feed_dict)

                duration = time.time() - start_time
                print('classifier Time %.3f' % duration)  

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing test data.',
        default='./datasets/my_dataset/test/')
    parser.add_argument('--save_path', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./datasets/save_mat/emb_mat_10k_test.mat')
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='./models/20170512-110547.pb')
    parser.add_argument('--classifier_filename',  type=str,
        help='Classifier model file name as a pickle (.pkl) file. ' ,
        default='./models/my_classifier.pkl')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=1000)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
