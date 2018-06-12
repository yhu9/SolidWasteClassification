#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import math
import sys
import os

#Python Modules
import constants
import featureReader

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################

####################################################################################################################################
#Helper Functions
####################################################################################################################################

####################################################################################################################################
#######################################################################################
#FLAGS WE USE FOR UNDERSTANDING USER INPUT
#IF HSV FLAG THEN DO HSV ON THE IMAGE
#IF HSVSEGFLAG THEN DO HSV SEGMENTATION BUT KEEP BGR BLOB ANALYSIS

flags = {}
flags['train'] = 'train' in sys.argv
flags['test'] = 'test' in sys.argv
flags['pca'] = 'pca' in sys.argv
#######################################################################################

#Main Function
def main(unused_argv):

    if not os.path.exists('logs'):
        os.makedirs('logs')

    #check the number of arguments given with running the program
    #must be at least two
    if len(sys.argv) >= 2:

        #if we are training the model on extracted instances
        if flags['train']:
            featurefile = sys.argv[2]

            if os.path.isfile(featurefile):
                instances,labels = featureReader.genFromText(featurefile)
                print("feature file successfully read: %s" % featurefile)
            else:
                print("could not open feature file path: %s" % featurefile)
                sys.exit()

            #check if the pca flag is set so we know the feature count is based on pca or the entire feature length
            if flags['pca']:
                pca = featureReader.getPCA(instances,featurelength=constants.PCA_LENGTH)
                featurelength = pca.n_components_
                print('pca generated')
            else:
                featurelength = len(instances[0])

        #if we are testing the model on new instances
        elif flags['test']:
            #read the image from the image file
            if os.path.isfile(sys.argv[2]):
                image = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
            else:
                print("image could not be read!")
                sys.exit()

            #model name
            modelpath = sys.argv[3]
            tokens = modelpath.split('_')
            hogflag = 'hog' in tokens[0]
            gaborflag = 'gabor' in tokens[0]
            colorflag = 'color' in tokens[0]
            sizeflag = 'size' in tokens[0]
            hsvflag = 'hsv' in tokens[0]
            hsvsegflag = 'hsvseg' in tokens[1]

            #extract the testing instances from the image
            #look at the model naming convention to see what features we are extracting
            fout = "ms_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + ".png"
            blobinstances,markers,markerlabels = featureReader.createTestingInstancesFromImage(image,hsvseg=hsvsegflag,hog=hogflag,gabor=gaborflag,color=colorflag,size=sizeflag,hsv=hsvflag,filename=fout)
            print("features extracted")

            #apply pca on the instances
            #get the pca from the training instances
            if flags['pca']:
                featurefile = sys.argv[sys.argv.index('pca') + 1]
                if os.path.isfile(featurefile):
                    instances,labels = featureReader.genFromText(featurefile)
                    print("feature file successfully read: %s" % featurefile)
                else:
                    print("could not open feature file path: %s" % featurefile)
                    sys.exit()

                #create the pca from the instances read in during training
                instances,labels = featureReader.genFromText(featurefile)
                pca = featureReader.getPCA(instances,featurelength=constants.PCA_LENGTH)

                #apply pca on the new blobs found
                new_instances = featureReader.applyPCA(pca,blobinstances)
                featurelength = pca.n_components_
                print('pca generated')
            else:
                new_instances = blobinstances
                featurelength = len(blobinstances[0])

        #################################################################################################################
        #################################################################################################################
        #Define our Neural Network from scratch
        #################################################################################################################
        #################################################################################################################
        x = tf.placeholder('float',[None,featurelength])
        y = tf.placeholder('float',[None,constants.NN_CLASSES])
        weights = {}
        biases = {}

        #create our first fully connected layer
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc1'] = tf.Variable(tf.random_normal([featurelength,constants.NN_FULL1]))
                biases['b_fc1'] = tf.Variable(tf.random_normal([constants.NN_FULL1]))
                layer_1 = tf.add(tf.matmul(x,weights['W_fc1']),biases['b_fc1'])
                fc1= tf.nn.relu(layer_1)
                #fullyConnected = tf.nn.dropout(fullyConnected,constants.KEEP_RATE)
            tf.summary.histogram('activation1',fc1)

        #create our first fully connected layer
        with tf.name_scope('Fully_Connected_2'):
            with tf.name_scope('activation'):
                weights['W_fc2'] = tf.Variable(tf.random_normal([constants.NN_FULL1,constants.NN_FULL2]))
                biases['b_fc2'] = tf.Variable(tf.random_normal([constants.NN_FULL2]))
                layer_2 = tf.add(tf.matmul(fc1,weights['W_fc2']),biases['b_fc2'])
                fc2= tf.nn.relu(layer_2)
                #fullyConnected = tf.nn.dropout(fullyConnected,constants.KEEP_RATE)
            tf.summary.histogram('activation2',fc2)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([constants.NN_FULL2,constants.NN_CLASSES]))
            biases['out'] = tf.Variable(tf.random_normal([constants.NN_CLASSES]))
            predictions = tf.matmul(fc2,weights['out'])+biases['out']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
            correct_prediction = tf.cast(correct_prediction,tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        #prediction operation
        tf.summary.scalar('accuracy',accuracy)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #################################################################################################################
        #################################################################################################################
        #DECIDE WHAT WE DO WITH THE MODEL INTIALIZED
        #################################################################################################################
        #################################################################################################################
        if flags['train']:
            #Run the session/NN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                sess.run(init)
                merged = tf.summary.merge_all()


                #apply pca on the instances extracted from the feature file
                if flags['pca']:
                    modelpath = os.path.splitext(os.path.basename(featurefile))[0][9:] + "_pcamodel"
                    logdir = "logs/log_"+modelpath+".txt"
                    new_instances = featureReader.applyPCA(pca,instances,log_dir=logdir)
                    print('features reduced from %i to %i' % (len(instances[0]),pca.n_components_))
                else:
                    new_instances = instances
                    modelpath = os.path.splitext(os.path.basename(featurefile))[0][9:] + "_model"
                    logdir = "logs/log_"+modelpath+".txt"

                #create and figure out where we save the information to
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                savedir = os.path.join(modelpath,"nn_model.ckpt")

                #separate the instances into its proper categories

                #training of the model
                acc = 0.00;
                for epoch in range(constants.NN_EPOCHS):

                    #get an image batch for each category and train on it
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,new_instances,labels)
                    optimizer.run(feed_dict={x: batch_x, y: batch_y})

                    #evaluate the model using a test set
                    if epoch % 1 == 0:
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,new_instances,labels)
                        accnew = accuracy.eval({x: eval_x, y: eval_y})

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,savedir)
                            print("highest accuracy found! model saved")

                        print('epoch: ' + str(epoch) + '     ' +
                                'accuracy: ' + str(accnew))
                        with open(logdir,'a') as log_out:
                            log_out.write('epoch: ' + str(epoch) + '     ' + 'accuracy: ' + str(accnew) + '\n')

                #PRINT OUT TO CONSOLE AND LOG THE HIGHEST ACCURACY ACHIEVED
                OUTPUT = "HIGEST ACCURACY ACHIEVED: " + str(acc)
                with open(logdir,'a') as log_out:
                    log_out.write(OUTPUT)
                    log_out.write('\n')
                print(OUTPUT)
                print("Model saved in file: %s" % save_path)

        #testing method requires an image to be tested
        #testing method needs a saved check point directory (model)
        #and any other flags
        elif flags['test']:

            '''
            print("--------------------------------------------------------\n\n")
            print("for running testing mode: ")
            print("\n\n******* PAREMETERS ARE OPTIONAL *******")
            print("featurelength must be prepended with -n")
            print("--------------------------------------------------------\n\n")
            print("test IMAGEPATH MODELPATH *FEATUREPATH")
            print("\n\n")
            print("--------------------------------------------------------\n\n")
            '''

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                #get the directory of the checkpoint
                ckpt_dir = sys.argv[3]
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #run the instances extracted on the learned model and get raw and max prediction
                rawpredictions = predictions.eval({x:new_instances})
                predictions = rawpredictions.argmax(axis=1)
                print("predictions made")
                print(predictions)
                rawname = "rawoutput_" + str(os.path.basename(os.path.abspath(os.path.join(sys.argv[3],'../')))[0]) + ".txt"
                rawfile = os.path.join('logs',rawname)
                with open(rawfile,'w') as fout:
                    for raw,cat,mark in zip(rawpredictions,predictions,markerlabels):
                        fout.write(str("cat: " + str(cat) + '    mark: ' + str(mark) + '    raw: '))
                        for val in raw:
                            fout.write(str(val) + ',')
                        fout.write('\n')

                #create mask from predictions
                print("writing results")
                h,w = image.shape[:2]
                best_guess = np.full((h,w),-1)
                for l,p in zip(markerlabels,predictions):
                    best_guess[markers == l] = p

                #write the results
                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                fileout = os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"
                featureReader.outputResults(image,np.array(best_guess),fout=fileout)

                print("segmentation results successfully saved to %s" % fileout)

        elif sys.argv[1] == 'debug':

            print("nothing to debug")

        else:
            print("test image_filepath features_filepath model_filepath")
            print("train features_filepath")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train,see)")
        print("argv[2]: image direcory to train/test on")
        print("argv[3]: *OPTIONAL* model directory if testing")
        print("need (train,img_dir) or (test,img_dir,model)")

if __name__ == "__main__":
    tf.app.run()
