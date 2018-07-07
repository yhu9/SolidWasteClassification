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
import pickle

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
flags['test2'] = 'test2' in sys.argv
flags['pca'] = 'pca' in sys.argv
flags['lda'] = 'lda' in sys.argv
#######################################################################################

#Main Function
def main(unused_argv):

    if not os.path.exists('log'):
        os.makedirs('log')

    #check the number of arguments given with running the program
    #must be at least two
    if len(sys.argv) >= 3 and (sys.argv[1] == 'test' or sys.argv[1] == 'train'):

        #if we are training the model on extracted instances
        if flags['train']:
            featurefile = sys.argv[2]

            if os.path.isfile(featurefile):
                instances,labels = featureReader.genFromText(featurefile)
                print("feature file successfully read: %s" % featurefile)

                #if appplying pca
                if flags['pca']:
                    pca = featureReader.getPCA(instances)
                    new_instances = pca.transform(instances)
                    pickle.dump(pca,open(os.path.splitext(os.path.basename(featurefile))[0] + '_pca.pkl','wb'))

                #if appplying lda
                elif flags['lda']:
                    lda = featureReader.getLDA(instances,labels)
                    new_instances = lda.transform(instances)
                    pickle.dump(lda,open(os.path.splitext(os.path.basename(featurefile))[0] + '_lda.pkl','wb'))

                #if doing normal training
                else:
                    new_instances = instances
            else:
                print("could not open feature file path: %s" % featurefile)
                sys.exit()

            #check if the pca flag is set so we know the feature count is based on pca or the entire feature length
            featurelength = len(new_instances[0])

        #if we are testing the model on new instances
        elif flags['test'] or flags['test2']:

            #model name
            modelpath = sys.argv[3]
            tokens = modelpath.split('_')
            colorflag = 'color' in tokens[1]
            gaborflag = 'gabor' in tokens[1]
            hogflag = 'hog' in tokens[1]
            hsvflag = 'hsv' in tokens[1]
            sizeflag = 'size' in tokens[1]
            hsvsegflag = 'hsvseg' in tokens[2]

            print('color: %s' % colorflag)
            print('gabor: %s' % gaborflag)
            print('hog: %s' % hogflag)
            print('hsv: %s' % hsvflag)
            print('size: %s' % sizeflag)
            print('hsvseg: %s' % hsvsegflag)

            #extract the testing instances from the image
            #look at the model naming convention to see what features we are extracting
            #read the image from the image file
            if flags['test']:
                if os.path.isfile(sys.argv[2]):
                    tmp = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                    image = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
                    ms_out = "ms_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + ".png"
                    blobinstances,markers,markerlabels = featureReader.createTestingInstancesFromImage(image,hsvseg=hsvsegflag,hog=hogflag,gabor=gaborflag,color=colorflag,size=sizeflag,hsv=hsvflag,filename=ms_out)

                else:
                    print("image could not be read!")
                    sys.exit()

            #apply pca on the instances
            #get the pca from the training instances
            if flags['pca']:
                pcaID = sys.argv.index('pca') + 1
                pca = featureReader.loadPCA(sys.argv[pcaID])
                new_instances = pca.transform(blobinstances)
                featurelength = pca.n_components_
                print('pca generated. %i components' % featurelength)
            elif flags['lda']:
                ldaID = sys.argv.index('lda') + 1
                lda = featureReader.loadLDA(sys.argv[ldaID])
                new_instances = lda.transform(blobinstances)
                featurelength = len(lda.classes_) - 1
                print('lda generated')
            else:
                new_instances = blobinstances
                featurelength = len(blobinstances[0])

        #################################################################################################################
        #################################################################################################################
        #Define our Neural Network from scratch
        #################################################################################################################
        #################################################################################################################

        weights = {}
        biases = {}
        with tf.name_scope('inputlayer'):
            x = tf.placeholder('float',[None,featurelength])
            y = tf.placeholder('float',[None,constants.NN_CLASSES])

        #create our first fully connected layer
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc1'] = tf.Variable(tf.random_normal([featurelength,constants.NN_FULL1]))
                biases['b_fc1'] = tf.Variable(tf.random_normal([constants.NN_FULL1]))
                layer_1 = tf.add(tf.matmul(x,weights['W_fc1']),biases['b_fc1'])
                fc1 = tf.nn.relu(layer_1)

        #create our first fully connected layer
        with tf.name_scope('Fully_Connected_2'):
            with tf.name_scope('activation'):
                weights['W_fc2'] = tf.Variable(tf.random_normal([constants.NN_FULL1,constants.NN_FULL2]))
                biases['b_fc2'] = tf.Variable(tf.random_normal([constants.NN_FULL2]))
                layer_2 = tf.add(tf.matmul(fc1,weights['W_fc2']),biases['b_fc2'])
                fc2 = tf.nn.relu(layer_2)

        #third fully connected layer
        with tf.name_scope('Fully_Connected_3'):
            with tf.name_scope('activation3'):
                weights['w_fc3'] = tf.Variable(tf.random_normal([constants.NN_FULL2,constants.NN_FULL3]))
                biases['b_fc3'] = tf.Variable(tf.random_normal([constants.NN_FULL3]))
                layer_3 = tf.add(tf.matmul(fc2,weights['w_fc3']),biases['b_fc3'])
                fc3= tf.nn.relu(layer_3)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([constants.NN_FULL3,constants.NN_CLASSES]))
            biases['out'] = tf.Variable(tf.random_normal([constants.NN_CLASSES]))
            predictions = tf.matmul(fc3,weights['out'])+biases['out']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions,labels=y)
            cost_sum = tf.reduce_mean(cost)
            tf.summary.scalar('cost',cost_sum)
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
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU':0})) as sess:
                sess.run(init)
                training_writer = tf.summary.FileWriter('./log/training',sess.graph)
                testing_writer = tf.summary.FileWriter('./log/testing',sess.graph)

                #apply pca on the instances extracted from the feature file
                modelpath = os.path.splitext(os.path.basename(featurefile))[0] + "_model"
                if flags['lda']:
                    modelpath += '_lda'
                elif flags['pca']:
                    modelpath += '_pca'

                logdir = "log/log_"+modelpath+".txt"

                #create and figure out where we save the information to
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                savedir = os.path.join(modelpath,"nn_model.ckpt")

                #training of the model
                acc = 0.00;
                merged = tf.summary.merge_all()
                for epoch in range(constants.NN_EPOCHS):

                    #get an image batch for each category and train on it. write out the summary.
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,new_instances,labels)
                    summary = sess.run([merged,optimizer],feed_dict={x: batch_x, y: batch_y})
                    training_writer.add_summary(summary[0],epoch)
                    training_writer.flush()

                    #evaluate the model using a test set
                    if epoch % 1 == 0:
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,new_instances,labels)
                        summary,accnew = sess.run([merged,accuracy],feed_dict={x:eval_x, y:eval_y})
                        testing_writer.add_summary(summary,epoch)
                        testing_writer.flush()

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
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU':0})) as sess:
                #get the directory of the checkpoint
                ckpt_dir = sys.argv[3]
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #run the instances extracted on the learned model and get raw and max prediction
                rawpredictions = predictions.eval({x:new_instances})
                predictions = rawpredictions.argmax(axis=1)
                rawname = "rawoutput_" + str(os.path.basename(os.path.abspath(os.path.join(sys.argv[3],'../')))[0]) + ".txt"
                rawfile = os.path.join('log',rawname)

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
                if not os.path.isdir('results'):
                    os.makedirs('results')
                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                fileout = os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"
                fileout = os.path.join('results',fileout)
                featureReader.outputResults(image,np.array(best_guess),fout=fileout)

                print("segmentation results successfully saved to %s" % fileout)

        elif flags['test2']:
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU':0})) as sess:
                #get the directory of the checkpoint
                ckpt_dir = sys.argv[3]
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                for i in range(6):
                    eval_x,eval_y = featureReader.getBatch(600,new_instances,labels)
                    accnew = accuracy.eval({x: eval_x, y: eval_y})
                    print("TRIAL %i         ACCURACY: %.4f" % (i,accnew))


        else:
            print("test image_filepath features_filepath model_filepath")
            print("train features_filepath")

    elif sys.argv[1] == 'debug':
        if os.path.isfile(sys.argv[2]):
            tmp = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
            image = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
            ms_out = "ms_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + ".png"

            blobinstances,markers,markerlabels = featureReader.createTestingInstancesFromImage(image,hsvseg=False,hog=True,gabor=True,color=True,size=True,hsv=True,filename=ms_out)
            np.save('blobs',blobinstances)
            quit()
        else:
            print("image could not be read!")
            sys.exit()

        print('debugging')
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train,see)")
        print("argv[2]: image direcory to train/test on")
        print("argv[3]: *OPTIONAL* model directory if testing")
        print("need (train,img_dir) or (test,img_dir,model)")

if __name__ == "__main__":
    tf.app.run()
