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

from multiprocessing import Pool
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


#######################################################################################
#######################################################################################
#Main Function
def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        y = tf.placeholder('float',[None,constants.CNN_CLASSES])
        weights = {}
        biases = {}

        #local convolution pathway
        weights['W_local1'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_local1'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv1 = tf.nn.conv2d(x,weights['W_local1'],strides=[1,1,1,1],padding='SAME',name='local1')
        local1 = tf.nn.relu(conv1 + biases['b_local1'])

        weights['W_local2'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL2]))
        biases['b_local2'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL2]))
        conv2 = tf.nn.conv2d(local1,weights['W_local2'],strides=[1,1,1,1],padding='SAME',name='local2')
        local2 = tf.nn.relu(conv2 + biases['b_local2'])

        #global convolution pathway
        weights['W_global1'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL1]))
        biases['b_global1'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL1]))
        conv1 = tf.nn.conv2d(x,weights['W_global1'],strides=[1,1,1,1],padding='SAME',name='global1')
        global1 = tf.nn.relu(conv1 + biases['b_global1'])

        #merge outputs of both convolution paths
        all_activations = tf.concat([local2,global1],3)

        #create our first fully connected layer
        #magic number = width * height * n_convout
        magic_number1 = int(constants.IMG_SIZE * constants.IMG_SIZE * constants.CNN_LOCAL2)
        magic_number2 = int(constants.IMG_SIZE * constants.IMG_SIZE * constants.CNN_GLOBAL1)
        magic_number = magic_number1 + magic_number2
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc'] = tf.Variable(tf.random_normal([magic_number,constants.CNN_FULL1]))
                biases['b_fc'] = tf.Variable(tf.random_normal([constants.CNN_FULL1]))
                layer1_input = tf.reshape(all_activations,[-1,magic_number])
                fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
                #fullyConnected = tf.nn.dropout(fullyConnected,constants.KEEP_RATE)
            tf.summary.histogram('activations_3',fullyConnected)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([constants.CNN_FULL1,constants.CNN_CLASSES]))
            biases['out'] = tf.Variable(tf.random_normal([constants.CNN_CLASSES]))
            predictions = tf.matmul(fullyConnected,weights['out'])+biases['out']

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
        predict_op = tf.argmax(predictions,1)
        tf.summary.scalar('accuracy',accuracy)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#helper functions

        def outputResults(image,mask,fout='segmentation.png'):
            #create the segmented image
            canvas = image.copy()
            canvas[mask == -1] = [0,0,0]
            canvas[mask == 0] = [0,0,255]
            canvas[mask == 1] = [0,255,0]
            canvas[mask == 2] = [255,0,0]
            canvas[mask == 3] = [0,255,255]
            canvas[mask == 4] = [255,0,255]
            canvas[mask == 5] = [255,255,0]

            #show the original image and the segmented image and then save the results
            cv2.imwrite(fout,canvas)

            #count the percentage of each category
            cat1_count = np.count_nonzero(mask == 0)
            cat2_count = np.count_nonzero(mask == 1)
            cat3_count = np.count_nonzero(mask == 2)
            cat4_count = np.count_nonzero(mask == 3)
            cat5_count = np.count_nonzero(mask == 4)
            cat6_count = np.count_nonzero(mask == 5)
            total = cat1_count + cat2_count + cat3_count + cat4_count + cat5_count + cat6_count

            #get the percentage of each category
            p1 = cat1_count / total
            p2 = cat2_count / total
            p3 = cat3_count / total
            p4 = cat4_count / total
            p5 = cat5_count / total
            p6 = cat6_count / total

            #output to text file
            with open('results.txt','a') as f:
                f.write("\nusing model: %s\n" % sys.argv[3])
                f.write("evaluate image: %s\n\n" % sys.argv[2])
                f.write("--------------------------------------------------------------------------------------\n")
                f.write("%s : %f.2\n" % (constants.CAT1,p1))
                f.write("%s : %f.2\n" % (constants.CAT2,p2))
                f.write("%s : %f.2\n" % (constants.CAT3,p3))
                f.write("%s : %f.2\n" % (constants.CAT4,p4))
                f.write("%s : %f.2\n" % (constants.CAT5,p5))
                f.write("%s : %f.2\n" % (constants.CAT6,p6))
                f.write("--------------------------------------------------------------------------------------\n")
                f.write("------------------------------------END-----------------------------------------------\n")
                f.write("--------------------------------------------------------------------------------------\n")

                greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

                #f.write out to the terminal what the most common category was for the image
                if(greatest == cat1_count):
                    f.write("\nthe most common category is: " + constants.CAT1)
                elif(greatest == cat2_count):
                    f.write("\nthe most common category is: " + constants.CAT2)
                elif(greatest == cat3_count):
                    f.write("\nthe most common category is: " + constants.CAT3)
                elif(greatest == cat4_count):
                    f.write("\nthe most common category is: " + constants.CAT4)
                elif(greatest == cat5_count):
                    f.write("\nthe most common category is: " + constants.CAT5)
                elif(greatest == cat6_count):
                    f.write("\nthe most common category is: " + constants.CAT6)
                else:
                    f.write("\nsorry something went wrong counting the predictions")


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#operates on the inference created above given the mode
        #training mode trained on the image
        if(sys.argv[1] == 'trainpix'):
            #read the image

            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                sess.run(init)
                merged = tf.summary.merge_all()

                #saver
                if not os.path.exists('pixelmodel'):
                    os.makedirs('pixelmodel')
                acc = 0.00

                #run the training
                for epoch in range(constants.CNN_EPOCHS):

                    #get an image batch
                    batch_x,batch_y = featureReader.getPixelBatch(constants.BATCH_SIZE)
                    eval_x,eval_y = featureReader.getPixelBatch(constants.BATCH_SIZE)

                    optimizer.run(feed_dict={x: batch_x, y: batch_y})

                    if epoch % 1 == 0:
                        acc = accuracy.eval({x: eval_x, y: eval_y})

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,'./pixelmodel/two-path_model.ckpt')
                            print("highest accuracy found! model saved")

                        print('epoch: ' + str(epoch) + '     ' +
                                'accuracy: ' + str(acc))
                        with open("two-path_log.txt",'a') as log_out:
                            log_out.write('epoch: ' + str(epoch) + '     ' + 'accuracy: ' + str(acc) + '\n')

                #saver
                print("Model saved in file: %s" % save_path)

        elif(sys.argv[1] == 'trainseg' and len(sys.argv) > 2):
            #check if segmentation file path exists
            filepath = sys.argv[2]
            if not os.path.isdir(filepath):
                print("ERROR!!! %s IS NOT A DIRECTORY" % filepath)
                sys.exit()

            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                sess.run(init)
                merged = tf.summary.merge_all()

                #saver
                if not os.path.exists('segmentmodel'):
                    os.makedirs('segmentmodel')
                acc = 0.00

                #run the training
                for epoch in range(constants.CNN_EPOCHS):

                    #get a training batch and an evaluation batch all batches are randomly picked from the same file directory
                    batch_x,batch_y = featureReader.getSegmentBatch(constants.BATCH_SIZE,filepath)
                    eval_x,eval_y = featureReader.getSegmentBatch(constants.BATCH_SIZE,filepath)

                    #run the training
                    optimizer.run(feed_dict={x: batch_x, y: batch_y})

                    #run the evaluation and print the results to the screen
                    if epoch % 1 == 0:
                        accnew = accuracy.eval({x: eval_x, y: eval_y})

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,'./segmentmodel/two-path_model.ckpt')
                            print("highest accuracy found! model saved")

                        print('epoch: ' + str(epoch) + '     ' +
                                'accuracy: ' + str(accnew))
                        with open("two-path_log.txt",'a') as log_out:
                            log_out.write('epoch: ' + str(epoch) + '     ' + 'accuracy: ' + str(accnew) + '\n')

                print("Model saved in file: %s" % save_path)

        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            if os.path.isfile(sys.argv[2]):
                image = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #we recreate the image by painting the best_guess mask on a blank canvas with the same shape as image
                #IM REALLY SORRY ABOUT THIS CODE

                #initialize counters and the height and width of the image being tested.
                #constants.IMG_SIZE is the img size the learned model uses for classifiying a pixel.
                #NOT THE actual size of the image being tested
                h,w = image.shape[:2]
                count = 0
                count2 = 0
                best_guess = np.full((h,w),-1)
                tmp = []
                i0 = int(constants.IMG_SIZE / 2)
                j0 = int(constants.IMG_SIZE / 2)

                #GO THROUGH EACH PIXEL WITHOUT THE EDGES SINCE WE NEED TO MAKE SURE EVERY PART OF THE PIXEL AREA
                #BEING SENT TO THE MODEL IS PART OF THE IMAGE
                for i in range(int(constants.IMG_SIZE / 2),int(len(image) - (constants.IMG_SIZE / 2))):
                    for j in range(int(constants.IMG_SIZE / 2),int(len(image[0]) - (constants.IMG_SIZE / 2))):

                        #get the bounding box around the pixel to send to the training
                        box = image[i-int(constants.IMG_SIZE / 2):i+int(constants.IMG_SIZE / 2),j-int(constants.IMG_SIZE / 2):j+int(constants.IMG_SIZE / 2)]

                        #append the box to a temporary array
                        tmp.append(box)

                        #once the temporary array is the same size as the batch size, run the testing on the batch
                        if(len(tmp) == constants.BATCH_SIZE or count == ((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE)) - 1):
                            batch = np.array(tmp)
                            mask = predict_op.eval({x:batch})

                            #now we go through the mask and append the values to the correct position of best_guess which is a copy of
                            #the original image except all the values are -1
                            for val in mask:
                                best_guess[i0,j0] = val
                                if j0 == (w - int(constants.IMG_SIZE/2)) - 1:
                                    j0 = int(constants.IMG_SIZE / 2)
                                    i0 += 1
                                else:
                                    j0 += 1

                            #give console output to show progress
                            print('%i out of %i complete' % (count2,math.ceil(int((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE) / constants.BATCH_SIZE))))
                            #empty tmporary array
                            tmp = []
                            count2 += 1
                        count += 1

                outputResults(image,np.array(best_guess),fout='pixel_segmentation.png')

        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test2' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                #initialize variables
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #read the image
                img = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)

                #segment the image
                unsupervised_segments, markers = featureReader.cnn_readOneImg2(sys.argv[2])
                unique_markers = np.unique(markers)
                basefile = os.path.basename(sys.argv[2])
                fname = os.path.splitext(basefile)[0]
                cv2.imwrite(fname + '_unsupervised_segmentation.png', unsupervised_segments)

                #lose some part of the segmentation to padding
                h,w = markers.shape[:2]
                paddedMarks = np.full((h,w),-1)
                paddedMarks[int(constants.IMG_SIZE / 2):h - int(constants.IMG_SIZE / 2),int(constants.IMG_SIZE / 2): w - int(constants.IMG_SIZE / 2)] = markers[int(constants.IMG_SIZE / 2):h - int(constants.IMG_SIZE / 2),int(constants.IMG_SIZE / 2): w - int(constants.IMG_SIZE / 2)]

                #go through each unique marker and find the classification for that segment
                tmp = []
                for segcounter,uq_mark in enumerate(unique_markers[1:]):
                    count = 0
                    cats = [0,0,0,0,0,0]
                    row,col = np.where(paddedMarks == uq_mark)
                    print(str(len(row)) + "," + str(len(col)))
                    for i,j in zip(row,col):
                        box = img[i-int(constants.IMG_SIZE / 2):i+int(constants.IMG_SIZE / 2),j-int(constants.IMG_SIZE / 2):j+int(constants.IMG_SIZE / 2)]

                        #append the box to a temporary array
                        tmp.append(box)

                        #once the temporary array is the same size as the batch size, run the testing on the batch
                        if(len(tmp) == constants.BATCH_SIZE or count == len(row) - 1):
                            batch = np.array(tmp)
                            mask = predict_op.eval({x:batch})
                            for val in mask:
                                if(val == 0):
                                    cats[0] += 1
                                elif(val == 1):
                                    cats[1] += 1
                                elif(val == 2):
                                    cats[2] += 1
                                elif(val == 3):
                                    cats[3] += 1
                                elif(val == 4):
                                    cats[4] += 1
                                elif(val == 5):
                                    cats[5] += 1

                                tmp = []

                            #give console output to show progress
                            print('BATCH %i out of %i ------> for segment %i of %i'  % (count,int(len(row) / constants.BATCH_SIZE),segcounter,len(unique_markers)))
                        count += 1

                    #find the category with highest activation count predicted by the model
                    index = cats.index(max(cats))
                    if(max(cats) == 0):
                        markers[markers == uq_mark] = -1
                    elif(index == 0):
                        markers[markers == uq_mark] = 0
                    elif(index == 1):
                        markers[markers == uq_mark] = 1
                    elif(index == 2):
                        markers[markers == uq_mark] = 2
                    elif(index == 3):
                        markers[markers == uq_mark] = 3
                    elif(index == 4):
                        markers[markers == uq_mark] = 4
                    elif(index == 5):
                        markers[markers == uq_mark] = 5

                    if(max(cats) > 0):
                        ratio = max(cats) / sum(cats)
                    else:
                        ratio = 0.0

                    print("SEGMENT PROCESSED: %i of %i          category was %i            with %f sureity" % (segcounter,len(unique_markers),index,ratio))

                outputResults(img,markers,fout='majoritypixel_segmentation.png')

        elif(sys.argv[1] == 'debug'):

            image = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
            botched_img = smartcut(image)
            print(botched_img.shape)

        else:
            print("trainpix [NONE]")
            print("trainseg [NONE]")
            print("test [image_filepath] [model_filepath]")
            print("test2 [image_filepath] [model_filepath]")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,test2,trainpix,trainseg)")

if __name__ == "__main__":
    tf.app.run()
