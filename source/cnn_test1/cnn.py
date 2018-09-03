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
        f.write("%s : %f\n" % (constants.CAT1,p1))
        f.write("%s : %f\n" % (constants.CAT2,p2))
        f.write("%s : %f\n" % (constants.CAT3,p3))
        f.write("%s : %f\n" % (constants.CAT4,p4))
        f.write("%s : %f\n" % (constants.CAT5,p5))
        f.write("%s : %f\n" % (constants.CAT6,p6))
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
#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if not os.path.exists('log'):
        os.makedirs('log')

    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x1 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x2 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x3 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x4 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x5 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x6 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        y = tf.placeholder('float',[None,constants.CLASSES])
        y1 = tf.placeholder('float',[None,1])
        y2 = tf.placeholder('float',[None,1])
        y3 = tf.placeholder('float',[None,1])
        y4 = tf.placeholder('float',[None,1])
        y5 = tf.placeholder('float',[None,1])
        y6 = tf.placeholder('float',[None,1])

        weights = {}
        biases = {}

        #magic number = width * height * n_convout
        magic_number = int((constants.CNN_LOCAL1 + constants.CNN_GLOBAL) * (constants.IMG_SIZE * constants.IMG_SIZE))


        #tree matter convolution
        with tf.name_scope('treematter'):
            weights['w_treematter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_treematter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            tree_conv1 = tf.nn.conv2d(x1,weights['w_treematter'],strides=[1,1,1,1],padding='SAME',name='tree_tree1')
            tree1 = tf.nn.relu(tree_conv1 + biases['b_treematter'])
            weights['w_treematter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_treematter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            tree_conv2 = tf.nn.conv2d(tree1,weights['w_treematter_p1b'],strides=[1,1,1,1],padding='SAME',name='tree_tree2')
            tree2 = tf.nn.relu(tree_conv2 + biases['b_treematter_p1b'])
            weights['w_treematter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_treematter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gtree_conv1 = tf.nn.conv2d(x1,weights['w_treematter_global'],strides=[1,1,1,1],padding='SAME',name='global_tree')
            tree3 = tf.nn.relu(gtree_conv1 + biases['b_treematter_global'])
            tree_activations = tf.concat([tree2,tree3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(tree_activations,[-1,magic_number])
            predictions1 = tf.matmul(output1,weights['out1'])+biases['out1']

        #plywood matter convolution
        with tf.name_scope('plywood'):
            weights['w_plywoodmatter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_plywoodmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            plywood_conv1 = tf.nn.conv2d(x2,weights['w_plywoodmatter'],strides=[1,1,1,1],padding='SAME',name='plywood_plywood1')
            plywood1 = tf.nn.relu(plywood_conv1 + biases['b_plywoodmatter'])
            weights['w_plywoodmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_plywoodmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            plywood_conv2 = tf.nn.conv2d(plywood1,weights['w_plywoodmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='plywood_plywood2')
            plywood2 = tf.nn.relu(plywood_conv2 + biases['b_plywoodmatter_p1b'])
            weights['w_plywoodmatter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_plywoodmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gplywood_conv1 = tf.nn.conv2d(x2,weights['w_plywoodmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_plywood')
            plywood3 = tf.nn.relu(gplywood_conv1 + biases['b_plywoodmatter_global'])
            plywood_activations = tf.concat([plywood2,plywood3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(plywood_activations,[-1,magic_number])
            predictions2 = tf.matmul(output1,weights['out1'])+biases['out1']

        #cardboard matter convolution
        with tf.name_scope('cardboard'):
            weights['w_cardboardmatter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_cardboardmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            cardboard_conv1 = tf.nn.conv2d(x3,weights['w_cardboardmatter'],strides=[1,1,1,1],padding='SAME',name='cardboard_cardboard1')
            cardboard1 = tf.nn.relu(cardboard_conv1 + biases['b_cardboardmatter'])
            weights['w_cardboardmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_cardboardmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            cardboard_conv2 = tf.nn.conv2d(cardboard1,weights['w_cardboardmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='cardboard_cardboard2')
            cardboard2 = tf.nn.relu(cardboard_conv2 + biases['b_cardboardmatter_p1b'])
            weights['w_cardboardmatter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_cardboardmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gcardboard_conv1 = tf.nn.conv2d(x3,weights['w_cardboardmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_cardboard')
            cardboard3 = tf.nn.relu(gcardboard_conv1 + biases['b_cardboardmatter_global'])
            cardboard_activations = tf.concat([cardboard2,cardboard3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(cardboard_activations,[-1,magic_number])
            predictions3 = tf.matmul(output1,weights['out1'])+biases['out1']

        #bottles matter convolution
        with tf.name_scope('bottles'):
            weights['w_bottlesmatter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_bottlesmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            bottles_conv1 = tf.nn.conv2d(x4,weights['w_bottlesmatter'],strides=[1,1,1,1],padding='SAME',name='bottles_bottles1')
            bottles1 = tf.nn.relu(bottles_conv1 + biases['b_bottlesmatter'])
            weights['w_bottlesmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_bottlesmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            bottles_conv2 = tf.nn.conv2d(bottles1,weights['w_bottlesmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='bottles_bottles2')
            bottles2 = tf.nn.relu(bottles_conv2 + biases['b_bottlesmatter_p1b'])
            weights['w_bottlesmatter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_bottlesmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gbottles_conv1 = tf.nn.conv2d(x4,weights['w_bottlesmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_bottles')
            bottles3 = tf.nn.relu(gbottles_conv1 + biases['b_bottlesmatter_global'])
            bottles_activations = tf.concat([bottles2,bottles3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(bottles_activations,[-1,magic_number])
            predictions4 = tf.matmul(output1,weights['out1'])+biases['out1']

        #pbag matter convolution
        with tf.name_scope('plasticbags'):
            weights['w_pbagmatter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_pbagmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            pbag_conv1 = tf.nn.conv2d(x5,weights['w_pbagmatter'],strides=[1,1,1,1],padding='SAME',name='pbag_pbag1')
            pbag1 = tf.nn.relu(pbag_conv1 + biases['b_pbagmatter'])
            weights['w_pbagmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_pbagmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            pbag_conv2 = tf.nn.conv2d(pbag1,weights['w_pbagmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='pbag_pbag2')
            pbag2 = tf.nn.relu(pbag_conv2 + biases['b_pbagmatter_p1b'])
            weights['w_pbagmatter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_pbagmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gpbag_conv1 = tf.nn.conv2d(x5,weights['w_pbagmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_pbag')
            pbag3 = tf.nn.relu(gpbag_conv1 + biases['b_pbagmatter_global'])
            pbag_activations = tf.concat([pbag2,pbag3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(pbag_activations,[-1,magic_number])
            predictions5 = tf.matmul(output1,weights['out1'])+biases['out1']

        #bbag matter convolution
        with tf.name_scope('black_bag'):
            weights['w_bbagmatter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
            biases['b_bbagmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            bbag_conv1 = tf.nn.conv2d(x6,weights['w_bbagmatter'],strides=[1,1,1,1],padding='SAME',name='bbag_bbag1')
            bbag1 = tf.nn.relu(bbag_conv1 + biases['b_bbagmatter'])
            weights['w_bbagmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
            biases['b_bbagmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
            bbag_conv2 = tf.nn.conv2d(bbag1,weights['w_bbagmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='bbag_bbag2')
            bbag2 = tf.nn.relu(bbag_conv2 + biases['b_bbagmatter_p1b'])
            weights['w_bbagmatter_global'] = tf.Variable(tf.random_normal([13,13,3,constants.CNN_GLOBAL]))
            biases['b_bbagmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
            gbbag_conv1 = tf.nn.conv2d(x6,weights['w_bbagmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_bbag')
            bbag3 = tf.nn.relu(gbbag_conv1 + biases['b_bbagmatter_global'])
            bbag_activations = tf.concat([bbag2,bbag3],3)
            weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
            biases['out1'] = tf.Variable(tf.random_normal([1]))
            output1 = tf.reshape(bbag_activations,[-1,magic_number])
            predictions6 = tf.matmul(output1,weights['out1'])+biases['out1']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions1,labels=y1))
            cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions2,labels=y2))
            cost3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions3,labels=y3))
            cost4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions4,labels=y4))
            cost5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions5,labels=y5))
            cost6 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions6,labels=y6))
            cost_sum1 = tf.summary.scalar('treecost',cost1)
            cost_sum2 = tf.summary.scalar('plywoodcost',cost2)
            cost_sum3 = tf.summary.scalar('cardboardcost',cost3)
            cost_sum4 = tf.summary.scalar('bottlescost',cost4)
            cost_sum5 = tf.summary.scalar('pbagcost',cost5)
            cost_sum6 = tf.summary.scalar('bbagcost',cost6)
        with tf.name_scope('optimizer'):
            optimizer1= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost1)
            optimizer2= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost2)
            optimizer3= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost3)
            optimizer4= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost4)
            optimizer5= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost5)
            optimizer6= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost6)
        with tf.name_scope('accuracy'):
            correct_prediction1 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions1)),y1),tf.float32)
            correct_prediction2 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions2)),y2),tf.float32)
            correct_prediction3 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions3)),y3),tf.float32)
            correct_prediction4 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions4)),y4),tf.float32)
            correct_prediction5 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions5)),y5),tf.float32)
            correct_prediction6 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions6)),y6),tf.float32)

            accuracy1 = tf.reduce_mean(correct_prediction1)
            accuracy2 = tf.reduce_mean(correct_prediction2)
            accuracy3 = tf.reduce_mean(correct_prediction3)
            accuracy4 = tf.reduce_mean(correct_prediction4)
            accuracy5 = tf.reduce_mean(correct_prediction5)
            accuracy6 = tf.reduce_mean(correct_prediction6)
            acc_sum1 = tf.summary.scalar('tree_accuracy',accuracy1)
            acc_sum2 = tf.summary.scalar('plywood_accuracy',accuracy2)
            acc_sum3 = tf.summary.scalar('cardboard_accuracy',accuracy3)
            acc_sum4 = tf.summary.scalar('bottles_accuracy',accuracy4)
            acc_sum5 = tf.summary.scalar('pbag_accuracy',accuracy5)
            acc_sum6 = tf.summary.scalar('bbag_accuracy',accuracy6)

        #stack all convoluted outputs from each model
        with tf.name_scope('all_combined'):
            stacked = tf.concat([tree_activations,plywood_activations,cardboard_activations,bottles_activations,pbag_activations,bbag_activations],3)
            convcount = constants.CLASSES * (constants.CNN_LOCAL1 + constants.CNN_GLOBAL)
            all_raws = tf.reshape(stacked,[-1,convcount * constants.IMG_SIZE * constants.IMG_SIZE])
            weights['w_all1'] = tf.Variable(tf.random_normal([convcount * constants.IMG_SIZE * constants.IMG_SIZE,constants.CNN_FULL]))
            biases['b_all1'] = tf.Variable(tf.random_normal([constants.CNN_FULL]))
            fc_1 = tf.matmul(all_raws,weights['w_all1']) + biases['b_all1']
            fc_activation1 = tf.nn.relu(fc_1)
            weights['w_all2'] = tf.Variable(tf.random_normal([constants.CNN_FULL,constants.CNN_FULL]))
            biases['b_all2'] = tf.Variable(tf.random_normal([constants.CNN_FULL]))
            fc_2 = tf.matmul(fc_activation1,weights['w_all2']) + biases['b_all2']
            fc_activation2 = tf.nn.relu(fc_2)
            weights['w_all'] = tf.Variable(tf.random_normal([constants.CNN_FULL,constants.CLASSES]))
            biases['b_all'] = tf.Variable(tf.random_normal([constants.CLASSES]))
            predictions_final = tf.matmul(fc_activation2,weights['w_all']) + biases['b_all']
            predictions_out = tf.nn.sigmoid(predictions_final)
            all_cost = tf.nn.softmax_cross_entropy_with_logits(logits=predictions_final,labels=y)
            all_reduced = tf.reduce_mean(all_cost)
            allcost_sum = tf.summary.scalar('all_cost',all_reduced)

            var_list1 = [weights['w_all'],biases['b_all'],weights['w_all1'],weights['w_all2'],biases['b_all1'],biases['b_all2']]
            all_optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(all_cost,var_list=var_list1)
            all_correct = tf.cast(tf.equal(tf.argmax(predictions_final,1),tf.argmax(y,1)),tf.float32)
            all_accuracy = tf.reduce_mean(all_correct)
            allacc_sum= tf.summary.scalar('all_accuracy',all_accuracy)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#helper functions

        #training mode trained on the image
        if(sys.argv[1] == 'train'):
            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                merged = tf.summary.merge_all()
                training_writer = tf.summary.FileWriter('./log/training',sess.graph)
                testing_writer = tf.summary.FileWriter('./log/testing',sess.graph)
                sess.run(init)

                #train the model
                acc = 0.00;
                modelpath = "model"
                logdir = 'log/traininglog.txt'
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                if not os.path.exists('log'):
                    os.makedirs('log')

                for epoch in range(constants.CNN_EPOCHS):

                    #get an image batch and train each model separately
                    batch_x1,batch_y1 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='treematter')
                    batch_x2,batch_y2 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='plywood')
                    batch_x3,batch_y3 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='cardboard')
                    batch_x4,batch_y4 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='bottles')
                    batch_x5,batch_y5 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='trashbag')
                    batch_x6,batch_y6 = featureReader.getTrainingBatch(constants.BATCH_SIZE,catname='blackbag')


                    #merge summaries
                    merged1 = tf.summary.merge([acc_sum1,acc_sum2,acc_sum3,acc_sum4,acc_sum5,acc_sum6,cost_sum1,cost_sum2,cost_sum3,cost_sum4,cost_sum5,cost_sum6])
                    summary = sess.run([merged1,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,optimizer1,optimizer2,optimizer3,optimizer4,optimizer5,optimizer6],feed_dict={x1:batch_x1,x2:batch_x2,x3:batch_x3,x4:batch_x4,x5:batch_x5,x6:batch_x6,y1:batch_y1,y2:batch_y2,y3:batch_y3,y4:batch_y4,y5:batch_y5,y6:batch_y6})
                    training_log = summary[0]
                    acc1 = summary[1]
                    acc2 = summary[2]
                    acc3 = summary[3]
                    acc4 = summary[4]
                    acc5 = summary[5]
                    acc6 = summary[6]
                    training_writer.add_summary(training_log,epoch)

                    #run the final optimizer
                    batch_x,batch_y = featureReader.getTrainingBatch(constants.BATCH_SIZE)
                    merged2 = tf.summary.merge([allcost_sum,allacc_sum])
                    summary = sess.run([merged2,all_reduced,all_optimizer],feed_dict={x1: batch_x,x2: batch_x,x3: batch_x,x4: batch_x,x5: batch_x,x6: batch_x, y: batch_y})
                    training_writer.add_summary(summary[0],epoch)

                    #evaluate the models separately using a test set
                    if epoch % 1 == 0:
                        #record summaries
                        #evaluate overall model on test set
                        eval_x,eval_y,eval_y1,eval_y2,eval_y3,eval_y4,eval_y5,eval_y6 = featureReader.getTestingBatch(constants.BATCH_SIZE)
                        merged2 = tf.summary.merge([allacc_sum,allcost_sum,cost_sum1,cost_sum2,cost_sum3,cost_sum4,cost_sum5,cost_sum6,acc_sum1,acc_sum2,acc_sum3,acc_sum4,acc_sum5,acc_sum6])
                        accnew,summary = sess.run([all_accuracy,merged2],feed_dict={x1: eval_x,x2: eval_x,x3: eval_x,x4: eval_x,x5: eval_x,x6: eval_x, y: eval_y, y1: eval_y1, y2: eval_y2, y3: eval_y3, y4: eval_y4, y5: eval_y5, y6: eval_y6})

                        #write the summary
                        testing_writer.add_summary(summary,epoch)

                        #record summaries
                        #summary = sess.run(merged,feed_dict={x1:eval_x, x2:eval_x,x3: eval_x,x4:eval_x,x5:eval_x,x6:eval_x,y:eval_y})
                        #training_writer.add_summary(summary,epoch)

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,'model/cnn_model.ckpt')
                            print("highest accuracy found! model saved")

                        print('epoch: %i  treematter: %.4f  plywood: %.4f  cardboard: %.4f   bottles: %.4f  trashbag: %.4f  blackbag: %.4f  all: %.4f' % (epoch,acc1,acc2,acc3,acc4,acc5,acc6,accnew))
                        with open(logdir,'a') as log_out:
                            log_out.write('epoch: %i   treematter: %.4f  plywood: %.4f  cardboard: %.4f  bottles: %.4f  trashbag: %.4f    blackbag: %.4f  all: %.4f\n' % (epoch,acc1,acc2,acc3,acc4,acc5,acc6,accnew))


        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            if os.path.isfile(sys.argv[2]):
                tmp = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                h,w = tmp.shape[:2]
                if(h >= constants.FULL_IMGSIZE or w >= constants.FULL_IMGSIZE):
                    image = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
                else:
                    image = tmp

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #we recreate the image by painting the best_guess mask on a blank canvas with the same shape as image
                #initialize counters and the height and width of the image being tested.
                #constants.IMG_SIZE is the img size the learned model uses for classifiying a pixel.
                #NOT THE actual size of the image being tested
                h,w = image.shape[:2]
                count = 0
                count2 = 0
                best_guess = np.full((h,w),-1)
                raw_guess = np.full((h,w,6),0)
                tmp = []
                i0 = int(constants.IMG_SIZE / 2)
                j0 = int(constants.IMG_SIZE / 2)

                #define our log file and pixel segmentation file name
                if not os.path.exists('results'):
                    os.mkdir('results')

                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                logname = "results/rawoutput_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + '_' + modelname + ".txt"
                seg_file = 'results/' + os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"

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
                            rawpredictions = predictions_final.eval({x1:batch, x2:batch, x3:batch, x4:batch, x5:batch, x6:batch})
                            mask = rawpredictions.argmax(axis=1)

                            #now we go through the mask and insert the values to the correct position of best_guess which is a copy of
                            #the original image except all the values are -1
                            for raw,cat in zip(rawpredictions,mask):
                                best_guess[i0,j0] = cat
                                raw_guess[i0,j0] = raw
                                if j0 == (w - int(constants.IMG_SIZE/2)) - 1:
                                    j0 = int(constants.IMG_SIZE / 2)
                                    i0 += 1
                                else:
                                    j0 += 1

                            #give console output to show progress
                            outputResults(image,np.array(best_guess),fout=seg_file)
                            print('%i out of %i complete' % (count2,math.ceil(int((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE) / constants.BATCH_SIZE))))
                            #empty tmporary array
                            tmp = []
                            count2 += 1
                        count += 1

                np.save(logname,raw_guess)
        else:
            print("train ")
            print("trainseg ")
            print("test [image_filepath] [model_filepath]")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train)")

if __name__ == "__main__":
    tf.app.run()
