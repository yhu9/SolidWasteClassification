import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import feature
import cv2
import os
import constants as C
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import TensorBoard,ModelCheckpoint

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#given an image and its mask writes the results as fout
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
    cat0_count = np.count_nonzero(mask == -1)
    cat1_count = np.count_nonzero(mask == 0)
    cat2_count = np.count_nonzero(mask == 1)
    cat3_count = np.count_nonzero(mask == 2)
    cat4_count = np.count_nonzero(mask == 3)
    cat5_count = np.count_nonzero(mask == 4)
    cat6_count = np.count_nonzero(mask == 5)
    total = cat1_count + cat2_count + cat3_count + cat4_count + cat5_count + cat6_count + cat0_count

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
        f.write("%s : %f\n" % (C.CAT1,p1))
        f.write("%s : %f\n" % (C.CAT2,p2))
        f.write("%s : %f\n" % (C.CAT3,p3))
        f.write("%s : %f\n" % (C.CAT4,p4))
        f.write("%s : %f\n" % (C.CAT5,p5))
        f.write("%s : %f\n" % (C.CAT6,p6))
        f.write("--------------------------------------------------------------------------------------\n")
        f.write("------------------------------------END-----------------------------------------------\n")
        f.write("--------------------------------------------------------------------------------------\n")

        greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

        #f.write out to the terminal what the most common category was for the image
        if(greatest == cat1_count):
            f.write("\nthe most common category is: " + C.CAT1)
        elif(greatest == cat2_count):
            f.write("\nthe most common category is: " + C.CAT2)
        elif(greatest == cat3_count):
            f.write("\nthe most common category is: " + C.CAT3)
        elif(greatest == cat4_count):
            f.write("\nthe most common category is: " + C.CAT4)
        elif(greatest == cat5_count):
            f.write("\nthe most common category is: " + C.CAT5)
        elif(greatest == cat6_count):
            f.write("\nthe most common category is: " + C.CAT6)
        else:
            f.write("\nsorry something went wrong counting the predictions")

#generate predictions on the image using the trained network
def generate_prediction(imgfile, network):

    #extract features from blobs
    print("Getting features...")
    image = cv2.imread(imgfile,cv2.IMREAD_COLOR)
    h,w = image.shape[:2]
    if h > C.FULL_IMGSIZE or w > C.FULL_IMGSIZE:
        image = cv2.resize(image,(C.FULL_IMGSIZE,C.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
    x_test,y_test,markers = feature.extractImage(image,imgfile)

    #get predictions
    print("Getting Predictions...")
    rawpredictions = network.predict_on_batch(x_test)
    predictions = rawpredictions.argmax(axis=1)

    #create mask from predictions
    print("writing results")
    h,w = image.shape[:2]
    best_guess = np.full((h,w),-1)
    for l,p in zip(np.unique(markers),predictions):
        best_guess[markers == l] = p

    #write the results as an image
    if not os.path.isdir('results'):
        os.makedirs('results')
    fileout = 'learnedseg_nn_' + os.path.splitext(os.path.basename(imgfile))[0]  + ".png"
    fileout = os.path.join('results',fileout)
    outputResults(image,np.array(best_guess),fout=fileout)

    #save the raw file
    if not os.path.isdir('raws'):
        os.makedirs('raws')
    filename ='rawoutput_nn_' + os.path.splitext(os.path.basename(imgfile))[0] + '.txt'
    full_dir = os.path.join('raws',filename)
    with open(full_dir,'w') as fout:
        for raw,cat,mark in zip(rawpredictions,predictions,np.unique(markers)):
            fout.write(str("cat: " + str(cat) + '    mark: ' + str(mark) + '    raw: '))
            for val in raw:
                fout.write(str(val) + ',')
            fout.write('\n')

#################################################################################################
#################################################################################################
#################################################################################################



#main method
if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1] == 'train':

        #define the model
        model = Sequential()
        model.add(Dense(units=100,activation='tanh', input_dim=12))
        model.add(Dropout(0.5))
        model.add(Dense(units=100,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(units=6,activation='tanh'))

        #create or log and model save locations
        if not os.path.isdir('model'):
            os.makedirs('model')
        if not os.path.isdir('log'):
            os.makedirs('log')

        #initialize model
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir="log/{}".format(time()))
        checkpoint = ModelCheckpoint('model/cnn_model.ckpt', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        #get the training data
        #train_x,train_y = feature.getTrainingBatch(100)
        #print('training data acquired')

        #get the validation data
        #valid_x,valid_y = feature.getTestingBatch()
        #print('testing batch acquired')

        #create our training batch generator
        def generator(n):
            while True:
                batch_x,batch_y = feature.getTrainingBatch(n)
                yield batch_x,batch_y

        #create our testing batch generator
        valid_x,valid_y= feature.getTestingBatch()
        def validationGenerator(x,y):
            while True:
                yield x,y

        #fit the model
        print('begin training')
        model.fit_generator(generator(100),
                epochs=3000,
                steps_per_epoch=1,
                validation_data=validationGenerator(valid_x,valid_y),
                validation_steps=1,
                verbose=2,
                callbacks=[tensorboard,checkpoint])

    elif len(sys.argv) == 4 and sys.argv[1] == 'test':

        if os.path.exists(sys.argv[2]):
            model = keras.models.load_model(sys.argv[3])

            generate_prediction(sys.argv[2],model)
        else:
            print('ooops this file does not exists %s' % sys.argv[2])

    else:
        print('error! wrong arguments to nn.py')
        print('expecting:')
        print('python nn.py train')
        print('python nn.py test [img_path] [model_path]')


