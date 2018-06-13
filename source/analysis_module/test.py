#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import sys
import segmentModule
import extractionModule as analyze
import numpy as np
import cv2
import time
import re
import gabor_threads_roi as gabor
import os
from multiprocessing import Process
from multiprocessing import Manager
################################################################################################
#GLOBAL FLAG VARIABLES
#flags that are handled
showflag = 'show' in sys.argv
hogflag = 'hog' in sys.argv
binflag = 'bin' in sys.argv
sizeflag = 'size' in sys.argv
colorflag = 'color' in sys.argv
gaborflag = 'gabor' in sys.argv
hsvflag = 'hsv' in sys.argv
msflag = 'meanshift' in sys.argv
msbinflag = 'meanshiftbin' in sys.argv
fjmsflag = 'fjmeanshift' in sys.argv
dbscanflag = 'dbscan' in sys.argv
pcaflag = 'pca' in sys.argv
ldaflag = 'lda' in sys.argv

#############################################################################################################
#Check system argument length and mode
#if mode is bin do 3d color binning
start_time = time.time()

#applies the correct mode of operation given user input
def evaluate(full_path,mode,SHOWFLAG=False):
    #check if the image was read in correctly
    original = cv2.imread(full_path,cv2.IMREAD_COLOR)
    if original is None:
        print('invalid image! Could not open: %s' % full_path)

    #if mode is size
    if mode == 'size':
        combined_filename = sys.argv[1]

        # Generate and save blob size for this blob we assume black as background
        size = analyze.extractBlobSize(original)
        #print('--------------SIZE---------------')
        return size

    #if mode is hog, show hog feature vector of image
    elif mode == 'hog':
        hist = analyze.extractHOG(original,False)
        featurevector = hist.flatten()
        norm = analyze.normalize(featurevector)
        #print('-------------HOG----------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector)
        return norm

    #if mode is gabor, extract gabor feature from image using several orientations
    elif mode == 'gabor':
        orientations = 16
        filters = gabor.build_filters(orientations)
        combined_filename = sys.argv[1]

        # Generate and save ALL hogs for this image
        result = gabor.run_gabor(original, filters, combined_filename, orientations, mode='training')
        featurevector = result.flatten()[1:]
        norm = analyze.normalize(featurevector)
        #print('--------------Gabor---------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector,'r--')
        return norm

    #if mode is color, show color histogram of image
    elif mode == 'color':
        hist = analyze.extractColorHist(original,False)
        #print('-------------Color----------------')
        norm = analyze.normalize(hist)
        if SHOWFLAG:
            analyze.displayHistogram(hist)
        return norm

    elif mode == 'bin':
        hist = analyze.extractbinHist(original,False)
        norm = analyze.normalize(hist)
        return norm

    elif mode == 'hsv':
        hsvimg = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hist = analyze.extractHSVHist(hsvimg,False)
        norm = analyze.normalize(hist)
        return norm


#takes a single image and extracts all features depending on flag constants
#based on user input
def evaluate_all(full_path,instances):
    #extract features for each image depending on the flag constants
    features = []
    if binflag:
        features.append(evaluate(full_path,'bin',SHOWFLAG=showflag))
    if colorflag:
        features.append(evaluate(full_path,'color',SHOWFLAG=showflag))
    if gaborflag:
        features.append(evaluate(full_path,'gabor',SHOWFLAG=showflag))
    if hogflag:
        features.append(evaluate(full_path,'hog',SHOWFLAG=showflag))
    if hsvflag:
        features.append(evaluate(full_path,'hsv',SHOWFLAG=showflag))
    if sizeflag:
        features.append(evaluate(full_path,'size',SHOWFLAG=showflag))

    #create the full feature vector for the given instance image and push to instances
    #and also push the file name as the label for the instance
    full_vector = np.array([])
    for i in range(len(features)):
        full_vector = np.hstack((full_vector,features[i]))

    #get the label of the instance
    group = re.findall("treematter|plywood|cardboard|bottles|trashbag|blackbag|mixed",full_path)
    if(len(group) ==0):
        label = 'mixed'
    else:
        label = os.path.basename(group[0])

    #console output to show progress
    print("%s ---> DONE" % full_path)

    #save results to shared variabl
    if instances is not None:
        instances.append([full_vector,label])

#CHECK USER INPUT FOR THE PROPER FLAGS AND APPLY THE CORRECT ANALYSIS DEPENDING ON THE FLAGS GIVEN
#IF THE FIRST INPUT IS A DIRECTORY THEN WE APPLY ANALYSIS ON ALL IMAGES IN THE DIRECTORY
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
if __name__ == '__main__':

    if len(sys.argv) >= 3:

        #if user input is a directory apply to all images in directory
        if os.path.isdir(sys.argv[1]):
            #initialize list of instances
            instances = []
            labels = []
            count = 1
            myfiles = os.listdir(sys.argv[1])
            dircount = len(myfiles)

            #prepend the file directory so we have a list of full file directories to supply to the evaluate_all() function
            mylist = [os.path.join(sys.argv[1],f) for f in myfiles]

            #multi process the images in mylist of files through a shared variable of the manager class
            manager = Manager()
            values = manager.list()
            jobs = []

            #run all jobs
            tmpcount = 0
            max_processes = 20
            for filepath in mylist:
                tmpcount += 1
                p = Process(target=evaluate_all,args=(filepath,values))
                jobs.append(p)
                p.start()

                if tmpcount % max_processes == (max_processes - 1):
                    for j in jobs:
                        j.join()

            #join all jobs
            for j in jobs:
                j.join()

            #extract feature vector instances and labels separately
            instances = np.array([i[0] for i in values])
            labels = [[i[1]] for i in values]

            #we have to normalize just the sizes across all instances
            if(sizeflag):
                instances[:,0] = analyze.normalize(instances[:,0])

            #figure out what features were processed during the whole thing and name the file appropriately
            mode_op = ""
            for flag,name in zip([sizeflag,hogflag,gaborflag,colorflag,hsvflag],['size','hog','gabor','color','hsv']):
                if flag:
                    mode_op = mode_op + name

            #write the data results out to a text file in working directory
            #we can't do pca before hand because it doesn't save the eigen values which we need for testing a new segment
            #if pcaflag:
            #    basedir = os.path.basename(os.path.normpath(sys.argv[1]))
            #    featurefile = 'pcafeatures_' + mode_op + "_" + str(basedir)
            #    pcafile = 'pcaanalysis_' + str(basedir) + '.txt'
            #    new_instances = analyze.pcaAnalysis(instances,fnameout=pcafile)
            #    print("FEATURES REDUCED FROM %i to %i" % (len(instances[0]),len(new_instances[0])))
            #    analyze.writeFeatures(new_instances,fnameout=featurefile,label=np.array(labels))
            #else:
            basedir = os.path.basename(os.path.normpath(sys.argv[1]))
            featurefile = 'features_' + mode_op + "_" + str(basedir)
            analyze.writeFeatures(instances,fnameout=featurefile,label=np.array(labels))

    #if less than 3 args given
    else:
        print "wrong number of files as arguments expecting 3:"
        print "argv1 = image file/directory"
        print "argv2 + = modes of operation"
        sys.exit()

    #find out execution time
    print("--- %s seconds ---" % (time.time() - start_time))

