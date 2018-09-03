KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "ops_logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants
MIXED_FILE = "../categories/mixed/all/mixed14.JPG"
GROUND_TRUTH = "../categories/mixed/gt/mixed14_gt.png"
segment_dir = "../bgrsegmentation_mixedtiled/"
CAT1            = "treematter"
CAT2            = "plywood"
CAT3            = "cardboard"
CAT4            = "bottles"
CAT5            = "trashbag"
CAT6            = "blackbag"
TREEMATTER = [0,0,255]
PLYWOOD = [0,255,0]
CARDBOARD = [255,0,0]
BLACKBAG = [255,255,0]
TRASHBAG = [255,0,255]
BOTTLES = [0,255,255]
CATS=[TREEMATTER,PLYWOOD,CARDBOARD,BLACKBAG,TRASHBAG,BOTTLES]
CAT1_ONEHOT     = [1,0,0,0,0,0]
CAT2_ONEHOT     = [0,1,0,0,0,0]
CAT3_ONEHOT     = [0,0,1,0,0,0]
CAT4_ONEHOT     = [0,0,0,1,0,0]
CAT5_ONEHOT     = [0,0,0,0,1,0]
CAT6_ONEHOT     = [0,0,0,0,0,1]
CATS_ONEHOT = [CAT1_ONEHOT,CAT2_ONEHOT,CAT3_ONEHOT,CAT4_ONEHOT,CAT5_ONEHOT,CAT6_ONEHOT]
LEARNING_RATE = 0.001               #Learning rate for training the NN
NN_CLASSES      = 6


FULL_IMGSIZE = 1000
NN_FULL1   = 100                #Number of features output for fully connected layer1
NN_FULL2   = 100                #Number of features output for fully connected layer1
NN_FULL3   = 100                #Number of features output for fully connected layer1
IMG_DEPTH   = 3
KEEP_RATE = 0.50
DECOMP_LENGTH = 0.99               #components that allow 99% of data described



