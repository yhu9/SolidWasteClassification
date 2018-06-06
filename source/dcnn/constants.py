

KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants
cat1_dir = "../categories/treematter/ingroup/"
cat2_dir = "../categories/plywood/ingroup/"
cat3_dir = "../categories/cardboard/ingroup/"
cat4_dir = "../categories/bottles/ingroup/"
cat5_dir = "../categories/trashbag/ingroup/"
cat6_dir = "../categories/blackbag/ingroup/"
CAT1            = "treematter"
CAT2            = "plywood"
CAT3            = "cardboard"
CAT4            = "bottles"
CAT5            = "trashbag"
CAT6            = "blackbag"
CAT1_ONEHOT     = [1,0,0,0,0,0]
CAT2_ONEHOT     = [0,1,0,0,0,0]
CAT3_ONEHOT     = [0,0,1,0,0,0]
CAT4_ONEHOT     = [0,0,0,1,0,0]
CAT5_ONEHOT     = [0,0,0,0,1,0]
CAT6_ONEHOT     = [0,0,0,0,0,1]
LEARNING_RATE = 0.01               #Learning rate for training the CNN
CNN_LOCAL1 = 64                  #Number of features output for conv layer 1
CNN_LOCAL2 = 64                  #Number of features output for conv layer 1
CNN_GLOBAL1 = 64                  #Number of features output for conv layer 2
CNN_CLASSES      = 6
CNN_EPOCHS       = 1000
CNN_FULL1   = 500                #Number of features output for fully connected layer1
IMG_SIZE = 56
FULL_IMGSIZE = 1000
IMG_DEPTH   = 3
KEEP_RATE = 0.85
BATCH_SIZE = 500
