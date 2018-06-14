


KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "ops_logs"             #Directory where the logs would be stored for visualization of the training

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
LEARNING_RATE = 0.001               #Learning rate for training the NN
NN_CLASSES      = 6
NN_EPOCHS       = 1000


NN_FULL1   = 200                #Number of features output for fully connected layer1
NN_FULL2   = 200                #Number of features output for fully connected layer1
NN_FULL3   = 200                #Number of features output for fully connected layer1
IMG_SIZE = 56
IMG_DEPTH   = 3
KEEP_RATE = 0.85
BATCH_SIZE = 1000
DECOMP_LENGTH = 0.99               #components that allow 99% of data described

