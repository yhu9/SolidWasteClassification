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

DECOMP_LENGTH = 0.99               #0.99 % of when I did it with hsv no bg image
