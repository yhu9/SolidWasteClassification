#!/bin/bash

CARDBOARD_DIR="categories/cardboard/ingroup/"
TREEMATTER_DIR="categories/treematter/ingroup/"
PLYWOOD_DIR="categories/plywood/ingroup/"
TRASHBAG_DIR="categories/trashbag/ingroup/"
BOTTLES_DIR="categories/bottles/ingroup/"
BLACKBAG_DIR="categories/blackbag/ingroup/"
GROUND_DIR="categories/ground/ingroup/"
MIXED_DIR="categories/mixed/all/"
OUT_DIR1="bgrsegmentation_mixedtiled"

mkdir $OUT_DIR1
#python save_segments.py savetile $CARDBOARD_DIR $OUT_DIR1 &
#python save_segments.py savetile $TREEMATTER_DIR $OUT_DIR1 &
#python save_segments.py savetile $PLYWOOD_DIR $OUT_DIR1 &
#python save_segments.py savetile $BOTTLES_DIR $OUT_DIR1 &
#python save_segments.py savetile $BLACKBAG_DIR $OUT_DIR1 &
#python save_segments.py savetile $TRASHBAG_DIR $OUT_DIR1 &
python save_segments.py savetile $MIXED_DIR $OUT_DIR1 &
wait

python save_segments.py rotate $OUT_DIR1

echo "Done extracting blobs"
