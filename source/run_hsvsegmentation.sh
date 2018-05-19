#!/bin/bash

CARDBOARD_DIR="categories/cardboard/ingroup/"
TREEMATTER_DIR="categories/treematter/ingroup/"
PLYWOOD_DIR="categories/plywood/ingroup/"
TRASHBAG_DIR="categories/trashbag/ingroup/"
BOTTLES_DIR="categories/bottles/ingroup/"
BLACKBAG_DIR="categories/blackbag/ingroup/"
GROUND_DIR="categories/ground/ingroup/"
MIXED_DIR="categories/mixed/"
OUT_DIR1="hsvsegmentation_nobg"
OUT_DIR2="hsvsegmentation_withbg"

#mkdir $OUT_DIR1
#python save_segments.py savehsv $CARDBOARD_DIR $OUT_DIR1 &
#python save_segments.py savehsv $TREEMATTER_DIR $OUT_DIR1 &
#python save_segments.py savehsv $PLYWOOD_DIR $OUT_DIR1 &
#python save_segments.py savehsv $BOTTLES_DIR $OUT_DIR1 &
#python save_segments.py savehsv $BLACKBAG_DIR $OUT_DIR1 &
#python save_segments.py savehsv $TRASHBAG_DIR $OUT_DIR1 &

mkdir $OUT_DIR2
python save_segments.py savehsv $CARDBOARD_DIR $OUT_DIR2 showbg &
python save_segments.py savehsv $TREEMATTER_DIR $OUT_DIR2 showbg &
python save_segments.py savehsv $PLYWOOD_DIR $OUT_DIR2 showbg &
python save_segments.py savehsv $BOTTLES_DIR $OUT_DIR2 showbg &
python save_segments.py savehsv $BLACKBAG_DIR $OUT_DIR2 showbg &
python save_segments.py savehsv $TRASHBAG_DIR $OUT_DIR2 showbg &

wait

echo "done exracting blobs"
