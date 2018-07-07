#!/bin/bash

CARDBOARD_DIR="categories/cardboard/ingroup/"
TREEMATTER_DIR="categories/treematter/ingroup/"
PLYWOOD_DIR="categories/plywood/ingroup/"
TRASHBAG_DIR="categories/trashbag/ingroup/"
BOTTLES_DIR="categories/bottles/ingroup/"
BLACKBAG_DIR="categories/blackbag/ingroup/"
GROUND_DIR="categories/ground/ingroup/"
OUT_DIR1="bgrsegmentation_nobg"

mkdir $OUT_DIR1
python save_segments.py save $CARDBOARD_DIR $OUT_DIR1 &
python save_segments.py save $TREEMATTER_DIR $OUT_DIR1 &
python save_segments.py save $PLYWOOD_DIR $OUT_DIR1 &
python save_segments.py save $BOTTLES_DIR $OUT_DIR1 &
python save_segments.py save $BLACKBAG_DIR $OUT_DIR1 &
python save_segments.py save $TRASHBAG_DIR $OUT_DIR1 &
wait

python save_segments.py rotate $OUT_DIR1

echo "Done extracting blobs"
