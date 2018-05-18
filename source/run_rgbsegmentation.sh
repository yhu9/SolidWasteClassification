#!/bin/bash

CARDBOARD_DIR="categories/cardboard/ingroup/"
TREEMATTER_DIR="categories/treematter/ingroup/"
PLYWOOD_DIR="categories/plywood/ingroup/"
TRASHBAG_DIR="categories/trashbag/ingroup/"
BOTTLES_DIR="categories/bottles/ingroup/"
BLACKBAG_DIR="categories/blackbag/ingroup/"
GROUND_DIR="categories/ground/ingroup/"
MIXED_DIR="categories/mixed/"

mkdir segments
python save_segments.py save $CARDBOARD_DIR &
python save_segments.py save $TREEMATTER_DIR &
python save_segments.py save $PLYWOOD_DIR &
python save_segments.py save $BOTTLES_DIR &
python save_segments.py save $BLACKBAG_DIR &
python save_segments.py save $TRASHBAG_DIR &&
mv segments bgrsegmentation_nobg

mkdir fullsegmentation_bgr
mv segmentation_* fullsegmentation_bgr
