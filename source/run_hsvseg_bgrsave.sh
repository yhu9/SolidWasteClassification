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
python save_segments.py bgrhsv $CARDBOARD_DIR &
python save_segments.py bgrhsv $TREEMATTER_DIR &
python save_segments.py bgrhsv $PLYWOOD_DIR &
python save_segments.py bgrhsv $BOTTLES_DIR &
python save_segments.py bgrhsv $BLACKBAG_DIR &
python save_segments.py bgrhsv $TRASHBAG_DIR &&
mv segments hsvsegmentation_bgrsegment_nobg

mkdir fullsegmentation_hsv
mv segmentation_* fullsegmentation_hsv
