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
python save_segments.py savehsv $CARDBOARD_DIR &
python save_segments.py savehsv $TREEMATTER_DIR &
python save_segments.py savehsv $PLYWOOD_DIR &
python save_segments.py savehsv $BOTTLES_DIR &
python save_segments.py savehsv $BLACKBAG_DIR &
python save_segments.py savehsv $TRASHBAG_DIR &&
mv segments hsvsegmentation_nobg

mkdir fullsegmentation_hsv
mv segmentation_* fullsegmentation_hsv
