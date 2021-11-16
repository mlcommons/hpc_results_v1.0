#!/bin/bash

python -m tools.convert_tfrecord_to_numpy.py -i ${DATA_SRC_DIR}/train -o ${DATA_DST_DIR}/train -c gzip
python -m tools.convert_tfrecord_to_numpy.py -i ${DATA_SRC_DIR}/validation -o ${DATA_DST_DIR}/validation -c gzip

ls -1 ${DATA_DST_DIR}/train | grep "_data.npy" | sort > ${DATA_DST_DIR}/train/files_data.lst
ls -1 ${DATA_DST_DIR}/validation | grep "_data.npy" | sort > ${DATA_DST_DIR}/validation/files_data.lst
ls -1 ${DATA_DST_DIR}/train | grep "_label.npy" | sort > ${DATA_DST_DIR}/train/files_label.lst
ls -1 ${DATA_DST_DIR}/validation | grep "_label.npy" | sort > ${DATA_DST_DIR}/validation/files_label.lst