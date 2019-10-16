#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
WL_FOLDER="${WORK_DIR}/datasets/wl_data"
INIT_MODEL="${WL_FOLDER}/init_models_mobilenet"
TRAIN_DIR="${WL_FOLDER}/train"
DATASET="${WL_FOLDER}/ny/tfrecord"

mkdir -p "${TRAIN_DIR}"


NUM_ITERATIONS=500000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=16 \
  --dataset="wl" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=True \
  --tf_initial_checkpoint="${INIT_MODEL}/model.ckpt-20265" \
  --train_logdir="${TRAIN_DIR}" \
  --dataset_dir="${DATASET}"
