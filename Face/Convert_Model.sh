#!/bin/bash
#Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
#SPDX-License-Identifier: Apache-2.0
source /opt/intel/openvino/bin/setupvars.sh

#Set below Model Parameter
export ONNX_MODEL_NAME=Tiny-Yolov2_face.onnx
#-------

export MODEL_DIR=$(dirname $0)
export MODEL_OPT_DIR=$INTEL_CVSDK_DIR/deployment_tools
export ONNX_PATH=$MODEL_DIR/$ONNX_MODEL_NAME
export OUTPUT_DIR=$MODEL_DIR

#python3 $MODEL_OPT_DIR/model_optimizer/mo.py  --data_type FP32 --input_model $ONNX_PATH --output_dir $OUTPUT_DIR/FP32
python3 $MODEL_OPT_DIR/model_optimizer/mo.py  --data_type FP16 --input_model $ONNX_PATH --output_dir $OUTPUT_DIR/FP16