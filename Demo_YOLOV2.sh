#!/bin/bash
#Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
#SPDX-License-Identifier: Apache-2.0
echo "SELECT MODEL:"
PS3='Please enter your choice: '
options=("Face" "Object")
select opt in "${options[@]}"
do
    case $opt in
        "Face")
            echo "you chose choice Face"
            export MODEL_DIR="$(dirname $0)/Face"
            export OPENVINO_MODEL_NAME=Tiny-Yolov2_face
	    break
            ;;
        "Object")
            echo "you chose choice Object"
            export MODEL_DIR="$(dirname $0)/Objects"
            export OPENVINO_MODEL_NAME=Tiny-Yolov2
	    break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

echo "SELECT DEVICE:"
PS3='Please enter your choice: '
options=("CPU" "GPU" "MYRIAD")
select opt in "${options[@]}"
do
    case $opt in
        "CPU")
            echo "you chose choice CPU"
            export INF_DEVICE=CPU
	    break
            ;;
        "GPU")
            echo "you chose choice GPU"
            export INF_DEVICE=GPU
	    break
            ;;
        "MYRIAD")
            echo "you chose choice MYRIAD"
            export INF_DEVICE=MYRIAD
	    break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

# PRECISION
export PRECISION=FP16
#-------


source /opt/intel/openvino/bin/setupvars.sh
echo $MODEL_DIR -n $OPENVINO_MODEL_NAME -d $INF_DEVICE -p $PRECISION
python3 SAS_YoloV2_OpenVINO_Demo_V2.py -i cam -m $MODEL_DIR -n $OPENVINO_MODEL_NAME -d $INF_DEVICE -p $PRECISION
