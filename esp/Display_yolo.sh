#!/bin/bash
#Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
#SPDX-License-Identifier: Apache-2.0

cd $(dirname $0)
source /opt/intel/openvino/bin/setupvars.sh

export DFESP_HOME=/opt/sas/SASEventStreamProcessingEngine/6.2
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$DFESP_HOME/lib:$DFESP_HOME/lib/tk:$DFESP_HOME/ssl/lib:$LD_LIBRARY_PATH
python3 display_yolo_v3.py -w 1280 --flip
