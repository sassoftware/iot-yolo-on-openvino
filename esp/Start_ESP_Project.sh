#!/bin/bash
#Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
#SPDX-License-Identifier: Apache-2.0

cd $(dirname $0)
source /opt/intel/openvino/bin/setupvars.sh

export DFESP_HOME=/opt/sas/SASEventStreamProcessingEngine/6.2
export LD_LIBRARY_PATH=$DFESP_HOME/lib:$DFESP_HOME/lib/tk:$DFESP_HOME/ssl/lib:$LD_LIBRARY_PATH
export DFESP_CONFIG=$DFESP_HOME/etc

export MODEL_HOME=../
export PYTHONPATH=.:/usr/lib/python3.6:/usr/lib/python3.6/site-packages:$MODEL_HOME/Functions:$PYTHONPATH

export MAS_PYPATH=/usr/bin/python3
export MAS_M2PATH=$DFESP_HOME/lib/tk/misc/embscoreeng/mas2py.py

$DFESP_HOME/bin/dfesp_xml_server -http 30001 -pubsub 30003 -model file://yoloV2OpenVINO.xml