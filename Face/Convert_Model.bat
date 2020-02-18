:: Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
:: SPDX-License-Identifier: Apache-2.0
call "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

::Set below Model Parameters
set ONNX_MODEL_NAME=Tiny-Yolov2_face.onnx
::-------

set MODEL_DIR=%cd%
set MODEL_OPT_DIR=%INTEL_CVSDK_DIR%/deployment_tools
set ONNX_PATH=%MODEL_DIR%/%ONNX_MODEL_NAME%
set OUTPUT_DIR=%MODEL_DIR%

::python "%MODEL_OPT_DIR%/model_optimizer/mo.py"  --data_type FP32 --input_model "%ONNX_PATH%" --output_dir "%OUTPUT_DIR%/FP32"
python "%MODEL_OPT_DIR%/model_optimizer/mo.py"  --data_type FP16 --input_model "%ONNX_PATH%" --output_dir "%OUTPUT_DIR%/FP16"
