:: Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
:: SPDX-License-Identifier: Apache-2.0
@ECHO OFF
ECHO SELECT MODEL:
ECHO 1. Face Detection
ECHO 2. Object Detection
ECHO.

CHOICE /C 12 /M "Enter your choice:"

:: Note - list ERRORLEVELS in decreasing order
IF ERRORLEVEL 2 GOTO=OBJECT
IF ERRORLEVEL 1 GOTO=FACE

:OBJECT
ECHO CPU
set MODEL_DIR=%cd%\Objects
set OPENVINO_MODEL_NAME=Tiny-Yolov2
GOTO DEVICE

:FACE
ECHO FACE
set MODEL_DIR=%cd%\Face
set OPENVINO_MODEL_NAME=Tiny-Yolov2_face
GOTO DEVICE


:DEVICE
ECHO.
ECHO SELECT DEVICE:
ECHO 1. CPU
ECHO 2. GPU
ECHO 3. MYRIAD
ECHO.

CHOICE /C 123 /M "Enter your choice:"

:: Note - list ERRORLEVELS in decreasing order
IF ERRORLEVEL 3 GOTO MYRIAD
IF ERRORLEVEL 2 GOTO=GPU
IF ERRORLEVEL 1 GOTO=CPU


:MYRIAD
ECHO MYRIAD
set INF_DEVICE=MYRIAD
GOTO Run

:GPU
ECHO GPU
set INF_DEVICE=GPU
GOTO Run

:CPU
ECHO CPU
set INF_DEVICE=CPU
GOTO Run

:Run


:: PRECISION
set PRECISION=FP16
::-------

call "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
python SAS_YoloV2_OpenVINO_Demo_v2.py -i cam -m %MODEL_DIR% -n %OPENVINO_MODEL_NAME% -d %INF_DEVICE% -p %PRECISION%
pause