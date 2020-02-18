#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved. 
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# ----------------------------------------- Libraries import -------------------------------
#Backward compatibility with python 2.6
from __future__ import print_function, division

#Import OpenCV before mali as different order will cause issue on RockPro64
import cv2
import numpy as np
import sys, os, time
import math
import logging
import importlib
from argparse import ArgumentParser, SUPPRESS

# ----------------------------------------- Inference Runtime import --------------------------
from openvino.inference_engine import IENetwork, IEPlugin
framework = "OpenVINO"

#Run in ESP mode, default is true to be set to false by __main__
isESP = False

# ----------------------------------------- Global Variables -------------------------------


#Global flags for async mode
is_async_mode = False
cur_request_id = 0
next_request_id = 1

#Global variable needed to maintain inference object between ESP call
net = None
exec_net = None
#Global variable needed to persist previous inference result if new are not yet ready
box = None
classes = None
scores = None


#Set real script path so you could call the script from another directory

#Logs Settings
logfilename='python.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler(logfilename, mode='w'),
                        logging.StreamHandler()])
log = logging.getLogger()


# ---------------------------------Load Configurations and Model Parameters---------------------
def loadmodelconfig(model_dir, model_config_name):
    sys.path.append(model_dir)
    if model_config_name.endswith((".py")): model_config_name=model_config_name[:-3]
    return importlib.import_module(model_config_name, package=None)


# ---------------------------------Check if we are running from ESP ---------------------
if __name__ != '__main__':
    isESP = True
    import _config as cf
    is_async_mode = cf.is_async_mode

    #import model parameters
    param = loadmodelconfig(cf.model_dir, cf.model_config_name)



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--modeldir", help="Path model dir. Default ./Face.",
                      required=False, default="./Face", type=str)
    args.add_argument("-n", "--modelname", help="Name of the model .xml and .py configuration file. Default Tiny-Yolov2_face.",
                      required=False, default="Tiny-Yolov2_face", type=str)
    args.add_argument("-p", "--precision", help="Model Precision FP16 or FP32. Default FP16.",
                      required=False, default="FP16", type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file or camera device ID. (You could "
                                            "specify 'cam' to work with default device 0)", required=False,
                      default="cam" , type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD are"
                           " acceptable. The sample will look for a suitable plugin for the device specified. "
                           "The default value is CPU", default="CPU", type=str)
    return parser



def sigmoid(x):
    mysigmoid= lambda x: 0.5*math.tanh(0.5*x) + 0.5
    return  mysigmoid(x)


def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


# Malisiewicz et al.
# Adapted version from Python port by Adrian Rosebrock
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = (x - w / 2)
    x2 = (x + w / 2)
    y1 = (y - h / 2)
    y2 = (y + h / 2)

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def column(matrix, i):
    return [row[i] for row in matrix]


def decodeYoloOutput(out, confidenceThresh=0.4):
    blockSize = 32
    numClasses = param.yolov2_classes
    _, gridHeight, gridWidth = out.shape

    boxes = np.empty((0, 4), float)
    objects_class = np.asarray([], float)
    objects_score = np.asarray([])

    for cy in range(0, gridHeight):
        for cx in range(0, gridWidth):
            for b in range(0, param.yolov2_boxPerCell):
                channel = b * (numClasses + 5)
                confidence = sigmoid(out[channel + 4][cy][cx])

                if confidenceThresh < confidence:
                    classes = np.zeros(numClasses)
                    for c in range(0, numClasses):
                        classes[c] = out[channel + 5 + c][cy][cx]
                    classes = softmax(classes)
                    detectedClass = classes.argmax()

                    if confidenceThresh < classes[detectedClass] * confidence:
                        tx = out[channel][cy][cx]
                        ty = out[channel + 1][cy][cx]
                        tw = out[channel + 2][cy][cx]
                        th = out[channel + 3][cy][cx]

                        x = (float(cx) + sigmoid(tx)) * blockSize
                        y = (float(cy) + sigmoid(ty)) * blockSize

                        w = np.exp(tw) * blockSize * param.yolov2_anchors[2 * b]
                        h = np.exp(th) * blockSize * param.yolov2_anchors[2 * b + 1]

                        box = np.asarray([(x, y, w, h)])
                        boxes = np.append(boxes, box, axis=0)
                        objects_class = np.append(objects_class, param.yolov2_label[detectedClass])
                        objects_score = np.append(objects_score,  classes[detectedClass] * confidence)

                        log.info("Detect: " + param.yolov2_label[detectedClass] + ", confidence: " + str(confidence) +
                              ", class prob: " + str(classes[detectedClass]) + ", overall prob: "
                              + str(classes[detectedClass] * confidence))

    #check if any box has been detected
    if len(boxes):
        pick = non_max_suppression_fast(boxes, overlapThresh=0.45)

        boxes = boxes [pick]
        objects_class = objects_class[pick]
        objects_score = objects_score[pick]
        # make image coordinate size independent by dividing all conten of box array for current image size declared
        boxes = boxes / param.yolov2_yoloimgsize

    return boxes, objects_class, objects_score


def init_model(model, device, plugin_dir, cpu_extension):
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V2 based single input topologies"
    assert len(net.outputs) == 1, "Sample supports only YOLO V2 based single output topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net, num_requests=2)

    return net, exec_net


# ----------------------------------------------  Preparing inputs -----------------------------------------------
def preprocess_input(_image_):
    # start_time = time.time()
    input_blob = next(iter(net.inputs))

    # Read and pre-process input images
    input_shape = net.inputs[input_blob].shape

    log.info("ESP variable is:" + str(isESP))
    if isESP:
        in_frame = np.frombuffer(_image_, dtype=np.uint8)
        in_frame_np = cv2.imdecode(in_frame, cv2.IMREAD_COLOR)
        log.info(in_frame_np.shape)
    else:
        in_frame_np = _image_
        log.info(in_frame_np.shape)


    # Input normalization as required by SAS VDMML Exported ONNX
    # Divide the array value by 255 (color max value) to normalize the vector with value from 0 to 1
    in_frame_np = in_frame_np / 255

    in_frame_np = in_frame_np.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame_np = in_frame_np.reshape(input_shape)

    return input_blob, in_frame_np


def score(_image_):
    "Output: object_id, nObjects, x_box, y_box, w_box, h_box, classes, scores"

    # ----------------------------------------- 0. Import Global Variables -----------------------------------------
    #Gobal Variables
    global net
    global exec_net
    global cur_request_id
    global next_request_id
    global box
    global classes
    global scores

    # ----------------------------------------- 2. Loading model to the plugin -----------------------------------------
    if exec_net is None or net is None:
        net, exec_net = init_model(os.path.join(cf.model_dir, cf.precision, cf.model_name), cf.inf_device,
                                   cf.plugin_dir, cf.cpu_extension)
    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    # Start inference
    log.info("Start inference")
    if is_async_mode:
        request_id = next_request_id
    else:
        request_id = cur_request_id

    # ---  ENUMERATOR wait value form ie_iinfer_request.hpp --------
    # Wait until inference result becomes available (block current thread) * /
    #   RESULT_READY = -1,
    # Doesn't block or interrupt current thread and immediately returns inference status */
    #   STATUS_ONLY = 0,
    #   INFER_NOT_STARTED = -11,

    if exec_net.requests[request_id].wait(0) == -11:
        input_blob, in_frame_np = preprocess_input(_image_)
        exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame_np})

    if exec_net.requests[cur_request_id].wait(0) == 0:
        output = exec_net.requests[cur_request_id].outputs
        layer_name, out_blob = next(iter(output.items()))
        box, classes, scores = decodeYoloOutput(out_blob[0], param.yolov2_detectionThreshold)
        input_blob, in_frame_np = preprocess_input(_image_)
        exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame_np})

    if is_async_mode:
        cur_request_id, next_request_id = next_request_id, cur_request_id

    if (box is None) or (classes is None) or (scores is None):
        return 0, 0, None, None, None, None, None, None

    if len(classes) == 0:
        return 0, 0, None, None, None, None, None, None

    object_id = list(range(0, len(classes)))


    return object_id, len(classes), \
           column(box.tolist(), 0), column(box.tolist(), 1), column(box.tolist(), 2), column(box.tolist(), 3),\
           classes.tolist(), scores.tolist()


def highlightImage(num_items, x_list, y_list, w_list, h_list, classes, scores, image_in_current):

    image_h, image_w, _ = image_in_current.shape
    img_np = image_in_current

    #horizontal flip
    img_np = cv2.flip(img_np, 1)

    for i in range(0, num_items):
        try:
            obj_name = classes[i]
            prob = scores[i]
            probability = "(" + str(round(prob * 100, 2)) + "%)"
            x = x_list[i]
            y = y_list[i]
            w = w_list[i]
            h = h_list[i]

            x_min = int(image_w * (x - w / 2))
            x_max = int(image_w * (x + w / 2))
            y_min = int(image_h * (y - h / 2))
            y_max = int(image_h * (y + h / 2))

            #flip coordinates
            x_min_f = image_w - x_max
            x_max= image_w - x_min
            x_min = x_min_f

            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (155, 255, 255), 1)
            cv2.putText(img_np, obj_name + " " + probability, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h, (255, 255, 255), 1, cv2.LINE_AA)

        except:
            continue

    return img_np


def main_camera(input_stream):
    global is_async_mode
    cap = cv2.VideoCapture(input_stream)
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    if number_input_frames != 1:
        ret, frame = cap.read()
    else:
        is_async_mode = False

    # ----------------------------------------------- 1. Manage Camera Input -----------------------------------------------
    print("To close the application, press 'CTRL+C' or any key with focus on the output window")
    while cap.isOpened():
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()

        if not ret:
            break

        if is_async_mode:
            in_frame = cv2.resize(next_frame, (param.yolov2_yoloimgsize, param.yolov2_yoloimgsize))
        else:
            in_frame = cv2.resize(frame, (param.yolov2_yoloimgsize, param.yolov2_yoloimgsize))

        object_id, nobjects, x_box, y_box, w_box, h_box, classes, scores = score(in_frame)
        frame = highlightImage(nobjects, x_box, y_box, w_box, h_box, classes, scores, frame)

        # #Draw performance stats over frame
        async_mode_message = "Async mode is on, press TAB for off. Processing request {}".format(cur_request_id) if is_async_mode else \
             "Async mode is off, press TAB for on. Processing request {}".format(cur_request_id)
        exit_message = "Press ESC to close this demo."

        cv2.putText(frame, exit_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        origin_im_size = frame.shape[:-1]
        cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (10, 10, 200), 1)


        cv2.imshow("SAS Tiny YoloV2 Detection Results", frame)

        if is_async_mode:
           frame = next_frame

        key = cv2.waitKey(1)
        # ESC key
        if key == 27:
            break
        # Tab key
        if key == 9:
            exec_net.requests[cur_request_id].wait()
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = build_argparser().parse_args()
    class MyCFG:
        def __init__(self, model_dir, model_name, model_config_name, precision, inf_device, plugin_dir=None,
                     cpu_extension = None, is_async_mode=False):
            self.model_dir = model_dir
            self.model_name = model_name
            self.model_config_name = model_config_name
            self.precision = precision
            self.inf_device = inf_device
            self.plugin_dir = plugin_dir
            self.cpu_extension = cpu_extension
            self.is_async_mode = is_async_mode

    cf = MyCFG(args.modeldir, args.modelname + ".xml", args.modelname +".py", args.precision, args.device)
    param = loadmodelconfig(cf.model_dir, cf.model_config_name)
    input_stream = 0 if args.input == "cam" else args.input
    sys.exit(main_camera(input_stream,) or 0)
