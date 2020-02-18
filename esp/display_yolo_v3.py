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

#Description: Display ESP results for OpenVino inferencing Tiny Yolo V2 Model
#Require ESP installed on the same machine where runs and ENV variables set
#Leverage OpenCV to display results
#See command line parameters for additional information.

import os, platform, sysimport datetime, time, signal

import numpy as np
import cv2
import hashlib
from collections import deque
from argparse import ArgumentParser, SUPPRESS

## check ESP environment variables
if "DFESP_HOME" in os.environ:
  dfesphome = os.environ["DFESP_HOME"]
else:
  print("Error: Environment variable DFESP_HOME not set. Abort!")
  sys.exit(1)

if platform.system() == 'Linux':
  sys.path.append(dfesphome + '/lib')
else:
  sys.path.append(dfesphome + '/bin')


import ezpubsub as ezps

from collections import deque
deltaTimeQueue = deque()
prevTime = -1

args = None
cv2_win_name = "SAS Tiny YOLO V2 Viewer"

videoWriter = None
loop = 0
maxloop = 3000

stayInLoop = True
currentImg = None
toberesized = False


def build_argparser():
  parser = ArgumentParser(add_help=False)
  args = parser.add_argument_group('Options')
  args.add_argument('-C', '--oneColor', nargs='?', const=True, default=False, type=bool, required=False,
                    help='label background using one color')
  args.add_argument('-w', '--width', default=None, type=int, required=False,  help='Output width of image. Scale will be preserved')
  #args.add_argument('-h', '--height', default=480, type=int, required=False,  help='output height of image')
  args.add_argument('-t', '--probThres', default=0.25, type=float, required=False,
                    help='probability threshold to filter')
  args.add_argument('-i', '--ipAddr', default='localhost', type=str, required=False,
                    help='Ip Address default: localhost')
  args.add_argument('-p', '--port', default='30003', type=str, required=False,  help='Pub/Sub Port default: 30003')
  args.add_argument('-e', '--espProj', default='yoloV2OpenVINO', type=str, required=False,
                    help='Project name default: yoloV2OpenVINO')
  args.add_argument('-q', '--cq', default='cq', type=str, required=False,
                    help='Continuous Query name default: cq')
  args.add_argument('-s', '--sw', default='w_score', type=str, required=False,
                    help='Score windows name default: w_score')
  args.add_argument('-f', '--frame', default='image_in', type=str, required=False,
                    help='Image field name default: image_in')
  args.add_argument('-a', '--autosize', nargs='?', const=True, default=False, type=bool, required=False,
                      help='Set CV window in autosize mode. Default True')
  args.add_argument('--fullscreen', nargs='?', const=True, default=False, type=bool, required=False,
                      help='Set CV window in fullsize (override autosize mode). Default False')
  args.add_argument('--showfps', nargs='?', const=True, default=False, type=bool, required=False,
                      help='Set CV window in fullsize (override autosize mode). Default False')
  args.add_argument('--flip', nargs='?', const=True, default=False, type=bool, required=False,
                      help='Flip camera (Mirror mode). Default False')
  args.add_argument('-v', '--video_out', default=None,
                    type=str, required=False,  help='Output Video path')
  args.add_argument('--noshow', nargs='?', const=True, default=False, type=bool, required=False,
                    help='Hide OpenCV output. Usefull to register video from a remote server')
  args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                    help='Show this help message and exit.')

  return parser


def videoOutput(frame_in, exitfunc = False):
    global videoWriter
    global loop

    if loop > maxloop or exitfunc:
      videoWriter.release()
      videoWriter = None
      loop = 0
      return

    if videoWriter is None:
      print("Video writer initialization.")
      loop += 1
      fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
      videoWriter = cv2.VideoWriter()
      framerate = 30 / 4
      width = frame_in.shape[1]  # for detection model
      height = frame_in.shape[0]
      name = args.video_out #output_video + extension
      success = videoWriter.open(name, fourcc,
                                 framerate, (width, height), True)
    if frame_in is not None:
      videoWriter.write(frame_in)


def highlightImage(row, objectId, fps):
  global currentImg
  tableau10 = [(31, 119, 180), (255, 127, 14),
               (127, 127, 127), (188, 189, 34),
               (148, 103, 189), (140, 86, 75),
               (227, 119, 194), (44, 160, 44),
               (214, 39, 40), (23, 190, 207)]

  color_palette = tableau10
  n_colors = len(color_palette)

  if args.frame in row.keys():
    imageBlob = row[args.frame]
    if imageBlob is None:
      currentImg = None
      return
  else:
    currentImg = None
    return

  #Each received row contains only the data of a single detected object
  #If there are multiple object detected, the ObjectID is also incremented and return to 0 when another frame is analyzed.
  #This code store the image each time an objectId == 0 is found and keep drawing bounding box on the same image till
  #all detected object are received.
  if objectId == 0:
    nparr = np.frombuffer(imageBlob, dtype=np.uint8)
    currentImg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if args.width is not None:
      image_h, image_w, _ = currentImg.shape
      height = image_h * (args.width / image_w)
      currentImg = cv2.resize(currentImg, (int(args.width), int(height)), cv2.INTER_LINEAR)

    if args.flip:
      #Flip horizzontaly Mirror effect
      currentImg = cv2.flip(currentImg, 1)

  if 'nObjects' in row.keys():
    numberOfObjects = row['nObjects']
    if numberOfObjects == 0:
      return
  else:
    return

  image_h, image_w, _ = currentImg.shape
  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.6
  thickness = 1

  if objectId == 0:
    ## put current timestamp
    text = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    #font_scale = 0.6
    size = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width  = int(size[0][0])
    text_height = int(size[0][1])
    cv2.putText(currentImg, text, (image_w - text_width - 2, image_h - text_height), font_face, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(currentImg, text, (image_w - text_width - 2, image_h - text_height), font_face, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

    ## put FPS
    text = "FPS=%.2f FrameId=%d" % (fps, row['id'])
    #font_scale = 1
    size = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width  = int(size[0][0])
    text_height = int(size[0][1])
    cv2.putText(currentImg, text, (+5, image_h - text_height), font_face, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(currentImg, text, (+5, image_h - text_height), font_face, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)


  obj = row['classes']
  prob = float(row['scores'])
  probability = "(" + str(round(prob * 100, 1)) + "%)"
  x = float(row['x_box'])
  y = float(row['y_box'])
  width = float(row['w_box'])
  height = float(row['h_box'])

  if prob < args.probThres:
    return

  if args.oneColor:
    color_idx = 0
  else:
    color_idx = int(hashlib.sha1(obj.encode()).hexdigest(), 16) % n_colors

  box_color = (color_palette[color_idx][2], color_palette[color_idx][1], color_palette[color_idx][0]) #(b,g,r)

  x_min = int(image_w * (x - width / 2))
  y_min = int(image_h * (y - height/ 2))
  x_max = int(image_w * (x + width / 2))
  y_max = int(image_h * (y + height/ 2))

  if args.flip:
    # flip coordinates
    x_min_f = image_w - x_max
    x_max = image_w - x_min
    x_min = x_min_f

  ## draw bounding box
  cv2.rectangle(currentImg, (x_min, y_min), (x_max, y_max), box_color, 1)

  ## draw object label
  text = obj.strip() + " " + probability
  if sum(box_color)/3 < 140:
    text_color = (255, 255, 255) #(b,g,r)
  else:
    text_color = (16, 16, 16) #(b,g,r)
  size = cv2.getTextSize(text, font_face, font_scale, thickness)

  text_width  = int(size[0][0])
  text_height = int(size[0][1])
  line_height = size[1]
  margin = 2

  text_x = x_min + margin
  text_y = y_min - line_height - margin

  # draw a filled rectangle around text
  cv2.rectangle(currentImg, (text_x - margin, text_y + line_height + margin),
                (text_x + text_width + margin, text_y - text_height - margin), box_color, -1)
  cv2.putText(currentImg, text, (text_x, text_y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)


def subCallbackCbFunc(row):
  global prevTime
  global deltaTimeQueue
  global stayInLoop
  global toberesized

  frameId = row['id']
  objectId = row['object_id']

  ## print out timing log     
  curDateTime = datetime.datetime.now()
  curTime = time.perf_counter()
  if prevTime != -1:
    deltaTime = (curTime - prevTime) * 1000
    deltaTimeQueue.appendleft(deltaTime)
    if len(deltaTimeQueue) > 100:
      deltaTimeQueue.pop()
    avgDeltaTime = sum(deltaTimeQueue)/len(deltaTimeQueue)
    fps = 1000 / avgDeltaTime
    print("FrameId: %d\t Current Time: %s\tDelta Time: %.2fms\tAvg Delta Time: %.2fms" % (frameId, str(curDateTime),
                                                                                          deltaTime, avgDeltaTime))
  else:
    deltaTime = 0
    avgDeltaTime = 0
    fps = 0
    print("FrameId: %d\t Current Time: %s" % (frameId, str(curTime)))

  prevTime = curTime

  if (currentImg is not None) and (objectId == 0):
    if args.video_out is not None:
      videoOutput(currentImg)

    if not (display is None or len(display) == 0):
      if not args.noshow:
        cv2.imshow(cv2_win_name, currentImg)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key to stop
          if args.video_out is not None:
            videoOutput(None, True)
          stayInLoop = False

  highlightImage(row, objectId, fps)

  #fix small windows issue in case of cv2.WINDOW_NORMAL
  if toberesized:
    image_h, image_w, _ = currentImg.shape
    cv2.resizeWindow(cv2_win_name, image_w, image_h)
    toberesized = False

  return

def subCallbackErr(err):
  global stayInLoop
  print("Error:" + str(err))
  stayInLoop = False

def main():
  global toberesized
  if args.fullscreen:
      cv2.namedWindow(cv2_win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      cv2.setWindowProperty(cv2_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  elif args.autosize:
      cv2.namedWindow(cv2_win_name, cv2.WINDOW_AUTOSIZE)
  else:
     cv2.namedWindow(cv2_win_name, cv2.WINDOW_NORMAL)
     toberesized=True

  try:
      sub = ezps.Subscriber(url, on_event=subCallbackCbFunc, on_error=subCallbackErr)
      while stayInLoop:
          time.sleep(0.02)

  except KeyboardInterrupt:
      if args.video_out is not None:
        videoOutput(None, True)
  except SystemExit:
      if args.video_out is not None:
        videoOutput(None, True)
  finally:  
      raise SystemExit


if __name__ == '__main__':
  if "DISPLAY" in os.environ:
    display = os.environ["DISPLAY"]
    print("Note: Images will be displayed at " + display)
  elif platform.system() == "Windows":
    print("Note: Images will be displayed at main display")
    display = "Windows"
  else:
    print("Warning: Environment variable DISPLAY not set. No images will be shown.")
    display = None

  args = build_argparser().parse_args()

  url = "dfESP://" + args.ipAddr + ":" + args.port + "/" + args.espProj + "/" + args.cq + "/" + args.sw
  print("Connecting to:" + url)

  main()
