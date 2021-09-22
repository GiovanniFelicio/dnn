import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import zipfile
import argparse
from utils import utils, opencvUtils
from utils.exceptions import VideoException
import imghdr

class Main():
    def __init__(self, cfg, weights, img, video, img_size=416, video_size=600, conf_thres=0.5, nms_thres=0.45):
        self.img = img
        self.video = video
        self.img_size = img_size
        self.video_size = video_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        labels_path = os.path.sep.join(['cfg', 'coco.names'])
        self.labels = open(labels_path).read().strip().split('\n')

        self.net = cv2.dnn.readNet(cfg, weights)

        np.random.seed(19)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]    

        if img is not None:
            self.__detectImage()
        elif video is not None:
            self.__detectVideo()

    def __detectVideo(self):
        directoryVideo = os.path.join('video', self.video)
        cap = cv2.VideoCapture(directoryVideo)
        conectado, video = cap.read()

        if not cap.isOpened():            
            raise VideoException('Error connecting video')

        width, height = opencvUtils.reshape(video)

        directoryPredictions = os.path.join('predictions', str(int(round(time.time() * 1000))))
        os.makedirs(directoryPredictions)

        path = os.path.join(directoryPredictions, 'resultado.avi')

        fourcc = cv2.VideoWriter_fourcc(*'XVID') # MP4V

        video_output = cv2.VideoWriter(path, fourcc, 24, (width, height))

        while (cv2.waitKey(1) < 0):            
            connected, frame = cap.read()
            if not connected:
                raise VideoException('Error connecting video')
            
            frame = cv2.resize(frame, (width, height))
            try:
                (h,w) = frame.shape[:2]
            except:
                print('Erro')
                continue

            layer_outputs = opencvUtils.blobImage(frame, self.net, self.ln)

            boxes, confiancas, idClasses, objs = self.__test(layer_outputs, [h,w])

            frame = opencvUtils.getDetections(objs, boxes, frame, '.jpg', self.labels, self.colors, idClasses, confiancas, True, False)

            video_output.write(frame)

        video_output.release()
        cv2.destroyAllWindows()

    def __detectImage(self):
        paths = utils.listDir(self.img)
        for i in paths:
            try:
                extension = os.path.splitext(i)[1]
                image = cv2.imread(i)
                image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            except:
                print('erron when load image: ' + i)
                continue
        
            h, w = image.shape[:2]

            layer_outputs = opencvUtils.blobImage(image, self.net, self.ln)

            boxes, confiancas, idClasses, objs = self.__test(layer_outputs, [w,h])

            image = opencvUtils.getDetections(objs, boxes, image, extension, self.labels, self.colors, idClasses, confiancas, True, False)

            opencvUtils.showImage(image)

    def __test(self, layer_outputs, dimensions):
        boxes = []
        confiancas = []
        idClasses = []
        w,h = dimensions

        for output in layer_outputs:
            for detection in output:
                boxes, confiancas, idClasses = utils.detections(detection, layer_outputs, self.conf_thres, boxes,confiancas, idClasses, [w,h])

        objs = cv2.dnn.NMSBoxes(boxes, confiancas, self.conf_thres, self.nms_thres)

        return boxes, confiancas, idClasses, objs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--img', type=str, help='path to images')
    parser.add_argument('--video', type=str, help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--video-size', type=int, default=600, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()

    main = Main(opt.cfg,
                opt.weights,
                opt.img,
                opt.video,
                img_size=opt.img_size,
                video_size=opt.video_size,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres)

'''
  imagem_cp = frame.copy() 
  net, frame, layerOutputs = blob_imagem(net, frame)
  caixas = []       
  confiancas = []   
  IDclasses = []    

  for output in layerOutputs:
    for detection in output:
      caixas, confiancas, IDclasses = deteccoes(detection, threshold, caixas, confiancas, IDclasses)

  objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, threshold_NMS)

  if len(objs) > 0:
    for i in objs.flatten():
      frame, x, y, w, h = funcoes_imagem(frame, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=False)
      objeto = imagem_cp[y:y + h, x:x + w]
'''