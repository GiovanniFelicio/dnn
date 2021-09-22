import numpy as np
import matplotlib.pyplot as plt
import os

def detections(detection, layer_outputs, conf_thres, boxes, confiancas, idClasses, imgShape):
    w, h = imgShape
    scores = detection[5:]
    classeId = np.argmax(scores)
    confianca = scores[classeId]
    if confianca > conf_thres:
        caixa = detection[0:4] * np.array([w, h, w, h])
        (centerX, centerY, width, height) = caixa.astype('int')

        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        boxes.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        idClasses.append(classeId)

    return boxes,confiancas,idClasses

def check_negative(n):
    if n < 0:
        return 0
    
    return n

def listDir(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]
