import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import utils

def showImage(img, axis='off'):
    plt.imshow(img)
    plt.axis(axis)
    plt.show()


def reshape(obj, max_width=600):
    if obj.shape[1] > max_width:
        proporcao = obj.shape[1] / obj.shape[0]
        width = max_width
        height = int(width / proporcao)
    else:
        width = obj.shape[1]
        height = obj.shape[0]

    return width, height


def getDetections(objs, boxes, img, extension,labels, colors, idClasses, confiancas, put_bouding_boxe = False, save_detection = False):
    if len(objs) > 0:
        if save_detection:
            directory = os.path.join('predictions', str(int(round(time.time() * 1000))))
            os.makedirs(directory)
            
        idx = 0
        for i in objs.flatten():
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])           

            x = utils.check_negative(x)
            y = utils.check_negative(y)

            if save_detection:
                filename = str(idx) + extension
                path = os.path.join(directory, filename)
                saveDetection(img, [x,y,w,h], path)

            if put_bouding_boxe:
                putBoudingBoxes(img,labels, colors, idClasses, confiancas, i, [x,y,w,h])

            idx+=1

    return img

def putBoudingBoxes(img, labels, colors, idClasses, confiancas, position, dimensions):

    x,y,w,h = dimensions

    color = [int(c) for c in colors[idClasses[position]]]

    background = np.full((img.shape), (0,0,0), dtype=np.uint8)

    text = "{}: {:.4f}".format(labels[idClasses[position]], confiancas[position])

    cv2.putText(background, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    fx,fy,fw,fh = cv2.boundingRect(background[:,:,2])

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) 

    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, -1) 
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 3) 
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img

def save(img, path):
    if img is not None and path is not None:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def saveDetection(img, dimensions, path):
    if img is not None and dimensions is not None:
        x,y,w,h = dimensions
        detection = img[y:y + h, x:x + w]

        save(detection, path)

def blobImage(img, net, ln):
    inicio = time.time()
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    fim = time.time()

    print('YOLO levou: {:.2f} segundos'.format(fim - inicio))

    return layer_outputs