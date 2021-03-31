import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet('weights\yolov4-tiny-custom_best.weights','cfg\yolov4-tiny-custom.cfg')
    classes = []
    with open('classes\classes.txt','r') as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(88, 3))
    return net, classes, colors, output_layers

def load_img(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def cropping_image(image):
    height, width, channels = image.shape
    net, classes, colors, output_layers = load_yolo()
    blob, outputs = detect_objects(image, net, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    nms_boxes = []
    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors[i]
            label = str(classes[class_ids[i]])
            crop_img = image[y:y+h,x:x+w]
            nms_boxes.append(boxes[i])
            labels.append(label)
    return nms_boxes,labels
    
def draw_labels(nms_boxes, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(nms_boxes)):
        x, y, w, h = nms_boxes[i]
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        label = str(classes[class_ids[i]])
        cv2.putText(img, label, (x, y-5), font, 1, color,1)
    return img
