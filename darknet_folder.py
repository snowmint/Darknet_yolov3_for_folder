from ctypes import *
import math
import random
import os #for read folder
import sys #for use argv[]
import cv2 #for drawing bounding box
import time #for timing

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    
    # Timing #############################################################
    start = time.time()
    predict_image(net, im) #truly detect
    end = time.time()
    print("$Timing:[", end-start, "]$")
    ######################################################################
    
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    
    n = len(sys.argv)
    meta_path = str(sys.argv[1])
    cfg_path = str(sys.argv[2])
    weight_path = str(sys.argv[3])
    in_image_path = str(sys.argv[4])
    out_image_path = str(sys.argv[5])
    print("cfg:", cfg_path, "  weight:", weight_path, "  in_image_path:", in_image_path, "  out_image_path:", out_image_path)

    net = load_net(cfg_path.encode('utf-8'), weight_path.encode('utf-8'), 0)
    meta = load_meta(meta_path.encode('utf-8'))

    for filename in os.listdir(in_image_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            full_path_of_image = in_image_path+filename
            print("$Filename:[", filename ,"]$")

            r = detect(net, meta, full_path_of_image.encode('utf-8'))

            img = cv2.imread(full_path_of_image)
            len_r = len(r)
            print(len_r)
            
            for i in range(len_r):
                here_color = random_color()
                here_x, here_y, here_w, here_h = r[i][2]
                cv2.rectangle(img,(int(here_x*0.8), int(here_y*0.8)), (int(here_x+here_w), int(here_y+here_h)), here_color, 3)
                text = str([i][0]) + " | " + str(r[i][1])
                cv2.putText(img, text, (int(here_x), int(here_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, here_color, 1)
            if not os.path.isdir(out_image_path):
                os.makedirs(out_image_path)
                print("create out directory: " + out_image_path)

            cv2.imwrite(out_image_path+"predict_"+filename, img)

            print(r)
            print("==============================================================================")

    #net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    #meta = load_meta("cfg/coco.data")
    #r = detect(net, meta, "data/dog.jpg")
    

