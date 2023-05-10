from flask_socketio import SocketIO
from flask import Flask, render_template, url_for, redirect, request
from random import random
from assis_function import get_sparse_opticalflow,focus_on_region,cal_ROI
from joblib import load
from threading import Thread, Event, Lock
import base64
import eventlet





###############################
import os
import platform
import time
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (Profile, check_file, check_img_size, check_imshow, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


############################################


global ip
ip = input('Enter ip address local machine: ')


app = Flask(__name__)
# socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)
socketio = SocketIO(app)
# thread = Thread()
# thread_stop_event = Event()
global value1
global value2
value1 = "Loading..."
value2 = "Loading..."
global idd
idd = 0
global list_user
list_users = []

#######################################






@smart_inference_mode()
def proc1(
        weights=ROOT / 'best_3.pt',  # model path or triton URL
        clf_weights=ROOT / 'svm/forest_decision.sav',
        transform=ROOT / 'transform/forest_scaler.bin',
        source=None,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=5,  # video frame-rate stride
        estimate=False,
        lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4),
):
#########################################
    tracks = []
    init_flag = True
    density_label = ["Sparse", "Medium", "Dense"]
    frame_id = 0
    global value1
    global idd
#########################################
    # try:
    if estimate:
        clf = load(clf_weights)
        sc = load(transform)
    source = str(source)
    if source.isnumeric():
        cam = int(source)
        points, view_mask = focus_on_region(cam)
    else:
        points, view_mask = focus_on_region(source)
    areaROI = cal_ROI(points)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(view_mask, source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, view_mask, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    get_prv_frame = False
    start = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        frame_id+=1
        if not get_prv_frame:
            prev_frame = im0s.copy()
            get_prv_frame = True
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None] 
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                opf, tracks, vel = get_sparse_opticalflow(prev_frame[i], im0, tracks, 4, lk_params, feature_params)
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                opf, tracks, vel = get_sparse_opticalflow(prev_frame, im0, tracks, 4, lk_params, feature_params)
                vel = round(vel, 5)
            p = Path(p)  # to Path
            annotator = Annotator(opf, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    annotator.box_label(xyxy, None, color=colors(c, True))
            opf = annotator.result()
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            end = time.time()

            if estimate :
                if init_flag:
                    density = "Loading..."
                    if int(end - start)%15==0 and int(end - start)!=0:
                        sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                        density = "Density: "+ density_label[int(clf.predict(sample))-1]
                        init_flag = False

                elif int(end - start)%5==0:
                    sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                    density = str(density_label[int(clf.predict(sample))-1])
                cv2.putText(opf, density, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # cv2.putText(opf, "Time stamp: "+str(int(end - start)), (int(opf.shape[1]/2+10), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            prev_frame = im0s.copy()
            value1 = density
            opf = cv2.resize(opf, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', opf)[1].tobytes()
            frame = base64.encodebytes(frame).decode("utf-8")
            socketio.emit(f'result1', {'dens': density,'obs': frame}, namespace='/admin')
            eventlet.sleep(0.001)
@smart_inference_mode()
def proc2(
        weights=ROOT / 'best_3.pt',  # model path or triton URL
        clf_weights=ROOT / 'svm/forest_decision.sav',
        transform=ROOT / 'transform/forest_scaler.bin',
        source=None,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=5,  # video frame-rate stride
        estimate=False,
        lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4),
):
#########################################
    tracks = []
    init_flag = True
    density_label = ["Sparse", "Medium", "Dense"]
    frame_id = 0
    global value2
    global idd
#########################################
    if estimate:
        clf = load(clf_weights)
        sc = load(transform)
    source = str(source)
    if source.isnumeric():
        cam = int(source)
        points, view_mask = focus_on_region(cam)
    else:
        points, view_mask = focus_on_region(source)
    areaROI = cal_ROI(points)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(view_mask, source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, view_mask, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    get_prv_frame = False
    start = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        frame_id+=1
        if not get_prv_frame:
            prev_frame = im0s.copy()
            get_prv_frame = True
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None] 
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                opf, tracks, vel = get_sparse_opticalflow(prev_frame[i], im0, tracks, 4, lk_params, feature_params)
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                opf, tracks, vel = get_sparse_opticalflow(prev_frame, im0, tracks, 4, lk_params, feature_params)
                vel = round(vel, 5)
            p = Path(p)  # to Path
            annotator = Annotator(opf, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    annotator.box_label(xyxy, None, color=colors(c, True))
            opf = annotator.result()
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            end = time.time()

            if estimate:
                if init_flag:
                    density = "Loading..."
                    if int(end - start)%15==0 and int(end - start)!=0:
                        sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                        density = "Density: "+ density_label[int(clf.predict(sample))-1]
                        init_flag = False

                elif int(end - start)%5==0:
                    sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                    density = str(density_label[int(clf.predict(sample))-1])
                cv2.putText(opf, density, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # cv2.putText(opf, "Time stamp: "+str(int(end - start)), (int(opf.shape[1]/2+10), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            prev_frame = im0s.copy()
            value2 = density
            opf = cv2.resize(opf, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', opf)[1].tobytes()
            frame = base64.encodebytes(frame).decode("utf-8")
            socketio.emit(f'result2', {'dens': density,'obs': frame}, namespace='/admin')
            eventlet.sleep(0.001)

def transmit1():
    global result
    params = result
    params['clf_weights'] = 'svm/smote_forest.sav'
    params['transform'] = 'transform/smote_stdscaler.bin'
    params['imgsz'] = [640, 640]
    params['classes'] = None
    params['vid_stride'] = int(params['vid_stride'])
    return proc1(**params)
    
def transmit2():
    global result
    params = result
    params['clf_weights'] = 'svm/smote_forest.sav'
    params['transform'] = 'transform/smote_stdscaler.bin'
    params['imgsz'] = [640, 640]
    params['classes'] = None
    params['vid_stride'] = int(params['vid_stride'])
    return proc2(**params)

@app.route('/')
def index():
    global ip
    domain = "http://"+ip+":9099/moderator"
    return render_template("index_loop.html", domain=domain)

@app.route('/moderator', methods=['POST', 'GET'])
def res():
    global result
    global ip
    if request.method=='POST':
        result = request.form.to_dict()
        return render_template("loop.html", ip=ip)
    return redirect(url_for('client_side'))

@app.route('/client', methods=['GET'])
def client_side():
    global value1, value2
    return render_template("index3_1_get.html", value=[value1,value2], broadcast=True)

@socketio.on('connect', namespace='/admin')
def res_connect():
    global idd
    idd+=1
    if idd==1:
        socketio.start_background_task(transmit1)
    if idd==2:
        socketio.start_background_task(transmit2)
    return render_template('index_loop.html')

@socketio.on('disconnect', namespace='/admin')
def res_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app, host=ip, port=9099)

# cmd: ipconfig -> kéo xuống cuối cùng gán ip = IPv4_address (ví dụ: ip = 192.168.1.77)
# Chạy trên web: http:// + ip + :9099/ (Chạy video 1, chạy video 2 vẫn nhập lại domain này)
# Chọn Video (chỉ có 2 cái để test) -> xem các video có sãn trong folder vids
# Để ý task bar, chương trình sẽ hiện ảnh lên yêu cầu người dùng chọn vùng
# -> Click chuột khoanh vùng cần detect và nhấn "d" để chạy chương trình
# Sau khi web chạy, trong android, ta sửa địa chỉ ip thành "http:// + ip + :9099/client"
