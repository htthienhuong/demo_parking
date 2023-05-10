 # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
from shapely.geometry.polygon import Polygon
import numpy as np
from scipy.spatial import distance as dist
import argparse
import os
import platform
import time
import sys
from pathlib import Path
import csv
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


from joblib import load



def get_sparse_opticalflow(prev_gray, frame, tracks, track_len, lk_params, feature_params):

    vis = frame.copy()
    mm1 = 0
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    vel=0

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 80), -1)
        tracks = new_tracks
        ptn1 = 0
        for tr in tracks:
            ptn1 += 1
            mm1 += dist.euclidean(tr[0], tr[1])
            vel = mm1/ptn1
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (255, 0, 80))

    mask = np.zeros_like(frame_gray)
    mask[:] = 255
    for x, y in [np.int32(tr[-1]) for tr in tracks]:
        cv2.circle(mask, (x, y), 3, 0, -1)
    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])
    return vis, tracks, vel


def handle_left_click (event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (255,255,255), -1)
    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 255, 255), thickness=2)
    return frame

def focus_on_region(source):
    points=[]
    cap = cv2.VideoCapture(source)
    while (True):
        ret, first_frame = cap.read()

        first_frame = draw_polygon(first_frame, points)

        if  cv2.waitKey(0) & 0xFF == ord('d'): 
            points.append(points[0])

            cal_mask = np.zeros_like(first_frame[:, :, 0])
            view_polygon = np.array(points)
            cal_polygon = view_polygon

            cv2.fillConvexPoly(cal_mask, view_polygon, 1)

            cv2.destroyAllWindows()
            return points, cal_mask

        cv2.imshow("Frame", first_frame)
        cv2.setMouseCallback("Frame", handle_left_click, points)

def save_csv(csv_file, header, data):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    f.close()


def cal_ROI(poly):
    polygon = Polygon(poly)
    return polygon.area
# Dùng yolo: phân biệt người và bikerider
# Dùng optical flow: đo mật độ lúc đông vì yolo không detect tốt lúc đông

# @smart_inference_mode()
def run(
        weights=ROOT / 'best_3.pt',  # model path or triton URL
        clf_weights=ROOT / 'svm/forest_decision.sav',
        transform=ROOT / 'transform/forest_scaler.bin',
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # view_img=False,  # show results
        # save_txt=False,  # save results to *.txt
        # nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        # project=ROOT / 'runs/detect',  # save results to project/name
        # name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        estimate=False,
        lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),

        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4)
        # feature_params = dict( maxCorners = 500,
        #     qualityLevel = 0.3,
        #     minDistance = 7,
        #     blockSize = 7 )
):
#########################################

    tracks = []
    init_flag = True
    density_label = ["Sparse", "Medium", "Dense"]
    # Num.Boxes: Số lượng xe (opf detect tất cả không chỉ xe)
    # Num.Features: Để đo vận tốc và độ phân tán của các xe
    # Vel: Vận tốc xe 
    # ROI: Diện tích
    # Categories: Label

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

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(view_mask, source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, view_mask, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    get_prv_frame = False

    start = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        
        
        # print("Out loop")
        if not get_prv_frame:

            prev_frame = im0s.copy()

            get_prv_frame = True


        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                    # Inference
        # im = cv2.bitwise_and(im, im, mask=view_mask)


        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
            # print("pred , ", pred)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            # print("In loop")
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                opf, tracks, vel = get_sparse_opticalflow(prev_frame[i], im0, tracks, 4, lk_params, feature_params)

            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                opf, tracks, vel = get_sparse_opticalflow(prev_frame, im0, tracks, 4, lk_params, feature_params)
                vel = round(vel, 5)
            p = Path(p)  # to Path
            # cur_frame = cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY)

            annotator = Annotator(opf, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    annotator.box_label(xyxy, None, color=colors(c, True))

            # Stream results
            opf = annotator.result()
            
            # if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # Velocity

            end = time.time()

            if estimate :
                if init_flag:
                    density = "Density: Loading..."
                    if int(end - start)%15==0 and int(end - start)!=0:
                        sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                        density = "Density: "+ density_label[int(clf.predict(sample))-1]
                        init_flag = False


                elif int(end - start)%5==0:
                    sample = sc.transform([[len(det), len(tracks), vel, areaROI]])
                    density = "Density: "+ density_label[int(clf.predict(sample))-1]

                cv2.putText(opf, density, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # cv2.putText(opf, "Time stamp: "+str(int(end - start)), (int(opf.shape[1]/2+10), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("ofp", opf)

            prev_frame = im0s.copy()
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # save_csv("decision4.csv",header, info_4_decision)
                exit()





    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--clf_weights', type=str, default=ROOT / 'svm/finalized_model.sav', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--transform', type=str, default=ROOT / 'transform/mm_scaler.sav', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--estimate', action='store_true', default=False, help='estimate density')
    

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# python velocity.py --weights best_3.pt --source "vids/lib.mp4"  --vid-stride 5 --estimate --clf_weights svm/forest_decision.sav --transform transform/forest_scaler.bin


