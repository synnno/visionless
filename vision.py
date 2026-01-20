import cv2
import numpy as np
import random
import math
import os
import argparse
import time
import torch
from ultralytics import YOLO

class YOLODirectML:
    def __init__(self, model_path, class_names):
        import onnxruntime as ort
        
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_mem_pattern = False
        
        self.session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider'], sess_options=so)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.classes = class_names
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def predict(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        predictions = np.squeeze(outputs[0]).T
        
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(scores) == 0:
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        
        input_boxes = boxes.copy()
        input_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        input_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        
        indices = cv2.dnn.NMSBoxes(input_boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        
        results = []
        for i in indices:
            idx = int(i) if np.isscalar(i) else int(i[0])
            box = input_boxes[idx]
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = float(scores[idx])
            cls = int(class_ids[idx])
            lbl = self.classes[cls]
            results.append([x, y, w, h, cls, conf, lbl])
            
        return results

class tracker:
    def __init__(self):
        self.centers = {}
        self.id_count = 0
    
    def update(self, rects):
        objects_bbs_ids = []
        for rect in rects:
            x, y, w, h, cls, conf, lbl = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            same_object_detected = False
            for id, pt in self.centers.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:
                    self.centers[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, cls, conf, lbl, id])
                    same_object_detected = True
                    break
            
            if not same_object_detected:
                self.centers[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, cls, conf, lbl, self.id_count])
                self.id_count += 1
        
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, _, object_id = obj_bb_id
            center = self.centers[object_id]
            new_center_points[object_id] = center
        
        self.centers = new_center_points.copy()
        return objects_bbs_ids

def corners(img, x, y, w, h, c, l=15, t=1):
    if x < 0: x = 0
    if y < 0: y = 0
    cv2.line(img, (x, y), (x + l, y), c, t, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + l), c, t, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w - l, y), c, t, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + l), c, t, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x + l, y + h), c, t, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - l), c, t, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w - l, y + h), c, t, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - l), c, t, cv2.LINE_AA)

def check(b, objs):
    bx, by, bw, bh = b
    for o in objs:
        if (bx < o[0] + o[2] and bx + bw > o[0] and by < o[1] + o[3] and by + bh > o[1]):
            return True
    return False

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input video file')
    parser.add_argument('--output', type=str, default='pro_vision_tiktok.mp4', help='output video file')
    args = parser.parse_args()

    v = args.input
    if not v:
        files = ['input.mov', 'input.mp4', 'input.avi', 'input.mkv']
        for f in files:
            if os.path.exists(f):
                v = f
                break
            
    if not v or not os.path.exists(v):
        print("no video found")
        return

    print("loading model...")
    use_onnx = False
    try:
        try:
            import onnxruntime as ort
            if 'DmlExecutionProvider' in ort.get_available_providers():
                use_onnx = True
        except ImportError:
            pass

        if torch.cuda.is_available():
            model = YOLO("yolov8x.pt")
            print(f"using cuda: {torch.cuda.get_device_name(0)}")
        elif use_onnx:
            print("using directml (amd gpu) via onnx")
            model_file = "yolov8x_fp16.onnx"
            if not os.path.exists(model_file):
                print("exporting model to onnx (fp16) for amd gpu support (this runs once)...")
                temp_model = YOLO("yolov8x.pt")
                temp_model.export(format="onnx", imgsz=[640,640], dynamic=False, half=True)
                if os.path.exists("yolov8x.onnx"):
                    os.rename("yolov8x.onnx", model_file)
                del temp_model
            
            temp = YOLO("yolov8x.pt")
            names = temp.names
            del temp
            
            model = YOLODirectML(model_file, names)
        else:
            model = YOLO("yolov8x.pt")
            print("using cpu (optimized)")
    except Exception as e:
        print(f"model error: {e}")
        return

    cap = cv2.VideoCapture(v)
    
    start_time = time.time()
    frame_cnt = 0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    tw = 1080
    th = 1920
    
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (tw, th))
    sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10, detectShadows=False)
    trk = tracker()
    
    sx = tw / ow
    sy = th / oh
    idx = 0
    
    detect_every = 4
    last_rects = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_cnt += 1
        if frame_cnt % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_cnt / elapsed
            print(f"processing... frame {idx} ({fps:.1f} fps)", end='\r')
        
        raw_rects = []
        
        if idx % detect_every == 0:
            new_rects = []
            if use_onnx:
                dml_rects = model.predict(frame)
                scale_x = tw / 640
                scale_y = th / 640
                
                for r in dml_rects:
                    x, y, w, h, cls, conf, lbl = r
                    nx = int(x * scale_x)
                    ny = int(y * scale_y)
                    nw = int(w * scale_x)
                    nh = int(h * scale_y)
                    new_rects.append([nx, ny, nw, nh, cls, conf, lbl])
            elif torch.cuda.is_available():
                res = model(frame, verbose=False, stream=True, device=0)
                for r in res:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy()
                        b[0] *= sx
                        b[1] *= sy
                        b[2] *= sx
                        b[3] *= sy
                        cls = int(box.cls[0])
                        lbl = model.names[cls]
                        conf = float(box.conf[0])
                        if conf > 0.25:
                            x = int(b[0])
                            y = int(b[1])
                            w = int(b[2] - b[0])
                            h = int(b[3] - b[1])
                            new_rects.append([x, y, w, h, cls, conf, lbl])
            else:
                res = model(frame, verbose=False, stream=True)
                for r in res:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy()
                        b[0] *= sx
                        b[1] *= sy
                        b[2] *= sx
                        b[3] *= sy
                        cls = int(box.cls[0])
                        lbl = model.names[cls]
                        conf = float(box.conf[0])
                        if conf > 0.25:
                            x = int(b[0])
                            y = int(b[1])
                            w = int(b[2] - b[0])
                            h = int(b[3] - b[1])
                            new_rects.append([x, y, w, h, cls, conf, lbl])
            
            last_rects = new_rects
            raw_rects.extend(new_rects)
        else:
            raw_rects.extend(last_rects)
        
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        mask = sub.apply(blur)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k2, iterations=2)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) > 20:
                x, y, w, h = cv2.boundingRect(c)
                hx = int(x * sx)
                hy = int(y * sy)
                hw = int(w * sx)
                hh = int(h * sy)
                
                if not check((hx, hy, hw, hh), raw_rects):
                    lbl = "unknown" if hw * hh > 2000 else "micro"
                    raw_rects.append([hx, hy, hw, hh, -1, 0.0, lbl])
        
        dobjs = trk.update(raw_rects)

        disp = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_CUBIC)
        overlay = np.zeros_like(disp)
        
        cv2.putText(overlay, "https://synucu.cat", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        
        cat_eyes = ["o.o", "O.O", "-.-", ">.<", "^.^", "0.0", "x.x"]
        eye = cat_eyes[(idx // 24) % len(cat_eyes)]
        
        cat_lines = [
            " /\\_/\\",
            f"( {eye} )",
            " > ^ <"
        ]
        
        cy = 160
        for line in cat_lines:
            cv2.putText(overlay, line, (50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
            cy += 30
        
        for i in range(len(dobjs)):
            x1, y1, w1, h1, c1, conf1, lbl1, id1 = dobjs[i]
            cx1 = x1 + w1 // 2
            cy1 = y1 + h1 // 2
            
            if lbl1 == "micro" and len(dobjs) > 80 and random.random() < 0.5: continue

            c_count = 0
            for j in range(i + 1, len(dobjs)):
                if c_count > 2: break
                x2, y2, w2, h2, c2, conf2, lbl2, id2 = dobjs[j]
                cx2 = x2 + w2 // 2
                cy2 = y2 + h2 // 2
                
                if lbl1 == "micro" and lbl2 == "micro": continue

                dst = math.hypot(cx1 - cx2, cy1 - cy2)
                if dst < 400:
                    c_count += 1
                    col = (150, 150, 150)
                    if lbl1 != "micro" and lbl2 != "micro" and lbl1 != lbl2:
                        col = (255, 50, 255)
                    elif lbl1 == lbl2:
                        col = (0, 255, 200)
                        
                    cv2.line(overlay, (cx1, cy1), (cx2, cy2), col, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.7, disp, 1.0, 0, disp)
        
        for obj in dobjs:
            x, y, w, h, cls, conf, lbl, id = obj
            cx = x + w // 2
            cy = y + h // 2
            
            if lbl == "micro":
                cv2.circle(disp, (cx, cy), 1, (200, 200, 200), -1, cv2.LINE_AA)
                if random.random() < 0.05:
                    corners(disp, x, y, w, h, (100, 100, 100), 4, 1)
                continue
            
            col = (0, 255, 0)
            if lbl == "unknown": col = (0, 0, 255)
            elif lbl == "person": col = (0, 255, 255)
            elif lbl in ["car", "truck", "bus"]: col = (0, 165, 255)
                
            corners(disp, x, y, w, h, col, int(min(w, h)/3), 2)
            cv2.circle(disp, (cx, cy), 4, col, 1, cv2.LINE_AA)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt = f"{lbl} {int(conf*100)}%"
            (tw_txt, th_txt), _ = cv2.getTextSize(txt, font, 0.5, 1)
            
            cv2.rectangle(disp, (x, y - th_txt - 10), (x + tw_txt + 10, y), col, -1)
            cv2.putText(disp, txt, (x + 5, y - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(disp, f"id:{id}", (x, y + h + 15), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(disp)
        idx += 1

    cap.release()
    out.release()
    print("done")

if __name__ == "__main__":
    run()