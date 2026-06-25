import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from collections import deque, defaultdict
from ultralytics import YOLO
import onnxruntime as ort
from PIL import ImageFont, ImageDraw, Image
import os
import sys

# [1. 전역 설정 및 경로 정의]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "models")
EMOTION_MODEL_DIR = os.path.join(BASE_DIR, "data", "emotion")

YOLO_SEG_ONNX = os.path.join(ROOT_DIR, "yolov8n-seg.onnx")
YOLO_POSE_ONNX = os.path.join(ROOT_DIR, "best.onnx")
V32_ONNX_PATH = os.path.join(ROOT_DIR, "vitpose-l-ap10k.onnx")
DOG_ONNX_PATH = os.path.join(EMOTION_MODEL_DIR, "dog_emotion_v1.onnx")
CAT_ONNX_PATH = os.path.join(EMOTION_MODEL_DIR, "cat_emotion_v1.onnx")
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

DOG_LABELS = ["편안함", "즐거움", "불안", "분노", "공포"]
CAT_LABELS = ["휴식", "장난", "스트레스", "공격성", "두려움"]
PET_NAMES = {15: "고양이", 16: "강아지", 0: "사람"}

SKELETON_V79 = [(0,1), (0,2), (2,3), (1,4), (4,5), (4,6), (5,7), (6,8), (4,13), (13,9), (13,10), (9,11), (10,12), (13,14)]
SKELETON_V32 = [(2,0), (2,1), (0,3), (1,3), (3,4), (3,5), (5,6), (6,7), (3,8), (8,9), (9,10), (4,11), (11,12), (12,13), (4,14), (14,15), (15,16)]
KP_LABELS_V79_KOR = {0: "코", 1: "이마", 2: "입", 3: "턱", 4: "목", 5: "R어깨", 6: "L어깨", 7: "R앞발", 8: "L앞발", 9: "R엉덩", 10: "L엉덩", 11: "R뒷발", 12: "L뒷발", 13: "꼬리시작", 14: "꼬리끝"}

class PetUltimateSystem:
    def __init__(self):
        available_providers = ort.get_available_providers()
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        self.device = "cuda" if 'CUDAExecutionProvider' in available_providers else "cpu"
        
        print(f">> [V60-3] 중복 ID 방지 통합 시스템 가동 (Device: {self.device})")
        print(f">> [V60-3] ONNX Runtime providers: {self.providers}")
        
        def create_session(path):
            if self.device == 'cuda':
                try:
                    return ort.InferenceSession(path, providers=['CUDAExecutionProvider'])
                except Exception as e:
                    print(f">> [V60-3] CUDA 세션 로드 실패, CPU로 폴백합니다: {e}")
                    self.device = 'cpu'
                    self.providers = ['CPUExecutionProvider']
            try:
                return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            except Exception as e:
                raise RuntimeError(f"ONNX 세션 로드 실패: {path} -> {e}")

        # 모델 로드
        self.yolo_seg = YOLO(os.path.abspath(YOLO_SEG_ONNX), task='segment')
        self.yolo_pose = YOLO(os.path.abspath(YOLO_POSE_ONNX), task='pose')
        if self.device == 'cuda':
            try:
                self.yolo_seg = self.yolo_seg.to(self.device)
                self.yolo_pose = self.yolo_pose.to(self.device)
            except Exception:
                pass
        
        self.vit_session = create_session(os.path.abspath(V32_ONNX_PATH))
        self.dog_emo = create_session(os.path.abspath(DOG_ONNX_PATH))
        self.cat_emo = create_session(os.path.abspath(CAT_ONNX_PATH))
        
        self.emotion_queues = defaultdict(lambda: deque(maxlen=30))
        self.last_analysis = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.target_brightness = 115

    def apply_smart_correction(self, frame):
        avg_b = np.mean(cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
        frame = cv2.convertScaleAbs(frame, alpha=self.target_brightness/(avg_b+1e-6), beta=0)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB); l,a,b = cv2.split(lab); cl = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    def draw_text_pil(self, img, text, pos, size=15, color=(255, 255, 255)):
        img_pil = Image.fromarray(img); draw = ImageDraw.Draw(img_pil)
        if os.path.exists(FONT_PATH):
            font = ImageFont.truetype(FONT_PATH, size); draw.text(pos, text, font=font, fill=color[::-1])
        return np.array(img_pil)

    def analyze_and_draw(self, frame):
        filtered = self.apply_smart_correction(frame)
        # iou 임계값을 높여 중복 박스 트래킹 억제
        results = self.yolo_seg.track(filtered, persist=True, iou=0.6, classes=[0, 15, 16], verbose=False)
        vis = filtered.copy(); h, w = frame.shape[:2]
        
        hud_w = int(w * 0.3)
        overlay = vis.copy(); cv2.rectangle(overlay, (w-hud_w, 0), (w, h), (0,0,0), -1)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

        obj_list = [] 
        used_centers = [] # 중복 위치 체크용 리스트

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy(); confs = results[0].boxes.conf.cpu().numpy()
            t_ids = results[0].boxes.id.int().cpu().tolist(); c_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for idx, (t_id, c_id, box, conf) in enumerate(zip(t_ids, c_ids, boxes, confs)):
                x1, y1, x2, y2 = map(int, box)
                
                # [V60-3 핵심] 같은 위치 중복 ID 필터링 (중심점 거리 기준)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                is_duplicate = False
                for prev_c in used_centers:
                    dist = np.sqrt((center[0]-prev_c[0])**2 + (center[1]-prev_c[1])**2)
                    if dist < 50: # 50픽셀 이내에 다른 ID가 이미 있다면 중복으로 간주
                        is_duplicate = True; break
                if is_duplicate: continue
                used_centers.append(center)

                roi = filtered[max(0,y1):y2, max(0,x1):x2]
                if roi.size == 0: continue

                if c_id == 0:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    vis = self.draw_text_pil(vis, f"사람 ID:{t_id}", (x1, y1-25), size=16); continue

                # Pose & Emotion (V60-2 로직 유지)
                v32_raw = self.vit_session.run(None, {self.vit_session.get_inputs()[0].name: cv2.resize(roi, (192, 256)).transpose(2,0,1)[np.newaxis,...].astype(np.float32)/255.0})[0]
                v32_kpts = np.array([[xh*(x2-x1)/48+x1, yh*(y2-y1)/64+y1] for yh, xh in [np.unravel_index(np.argmax(v32_raw[0, m]), (64,48)) for m in range(17)]])
                
                pose_res = self.yolo_pose(roi, verbose=False)
                if pose_res[0].keypoints is not None and len(pose_res[0].keypoints) > 0:
                    v79_kpts = pose_res[0].keypoints.data.cpu().numpy()[0].copy()
                    v79_kpts[:, 0] += x1; v79_kpts[:, 1] += y1
                else: v79_kpts = np.zeros((15,3))

                # 시각화 (가느다란 녹색선 & 파란색 점)
                for s, e in SKELETON_V32:
                    p1, p2 = tuple(v32_kpts[s][:2].astype(int)), tuple(v32_kpts[e][:2].astype(int))
                    if p1[0] > 0 and p2[0] > 0: cv2.line(vis, p1, p2, (255, 100, 100), 1)
                for pt in v32_kpts: cv2.circle(vis, tuple(pt[:2].astype(int)), 3, (255, 0, 0), -1)

                for s, e in SKELETON_V79:
                    p1, p2 = tuple(v79_kpts[s][:2].astype(int)), tuple(v79_kpts[e][:2].astype(int))
                    if p1[0] > 0 and p2[0] > 0: cv2.line(vis, p1, p2, (100, 255, 100), 1)
                for pt in v79_kpts: cv2.circle(vis, tuple(pt[:2].astype(int)), 4, (0, 255, 0), -1)

                # 감정 분석 (생략 없이 통합)
                match = conf * 100
                norm = v79_kpts.copy(); norm[:,0], norm[:,1] = (norm[:,0]-x1)/(x2-x1+1e-6), (norm[:,1]-y1)/(y2-y1+1e-6)
                self.emotion_queues[t_id].append(norm[:,:2].flatten())
                emo = "분석중.."
                if len(self.emotion_queues[t_id]) == 30:
                    seq = np.array(self.emotion_queues[t_id])
                    sess = self.dog_emo if c_id == 16 else self.cat_emo
                    out = sess.run(None, {sess.get_inputs()[0].name: (seq-seq[0])[np.newaxis,...].astype(np.float32)})[0]
                    emo = (DOG_LABELS if c_id == 16 else CAT_LABELS)[np.argmax(out)]
                
                head_ref = v79_kpts[1] if v79_kpts[1][0] > 0 else v79_kpts[0]
                if head_ref[0] > 0:
                    vis = self.draw_text_pil(vis, f"ID:{t_id} {PET_NAMES[c_id]} [{emo}] ({match:.1f}%)", (int(head_ref[0])-40, int(head_ref[1])-55), size=17, color=(0,255,0))
                obj_list.append({"tid": t_id, "cid": c_id, "emo": emo, "match": match, "idx": len(used_centers)})

            for o in obj_list:
                hud_text = f"{o['idx']}. {PET_NAMES[o['cid']]} ID:{o['tid']} [{o['emo']}] 매칭:{o['match']:.1f}%"
                vis = self.draw_text_pil(vis, hud_text, (w-hud_w+20, 60 + (o['idx']-1) * 55), size=17)
        self.last_analysis = obj_list
        return vis

    def run(self, source):
        cap = cv2.VideoCapture(source)
        orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
        window_name = "Pet Ultimate v60-3 (Anti-Double ID)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, int(1280 * orig_h / (orig_w + 1e-6)))

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            vis = self.analyze_and_draw(frame)
            curr_fps = 1.0 / (time.time() - start_time + 1e-6)
            cv2.rectangle(vis, (15, 15), (190, 65), (0, 0, 0), -1)
            cv2.putText(vis, f"FPS: {curr_fps:.1f}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(window_name, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    PetUltimateSystem().run(source=int(src) if str(src).isdigit() else src)
