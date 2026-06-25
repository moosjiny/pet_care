#!/usr/bin/env python3
import cv2
import importlib.util
import os
import sys
import time
from sakana_orchestrator import SakanaPetOrchestrator

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_emotion_v60-3_hybrid_integrated.py")
spec = importlib.util.spec_from_file_location("pet_emotion_v60_3_hybrid_integrated", SPEC_PATH)
pet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pet_module)
PetUltimateSystem = pet_module.PetUltimateSystem

class PetEmotionSakanaIntegrator:
    def __init__(self, source=0, interval=90):
        self.source = source
        self.pet_system = PetUltimateSystem()
        self.orchestrator = SakanaPetOrchestrator()
        self.interval = interval
        self.last_guidance = ""

    def build_vision_data(self, detections):
        if not detections:
            return {}
        top = detections[0]
        pet_type = "고양이" if top["cid"] == 15 else "강아지" if top["cid"] == 16 else "unknown"
        return {
            "pet_type": pet_type,
            "track_id": int(top["tid"]),
            "emotion": top["emo"],
            "confidence_score": float(top["match"]) / 100.0,
            "summary": f"{pet_type} 감정: {top['emo']} ({float(top['match']):.1f}%)",
            "detections": [
                {
                    "id": int(o["tid"]),
                    "type": "고양이" if o["cid"] == 15 else "강아지" if o["cid"] == 16 else "unknown",
                    "emotion": o["emo"],
                    "confidence": float(o["match"]),
                }
                for o in detections
            ]
        }

    def draw_guidance(self, frame, text):
        if not text:
            return frame
        lines = text.strip().split("\n")[:6]
        x, y = 20, 80
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Source 열기 실패: {self.source}")

        orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
        window_name = "Pet Emotion Hybrid + Sakana"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, int(1280 * orig_h / (orig_w + 1e-6)))

        frame_idx = 0
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            vis = self.pet_system.analyze_and_draw(frame)
            frame_idx += 1

            if frame_idx % self.interval == 0 and self.pet_system.last_analysis:
                vision_data = self.build_vision_data(self.pet_system.last_analysis)
                self.last_guidance = self.orchestrator.run_collaboration_loop(vision_data)

            vis = self.draw_guidance(vis, self.last_guidance)
            fps = 1.0 / (time.time() - start_time + 1e-6)
            cv2.putText(vis, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(window_name, vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    source = int(src) if str(src).isdigit() else src
    integrator = PetEmotionSakanaIntegrator(source=source)
    integrator.run()
