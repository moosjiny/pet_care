import importlib.util
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TARGET_FILE = CURRENT_DIR / "pet_emotion_v60-3_hybrid_sakana.py"

if not TARGET_FILE.exists():
    raise FileNotFoundError(f"Target script not found: {TARGET_FILE}")

spec = importlib.util.spec_from_file_location("pet_emotion_v60_3_hybrid_sakana_module", str(TARGET_FILE))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

PetEmotionSakanaIntegrator = module.PetEmotionSakanaIntegrator

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pet_emotion_v60-3_hybrid_sakana via importable wrapper")
    parser.add_argument("source", nargs="?", default=0, help="Video source path or camera index")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    integrator = PetEmotionSakanaIntegrator(source=source)
    integrator.run()
