#!/usr/bin/env python3
import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://thesis.hyperbook.com/api/papers/submit"

DEFAULT_SUBMISSION = {
    "title": "펫 감정 하이브리드 시스템 v60-3: Ollama GPU 메모리 공유 문제 해결",
    "author": "Moojoco",
    "abstract": "YOLO, ViTPose, ONNX 추론, LLM 오케스트레이션을 결합한 실시간 펫 감정 분석 하이브리드 시스템과 GPU 메모리 충돌 해결 전략입니다.",
    "body_md": "# 펫 감정 하이브리드 시스템 v60-3\n\n본 제출 문서는 `pet_care` 워크스페이스에 구현된 하이브리드 펫 감정 시스템과 GPU 메모리 병목 해결 방안을 설명합니다.\n\n## 시스템 개요\n\n- YOLO 객체 탐지로 펫, 얼굴, 손 인식\n- ViTPose 자세 추정으로 신체 키포인트 분석\n- ONNX 기반 감정 분류기로 펫 감정 추론\n- 중복 ID 필터링으로 다중 프레임에서 안정적 추적 유지\n- `sakana_orchestrator.py`의 LLM 오케스트레이션으로 기본 판정과 크리틱 검토 수행\n\n## GPU 메모리 병목 분석\n\n`pet_emotion_v60-3_hybrid_sakana.py`는 `SakanaPetOrchestrator`를 통해 로컬 Ollama API(`http://localhost:11434/api/generate`)에 질의합니다. 중요한 점은 `sakana` 코드가 Ollama 서버를 직접 실행하지 않는다는 것입니다.\n\n실제 병목은 다음과 같은 상황에서 발생했습니다.\n\n1. `ollama-server`가 GPU에서 약 3.8GB 메모리를 이미 점유한 상태\n2. `pet_emotion` 파이프라인이 추가로 ONNX CUDA 세션을 생성하려 시도\n3. `BFCArena::AllocateRawInternal` 메모리 할당 실패 발생\n\n이로 인해 `pet_emotion_v60-3_hybrid_sakana.py`는 GPU 폴백 상태로 전환되어 성능이 크게 저하되었습니다.\n\n## 수정한 내용 및 이유\n\n1. **`pet_emotion_v60_3_hybrid_sakana.py` 래퍼 추가**\n   - 원본 스크립트 파일명에 하이픈(`-`)이 포함되어 있어 Python 표준 모듈 임포트가 불가했습니다.\n   - 이를 해결하기 위해 `importlib.util` 기반 래퍼를 추가하여 파일 이름과 모듈 이름을 분리했습니다.\n   - 이 수정은 자동화 테스트와 코드 재사용을 위해 필요했습니다.\n\n2. **`submit_thesis.py`에 GPU 메모리 공유 문제 설명 보강**\n   - Ollama 서버가 `pet_emotion`과 동시에 GPU를 사용하면 메모리 부족이 발생한다는 점을 명확히 기술했습니다.\n   - VS Code 창 여부와는 무관하게, 실제 원인은 GPU 메모리 할당 충돌입니다.\n\n3. **테스트 절차를 명확히 정리**\n   - `VS Code`를 종료한 상태에서 `.venv` 활성화 후 실행하도록 절차를 작성했습니다.\n   - 이 절차는 환경 간 차이를 줄이고, GUI 세션이 남긴 상태 변수를 배제하기 위함입니다.\n\n## VS Code 종료 후 테스트 절차\n\n다음 절차는 VS Code를 닫은 이후에 터미널에서 직접 수행하도록 설계되었습니다.\n\n1. `cd /home/moos/dev_ws/pet_care`\n2. `source .venv/bin/activate`\n3. `python -m py_compile pet_emotion_v60_3_hybrid_sakana.py sakana_orchestrator.py pet_emotion_v60-3_hybrid_integrated.py submit_thesis.py`\n4. `python -c "import pet_emotion_v60_3_hybrid_sakana; print('OK')"`\n5. `nvidia-smi`로 현재 GPU 점유 상태를 확인하고, `ollama-server` 또는 다른 Python 프로세스가 GPU를 사용 중인지 점검\n6. 필요 시 `ollama-server`를 종료하거나 CPU 모드로 재실행하여 GPU 메모리를 확보\n7. `python pet_emotion_v60_3_hybrid_sakana.py ./data/cat_test.mp4`를 실행\n8. 실행 중 `exit code 137`이 발생하면 GPU 메모리 부족 또는 강제 종료가 원인이므로 추가로 `ollama-server`를 멈추고 다시 시도\n\n## 검증 및 테스트\n\n- `pet_emotion_v60_3_hybrid_sakana.py` 래퍼는 `python -m py_compile`와 `python -c "import pet_emotion_v60_3_hybrid_sakana; print('OK')"`에서 정상 동작을 확인했습니다.\n- VS Code를 닫은 상태에서 테스트하면 GUI 세션 관련 환경 변수나 별도 Python 프로세스의 간섭을 줄일 수 있습니다.\n\n## 확인된 Thesis 제출 API\n\nThesis 플랫폼은 다음을 통해 제출을 지원합니다:\n- `POST /api/papers/submit`\n- Bearer 토큰 인증: `Authorization: Bearer <THESIS_SUBMIT_TOKEN>`\n- `title`, `author`, `abstract`, `body_md`, `categories`, `tags`, `changelog`, 선택적 `slug`를 포함하는 JSON 본문\n\n## 워크스페이스 노트\n\n이 저장소는 CUDA 지원 YOLO 초기화, `models/` 및 `data/` 하위의 로컬 자산 관리, 실시간 펫 감정 인식을 위한 하이브리드 추론 파이프라인을 포함합니다.\n\n## 제출 메타데이터\n\n이번 제출은 `pet-emotion-hybrid-system-v60-3` 논문의 GPU 메모리 공유 문제 해결을 중심으로 한 버전 업데이트입니다.\n",
    "categories": ["robotics", "ai"],
    "tags": ["pet-emotion", "hybrid", "vision", "onnx", "yolo", "gpu"],
    "changelog": "Ollama GPU 메모리 공유 문제 및 sakana 통합 검증 내용 추가 제출입니다.",
    "slug": "pet-emotion-hybrid-system-v60-3-gpu-fix"
}


def get_token(args):
    if args.token:
        return args.token
    env_token = os.environ.get("THESIS_TOKEN_GUEST")
    if env_token:
        return env_token
    print("Error: Bearer token is required via --token or THESIS_TOKEN_GUEST environment variable.", file=sys.stderr)
    sys.exit(1)


def submit_paper(token, payload):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    request = Request(API_URL, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            print("Submission response:")
            print(response_body)
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        print(error_body, file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Submit a paper to thesis.hyperbook.com")
    parser.add_argument("--token", help="Bearer token for thesis submission")
    parser.add_argument("--dry-run", action="store_true", help="Show payload without submitting")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(DEFAULT_SUBMISSION, indent=2, ensure_ascii=False))
        return

    token = get_token(args)
    submit_paper(token, DEFAULT_SUBMISSION)


if __name__ == "__main__":
    main()
