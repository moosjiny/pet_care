import json
import requests

class SakanaPetOrchestrator:
    def __init__(self):
        # 로컬 Ollama 엔드포인트
        self.ollama_url = "http://localhost:11434/api/generate"
        self.orchestrator_model = "qwen2.5-coder:7b"
        self.critic_model = "deepseek-r1:8b"

    def _json_default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        return str(obj)

    def _query_ollama(self, model, prompt):
        """로컬 Ollama 모델에 질의하는 헬퍼 함수"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3} # 일관된 출력을 위해 낮은 온도로 세팅
        }
        try:
            response = requests.post(self.ollama_url, json=payload)
            return response.json().get("response", "")
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def run_collaboration_loop(self, vision_data):
        print(f"\n[1. Vision Data 수신]: {vision_data}")
        
        # 단계 1: Orchestrator(Qwen)가 기본 분석 및 초안 작성
        orchestrator_prompt = f"""
        당신은 반려동물 행동 분석 오케스트레이터입니다. 
        아래의 비전 센싱 데이터를 바탕으로 동물의 현재 감정 상태를 진단하고 수행할 행동 가이드를 작성하세요.
        최종 답변은 반드시 한국어로 작성해야 합니다.

        [비전 데이터]: {json.dumps(vision_data, default=self._json_default, ensure_ascii=False)}
        """
        initial_decision = self._query_ollama(self.orchestrator_model, orchestrator_prompt)
        print(f"\n[2. Orchestrator (Qwen)의 초안 판정]:\n{initial_decision}")

        # 단계 2: Critic(DeepSeek-R1)에게 비판 및 모순 검증 요청 (사카나 스타일의 상호 검증)
        critic_prompt = f"""
        당신은 동물 행동학 전문가이자 시스템 검증 에이전트입니다.
        오케스트레이터가 내린 판정에 논리적 모순이 있거나, 비전 데이터의 한계(예: 일시적 오탐 가능성)를 간과한 부분이 없는지 '비판적으로' 검토하십시오.
        생각(<think>) 과정은 자유롭게 하되, 최종 조언은 반드시 한국어로 작성하세요.

        [원시 비전 데이터]: {json.dumps(vision_data, default=self._json_default, ensure_ascii=False)}
        [오케스트레이터의 판정]: {initial_decision}
        """
        critic_feedback = self._query_ollama(self.critic_model, critic_prompt)
        print(f"\n[3. Critic (DeepSeek-R1)의 심층 추론 및 피드백]:\n{critic_feedback}")

        # 단계 3: 피드백을 반영한 최종 의사결정 (Consensus)
        final_prompt = f"""
        당신은 오케스트레이터입니다. 전문가(Critic)의 피드백을 반영하여 최종 행동 지침을 확정하세요.
        반드시 한국어로 최종 결론만 깔끔하게 출력하세요.

        [이전 판정]: {initial_decision}
        [전문가 피드백]: {critic_feedback}
        """
        final_decision = self._query_ollama(self.orchestrator_model, final_prompt)
        print(f"\n[4. 최종 조율된 집단지성 결론]:\n{final_decision}")
        return final_decision

# 로컬 테스트 실행부
if __name__ == "__main__":
    orchestrator = SakanaPetOrchestrator()
    
    # 기존 v60-3 파이썬 코드가 뽑아서 던져줄 가상의 '하이브리드 비전 데이터'
    mock_vision_data = {
        "pet_type": "cat",
        "track_id": 4,
        "tail_status": "raised_and_vibrating", # 꼬리가 바짝 서고 떨림
        "ear_status": "flattened",            # 귀가 양옆으로 눕혀짐 (마징가 귀)
        "confidence_score": 0.89
    }
    
    orchestrator.run_collaboration_loop(mock_vision_data)
