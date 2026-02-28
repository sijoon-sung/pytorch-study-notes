# PyTorch Deep Learning Research Notes

오비시디언 기반의 파이토치(PyTorch) 심층 연구 저장소입니다. 단순 요약을 넘어 수학적 증명과 시스템 최적화 관점에서 딥러닝을 분석합니다.

## 🔬 Research Pillars

모든 분석은 아래의 **3-Step Framework**를 따릅니다:

1.  **[Concept]**: 알고리즘의 수학적 증명, 비용 함수 분석, 시간/공간 복잡도(Big-O) 평가.
2.  **[Code]**: 효율적인 파이프라인 설계 및 PyTorch 기반 베이스라인 모듈화 구현.
3.  **[Experiment]**: 하이퍼파라미터 튜닝, 추론 속도(Latency) 최적화 및 시스템 아키텍처 관점의 한계 테스트.

---

## 📂 Repository Structure

```text
root/
├── MOC.md              # 연구 목차 및 진행 현황 (Map of Contents)
├── CHAP_X/             # 각 챕터별 연구 폴더
│   ├── Concept/        # 이론 및 수식 분석
│   ├── Code/           # 구현 코드 (Baseline)
│   └── Experiment/     # 최적화 및 한계 돌파 실험
└── README.md
```

**Last Updated**: 2026-02-28
