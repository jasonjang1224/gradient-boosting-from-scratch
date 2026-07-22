# Gradient Boosting from Scratch

Jason Jang

현대적인 Gradient Boosting Machine(GBM)을 학부 수준의 미적분학(도함수, 연쇄법칙, 테일러 전개)만 가정하고 밑바닥부터 유도·구현한 프로젝트입니다. 모든 수식을 한 줄씩 직접 유도하고, 각 구성 요소를 순수 NumPy로 구현했으며, 코드의 모든 줄을 그 수학적 근거로 다시 연결합니다.

이 문서와 코드는 **시중에 있는 어떤 자료보다도 쉬우면서도 엄밀하게** 설명하는 것을 목표로 작성되었습니다. "쉽다"는 것은 수식을 생략한다는 뜻이 아니라, 모든 단계를 건너뛰지 않고 왜 그런 형태가 되는지를 끝까지 따라갈 수 있다는 뜻이고, "엄밀하다"는 것은 논문에 등장하는 각 수식이 실제 코드의 어느 줄에 대응하는지까지 빠짐없이 보여준다는 뜻입니다.

## 무엇을 다루는가

Vanilla GBM에서 출발하여, 각 단계가 이전 단계의 어떤 구체적인 한계를 해결하는지 밝히며 다섯 가지 개선을 순서대로 도입합니다.

1. **Newton Step** — 1차 기울기(gradient)뿐 아니라 2차 미분(Hessian)까지 활용해 최적 리프 값을 구하는 뉴턴 방법
2. **Oblivious Trees** — 같은 깊이의 모든 노드가 동일한 분기 기준을 공유하는 대칭 트리 (CatBoost 방식)
3. **Ordered Target Statistics** — 범주형 변수를 순서(permutation) 기반으로 인코딩하여 타겟 누수(target leakage)를 방지
4. **Ordered Boosting** — 이전 트리의 예측이 자기 자신의 학습에 사용되며 생기는 예측 편향(prediction shift)을 순서 기반 기법으로 제거
5. **Posterior Sampling (SGLB)** — Stochastic Gradient Langevin Boosting으로 그래디언트에 노이즈를 주입해 베이지안 사후분포에서 샘플링하고, 여러 체인의 앙상블로 불확실성(epistemic uncertainty)까지 추정

## 저장소 구조

```
src/
  vanilla_gbm.py   기본 CART 스타일 결정 트리 + Vanilla GBM (1차 gradient만 사용)
  tree.py          Oblivious Tree — 뉴턴 스텝 기반 리프 값 계산, 깊이별 공유 분기
  loss.py          손실 함수: MSE, LogLoss (gradient / hessian / 초기 예측값)
  boosting.py       부스팅 전략: VanillaBoosting, OrderedBoosting, SGLB, SGLBEnsemble
  encoder.py       OrderedTargetEncoder — 순서 기반 타겟 통계 인코딩
  experiment.py    6가지 설정을 비교하는 재현 실험

gradient-boosting-from-scratch.pdf   전체 수식 유도와 설명 문서
```

## 실행

```bash
pip install numpy
python src/experiment.py
```

`experiment.py`는 `y = x0^2 + 0.5*x1 + noise` 형태의 합성 데이터에서 아래 6가지 구성을 학습/평가하여 Train/Test MSE를 비교합니다.

1. DecisionTree + MSE + Vanilla
2. ObliviousTree + MSE + Vanilla
3. ObliviousTree + MSE + Ordered Boosting
4. ObliviousTree + LogLoss + Ordered Boosting (이진 분류)
5. ObliviousTree + MSE + SGLB
6. ObliviousTree + MSE + SGLBEnsemble (10-chain 앙상블, 불확실성 추정 포함)

## 문서

`gradient-boosting-from-scratch.pdf`에 전체 수식 유도, 각 개선의 동기, 그리고 코드와의 대응 관계가 정리되어 있습니다. 외부 라이브러리(scikit-learn, XGBoost, CatBoost 등) 없이 순수 NumPy만으로 모든 구성 요소를 직접 구현했습니다.
