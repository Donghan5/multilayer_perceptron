🔢 A. Gradient Check (최우선)

 ❌ A1. Gradient check 구현
 ❌ A2. 각 레이어별 weight/bias gradient 비교 (relative error < 1e-5)
 ❌ A3. Softmax+CE 결합 부분도 check 통과

🎯 B. Sanity Check

 ❌ B1. 초기 loss ≈ -log(1/C) (Binary면 ~0.693)
 ❌ B2. 20개 샘플로 overfit 성공 (loss→0, acc→100%)
 ❌ B3. LR sweep [1e-4, 1e-3, 1e-2, 1e-1, 1.0] — sweet spot 존재 확인
 ❌ B4. Adam이 SGD보다 수렴 빠른지 확인

🧮 C. 수치 안정성

 ✅ C1. Softmax: exp(x - max(x)) shift — utils.py:44
 ✅ C2. Cross-entropy: log(p + 1e-15) epsilon — utils.py:59-60
 ❌ C3. NaN/Inf 체크 (loss, gradient, weights)
 ❌ C4. Gradient clipping 필요한지 확인 (터지면)

🔄 D. 학습 루프

 ✅ D1. 매 epoch 데이터 shuffle — model.py:100-101
 ✅ D2. Train/Val split 전에 정규화 아님 → Train fit → Val transform — model.py:86-91
 ❌ D3. Batch size 효과 확인 (1, 32, 전체)
 ✅ D4. Random seed 고정으로 재현 가능 — main.py:59-60

⚙️ E. Adam 구현

 ✅ E1. β1=0.9, β2=0.999, ε=1e-8 — optimizer.py:24
 ✅ E2. m, v 0으로 초기화 — optimizer.py:38-41
 ✅ E3. Timestep t 매 step 증가 — optimizer.py:42
 ✅ E4. Bias correction (m_hat, v_hat) 적용 — optimizer.py:54-57
 ✅ E5. 레이어별 m, v 독립 유지 — optimizer.py:48-52

🎲 F. SGD 구현

 ✅ F1. Mini-batch 제대로 분할 — model.py:106-108
 ❌ F2. Momentum 옵션 있으면 velocity 유지
 ❌ F3. Learning rate decay 옵션 (있다면)

🏗️ G. Weight 초기화

 ✅ G1. 전부 0 초기화 아님 (대칭성 문제) — network.py:66-72
 ✅ G2. ReLU면 He, tanh/sigmoid면 Xavier — network.py:66-72
 ✅ G3. Bias는 0 초기화 OK — network.py:77

🔀 H. Activation

 ✅ H1. ReLU: max(0, x), gradient는 x>0 ? 1 : 0 — utils.py:34-41
 ✅ H2. Sigmoid: 수치 안정 버전 — utils.py:18-21
 ✅ H3. Softmax: 마지막 레이어에만 — network.py:97
 ❌ H4. Dead ReLU 확인 (뉴런 대부분 0 출력하면 문제)

📉 I. Loss

 ✅ I1. Binary면 BCE, Multi-class면 CCE — utils.py:53-61
 ✅ I2. Batch 전체 평균 (sum 아님) — utils.py:61
 ✅ I3. Label 포맷 맞음 (one-hot vs index) — utils.py:73-79

📊 J. 평가

 ✅ J1. Train + Val loss 둘 다 기록 — model.py:120-131
 ✅ J2. Train + Val accuracy 둘 다 기록 — model.py:121-132
 ❌ J3. Confusion matrix 출력
 ❌ J4. Precision, Recall, F1 (특히 unbalanced면 필수)
 ✅ J5. Learning curve 시각화 저장 — main.py:51

💾 K. I/O

 ✅ K1. 모델 save (weights + architecture) — model.py:157-160
 ✅ K2. 모델 load 후 같은 성능 재현 — predict.py
 ✅ K3. 하이퍼파라미터 config/argparse로 분리 — main.py:9-20
 ⚠️ K4. Seed 재현성 최종 확인 — seed 설정됨, 최종 검증 미확인

🛡️ L. Edge Cases

 ❌ L1. 데이터 전부 같은 클래스면? (degenerate)
 ❌ L2. 결측치 있는 데이터 처리
 ❌ L3. 입력 스케일 다를 때 정규화 동작
 ❌ L4. 매우 작은 데이터셋 (n<10) 크래시 안 남

🎤 M. 디펜스 대비

 ❓ M1. Backprop 수식 손으로 유도 가능
 ❓ M2. "왜 이 초기화?" 답 준비
 ❓ M3. "왜 이 activation?" 답 준비
 ❓ M4. "Adam이 SGD보다 나은 이유" 답 준비
 ❓ M5. Overfitting 겪었을 때 대처법 설명 가능