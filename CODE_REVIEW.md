# Code Review - Multilayer Perceptron

> 시니어 AI 모델 개발자 관점의 코드 리뷰
> 최초 리뷰: 2026-04-14 | 최종 업데이트: 2026-04-27

---

## 목차

1. [버그 및 에러](#1-버그-및-에러)
2. [설계 및 구조 문제](#2-설계-및-구조-문제)
3. [파이썬 스타일](#3-파이썬-스타일-pythonic)
4. [추가 개선 권장사항](#4-추가-개선-권장사항)
5. [테스트 현황](#5-테스트-현황)
6. [요약 (우선순위별)](#6-요약-우선순위별)

---

## 1. 버그 및 에러

### ~~[Critical] `network.py:5` — Import 오타 (`standarize`)~~ ✅ 수정 완료

`standarize` → `standardize` 로 수정 완료.

---

### ~~[Critical] `backward`에서 activation_prime 입력값 혼동~~ ✅ 수정 완료 (2차 수정)

**변경 이력:**

| 시점 | `network.py` backward 코드 | 상태 |
|------|---------------------------|------|
| 최초 | `self.activation_prime(self.activations[-l])` (post-activation) | sigmoid ✅ / relu ✅ (우연히 동작) |
| 1차 수정 | `self.activation_prime(self.zs[-l])` (pre-activation) | relu ✅ / sigmoid ❌ |
| 2차 수정 (현재) | `self.activation_prime(self.activations[-l])` (post-activation) | relu ✅ / sigmoid ✅ |

**근거:**
- `sigmoid_prime(x)` = `x * (1 - x)` → post-activation 값(σ(z))을 기대
- `relu_prime(relu(z))` = `(relu(z) > 0)` = `(z > 0)` → pre/post 결과 동일
- Gradient check (67 tests)로 두 activation 모두 relative error < 1e-5 검증 완료

**현재 코드 (`network.py:121-122`):**
```python
for l in range(2, len(self.config.layers)):
    delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(self.activations[-l])
```

---

### ~~[Critical] `predict.py:39` — cross_entropy 호출 오류~~ ✅ 수정 완료

`one_hot_encode(y_raw)`를 적용하여 올바른 형태로 전달하도록 수정됨.

---

### ~~[Critical] 학습/추론 시 정규화(Standardization) 불일치~~ ✅ 수정 완료

`model.py:86-91`에서 `mean_train`/`std_train`을 저장하고, `fit()` 내부에서 직접 정규화 처리. `predict.py`도 동일 통계 사용.

---

### ~~[Critical] `main.py:80` — `MultilayerPerceptron` 클래스 미존재~~ ✅ 수정 완료

`model.Model(...)` 으로 교체 완료.

---

### ~~[Critical] `main.py:63` + `model.py:79` — 이중 정규화~~ ✅ 수정 완료

`X_raw`를 그대로 `model.fit()`에 전달. 정규화는 `model.fit()` 내부에서만 처리.

---

### ~~[Critical] Gradient를 batch_size로 나누지 않음~~ ✅ 수정 완료

`network.py:114`에서 `delta = (y_pred - y_true) / batch_size`로 출력층 delta를 batch_size로 나눔.
이 나눗셈이 backprop 전체에 전파되므로 모든 gradient가 올바르게 평균화됨.

---

### [Medium] `network.py:39-40, 49-50` — 존재하지 않는 속성 접근

```python
else:
    self.activation = self.config.activation
    self.activation_prime = self.config.activation_prime  # NetworkConfig에 없는 필드!
```

`NetworkConfig` dataclass에 `activation_prime`, `loss_prime` 필드가 없음.
커스텀 activation/loss를 문자열이 아닌 함수로 전달 시 `AttributeError` 발생.

**수정:** 커스텀 함수 지원을 제거하거나, `NetworkConfig`에 optional 필드를 추가.

---

### [Medium] `network.py:116` — 비-cross_entropy 출력층 backward

```python
else:
    delta = self.loss_prime(y_true, y_pred) * self.activation_prime(y_pred) / batch_size
```

MSE + softmax 조합 시 `self.activation_prime(y_pred)`가 softmax의 Jacobian이 아닌 은닉층 activation의 derivative를 사용.
Softmax derivative는 벡터→행렬 매핑이므로 element-wise 곱이 아닌 Jacobian 처리가 필요.

현재 프로젝트에서 cross_entropy만 사용하므로 실질적 영향 없음. MSE 사용 시 수정 필요.

---

### [Low] `utils.py:19-21` — sigmoid RuntimeWarning

```python
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),     # x=1000 → overflow in exp(-1000)은 안남
                    np.exp(x) / (1 + np.exp(x)))  # x=1000일 때 overflow 발생
```

`np.where`는 **양쪽 분기를 모두 평가**하므로, `x >= 0`인 경우에도 `np.exp(x)` 분기가 실행되어 overflow warning 발생.
최종 결과는 정확하나 경고가 noisy.

**수정:**
```python
def sigmoid(x):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
```

---

### ~~[Low] `optimizer.py:33` — 오타~~ ✅ 수정 완료

`timestap` → `timestep` 수정 완료.

---

### ~~[Medium] Adam in-place 뮤테이션~~ ✅ 수정 완료

`new_weights`, `new_biases` 리스트를 생성한 뒤 한 번에 할당.

---

### ~~[Medium] Adam 재훈련 시 상태 미초기화~~ ✅ 해결

`fit()` 호출 시마다 새 `Adam` 인스턴스 생성으로 해결.

---

## 2. 설계 및 구조 문제

### ~~불필요한 3단 추상화 계층~~ ✅ 수정 완료

```
(이전) MultilayerPerceptron → Model → Network
(현재) Model → Network
```

---

### ~~Weight initialization 전략 미사용~~ ✅ 수정 완료

`network.py:61-77`에서 He Uniform / Xavier Uniform 올바르게 구현:

```python
if self.config.weights_initializer == "heUniform":
    limit = np.sqrt(6 / input_dim)
    w = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
elif self.config.weights_initializer == "xavierUniform":
    limit = np.sqrt(6 / (input_dim + output_dim))
    w = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
```

Bias는 0으로 초기화. 테스트로 검증됨 (`test_activation.py`).

---

### ~~`random_seed` 미사용~~ ✅ 수정 완료

`main.py:59-60`에서 `np.random.seed(args.seed)` 적용. 재현성 테스트 통과 (`test_training.py::test_seed_deterministic_training`).

---

### ~~`split.py`가 프로젝트에서 미사용~~ ✅ 수정 완료

삭제됨. `main.py`에서 직접 split 처리.

---

### ~~Early stopping 하드코딩~~ ✅ 수정 완료

`model.py:21`에서 `early_stopping_rounds=10`을 `__init__` 파라미터로 노출.
Validation data 없을 시 경고 메시지 출력 (`model.py:60-61`).

---

### [Low] `model.py:37-48` — fit() docstring이 실제 시그니처와 불일치

docstring에 `learning_rate`, `epochs`, `batch_size`, `optimization` 등의 파라미터를 기술하지만, 실제로는 `__init__`에서 설정한 값을 사용. docstring 업데이트 필요.

---

### [Low] `network.py:17, 20-21` — 미사용 인스턴스 변수

```python
self.network = []      # 한번도 사용되지 않음
self.gw_history = []   # 한번도 사용되지 않음
self.gb_history = []   # 한번도 사용되지 않음
```

---

### [Low] `utils.py:7-10` — `standardize()` 함수 미사용

`model.py`에서 직접 numpy로 정규화 처리하므로 `standardize()` 함수는 dead code.

---

### [Low] `predict.py:4` — `import model` 섀도잉

```python
import model              # 모듈 임포트
...
def predict(data_path, model_path):
    ...
    model = pickle.load(f)  # 지역 변수가 모듈을 섀도잉
```

`model` 변수명이 `import model` 모듈을 섀도잉. 함수 내에서 `model` 모듈 접근 불가.
현재 함수 내에서 모듈을 사용하지 않으므로 실질적 버그는 아니지만, `loaded_model` 등으로 변경 권장.

---

## 3. 파이썬 스타일 (Pythonic)

### Docstring 위치 오류

`utils.py`, `optimizer.py`, `network.py`에서 docstring을 함수/클래스 **위에** 작성. 단순 문자열 리터럴이며 파이썬 docstring이 아님.

```python
# 현재 (잘못된 위치)
"""
    Sigmoid activation function
"""
def sigmoid(x):
    ...

# 올바른 위치
def sigmoid(x):
    """Sigmoid activation function."""
    ...
```

해당 파일: `utils.py:4, 12, 23, 29, 37`, `optimizer.py:3, 12, 20`, `network.py:102`

---

### Type hint 일관성 부재

| 함수 | 현재 상태 |
|------|-----------|
| `sigmoid(x)` | type hint 없음 |
| `sigmoid_prime(x)` | type hint 없음 |
| `relu(x)` | type hint 없음 |
| `relu_prime(x: np.ndarray)` | ✅ |
| `softmax(x: np.ndarray)` | ✅ |
| `cross_entropy(y_true, y_pred)` | ✅ (내부 docstring) |
| `one_hot_encode(y)` | type hint 없음 |

---

### 기타 스타일 이슈

- `model.py:2` — `import copy`는 `copy.deepcopy`만 사용 → `from copy import deepcopy`가 더 명시적
- `predict.py` import 순서가 PEP 8과 다름 (stdlib → third-party → local 순서)
- 탭/스페이스 혼용: `utils.py`에서 `sigmoid` 함수만 스페이스 4칸 들여쓰기, 나머지는 탭

---

## 4. 추가 개선 권장사항

| 항목 | 설명 | 우선도 | 상태 |
|------|------|--------|------|
| **Gradient clipping** | gradient explosion 방지 | 높음 | 미구현 (04_gradient_norms.ipynb로 모니터링 가능) |
| **L2 regularization** | overfitting 방지를 위한 weight decay | 높음 | 미구현 |
| **Learning rate scheduler** | 고정 LR만 지원 | 중간 | 미구현 |
| **SGD momentum** | 기본 SGD만 구현, momentum 미지원 | 중간 | 미구현 |
| **Batch normalization** | 학습 안정성 향상 | 중간 | 미구현 |
| **Logging** | `print` 대신 `logging` 모듈 | 낮음 | 미구현 |
| ~~**Reproducibility**~~ | seed 적용 | 높음 | ✅ 구현됨 |
| ~~**Data pipeline**~~ | 정규화 파라미터 직렬화 | 높음 | ✅ 구현됨 |
| ~~**Unit tests**~~ | 컴포넌트별 테스트 | 중간 | ✅ 67 tests (pytest) |
| **Malignant recall** | 의료 도메인 Recall >= 0.95 목표 | 높음 | 0.9149 (하이퍼파라미터 튜닝 필요) |

---

## 5. 테스트 현황

### 구조

```
tests/
├── conftest.py          # fixtures & helpers
├── test_gradient.py     # A: gradient check (10 tests)
├── test_activation.py   # H: activation + G: weight init (13 tests)
├── test_optimizer.py    # E: Adam + F: SGD (11 tests)
├── test_loss.py         # I: loss function (5 tests)
├── test_stability.py    # C: numerical stability (7 tests)
├── test_training.py     # D: training loop (3 tests)
├── test_io.py           # K: save/load/argparse (11 tests)
├── test_evaluation.py   # J: history/metrics (5 tests)
└── test_edge_cases.py   # L: edge cases (8 tests)
```

### 실행 결과 (2026-04-27)

```
Fast tests (evaluation 제외):  67/67 passed
Evaluation tests:              4/5 passed (recall 0.9149 < 0.95 미달)
```

### 실행 방법

```bash
./run_tests.sh          # 전체
./run_tests.sh fast     # evaluation 제외
./run_tests.sh gradient # 개별 모듈
```

---

## 6. 요약 (우선순위별)

### 해결 완료

| # | 항목 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | ~~Import 오타~~ ✅ | `network.py:5` | Critical |
| 2 | ~~정규화 불일치~~ ✅ | `model.py:86-91` | Critical |
| 3 | ~~NameError (MultilayerPerceptron)~~ ✅ | `main.py:85` | Critical |
| 4 | ~~이중 정규화~~ ✅ | `main.py`, `model.py` | Critical |
| 5 | ~~Gradient batch 정규화~~ ✅ | `network.py:114` | Critical |
| 6 | ~~backward activation_prime z vs a~~ ✅ | `network.py:122` | Critical |
| 7 | ~~Weight init 미구현~~ ✅ | `network.py:61-77` | Important |
| 8 | ~~cross_entropy 호출 오류~~ ✅ | `predict.py:35-37` | Important |
| 9 | ~~클래스 구조 단순화~~ ✅ | Model → Network | Improvement |
| 10 | ~~Adam in-place 뮤테이션~~ ✅ | `optimizer.py:44-63` | Medium |
| 11 | ~~Early stopping 하드코딩~~ ✅ | `model.py:21` | Medium |
| 12 | ~~Seed 재현성~~ ✅ | `main.py:59-60` | Important |
| 13 | ~~Unit tests 부재~~ ✅ | `tests/` (67 tests) | Medium |
| 14 | ~~오타 (timestap)~~ ✅ | `optimizer.py:33` | Low |

### 미해결

| # | 항목 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | Malignant recall < 0.95 | 하이퍼파라미터 튜닝 | Important |
| 2 | 커스텀 activation/loss 시 AttributeError | `network.py:39-40, 49-50` | Medium |
| 3 | MSE + softmax backward 미지원 | `network.py:116` | Medium |
| 4 | sigmoid RuntimeWarning (결과는 정확) | `utils.py:19-21` | Low |
| 5 | `predict.py` 변수명 모듈 섀도잉 | `predict.py:4, 11` | Low |
| 6 | 미사용 코드: `standardize()`, `network/gw_history` | `utils.py:7`, `network.py:17,20-21` | Low |
| 7 | Docstring 위치 오류 | `utils.py`, `optimizer.py`, `network.py` | Style |
| 8 | Type hint 일관성 부재 | `utils.py` 전반 | Style |
| 9 | `fit()` docstring 시그니처 불일치 | `model.py:37-48` | Style |
| 10 | 탭/스페이스 혼용 | `utils.py:18-21` | Style |
