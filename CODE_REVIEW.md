# Code Review - Multilayer Perceptron

> 시니어 AI 모델 개발자 관점의 코드 리뷰  
> 리뷰 일자: 2026-04-14

---

## 목차

1. [버그 및 에러](#1-버그-및-에러)
2. [설계 및 구조 문제](#2-설계-및-구조-문제)
3. [파이썬 스타일](#3-파이썬-스타일-pythonic)
4. [추가 개선 권장사항](#4-추가-개선-권장사항)
5. [요약 (우선순위별)](#5-요약-우선순위별)

---

## 1. 버그 및 에러

### [Critical] `network.py:5` — Import 오타 (`standarize`)

```python
# network.py:5
from utils import standarize, sigmoid, ...
```

`utils.py`에는 `standardize`로 정의되어 있으나, import 시 `standarize`(d 누락)로 참조합니다.  
**현재 이 코드는 import 시점에서 `ImportError`가 발생합니다.**

**수정:**
```python
from utils import standardize, sigmoid, ...
```

---

### [Critical] `backward`에서 activation_prime 입력값 혼동

`network.py:106` (출력층):
```python
delta = self.loss_prime(y_true, y_pred) * self.activation_prime(y_pred)
```

`network.py:114` (은닉층):
```python
delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(current_activation)
```

- `sigmoid_prime`은 activated value(`a`)를 받도록 구현되어 있어 현재 동작하지만, `relu_prime`은 본래 pre-activation 값(`z`)을 받아야 합니다.
- ReLU의 경우 `a > 0`과 `z > 0`이 동치이므로 우연히 동작하지만, **의도가 불분명**하고 Leaky ReLU 등으로 확장 시 즉시 깨집니다.
- 출력층 backward에서 비-cross_entropy 케이스에 `self.activation_prime`을 쓰는데, 출력층은 `output_activation`의 derivative를 사용해야 합니다.

**수정:** `self.zs`에 저장된 pre-activation 값을 사용하도록 변경하고, activation_prime 함수들도 `z`를 받도록 통일합니다.

```python
# 은닉층 backprop
z = self.zs[-l]
delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(z)
```

---

### [Critical] `predict.py:39` — cross_entropy 호출 오류

```python
y_true = np.array([1 if label == 'M' else 0 for label in y_raw])  # shape: (N,)
y_prob = probabilities[:, 1]                                        # shape: (N,)
loss = cross_entropy(y_true, y_prob)  # cross_entropy는 one-hot을 기대함!
```

- `cross_entropy` 함수는 one-hot encoded `y_true`를 기대하지만, 여기서는 scalar label(0/1)을 넘깁니다.
- 계산된 loss 값이 무의미하며, 반환되지도 않습니다 (dead code).

**수정:** one-hot encoding을 적용하거나, binary cross entropy 함수를 별도로 사용합니다.

---

### [Critical] 학습/추론 시 정규화(Standardization) 불일치

```python
# main_train.py:69 (학습)
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

# predict.py:30 (추론)
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

학습 시 사용한 `mean`/`std`를 저장하지 않고, 추론 시 **테스트 데이터 자체의** `mean`/`std`로 정규화합니다.  
Train/test 분포가 다를 경우 **완전히 다른 feature space**에서 예측하게 됩니다.

**수정:** 학습 시 `mean`/`std`를 모델과 함께 직렬화하여 저장하고, 추론 시 동일한 값을 사용합니다.

```python
# 학습 시
self.train_mean = X_raw.mean(axis=0)
self.train_std = X_raw.std(axis=0)
X = (X_raw - self.train_mean) / self.train_std

# 추론 시
X = (X_raw - model.train_mean) / model.train_std
```

---

### [Medium] `network.py:39-40` — 존재하지 않는 속성 접근

```python
else:
    self.activation = self.config.activation
    self.activation_prime = self.config.activation_prime  # NetworkConfig에 없는 필드!
```

`NetworkConfig` dataclass에는 `activation_prime`, `loss_prime` 필드가 정의되어 있지 않습니다.  
커스텀 activation/loss를 문자열이 아닌 함수로 넘기면 `AttributeError`가 발생합니다.

**수정:** `NetworkConfig`에 optional 필드를 추가하거나, 커스텀 함수 지원 로직을 제거합니다.

---

### [Medium] Gradient를 batch_size로 나누지 않음

`network.py:108-109`:
```python
nabla_w[-1] = np.dot(self.activations[-2].T, delta)
nabla_b[-1] = np.sum(delta, axis=0)
```

Mini-batch의 gradient를 `batch_size`로 나누지 않습니다.  
Batch size가 커지면 gradient magnitude도 비례하여 커져, learning rate 튜닝이 batch_size에 종속됩니다.

**수정:**
```python
batch_size = y_true.shape[0]
nabla_w[-1] = np.dot(self.activations[-2].T, delta) / batch_size
nabla_b[-1] = np.sum(delta, axis=0) / batch_size
```

---

### [Low] `optimizer.py:33` — 오타

```python
self.timestap = 0  # "timestep"이 올바른 표기
```

---

## 2. 설계 및 구조 문제

### 불필요한 3단 추상화 계층

```
MultilayerPerceptron → Model → Network
```

- `Model`은 단순히 `Network`에 학습 루프를 추가하는 역할
- `MultilayerPerceptron`은 `Model`의 얇은 래퍼

`MultilayerPerceptron`과 `Model`을 하나로 합치는 것이 자연스럽습니다.

---

### `MultilayerPerceptron.fit`의 죽은 코드

```python
# multilayer_perceptron.py:43-46
if hasattr(X, 'values'):
    input_size = X.shape[1]
else:
    input_size = X.shape[1]  # 양쪽 분기가 동일 → 분기 불필요
```

---

### Weight initialization 전략 미사용

`weight_init="HeUniform"` 파라미터를 받지만 실제로 사용하지 않습니다.

```python
# network.py:66-67 (실제 초기화)
scale = 0.1
self.weights.append(np.random.randn(input_dim, output_dim) * scale)
```

- ReLU activation → **He initialization** 필요
- Sigmoid activation → **Xavier initialization** 필요
- 현재 `scale=0.1` 고정값은 깊은 네트워크에서 vanishing gradient를 유발할 수 있음

**수정 예시 (He initialization):**
```python
scale = np.sqrt(2.0 / input_dim)
self.weights.append(np.random.randn(input_dim, output_dim) * scale)
```

---

### `random_seed` 미사용

`MultilayerPerceptron`에서 `random_seed` 파라미터를 받지만 `np.random.seed()`에 적용하지 않아 **재현성이 보장되지 않습니다**.

---

### `split.py`가 프로젝트에서 미사용

`main_train.py`에서 자체적으로 split을 수행하고 있어, `split.py`의 `split_config` 함수는 dead code입니다.

---

### Early stopping의 제한적 동작

- Validation data 없이 호출 시 early stopping 로직이 아예 실행되지 않음
- `best_weights`가 `None`인 채로 학습이 끝날 수 있음
- 이 자체는 버그는 아니지만, 명시적으로 문서화하거나 경고를 추가해야 함

---

## 3. 파이썬 스타일 (Pythonic)

### Docstring 위치 오류

`utils.py`, `optimizer.py`에서 docstring을 함수 **위에** 작성했습니다. 이는 파이썬 docstring이 아니라 단순 문자열 리터럴입니다.

```python
# 잘못된 위치 (현재)
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

---

### Type hint 일관성 부재

| 함수 | 현재 상태 |
|------|-----------|
| `sigmoid(x)` | type hint 없음 |
| `relu_prime(x: np.matrix)` | deprecated type `np.matrix` 사용 |
| `softmax(x: np.ndarray)` | 올바름 |

- `np.matrix`는 deprecated입니다. `np.ndarray`로 통일하세요.
- 모든 public 함수에 type hint를 일관되게 적용하세요.

---

### 기타 스타일 이슈

- `model.py`에서 `import copy`는 `copy.deepcopy`만을 위해 사용 → `from copy import deepcopy`가 더 명시적
- `predict.py`의 `import pandas as pd`와 `import pickle`이 있지만 상단 import 순서가 PEP 8과 다름 (stdlib → third-party → local 순서)
- 매직 넘버 (`scale = 0.1`, `epsilon = 1e-15`) → 상수로 정의 권장

---

## 4. 추가 개선 권장사항

| 항목 | 설명 | 우선도 |
|------|------|--------|
| **Gradient clipping** | gradient explosion 방지를 위해 추가 권장 | 높음 |
| **L2 regularization** | overfitting 방지를 위한 weight decay 미구현 | 높음 |
| **Learning rate scheduler** | 고정 LR만 지원. cosine annealing 등 추가 고려 | 중간 |
| **Batch normalization** | 학습 안정성을 위해 고려 | 중간 |
| **Logging** | `print` 대신 `logging` 모듈 사용 권장 | 낮음 |
| **Reproducibility** | `np.random.seed()` 또는 `np.random.Generator` 사용 | 높음 |
| **Data pipeline** | 정규화 파라미터(mean, std)를 모델과 함께 직렬화 | 높음 |
| **Unit tests** | 각 컴포넌트(forward, backward, optimizer)에 대한 테스트 부재 | 중간 |

---

## 5. 요약 (우선순위별)

| 순위 | 항목 | 위치 | 심각도 |
|------|------|------|--------|
| 1 | Import 오타 (`standarize` → `standardize`) | `network.py:5` | Critical |
| 2 | 추론 시 정규화 mean/std 불일치 | `predict.py:30` | Critical |
| 3 | Gradient를 batch_size로 정규화 | `network.py:108-109` | Critical |
| 4 | He/Xavier weight initialization 실제 구현 | `network.py:61-68` | Important |
| 5 | backward에서 z vs a 일관성 확보 | `network.py:106,114` | Important |
| 6 | cross_entropy 호출 오류 | `predict.py:39` | Important |
| 7 | 불필요한 3단 클래스 구조 단순화 | 전체 구조 | Improvement |
| 8 | Docstring 위치, type hint 일관성 | `utils.py`, 전체 | Style |
| 9 | 오타 수정 (`timestap` → `timestep`) | `optimizer.py:33` | Low |
