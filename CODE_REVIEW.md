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

### ~~[Critical] `network.py:5` — Import 오타 (`standarize`)~~ ✅ 수정 완료

`standarize` → `standardize` 로 수정 완료.

---

### ~~[Critical] `backward`에서 activation_prime 입력값 혼동~~ ✅ 수정 완료

`network.py:114` (은닉층) — **수정 전:**
```python
current_activation = self.activations[-l]
delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(current_activation)
```

**수정 후:**
```python
z = self.zs[-l]
delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(z)
```

- `relu_prime`은 pre-activation 값(`z`)을 받아야 하므로 `self.activations[-l]`(a) → `self.zs[-l]`(z)로 변경
- `sigmoid_prime`은 `a * (1 - a)` 공식이라 기존 동작에 영향 없음
- 잔여 이슈: 출력층(`network.py:106`) 비-cross_entropy 케이스는 `output_activation`의 derivative를 사용해야 하나 미수정

---

### ~~[Critical] `predict.py:39` — cross_entropy 호출 오류~~ ✅ 수정 완료

`one_hot_encode(y_raw)`를 적용하여 올바른 형태로 전달하도록 수정됨. loss 값도 정상 출력.

```python
# 수정 후 (predict.py:35-38)
y_true_one_hot = one_hot_encode(y_raw)
loss = cross_entropy(y_true_one_hot, probabilities)
print(f"Loss: {loss:.4f}")
```

---

### ~~[Critical] 학습/추론 시 정규화(Standardization) 불일치~~ ✅ 수정 완료

`model.py:76-81`에서 `mean_train`/`std_train`을 저장하고, `fit()` 내부에서 직접 정규화 처리.

```python
self.mean_train = x_train.mean(axis=0)
self.std_train = x_train.std(axis=0) + 1e-08
x_train = (x_train - self.mean_train) / self.std_train
if x_val is not None:
    x_val = (x_val - self.mean_train) / self.std_train
```

`predict.py`도 `model.mean_train` / `model.std_train`을 사용하므로 일관성 확보.

---

### ~~[Critical] `main.py:80` — `MultilayerPerceptron` 클래스 미존재 (NameError)~~ ✅ 수정 완료

`import model` 추가 및 `model.Model(...)` 으로 교체 완료. `MultilayerPerceptron` 참조 제거됨.

---

### ~~[Critical] `main.py:63` + `model.py:79` — 이중 정규화~~ ✅ 수정 완료

`main.py:63`의 수동 정규화 코드를 주석 처리하고, `X_raw`를 그대로 `model.fit()`에 전달하도록 수정됨. 정규화는 `model.fit()` 내부에서만 처리.

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

### ~~[Low] `optimizer.py:33` — 오타~~ ✅ 수정 완료

```python
self.timestep = 0  # "timestap" → "timestep" 수정 완료
```

---

### ~~[Medium] `optimizer.py:54-55` — Adam이 weights를 in-place 뮤테이션~~ ✅ 수정 완료

`new_weights`, `new_biases` 리스트를 생성한 뒤 한 번에 할당하는 방식으로 SGD와 통일됨.

```python
# Adam (새 객체 생성) ✅
new_weights.append(network.weights[i] - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon))
new_biases.append(network.biases[i] - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon))
...
network.weights = new_weights
network.biases = new_biases
```

---

### ~~[Medium] `optimizer.py:36` — Adam 재훈련 시 상태 미초기화~~ ✅ 효과적으로 해결

`model.py`의 `fit()`이 호출될 때마다 새 `Adam` 인스턴스를 생성하므로, 이전 `m_w`, `v_w`, `timestep`이 누적되는 문제가 실질적으로 해결됨.

```python
# model.py:64-67 — fit() 호출 시마다 새 optimizer 생성
if self.solver == "adam":
    optimizer = Adam(self.learning_rate)
else:
    optimizer = Sgd(self.learning_rate)
```

---

## 2. 설계 및 구조 문제

### ~~불필요한 3단 추상화 계층~~ ✅ 수정 완료

`MultilayerPerceptron`과 `Model`이 단일 `Model` 클래스로 통합됨.

```
(이전) MultilayerPerceptron → Model → Network
(현재) Model → Network
```

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

### ~~`split.py`가 프로젝트에서 미사용~~ ✅ 수정 완료

`split.py` 파일이 삭제됨. `main.py`에서 직접 split 처리.

---

### Early stopping의 제한적 동작

**문제 1: `early_stopping_rounds`가 외부에서 설정 불가 (model.py:134)**

`MultilayerPerceptron.__init__`에 파라미터가 없어 항상 10으로 하드코딩됩니다.

```python
# model.py:134 — 하드코딩
early_stopping_rounds = 10
```

사용자가 `MultilayerPerceptron(early_stopping_rounds=20)` 형태로 설정할 수 없습니다.

**수정:**
```python
# __init__ 파라미터 추가
def __init__(self, ..., early_stopping_rounds=10):
    ...
    self.early_stopping_rounds = early_stopping_rounds

# fit에서 self 참조
early_stopping_rounds = self.early_stopping_rounds
```

**문제 2: Validation data 없으면 early stopping이 완전히 비활성화 (model.py:69)**

```python
if x_val is not None and y_val is not None:
    ...
    if patience >= early_stopping_rounds:  # 이 블록 자체가 실행 안 됨
        ...
```

`x_val`/`y_val`을 넘기지 않으면 early stopping 로직 전체가 무시되며, 경고 없이 `epochs`까지 풀로 학습됩니다. `best_weights`도 `None`인 채 학습이 끝납니다.

**수정:** validation data가 없을 경우 명시적 경고를 출력하거나, train loss 기반 early stopping으로 fallback 처리합니다.

```python
if x_val is None or y_val is None:
    print("Warning: early_stopping_rounds is set but no validation data provided. Early stopping disabled.")
```

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
| `relu_prime(x: np.ndarray)` | ✅ `np.matrix` → `np.ndarray` 수정 완료 |
| `softmax(x: np.ndarray)` | 올바름 |

- ~~`np.matrix`는 deprecated입니다. `np.ndarray`로 통일하세요.~~ ✅ 수정 완료
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
| 1 | ~~Import 오타 (`standarize` → `standardize`)~~ ✅ | `network.py:5` 수정 완료 | Critical |
| 2 | ~~정규화 불일치~~ ✅ | `model.py:76-81` 수정 완료 | Critical |
| 3 | ~~`main.py`에서 `MultilayerPerceptron` NameError (미임포트)~~ ✅ | `main.py:80` 수정 완료 | Critical |
| 4 | ~~`main.py` + `model.fit()` 이중 정규화~~ ✅ | `main.py:63`, `model.py:79` 수정 완료 | Critical |
| 5 | Gradient를 batch_size로 정규화 | `network.py:108-109` | Critical |
| 6 | He/Xavier weight initialization 실제 구현 | `network.py:61-68` | Important |
| 7 | ~~backward에서 z vs a 일관성 확보~~ ✅ | `network.py:114` 수정 완료 (`106` 출력층 잔여) | Important |
| 8 | ~~cross_entropy 호출 오류~~ ✅ | `predict.py:39` 수정 완료 | Important |
| 9 | ~~불필요한 3단 클래스 구조~~ ✅ | `Model → Network`으로 통합 완료 | Improvement |
| 10 | Docstring 위치, type hint 일관성 (`relu_prime` ✅) | `utils.py`, 전체 | Style |
| 11 | ~~오타 수정 (`timestap` → `timestep`)~~ ✅ | `optimizer.py:33` 수정 완료 | Low |
| 12 | `early_stopping_rounds` 하드코딩 → `__init__` 파라미터로 노출 | `model.py:73` | Medium |
| 13 | Validation data 없을 시 early stopping 경고 없이 비활성화 | `model.py:122` | Medium |
| 14 | ~~Adam in-place 뮤테이션 → SGD와 방식 불일치~~ ✅ | `optimizer.py:59-60` 수정 완료 | Medium |
| 15 | ~~Adam 재훈련 시 m/v/timestep 상태 미초기화~~ ✅ | `fit()` 호출 시 새 인스턴스 생성으로 해결 | Medium |
