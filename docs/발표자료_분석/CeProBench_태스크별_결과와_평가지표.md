# CeProBench 태스크별 결과와 평가지표 정리

## 1. 전체 구조

CeProBench는 CeProAgents의 성능을 Knowledge, Concept, Parameter의 세 차원과 여섯 개 세부 태스크로 평가한다.

| 차원 | 태스크 | 핵심 평가 질문 |
|---|---|---|
| Knowledge | Knowledge Augment | 검색 결과를 이용해 정확하고 논리적이며 완전한 답변을 만들었는가? |
| Knowledge | Knowledge Extract | 문서의 엔티티와 관계 그래프를 얼마나 정확히 추출했는가? |
| Concept | Concept Parse | PFD 이미지의 장치와 연결을 얼마나 잘 읽었는가? |
| Concept | Concept Complete | 가려진 장치의 실제 타입을 후보 안에 포함했는가? |
| Concept | Concept Design | 자연어로 생성한 공정 그래프가 실행 가능하고 요구사항과 일치하는가? |
| Parameter | Parameter Optimize | Aspen 파라미터 조정으로 수율·순도·비용을 얼마나 개선했는가? |

논문의 평가식은 부록 24쪽의 `S3 Definition of Evaluation Metrics`에 정리되어 있다.

---

## 2. Knowledge Augment

### 2.1 입력과 출력

- 입력: 화학공학 질문
- 처리: Web, Knowledge Base, Knowledge Graph에서 관련 정보를 검색하고 Report Agent가 통합
- 출력: 자연어 답변
- 평가 데이터: 총 243개 Q&A
- 세부 분야: 공정 경로 선택, 촉매·조건 선택, 생산 방법·공정 흐름, 분리 방법 선택

### 2.2 평가 기준

다른 LLM이 질문 `Q`, 모델 답변 `A`, 기준 답변 `R`을 읽고 네 항목에 각각 0~100점을 부여한다.

\[
S_d=\operatorname{Judge}(Q,A,R\mid d)
\]

| 지표 | 평가 내용 |
|---|---|
| Correctness | 반응, 온도, 압력, 장치, 수치와 단위가 사실적으로 정확한가? 허구의 공정 단계를 만들지 않았는가? |
| Rationality | 원료→반응→분리 순서와 열역학·반응속도 설명이 논리적이며 자기모순이 없는가? |
| Clarity | 간결하고 읽기 쉬우며 전문용어를 적절히 사용했는가? |
| Completeness | 질문에 필요한 운전 조건, 촉매, 장치와 핵심 내용을 빠뜨리지 않았는가? |

평가 코드: `CeProAgents/CeProBench/utils/qa_eval.py`

### 2.3 주요 결과

- Gemini-3.0-Pro Correctness: 78.8점
- Gemini는 GPT-5.0보다 Correctness가 12.5점, Claude-4.5-Opus보다 9.4점 높음
- Gemini, GPT, Claude의 Rationality는 91점 이상
- Clarity는 모든 모델이 약 96점으로 차이가 작음
- Qwen-3.0-Max Rationality: 85.5점

Ablation 결과:

- Knowledge Base Agent 제거 시 선두 모델의 Correctness가 17.6점 감소
- Completeness가 16.2점 감소
- Knowledge Graph Agent 제거 시 Gemini의 Rationality가 92.7에서 81.2로 감소

### 2.4 해석상 주의점

이 평가는 정답 문자열 비교가 아니라 LLM-as-a-Judge 방식이다. Judge 모델의 문체 선호, 답변 길이 선호, 자기 모델 계열 선호 등이 점수에 영향을 줄 수 있다.

---

## 3. Knowledge Extract

### 3.1 입력과 출력

- 입력: 비정형 기술문서
- 출력: 엔티티 목록과 `(subject, predicate, object)` 관계 삼중항
- 목표: 문서를 Knowledge Graph로 변환

### 3.2 엔티티 평가

논문의 표기는 Accuracy이지만 계산식은 일반적인 Precision과 같다.

\[
A=\frac{N_{match}}{N_{pred}},\qquad
R=\frac{N_{match}}{N_{gt}}
\]

\[
F1=\frac{2AR}{A+R}
\]

- `Nmatch`: 정답과 매칭된 엔티티 수
- `Npred`: 예측한 엔티티 수
- `Ngt`: Ground Truth 엔티티 수
- `A`: 논문에서는 Accuracy라 부르지만 수학적으로 Precision
- `R`: Recall

실제 엔티티 매칭 방식:

1. 임베딩 유사도로 Top-5 후보 검색
2. 정확한 문자열이면 자동 일치
3. 유사도 0.98 이상이면 자동 일치
4. 유사도 0.65 미만이면 자동 불일치
5. 0.65~0.98이면 LLM Judge가 동의어·별칭인지 판정

평가 코드:

- `CeProAgents/CeProBench/utils/kg_entity_eval.py`
- `CeProAgents/CeProBench/utils/kg_graph_metrics.py`

### 3.3 그래프 평가

#### MEC: Mapping-based Edge Connectivity

정답 관계의 두 엔티티가 예측 그래프에서 경로로 연결되는 비율이다.

\[
MEC=\frac{\text{예측 그래프에서 연결된 정답 관계 수}}{\text{전체 정답 관계 수}}
\]

직접 간선이 아니어도 `A→X→B`처럼 경로로 이어져 있으면 연결된 것으로 처리될 수 있다. 높을수록 정답 그래프의 연결 구조를 많이 보존한 것이다.

#### MED: Mapping-based Edge Distance

매핑된 정답 엔티티 쌍이 예측 그래프에서 몇 단계 떨어져 있는지 최단 경로로 측정하고, 그래프의 평균 거리로 정규화한다.

\[
MED=\frac{1}{|E_{conn}|}\sum\frac{d(M(u),M(v))}{\bar d}
\]

MEC는 연결 여부를, MED는 연결 경로의 거리를 본다. MED는 일반적인 정확도처럼 단순히 높을수록 좋은 지표가 아니다.

### 3.4 주요 결과

| 모델 | Entity F1 |
|---|---:|
| Gemini-3.0-Flash | 0.431 |
| DeepSeek-V3 | 0.383 |
| GPT-5.0-Mini | 0.376 |
| Qwen-3.0-Max | 0.308 |
| Claude-4.5-Haiku | 0.127 |

Gemini-3.0-Flash는 Precision/Accuracy 0.340, Recall 0.631이었다. 많은 엔티티를 찾았지만 오추출도 적지 않았다는 뜻이다.

Graph MEC 결과:

- DeepSeek-V3: 0.348
- Qwen-3.0-Max: 0.324
- Gemini-3.0-Flash: 0.313
- GPT-5.0-Mini: 0.169

엔티티 인식은 Gemini가 가장 좋았지만 관계 구조 보존은 DeepSeek가 가장 높았다.

---

## 4. Concept Parse

### 4.1 입력과 출력

- 입력: PFD 이미지
- 출력: `equipments`와 `connections` JSON
- 데이터: PFD 113개, 장치 986개, 연결 1,172개

### 4.2 평가식

\[
A_{Eq}=\frac{N_{matched\ equipment}}{N_{predicted\ equipment}},\qquad
R_{Eq}=\frac{N_{matched\ equipment}}{N_{groundtruth\ equipment}}
\]

\[
A_{Cn}=\frac{N_{matched\ connection}}{N_{predicted\ connection}},\qquad
R_{Cn}=\frac{N_{matched\ connection}}{N_{groundtruth\ connection}}
\]

여기서도 `Accuracy`는 사실상 Precision이다.

장치는 `identifier`와 `type`을 결합한 문자열을 비교한다. 기본 유사도 임계값은 0.4이며 Python `SequenceMatcher`를 사용한다. 연결은 `source`와 `target`이 방향까지 맞아야 한다.

평가 코드: `CeProAgents/CeProBench/eval_concept_parse.py`

### 4.3 주요 결과

Gemini-3.0-Pro:

- Equipment Accuracy/Precision: 약 72%
- Equipment Recall: 약 88%
- Connection Precision과 Recall: 모두 65% 이상

GPT-5.0:

- Equipment Accuracy/Precision: 약 68%
- Equipment Recall: 약 37%
- Connection Accuracy/Precision: 약 47%

GPT는 추출한 장치의 정확성은 비교적 높지만 실제 장치를 많이 놓치는 보수적 경향을 보였다. Gemini는 Precision과 Recall이 모두 높아 균형이 좋았다.

### 4.4 무엇을 평가하지 않는가

- 장치 bounding box나 픽셀 위치
- 장치 포트의 좌표
- 심볼 검출 자체의 정확도
- 엄밀한 그래프 동형성

최종 JSON의 문자열과 연결 방향을 평가한다.

---

## 5. Concept Complete

### 5.1 입력과 출력

- 입력: 장치 하나가 마스킹된 PFD와 Parser의 JSON
- 데이터: 69개 masked PFD
- 출력: 가능한 장치 타입 후보 10개를 확률 순으로 나열한 `completion`

### 5.2 Top-K Accuracy

\[
A@K=\delta(y\in\hat{Y}_K)
\]

- `y`: 실제 가려진 장치 타입
- `ŶK`: 모델의 상위 K개 후보
- 정답이 상위 K개에 있으면 1, 없으면 0
- 모든 샘플의 0/1을 평균하여 Top-K Accuracy 계산

예를 들어 정답이 후보 2위에 있으면 Top-1은 0, Top-2 이상은 1이다.

코드는 소문자화와 공백 제거 후 정확한 타입 문자열 일치를 사용한다. 동의어나 유사 타입은 인정하지 않는다.

평가 코드: `CeProAgents/CeProBench/eval_concept_complete.py`

### 5.3 주요 결과

Gemini-3.0-Pro:

- Top-1: 약 40.6%
- Top-5: 약 68.1%
- Top-10: 약 69%

그림상 Top-10 결과:

- Claude-4.5-Opus: 약 65%
- QwenVL-3.0-Max: 약 45%
- GPT-5.0: 약 30%

Gemini는 약 40%의 사례에서 첫 후보로 정답을 맞혔고, 후보를 다섯 개까지 허용하면 약 68%에서 정답을 포함했다.

---

## 6. Concept Design

### 6.1 입력과 출력

- 입력: 자연어 공정 설명
- 출력: 장치와 연결로 구성된 P&ID JSON
- 데이터: 30개 사례
  - Simple 20개
  - Standard 6개
  - Hard 4개

### 6.2 판정 기준

LLM Judge가 자연어 요구사항과 생성 JSON을 비교해 세 범주로 판정한다.

#### Correct

- 심각한 공학 오류가 없음
- 자연어 공정 설명과 일치
- 요구된 recycle, purge, heating, cooling 등을 포함
- 흐름 방향과 장치 순서가 맞음

#### Valid / Reasonable but Incorrect

그래프는 실행 가능하지만 텍스트 요구와 완전히 일치하지 않는다. 예를 들어 purge, 냉각기, 태그, 연결 순서 등이 빠졌지만 전체 공정은 연결되어 있다.

#### Invalid

- 주 공정이 끊김
- 제품·배출 종점이 없음
- 고립된 장치나 맹목적인 recycle loop
- 순환에 필요한 펌프·압축기·밸브가 없음
- 분리 없이 반응기 출구가 같은 반응기로 직접 복귀
- 원료에서 제품까지 도달할 수 없음

평가 코드:

- `CeProAgents/CeProBench/eval_concept_generate.py`
- `CeProAgents/CeProBench/eval_generate_prompts.py`

### 6.3 평가식

\[
R_{valid}=\frac{N_{valid}}{N_{total}},\qquad
R_{correct}=\frac{N_{correct}}{N_{total}}
\]

그림에서는 Correct, Valid, Invalid를 서로 배타적인 세 범주로 표시한다. 전체 실행 가능 비율을 보고 싶으면 `Correct + Valid`로 해석하는 것이 자연스럽다.

### 6.4 주요 결과

| 모델 | 설정 | Correct | Valid | Invalid |
|---|---|---:|---:|---:|
| Gemini-3.0-Pro | Correction 없음 | 47% | 23% | 30% |
| Gemini-3.0-Pro | Full | 63% | 17% | 20% |
| GPT-5.0 | Correction 없음 | 37% | 37% | 27% |
| GPT-5.0 | Full | 53% | 33% | 13% |
| Claude-4.5-Opus | Full | 23% | 37% | 40% |
| DeepSeek-V3 | Full | 10% | 33% | 57% |
| Qwen-3.0-Max | Full | 10% | 37% | 53% |

Correction Agent를 사용하면 Gemini의 Correct가 47%에서 63%, GPT의 Correct가 37%에서 53%로 증가한다. 반올림 때문에 일부 행의 합계가 99% 또는 101%일 수 있다.

### 6.5 해석상 주의점

유효성과 정확성은 규칙 기반 시뮬레이터가 아니라 LLM Judge가 판정한다. Judge가 장문의 JSON에서 공학적 위반을 놓치거나 과도하게 엄격하게 판단할 가능성이 있다.

---

## 7. Parameter Optimize

### 7.1 입력과 출력

- 입력: Aspen Plus `.bkp`, P&ID JSON, 입력·출력 설정, 파라미터 범위, 최적화 목표
- 출력: 반복별 파라미터, Aspen 결과, Analyst 피드백
- 데이터: 20개 Aspen 시나리오
  - Yield Optimization 8개
  - Purity Optimization 9개
  - Comprehensive Economic Optimization 3개

조정 변수는 반응기 온도, 길이·부피, 체류시간, 증류탑 단수, 환류비, Feed Stage 등이다.

### 7.2 논문의 평가식

\[
r_Y=\frac{Y_{opt}}{Y_{init}},\qquad
r_P=\frac{P_{opt}}{P_{init}},\qquad
r_C=\frac{C_{opt}}{C_{init}}
\]

\[
r_{eff}=r_Yr_P,\qquad
r_{overall}=\frac{r_{eff}}{r_C}
\]

- Yield와 Purity가 증가하면 점수가 커짐
- Cost가 감소하면 `rC < 1`이 되어 Overall 점수가 커짐
- 최종 논문 그림은 이 비율을 모델 사이에서 0~1로 정규화한 값
- 논문에는 정확한 정규화식이 명시되어 있지 않음

### 7.3 실제 평가 코드

각 반복의 원점수는 다음처럼 계산한다.

\[
score=\frac{purity\times yield}{cost}
\]

전체 반복 중 이 값이 가장 큰 반복을 `best`로 선택한다. 따라서 마지막 반복이 아니라 중간 반복이 최종 평가값이 될 수 있다.

- Yield-only 태스크: purity를 1.0으로 고정
- Purity-only 태스크: yield를 100으로 고정
- Combined 태스크: 실제 purity와 yield를 모두 사용

평가 코드: `CeProAgents/CeProBench/eval_parameter_optimize.py`

### 7.4 주요 결과

- Gemini-3.0-Pro Effective Score: 약 0.99
- GPT-5.0 Purity Score: 0.71
- GPT-5.0 Cost Score: 0.65
- Claude-4.5-Opus Yield Score: 0.62
- DeepSeek-V3 Cost Score: 0.26
- DeepSeek-V3 Comprehensive Score: 0.26

모델별 반복 효율:

| 모델 | Best Iteration 중앙값 | Total Iterations 중앙값 | 해석 |
|---|---:|---:|---|
| Gemini-3.0-Pro | 약 4 | 약 7 | 좋은 값을 빠르게 찾고 비교적 적절히 종료 |
| GPT-5.0 | 약 5.5 | 약 19 | 좋은 값을 찾지만 반복을 오래 계속함 |
| Claude/DeepSeek/Qwen | 대체로 3 이내 | 약 5~5.5 | 일찍 종료하지만 종합점수가 낮아 조기 수렴 가능성 |

반복 횟수가 적다고 항상 좋은 것은 아니다. 낮은 점수로 일찍 끝나는 것은 효율이 아니라 premature convergence일 수 있다.

---

# 8. TP, FP, FN, TN과 혼동행렬

## 8.1 혼동행렬

| 실제값 \ 예측값 | Positive 예측 | Negative 예측 |
|---|---:|---:|
| 실제 Positive | TP: True Positive | FN: False Negative |
| 실제 Negative | FP: False Positive | TN: True Negative |

PFD 장치 추출을 기준으로 해석하면:

- TP(참양성): 실제 장치를 올바르게 추출
- FP(거짓양성·위양성): 실제로 없는 장치를 있다고 추출. 과다 추출 또는 환각
- FN(거짓음성·위음성): 실제 장치를 놓침. 누락 또는 미검출
- TN(참음성): 장치가 아닌 것을 장치가 아니라고 판단

장치·엔티티 리스트 추출에서는 모든 가능한 Negative 후보를 정의하지 않으므로 TN을 계산하기 어렵다.

## 8.2 일반적인 Accuracy

\[
Accuracy=\frac{TP+TN}{TP+FP+FN+TN}
\]

전체 판단 중 맞게 판단한 비율이다. 그러나 불균형 데이터에서는 성능을 과장할 수 있다. 1,000개 영역 중 장치가 10개일 때 모든 영역을 장치가 아니라고 예측하면 장치는 하나도 못 찾지만 Accuracy는 99%가 된다.

## 8.3 Precision

\[
Precision=\frac{TP}{TP+FP}
\]

모델이 찾았다고 말한 것 중 실제 정답의 비율이다.

- Precision이 낮음 → FP가 많음 → 환각·과다 추출이 많음
- 질문: “모델의 예측을 얼마나 믿을 수 있는가?”

## 8.4 Recall

\[
Recall=\frac{TP}{TP+FN}
\]

실제로 존재하는 대상 중 모델이 찾아낸 비율이다.

- Recall이 낮음 → FN이 많음 → 실제 장치·연결 누락이 많음
- 질문: “실제 대상을 얼마나 놓치지 않았는가?”

Recall은 Sensitivity 또는 True Positive Rate(TPR)라고도 한다.

## 8.5 F1-score

\[
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
\]

Precision과 Recall의 조화평균이다. 한쪽만 높고 다른 쪽이 낮으면 점수가 크게 낮아진다.

## 8.6 계산 예시

실제 PFD 장치 10개, 모델 예측 8개, 정답과 일치 6개라면:

- TP = 6
- FP = 8-6 = 2
- FN = 10-6 = 4
- TN = 정의하기 어려움

\[
Precision=6/8=0.75
\]

\[
Recall=6/10=0.60
\]

\[
F1=\frac{2\times0.75\times0.60}{0.75+0.60}=0.667
\]

| 지표 | 결과 | 의미 |
|---|---:|---|
| Precision | 75% | 모델이 찾은 장치 중 75%가 정답 |
| Recall | 60% | 실제 장치 중 60%를 찾음 |
| F1 | 66.7% | Precision과 Recall의 균형 점수 |
| 일반 Accuracy | 계산 불가 | TN이 정의되지 않음 |

CeProAgents 논문은 이 예시의 Precision 75%를 `Equipment Accuracy`라고 부른다.

## 8.7 위양성률과 위음성률

\[
FPR=\frac{FP}{FP+TN}=1-Specificity
\]

\[
FNR=\frac{FN}{TP+FN}=1-Recall
\]

Recall이 60%라면 FNR은 40%이며, 실제 장치의 40%를 놓쳤다는 뜻이다.

---

# 9. 관련 그래프

## 9.1 Confusion Matrix Heatmap

TP·FP·FN·TN 또는 여러 장치 클래스 사이의 오분류를 색으로 표시한다. 대각선 값이 클수록 올바른 분류가 많고, 비대각선 값이 클수록 특정 클래스 사이의 혼동이 많다.

CeProAgents는 장치 타입별 혼동행렬을 보고하지 않고 전체 장치·연결 매칭 수를 집계한다.

## 9.2 Precision-Recall Curve

- 가로축: Recall
- 세로축: Precision
- 판정 임계값을 바꾸며 Precision과 Recall의 변화를 표시
- 오른쪽 위에 가까울수록 좋음
- 곡선 아래 면적: AUPRC

불균형 데이터나 객체 검출에서는 ROC보다 PR Curve가 더 유용한 경우가 많다.

## 9.3 ROC Curve

- 가로축: FPR
- 세로축: TPR=Recall
- 왼쪽 위에 가까울수록 좋음
- 곡선 아래 면적: ROC-AUC

그러나 CeProAgents의 리스트 기반 장치 추출은 TN과 확률 점수를 명시하지 않아 ROC Curve를 직접 만들기 어렵다.

## 9.4 논문 Figure 3a

Concept Parsing의 장치와 연결에 대해:

- 가로축: Recall
- 세로축: 논문 표기의 Accuracy, 실제로는 Precision

오른쪽 위에 가까울수록 실제 장치를 많이 찾으면서 오추출도 적다.

## 9.5 논문 Figure S2c

문자열 매칭 임계값 `τ`를 0.3~0.9로 바꾸며 Node Precision, Node Recall, Edge Precision, Edge Recall을 측정한다. 이는 일반적인 분류기 PR Curve가 아니라 평가 매칭 기준을 엄격하게 했을 때 점수가 어떻게 바뀌는지 보여주는 민감도 분석이다.

---

# 10. 발표 핵심 문장

> TP는 실제 장치를 올바르게 찾은 경우, FP는 존재하지 않는 장치를 잘못 생성한 경우, FN은 실제 장치를 놓친 경우이며 TN은 장치가 아닌 것을 장치가 아니라고 판정한 경우다. Precision은 모델이 찾았다고 한 것 중 정답 비율이고 Recall은 실제 장치 중 찾아낸 비율이다. F1은 두 지표의 조화평균이다. CeProAgents의 Concept Parse와 Knowledge Extract에서는 모든 Negative 후보를 정의하지 않으므로 일반적인 TN 기반 Accuracy를 계산하기 어렵다. 논문이 Accuracy라고 표기한 `Nmatch/Npred`는 수학적으로 Precision에 해당한다.

짧은 암기 방식:

- Precision: 모델의 예측을 얼마나 믿을 수 있는가?
- Recall: 실제 대상을 얼마나 놓치지 않았는가?
- F1: Precision과 Recall을 함께 얼마나 잘 만족하는가?
- Accuracy: 전체 판단 중 얼마나 맞았는가?

