# Information theory 요약

## 정보이론이 다루는 것
- “관측/메시지에 들어있는 정보량”과 “불확실성”을 확률로 정량화하는 분야
- 머신러닝에서는
  - 분포의 불확실성(엔트로피),
  - 두 분포의 차이(KL divergence),
  - 최적 부호화/압축(코딩 길이),
  - 학습 목적함수(크로스엔트로피, NLL)
  같은 형태로 자주 등장한다.

---

## 정보량(Self-information)
- 어떤 사건 \(x\)가 일어났을 때 얻는 정보량:
  \[
  I(x) = -\log p(x)
  \]
- 확률이 작을수록(\(p(x)\downarrow\)) 정보량은 커진다(더 “놀라운” 사건).

---

## 엔트로피(Entropy): 평균 정보량 = 평균 놀라움
- 분포의 평균적인 불확실성:
  \[
  H(X) = \mathbb{E}[I(X)] = -\mathbb{E}_{x\sim p}\big[\log p(x)\big]
  \]
- 이산형:
  \[
  H(X) = -\sum_x p(x)\log p(x)
  \]
- 성질/직관
  - 균등분포일 때 엔트로피가 최대(가장 예측하기 어려움)
  - 한 값에 확률이 몰릴수록 엔트로피가 감소(더 확실함)
- 단위
  - \(\log_2\): bits
  - \(\ln\): nats

---

## 조건부 엔트로피(Conditional Entropy)
- \(Y\)를 알고 있을 때 \(X\)의 남은 불확실성:
  \[
  H(X\mid Y)= -\mathbb{E}_{(x,y)\sim p}\big[\log p(x\mid y)\big]
  \]
- 직관
  - 정보를 더 알면 불확실성은 줄어든다: 보통 \(H(X\mid Y)\le H(X)\)

---

## 결합 엔트로피와 체인 룰
- 두 변수의 전체 불확실성:
  \[
  H(X,Y)= -\mathbb{E}\big[\log p(x,y)\big]
  \]
- 체인 룰:
  \[
  H(X,Y)=H(X)+H(Y\mid X)=H(Y)+H(X\mid Y)
  \]

---

## 상호정보량(Mutual Information)
- \(X\)가 \(Y\)에 대해 제공하는 정보(의존성의 양):
  \[
  I(X;Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)
  \]
- 또 다른 형태:
  \[
  I(X;Y)=\mathbb{E}\left[\log \frac{p(x,y)}{p(x)p(y)}\right]
  \]
- 직관/성질
  - \(X\)와 \(Y\)가 독립이면 \(I(X;Y)=0\)
  - 항상 \(I(X;Y)\ge 0\)

---

## Cross Entropy (교차 엔트로피)

### 정의
\[
H(p,q) \;=\; -\mathbb{E}_{x\sim p}\big[\log q(x)\big]
\]
- 의미: **\(p\)에서 샘플이 나오는데**, 그걸 **\(q\)** 로 “설명/코딩”한다고 할 때 드는 평균 비용.

### 이산형(Discrete)
\[
H(p,q) = -\sum_x p(x)\log q(x)
\]

### 연속형(Continuous)
\[
H(p,q) = -\int p(x)\log q(x)\,dx
\]

---

## Perplexity
- 언어모델/시퀀스 모델에서 자주 쓰는 “평균적으로 몇 개의 선택지 중 하나를 고르는 수준인가”를 나타내는 지표.
- 정의(이산 분포, 로그 밑 2 기준):
  \[
  \mathrm{PPL}(p) = 2^{H(p)}
  \]
- 모델 \(q\)를 평가할 때(데이터 분포 \(p\)에서 샘플이 나온다고 보면) 보통 “크로스엔트로피 기반 퍼플렉서티”를 쓴다:
  \[
  \mathrm{PPL}(p,q)=\exp\big(H(p,q)\big)
  =\exp\Big(-\mathbb{E}_{x\sim p}[\log q(x)]\Big)
  \]
  (로그가 자연로그면 \(\exp\), \(\log_2\)면 \(2^{(\cdot)}\))
- 직관:
  - 낮을수록 좋음(모델이 데이터에 더 높은 확률을 줌)
  - 완벽 예측에 가까울수록 1에 가까워짐

---

## Kullback–Leibler Divergence
- 정의:
  \[
  D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right]
  =\mathbb{E}_{x\sim p}[\log p(x)]-\mathbb{E}_{x\sim p}[\log q(x)]
  \]
- 해석(코딩 관점):
  - \(p\)에서 나온 데이터를 \(q\)로 코딩할 때 “추가로 드는 평균 비트/정보량” (상대 엔트로피)
- 크로스엔트로피와 관계:
  \[
  H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)
  \]
  → \(p\)가 고정이면 크로스엔트로피 최소화는 KL 최소화와 동일

---

## Jensen’s inequality (젠슨 부등식)
- 볼록(convex) 함수 \(\phi\)에 대해:
  \[
  \phi(\mathbb{E}[X])\le \mathbb{E}[\phi(X)]
  \]
- 오목(concave) 함수는 부등호가 반대.
- 로그는 오목함수이므로:
  \[
  \log \mathbb{E}[X] \ge \mathbb{E}[\log X]
  \]
- 정보이론/변분추론에서 “로그 안으로 기대값을 집어넣기” vs “기대값 안으로 로그를 빼기” 차이를 만들며,
  ELBO 유도나 KL의 비음이 아닌 성질 증명에 핵심으로 쓰인다.

---

## KL의 non-negativity (항상 0 이상)
- 결론:
  \[
  D_{\mathrm{KL}}(p\|q)\ge 0,\quad \text{and } D_{\mathrm{KL}}(p\|q)=0 \Leftrightarrow p=q \text{ (a.e.)}
  \]
- 한 줄 증명 아이디어(젠슨 사용):
  - \(\log\)가 오목이므로 \(\mathbb{E}[\log Z]\le \log \mathbb{E}[Z]\)
  - \(Z=\frac{q(x)}{p(x)}\)로 두면
    \[
    \mathbb{E}_{p}\left[\log \frac{q(x)}{p(x)}\right]
    \le \log \mathbb{E}_{p}\left[\frac{q(x)}{p(x)}\right]
    =\log \int q(x)dx = 0
    \]
  - 양변에 마이너스를 붙이면
    \[
    \mathbb{E}_{p}\left[\log \frac{p(x)}{q(x)}\right]\ge 0
    \]
- 직관:
  - \(q\)가 \(p\)와 다르면 “맞는 곳엔 덜 주고, 틀린 곳엔 더 주는” 확률 배분 손해가 평균적으로 생긴다.

---

## Forward KL vs Reverse KL (왜 다르게 행동하나)
- Forward KL:
  \[
  D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{p}\big[\log p(x)-\log q(x)\big]
  \]
  - “\(p\)에서 자주 나오는 영역”에 대해 \(q\)가 확률을 충분히 주지 않으면 큰 페널티
  - 특징: **mode-covering** 경향 (여러 모드를 넓게 덮으려 함)
  - 특히 \(q(x)=0\)인데 \(p(x)>0\)이면 발산(무한대) → 지원집합(support) 미스매치에 매우 민감

- Reverse KL:
  \[
  D_{\mathrm{KL}}(q\|p)=\mathbb{E}_{q}\big[\log q(x)-\log p(x)\big]
  \]
  - “\(q\)가 샘플링하는 영역”에서 \(p\)가 낮으면 페널티
  - 특징: **mode-seeking** 경향 (한 모드에 붙어버리기 쉬움)
  - \(p\)의 여러 모드를 모두 덮기보다, “가장 설명 잘 되는 한 모드”를 고르는 쪽으로 갈 수 있음

- 실무 감각:
  - 최대우도(MLE)는 보통 \(D_{\mathrm{KL}}(p_{\text{data}}\|q_\theta)\)를 줄이는 방향(Forward KL 성격)
  - 변분추론(VI)은 종종 \(D_{\mathrm{KL}}(q_\phi(z)\|p(z\mid x))\) 형태(Reverse KL 성격)

---

## Mutual Information (상호정보량) 자세한 설명
- 정의(여러 형태가 모두 동일):
  \[
  I(X;Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)
  \]
  \[
  I(X;Y)=D_{\mathrm{KL}}(p(x,y)\|p(x)p(y))
  \]
  \[
  I(X;Y)=\mathbb{E}_{p(x,y)}\left[\log\frac{p(x,y)}{p(x)p(y)}\right]
  \]
- 해석:
  - \(X\)와 \(Y\)가 독립이면 \(p(x,y)=p(x)p(y)\)이므로 MI=0
  - MI가 크면 “서로를 많이 예측할 수 있다” (의존성이 강함)
- 포인트:
  - MI는 “선형”만 보는 상관계수보다 더 일반적(비선형 의존도도 반영)
  - 그러나 MI를 정확히 계산하려면 결합분포와 주변분포가 필요해서 고차원에서는 어렵다(추정 문제가 생김)

---

## InfoNCE (대조학습에서의 MI 하한)
- 대조학습(contrastive learning)은 “양성쌍(positive)”은 가깝게, “음성쌍(negative)”은 멀게 만드는 목적함수를 쓴다.
- 대표 손실(한 앵커 \(x\)에 대해 정답 \(y^+\), 음성 \(y^-\)들):
  \[
  \mathcal{L}_{\text{InfoNCE}}
  =-\mathbb{E}\left[\log \frac{\exp(s(x,y^+)/\tau)}
  {\exp(s(x,y^+)/\tau)+\sum_{i=1}^{N-1}\exp(s(x,y_i^-)/\tau)}\right]
  \]
  - \(s(\cdot,\cdot)\): 유사도(내적/코사인 등)
  - \(\tau\): temperature
- 중요한 이론적 메시지:
  - InfoNCE는 특정 조건에서 \(I(X;Y)\)의 **하한(lower bound)**을 최대화하는 것과 연결된다.
  - 음성 샘플 수 \(N\)이 클수록 MI 하한이 타이트해지는 방향으로 알려져 있다.
- 직관:
  - “진짜 짝(y⁺)”을 많은 후보(음성 포함) 중에서 맞히는 분류 문제로 바꾼 것
  - 잘 맞힐수록 \(X\)가 \(Y\)에 대한 정보를 많이 담고 있다고 볼 수 있다.

---

## MIC (Mutual Information estimation / Mutual Information Classifier 관점)
- 문맥마다 약어가 조금 애매하게 쓰일 수 있는데, 교재/슬라이드에서 보통 다음 중 하나 의미로 등장한다.
  - Mutual Information **estimation**: MI를 직접 추정하는 다양한 방법(예: MINE, NWJ, InfoNCE 계열)
  - Mutual Information **Classifier**(=contrastive classifier): “맞는 짝을 분류로 맞히기” 방식으로 MI 하한을 학습하는 프레임
- 핵심 아이디어는 공통적으로:
  - MI는 직접 계산이 어려우니,
  - 분류/대조 목적함수로 바꾸거나(InfoNCE),
  - 변분 하한(variational lower bound)을 최적화해 우회적으로 키운다.

---

## 머신러닝에서의 연결(핵심만)
- 최대우도추정(MLE)은 보통
  \[
  \min_\theta \; -\mathbb{E}_{x\sim p_{\text{data}}}[\log q_\theta(x)]
  \]
  즉 데이터 분포 \(p_{\text{data}}\)와 모델 \(q_\theta\) 사이의 크로스엔트로피를 줄이는 것.
- 분류 손실(크로스엔트로피), 변분추론(ELBO의 KL), 정보보틀넥/표현학습(상호정보) 등에서 반복 등장.

---

## 기억 포인트(짧게)
- 정보량: \(-\log p(x)\)
- 엔트로피: 평균 정보량(불확실성)
- 크로스엔트로피: “모델로 설명할 때 드는 평균 비용”
- KL: “모델이 진짜 분포를 얼마나 못 맞추는지” (항상 0 이상)
- 상호정보: “둘이 얼마나 의존적인지” (독립이면 0)
