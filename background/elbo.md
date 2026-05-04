# ELBO (Evidence Lower Bound) 정리

ELBO는 잠재변수(latent variable) 모델에서 **로그 우도 $\log p_\theta(x)$** 를 직접 최적화하기 어려울 때,
대신 **그 하한(lower bound)** 을 최대화하는 목적함수다.  
변분추론(VI)과 VAE는 같은 ELBO를 “해석/구현” 관점만 다르게 쓴다고 보면 된다.

---

## 왜 ELBO가 필요한가

잠재변수 $z$가 있는 생성모델은 보통

$ \displaystyle
p_\theta(x) = \int p_\theta(x,z)\,dz
$

이고 우리가 키우고 싶은 건

$
\log p_\theta(x) = \log \int p_\theta(x,z)\,dz
$

인데, 이 적분(또는 합)이 고차원에서 대부분 **계산 불가능**하다.

그래서 “적분을 직접 하지 않고도” 최적화할 수 있는 우회 목적함수로 ELBO를 만든다.

---

## 핵심 아이디어: 근사 posterior $q_\phi(z\mid x)$

진짜 posterior는

$ \displaystyle
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}
$

인데 $p_\theta(x)$ 자체가 적분이라 계산이 어렵다.

그래서 계산 가능한 분포 $q_\phi(z\mid x)$를 도입해
$p_\theta(z\mid x)$를 **근사**한다.

- $p_\theta(z\mid x)$: 진짜 posterior (하지만 계산 어려움)
- $q_\phi(z\mid x)$: 근사 posterior (학습으로 맞추는 분포)

---

## ELBO 유도 (Jensen’s inequality 버전)

$
\log p_\theta(x)=\log \int p_\theta(x,z)\,dz
$

여기에 $q_\phi(z\mid x)$를 곱하고 나누면

$ \displaystyle
\log p_\theta(x)
= \log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
= \log \mathbb{E}_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
$

로그는 오목함수이므로 (Jensen)

$
\log \mathbb{E}[A] \ge \mathbb{E}[\log A]
$

따라서

$ \displaystyle
\log p_\theta(x)\ge 
\mathbb{E}_{q_\phi(z\mid x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
$

오른쪽이 **ELBO**다:

$
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
$

---

## ELBO의 대표 형태 2가지

### (형태 A) “재구성 - KL” 형태 (VAE에서 가장 흔함)

모델을 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$로 분해하면

$
\mathrm{ELBO}(x)
= \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
$

- $\mathbb{E}[\log p_\theta(x\mid z)]$: **데이터 적합(재구성)**
- $D_{\mathrm{KL}}(q\|p)$: **정규화/규제 (posterior를 prior 쪽으로)**

VAE 구현에서 흔히 “recon loss + KL loss”로 나뉘는 게 이 형태다.
(보통은 최대화 대신 최소화로 바꿔서 **$-\mathrm{ELBO}$** 를 loss로 둔다.)

---

### (형태 B) “log joint + entropy” 형태 (VI에서 직관적)

$
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x,z)]
+H\big(q_\phi(z\mid x)\big)
$

여기서 엔트로피는

$
H(q)= -\mathbb{E}_q[\log q]
$

즉,
- 첫 항: joint를 높이고
- 둘째 항: $q$가 너무 날카롭게 한 점에 붙지 않도록(불확실성 유지)

라는 관점으로 볼 수 있다.

---

## ELBO가 “하한”인 이유 (핵심 항등식)

아래 항등식이 ELBO의 정체를 가장 정확히 보여준다:

$
\log p_\theta(x)
= \
\mathrm{ELBO}(x)
+
D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)
$

KL은 항상 0 이상이므로

$
\mathrm{ELBO}(x)\le \log p_\theta(x)
$

즉 ELBO는 **로그 우도의 lower bound**이다.

그리고 ELBO를 키우면(=로그 우도에 가까워지려면)
**bound gap**인 $D_{\mathrm{KL}}(q\|p_\theta)$도 같이 줄어드는 방향이다.

---

## KL의 non-negativity (왜 KL은 항상 0 이상인가)

$
D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{p}\left[\log\frac{p(x)}{q(x)}\right]\ge 0
$

한 줄 핵심 아이디어:
- $Z=\frac{q(x)}{p(x)}$로 두면 $\mathbb{E}_p[Z]=\int q(x)dx=1$
- 로그의 오목성(Jensen)으로 $\mathbb{E}[\log Z] \le \log \mathbb{E}[Z]=\log 1=0$
- 양변에 마이너스를 붙이면 $\mathbb{E}_p[\log(p/q)]\ge 0$

즉 “평균적으로는” 잘못된 분포 $q$로 코딩/설명하면 손해가 생긴다.

---

## Forward KL vs Reverse KL (ELBO/VI에서 왜 중요해지나)

두 KL은 방향이 다르다:

- Forward KL:
  $
  D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{p}[\log p-\log q]
  $
  - $p$가 자주 나오는 영역에서 $q$가 확률을 안 주면 큰 벌점
  - **mode-covering** 경향(모드들을 넓게 덮으려 함)

- Reverse KL:
  $
  D_{\mathrm{KL}}(q\|p)=\mathbb{E}_{q}[\log q-\log p]
  $
  - $q$가 샘플링하는 영역에서 $p$가 낮으면 벌점
  - **mode-seeking** 경향(한 모드에 붙기 쉬움)

왜 VI에서 자주 “Reverse KL”이 나오나?
- 많은 변분추론은 $D_{\mathrm{KL}}(q_\phi(z)\|p(z\mid x))$ 형태를 최소화하는데,
  이게 바로 Reverse KL 성격이라 **모드 하나를 선택**해 근사하는 경향이 생길 수 있다.

---

## 변분추론(VI) 관점에서 ELBO

VI의 목표는 “posterior 근사”가 핵심이다:

- 타깃: $p_\theta(z\mid x)$
- 근사: $q_\phi(z)$ 또는 $q_\phi(z\mid x)$

많은 VI는 다음을 최소화한다:

$
\min_\phi \; D_{\mathrm{KL}}\big(q_\phi(z)\,\|\,p_\theta(z\mid x)\big)
$

하지만 $p_\theta(z\mid x)$는 정규화 상수 $p_\theta(x)$가 필요해서 직접 계산이 어렵다.  
그래서 위 문제는 ELBO 최대화로 바뀐다(동치).

정리하면:
- VI는 “**posterior 근사 최적화 문제**”
- ELBO는 그 문제를 계산 가능하게 만든 “**실행 가능한 목적함수**”

---

## VAE 관점에서 ELBO

VAE는 VI를 신경망으로 구현한 형태다:

- prior: $p(z)$ (보통 $\mathcal{N}(0,I)$)
- decoder(생성): $p_\theta(x\mid z)$
- encoder(근사 posterior): $q_\phi(z\mid x)$

VAE 학습 목표:

$
\max_{\theta,\phi}\sum_i \mathrm{ELBO}(x_i)
$

실제로는 $-\mathrm{ELBO}$를 loss로 두고 최소화한다:

$ 
\mathcal{L}_{\text{VAE}}
= \
-\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+
D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
$

---

## Reparameterization trick (왜 필요하고 뭘 하는가)

문제: 
- $\mathbb{E}_{q_\phi(z\mid x)}[\cdot]$ 안에 $\phi$가 들어가면,
  샘플링 $z\sim q_\phi$ 때문에 gradient가 막힐 수 있다.

해결(가우시안의 경우):
- $q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x),\Sigma_\phi(x))$일 때
  $
  z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
  $
- 샘플링을 $\epsilon$로 “바깥으로” 빼서, $\phi$에 대한 미분이 가능하게 만든다.

---

## Mutual Information (MI)와 ELBO/VAE의 연결

상호정보량:

$
I(X;Z)=D_{\mathrm{KL}}(p(x,z)\|p(x)p(z))
=H(Z)-H(Z\mid X)
$

직관:
- $I(X;Z)$가 크면 $Z$가 $X$에 대한 정보를 많이 담고 있다(좋은 표현 학습일 수도 있음)
- 하지만 VAE는 KL 항 때문에 $q(z\mid x)$를 $p(z)$에 가깝게 만들려 하므로
  $Z$가 $X$ 정보를 덜 담게 되는 방향(“정보 병목”)이 생길 수 있다.

이게 흔히 말하는 **posterior collapse**와도 연결된다:
- KL 항이 너무 강하면 $q(z\mid x)\approx p(z)$가 되어 $z$가 $x$를 거의 안 담음
- 그러면 decoder가 $z$ 없이도 $x$를 예측해버리는 현상이 발생

---

## MLE와 NLL (ELBO를 loss로 바꾸기 전에 꼭 필요한 연결)

### 우리가 보통 하고 싶은 것: 모델이 데이터에 높은 확률을 주게 만들기
데이터가 \(x\)라고 할 때, 모델이 \(p_\theta(x)\)를 크게 주면
“모델이 이 데이터를 그럴듯하다고 본다”는 뜻이다.

---

### MLE (Maximum Likelihood Estimation): 우도 최대화
관측 데이터 \(x_1,\dots,x_N\)가 주어졌을 때,
모델 파라미터 \(\theta\)를 다음처럼 고른다:

\[
\hat{\theta}
=\arg\max_\theta \prod_{i=1}^{N} p_\theta(x_i)
\]

곱은 계산이 불편하니까 로그를 취해(단조 증가라 최적해는 같음)

\[
\hat{\theta}
=\arg\max_\theta \sum_{i=1}^{N}\log p_\theta(x_i)
\]

이게 **로그우도(log-likelihood) 최대화**다.

---

### NLL (Negative Log-Likelihood): 로그우도의 음수 = loss
딥러닝은 보통 "최소화"를 하므로,
최대화 문제를 최소화 문제로 바꿔서 학습한다.

\[
\arg\max_\theta \sum_{i=1}^{N}\log p_\theta(x_i)
\quad \Longleftrightarrow \quad
\arg\min_\theta \sum_{i=1}^{N} -\log p_\theta(x_i)
\]

여기서

\[
\mathrm{NLL}(x_i)= -\log p_\theta(x_i)
\]

즉 **NLL은 로그우도를 음수로 만든 값**이고,
학습에서 흔히 말하는 "loss"로 바로 쓴다.

---

### 직관 한 줄
- \(\log p_\theta(x)\)를 키우는 것 = 모델이 데이터에 높은 확률을 주게 만드는 것
- \(-\log p_\theta(x)\)를 줄이는 것 = 같은 목표를 "minimize" 형태로 바꾼 것

---

### (분류에서) Cross-Entropy와 NLL이 같은 이유 (짧게)
정답 클래스가 \(y\)이고 모델이 \(q_\theta(y\mid x)\)를 예측할 때,
샘플 1개에 대한 NLL은

\[
-\log q_\theta(y\mid x)
\]

이며, one-hot 정답 분포에 대한 cross-entropy와 동일한 형태가 된다.

---

## ELBO에서 Loss로 가는 과정 (학습 관점)

딥러닝 최적화는 보통 **loss를 최소화(minimize)** 한다.  
하지만 ELBO는 정의상 **최대화(maximize)** 하고 싶은 값이다.

따라서 학습에서는 ELBO를 그대로 “maximize”하기보다,
부호를 바꿔서 **Negative ELBO를 loss로 최소화**한다.

\[
\max_{\theta,\phi}\; \mathrm{ELBO}(x)
\quad\Longleftrightarrow\quad
\min_{\theta,\phi}\; \mathcal{L}(x)
\]
\[
\mathcal{L}(x) = -\mathrm{ELBO}(x)
\]

---

### 가장 흔한 형태 (VAE 구현에서 그대로 쓰는 형태)

ELBO가 다음과 같다면

\[
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
\]

loss는

\[
\mathcal{L}_{\text{VAE}}(x)
= \
-\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
\]

즉,

- 첫 항은 **Negative log-likelihood (NLL)** 이다  
  (데이터를 잘 설명할수록 \(\log p_\theta(x\mid z)\)가 커지므로, 앞에 마이너스를 붙여 작아지게 만든다)
- 둘째 항은 **KL penalty** 로 그대로 더한다  
  (posterior가 prior에서 너무 멀어지지 않게 하는 규제)

---

### 데이터셋 전체 objective

데이터 \(\{x_i\}_{i=1}^N\)에 대해

\[
\max_{\theta,\phi}\sum_{i=1}^N \mathrm{ELBO}(x_i)
\quad\Longleftrightarrow\quad
\min_{\theta,\phi}\sum_{i=1}^N \mathcal{L}_{\text{VAE}}(x_i)
\]

실제로는 미니배치 평균으로 학습한다.

---

### (실무 메모) 샘플링 때문에 기대값은 Monte Carlo로 근사

기대값은 보통 샘플 \(z^{(l)}\sim q_\phi(z\mid x)\)로 근사한다:

\[
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
\approx
\frac{1}{L}\sum_{l=1}^L \log p_\theta(x\mid z^{(l)})
\]

그래서 loss는 대략

\[
\mathcal{L}_{\text{VAE}}(x)
\approx
-\frac{1}{L}\sum_{l=1}^L \log p_\theta(x\mid z^{(l)})
+
D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
\]

(보통 \(L=1\)로도 잘 학습된다.)


## 실무에서 자주 쓰는 변형/트릭

- $\beta$-VAE:
  $
  \mathrm{ELBO}_\beta
  =\mathbb{E}[\log p_\theta(x\mid z)]-\beta\,D_{\mathrm{KL}}(q(z\mid x)\|p(z))
  $
  - $\beta>1$: 더 강한 규제(더 disentangle 경향, 대신 재구성 품질 저하 가능)
  - $\beta<1$: 정보 더 담게(재구성 좋아질 수 있으나 prior 정렬 약해짐)

- KL annealing:
  - 학습 초기에 KL 가중치를 0에서 서서히 1로 올려
    posterior collapse를 완화

- free bits / KL floor:
  - KL이 너무 빨리 0으로 가는 걸 막기 위해
    “최소 KL”을 보장하거나 일정량까지는 벌점을 덜 줌

---

## 요약 문장 (문서 맨 아래에 두기 좋은 버전)

- ELBO는 $\log p_\theta(x)$의 하한이며, Jensen 부등식으로 유도된다.
- $\log p_\theta(x)=\mathrm{ELBO}(x)+D_{\mathrm{KL}}(q(z\mid x)\|p_\theta(z\mid x))$ 이라서,
  ELBO를 최대화하면 로그우도를 키우면서 posterior 근사도 좋아지는 방향이 된다.
- VAE는 VI를 신경망(encoder/decoder)으로 구현한 것이고,
  목적함수는 “재구성 항 - KL 항”으로 해석된다.
- KL은 항상 0 이상이며(로그 오목성/Jensen),
  KL 방향에 따라 mode-covering(Forward) / mode-seeking(Reverse) 성향이 달라진다.

---

# ELBO → Loss 유도 과정 (한 줄씩)

우리가 궁극적으로 하고 싶은 건 데이터 우도 최대화, 즉 MLE다:

\[
\max_\theta \log p_\theta(x)
\]

하지만 잠재변수 \(z\)가 있으면

\[
\log p_\theta(x)=\log \int p_\theta(x,z)\,dz
\]

이 적분이 어려워서 \(\log p_\theta(x)\)를 직접 최대화하기 힘들다.

그래서 근사 posterior \(q_\phi(z\mid x)\)를 도입

\[
p_\theta(x)=\int p_\theta(x,z)\,dz=\int q_\phi(z \mid x) \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\]

이후 log를 붙이고 Jensen 부등식으로 고친다.

\[
\log p_\theta(x)
= \
\log \mathbb{E}_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\ge
\mathbb{E}_{q_\phi(z\mid x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\]

오른쪽을 ELBO라고 둔다:

\[
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
= \
\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
\]

그리고 \(p_\theta(x,z)=p_\theta(x\mid z)p(z)\)로 분해하면:

\[
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+\mathbb{E}_{q_\phi(z\mid x)}[\log p(z) - \log q_\phi (z \mid x)]
\]

KL divergense 식에 의해서
\[
\text{KL divergence : }\mathbb{E}_{x\sim p}[\log p(x)-\log q(x)] = D_{\mathrm{KL}}(p\|q)
\]

\[
\mathrm{ELBO}(x)
= \
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
\]

- 첫 항: encoder가 뽑은 $z$로 decoder가 $x$를 잘 만들어내게
- 둘째 항: encoder의 $q(z∣x)$가 prior $p(z)$에서 너무 벗어나지 않게(정규화)

여기까지가 “로그우도(\(\log p\))를 대신할 하한”을 만든 단계다.

---

ELBO 최대화를 loss 최소화로 바꾼다:

\[
\max_{\theta,\phi}\ \mathrm{ELBO}(x)
\quad \Longleftrightarrow \quad
\min_{\theta,\phi}\ -\mathrm{ELBO}(x)
\]

따라서 정의하는 loss는:

\[
\mathcal{L}_{\text{ELBO}}(x) \equiv -\mathrm{ELBO}(x)
\]

이를 전개하면

\[
\mathcal{L}_{\text{ELBO}}(x)
= \
-\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
\]

여기서 첫 항은 "조건부 로그우도"에 마이너스를 붙인 것이므로,
(샘플 1개 기준) **NLL**로 해석된다:

\[
\mathrm{NLL}(x\mid z) = -\log p_\theta(x\mid z)
\]

즉 최종적으로

- 재구성 항: \( \mathbb{E}_{q}[\mathrm{NLL}(x\mid z)] \)
- 정규화 항: \( D_{\mathrm{KL}}(q(z\mid x)\|p(z)) \)

을 더한 것이 VAE에서 쓰는 표준 loss가 된다.
