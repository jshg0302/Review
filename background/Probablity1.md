# Expectation
$\displaystyle \mathbb{E}[X]=\int_{\mathcal{X}} xp(x)dx$

$\displaystyle \mathbb{E}[X]=\sum_{x\in\mathcal{X}} xp(x)$

## The expectation of some function $f(x)$
$\displaystyle \mathbb{E}_{x \sim p(x)}[f]=\int f(x)p(x)dx$

$\displaystyle \mathbb{E}_{x \sim p(x)}[f]=\sum_x f(x)p(x)$

$\displaystyle \mathbb{E}_{x}[f|y]=\sum_x f(x)p(x|y)$

## Properites of Expectation

$\mathbb{E}[aX+b]=a\mathbb{E}[X]+b$
$\displaystyle \mathbb{E}[\sum_{i=1}^{n} X_i] = \sum_{i=1}^{n} \mathbb{E}[X_i]$

---

# Variance
$ \begin{aligned}
\mathbb{V}[X]=\mathbb{E}[(X-\mu)^2]
&= \int(x-\mu)^2p(x)dx \\
&= \int x^2p(x)-\int 2\mu xp(x) + \int \mu^2 p(x) \\
&= \int x^2p(x)- 2\mu \int xp(x) + \mu^2 \int p(x) \\
&= \mathbb{E}[X^2] - 2\mu^2 + \mu^2 \\
&= \mathbb{E}[X^2]-\mu^2 \\
\end{aligned}
$

---

# Common probablity distribution
### discrete
- Bernoulli
- Binomial
- Poisson
- Multinoulli

### continuous
- Gaussian
- Student t
- Gamma
- Chi-square
- Beta

---

## Bernoulli
- 1회 시행에서 결과과 성공(1)과 실패(0) 중 하나만 나오는 이산확률분포
- param : $\mu$
- $ Ber(x|\mu)=
\begin{cases}
\mu, & x=0\\
1-\mu, &x=0
\end{cases}
$
- $\mathbb{E}[X]=\mu$
- $\mathbb{V}[X]=\mu(1-\mu)$

---

## Logistic (or sigmoid) function
$y=ax+b$ 이러한 선형결합의 범위를 확률(0~1)로 맞춰주기 위해
$x \to \theta_0 + \theta_1 x \to \sigma(\theta_0 + \theta_1 x)$
$\sigma(t) = \frac{1}{1+e^{-t}} \to f(x;\theta)=\frac{1}{1+e^{-(\theta_0 + \theta_1 x)}}$

---

## Binomial (이항)
- 확률 $p$인 베르누이 시행을 독립적으로 $n$번 반복했을 때, 성공 횟수 $X$의 확률 분포.
- param : $n$, $p$
- $x_n \sim Ber(.|p)$ for $n=1:N$
- $\displaystyle s = \sum_{n=1}^{N} \mathbb{I}(x_n =1)$
- $Bin(s|N,p)=\binom{N}{s}p^s(1-p)^{N-s}$ where $\binom{N}{s}=\frac{N!}{(N-k)!k!}$
- $\mathbb{E}[X]=Np$
- $\mathbb{V}[X]=Np(1-p)$

---

## Poisson
- 어떤 시간/공간 안에 사건이 평균 $\lambda$번 발생할 떄, $k$번 발생할 확률
- param : $\lambda$
- $P(X=k|\lambda)=\frac{\lambda^ke^{-\lambda}}{k!}$
- $\mathbb{E}[X]=\lambda$
- $\mathbb{V}[X]=\lambda$
- ex) 1분 동안 교차로에 진입하는 차량이 6대일 때 ($\lambda=6$)
    - $P(X=0)=\frac{6^0e^{-6}}{0!}$
    - $P(X=8)=\frac{6^8e^{-8}}{8!}$

---

## Gaussian
- param : $\mu$, $\sigma$
- $\displaystyle N(x|\mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp(-\frac{1}{2\sigma^2}(x-\mu)^2)$

---

## Student t
- 가우시안 분포에서 outlier에 맞는 분포 (꼬리가 두꺼움)
- $\displaystyle \Tau(y|\mu, \sigma^2, \nu) \propto [1+\frac{1}{\nu}(\frac{y-\mu}{\sigma})^2]^{-(\frac{\nu+1}{2})}$
- 자유도 $\nu$가 무한이면 정규분포
- $\nu > 1, \mathbb{E}[X]=0$
- $\nu > 2, \mathbb{V}[X]=\frac{\nu}{\nu-2}$

---

## Laplace
- 이상치에 덜 민감하고 꼬리가 두꺼운 분포
- $\displaystyle Laplace(y|\mu,b)=\frac{1}{2b}e^{-\frac{|y-\mu|}{b}}$
- $\mathbb{E}[X]=\mu$
- $\mathbb{V}[X]=2b^2$

---

## Beta
- 어떤 확률 $p$에 대한 확신이 없는 경우($p$분포를 알고 싶은 경우)
- 확률/비율처럼 0~1 사이 값을 모델링할 때
예) 성공확률 p, 점유율, 정규화된 비율
- $\displaystyle Beta(x|a,b)=\frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1}$ 
where
$\displaystyle B(a,b)=\int_0^1 p^{a-1}(1-p)^{b-1}dp = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$
- $\mathbb{E}[X]=\frac{a}{a+b}$
- $\mathbb{V}(X)=\frac{ab}{(a+b)^2(a+b+1)}$

- a=b=1이면 uniform
- a>b이면 1쪽이 몰리고, b>a이면 0쪽에 몰림
- a<1,b<1이면 양 끝이 뾰족해지고, a>1, b>1이면 중앙이 뾰족

### Beta 분포에 왜 a-1, b-1이 있는지?
- 감마함수(팩토리얼)와 정확히 맞물리고, 정수일 때 해석·정규화·업데이트가 모두 깔끔해지기 때문
- 아무 정보가 없는 기본형. uniform a=b=1
### Gamma function $\Gamma(.)$
- 팩토리얼 $n!$을 실수/연속값까지 확장해준 함수
- $\Gamma(n)=(n-1)!,\quad n\in\mathbb{N}$

---

## Gamma
- 포아송 발생률 $\lambda$가 불확실할 때
- 양수(real positive) 값: 시간, 길이, 에너지, 강도 등
예) 대기시간, 사건 발생률(rate) 불확실성, 포아송 과정과 연결
- $\displaystyle Gamma(x|a,b)=\frac{b^a}{\Gamma(a)}x^{a-1}e^{-xb}$ \
 or
$\displaystyle Gamma(a,s)=\frac{1}{\Gamma(a)s^a}x^{a-1}e^{-\frac{x}{s}}$
- $\mathbb{E}[X]=\frac{a}{b}$ or $as$
- $\mathbb{V}(X)=\frac{a}{b^2}$ or $as^2$

- $a$ (shape): 모양 결정 (작으면 0 근처 뾰족, 커지면 종 모양처럼)
- $b$ (rate): 오른쪽 꼬리의 감쇠 정도 (클수록 빨리 감소 → 평균 작아짐)
- scale $s$: 커질수록 더 넓게 퍼짐(평균 커짐) ($s=\frac{1}{b}$)

---

## Special cases of the Gamma
### Exponential distribution
- 왜 쓰나? (이유)
    - **사건이 랜덤하게 발생**하는 상황에서 다음 사건까지의 대기시간을 모델링하기 좋음
    - 기억 없음(memoryless) 성질이 있음:
$P(X>s+t∣X>s)=P(X>t)$
→ “이미 $s$만큼 기다렸다고 해서 앞으로 더/덜 기다릴 확률이 바뀌지 않는” 상황에 잘 맞음
    - Poisson process에서 자연스럽게 등장:
“단위시간 사건 수가 Poisson”이면 “사건 간 시간 간격”은 Exponential

- $p(x)=
\begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & otherwise
\end{cases}
= Gamma(1,\lambda)
$

### Chi-squared distribution
- 왜 쓰나? (이유)
    - 분산(variance) 추론의 핵심 분포라서
    - “관측값과 기대값의 차이(제곱합)” 같은 형태가 자연스럽게 나오기 때문에
    - 그래서 통계에서 매우 자주 나오는 검정들이 카이제곱을 기반으로 함
- $\chi_\nu^2 = Gamma(\frac{\nu}{2},\frac{1}{2})$

---

## Monte Carlo approximation
- 어떤 분포 $p(x)$에 대해
$\displaystyle I=\mathbb{E_{x\sim p}[f(x)]}=\int f(x)p(x)dx$
이 적분을 계산하기 어려우면 
샘플 $x^{(1)}, ..., x^{(N)} \sim p(x)$를 뽑고
$\displaystyle \^I_N = \frac{1}{N}\sum_{i=1}^N f(x^{(i)})$
로 근사