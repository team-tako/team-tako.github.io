---
layout: post
title:  "Charformer: Fast Character Transformers via Gradient-based Subword Tokenization"
date:   2021-08-21T14:25:52-05:00
author: 보이차보이
categories: Papers

---

# Charformer: Fast Character Transformers viaGradient-based Subword Tokenization

## Introduction

### 서브워드<sup>subword</sup> 기반 토큰화의 한계점

- 빈도수<sup>frequency</sup>  에 의존하여 토큰을 서브워드로 나눈다. 
  - 어휘<sup>lexical</sup> 혹은 의미<sup>semantic</sup>적 유사성을 반영하지 못한다. 
  - 빈도수가 낮은 단어(rare words)나 변형<sup>perturbation</sup>에 취약하다. 
- 다언어 모델<sup>multilingual model</sup>의 경우 저자원<sup>low-resource</sup>언어는 여러개의 서브워드로 나뉘게 되는데, 이는 성능을 저하시킨다. 
- 사전학습<sup>pretraining</sup> 과 다운스트림 태스크에서의 단어 분포 차이가 발생한다. 
  - 모델이 사용되는 새로운 세팅에 맞게 알고리즘을 수정할 수 없다. 

### GBST(Gradient Based Subword Tokenization)

- end-to-end learning이 가능하다. 

- 대량의 레이블이 없는<sup>unlabeled</sup> 데이터를 통해서 잠재된<sup>latent</sup> 서브워드 표상<sup>representation</sup> 을 학습한다. 

- "GBST  learns  a  position-wise soft selection over candidate subword blocks by scoring them with a scoring network. In contrast to prior tokenization-free methods, GBST learns interpretable latent subwords, which enables easy inspection of lexical representations and is more efficient than other byte-based models "

  ![image-1](/_posts/img/Charformer-1.png)

- 성능<sup>performance</sup>

  - GLUE 및 문서 분류 태스크에서 `byte-level` 모델들을 능가하고, `BERT` 혹은 `T5`와 동일한 성능을 낸다.
  - 소셜 미디어 혐오 표현 분류<sup>toxicity detection</sup> 태스크에서 모든 모델들을 능가한다. 
  - 다언어 모델에서도 기존 방법들과 동일하거나 더 좋은 성능을 낸다. 

- 효율성<sup>efficiency</sup>

  - 동일한 크기의 `ByT5`보다 2배 더 메모리 효율이 좋고, 10%에서 93%더 빠르다.~~(10퍼에서93퍼?몬소리야)~~ 
  - `mT5`보다 학습 속도가 28% 빠르고, 1/3의 크기로도 동일한 성능을 낸다. 

- 오픈소스 [코드][https://github.com/google-research/google-research/tree/master/charformer]

## Charformer

### GBST

$$
X\in\mathbb{R}^{L\times d}
$$

​	$L$ : 입력 $char$ 의 길이

​	$d$ : $char$ 임베딩의 차원수<sup>dimension</sup>

- 모델이 $char$마다 가장 적합한 **서브워드 블록**을 선택함으로서 잠재된 서브워드 분할<sup>segmantation</sup> 방법을 학습할 수 있도록하는 것이 핵심 아이디어이다. 
- 서브워드 블록은 전체에서 $b$ 길이를 갖는 $char$ span을 의미한다. 

![image-2](/_posts/img/charformer-2.png)

- Strided-pooling function $F : \mathbb{R}^{b\times d}\in\mathbb{R}^d$
  $$
  X_b = [F(X_{(i:i+b)}); F(X_{(i+s):(i+s)+b});...]
  $$

  - $s = b$ , $b\in1,...,M$
  - $b$ : block size
  - $s$ : stride size
  - $M$ : maximum block size

- **Offsets** 위와 같은 방법으로 $b=2$ 일 때, [a,b], [c,d]가 stride되고, [b,c]의 subword는 만들어지지 않는 문제가 있다. 이를 해결하기 위해 offset을 문장 길이(s)만큼 더하면서 stride하게 되는데, 이때 연산량이 매우 많아지기 때문에 우리는 대신 X에 먼저 1D convolution을 한다. 

- **Intra-block Positions** Block 내에서의 $char$ 의 순서를 유지하기 위해서 F를 적용하기 이전에 `positional embedding`을 사용한다. 

- **Block Scoring Network** 각 block마다의 score를 생성하는 parameterized function

  - 모든 $char$ 에 대해서 가장 적합한 subword block $X_{b,i}$를 선택하도록 모델 학습.

  - 해당 $char$ 가 포함된 block 0부터 M까지를 candidate로 하는 softmax function 을 사용한다. 
    $$
    P_i = softmax([p_{0,i}, p_{1,i}...,p_{M,i}])
    $$

- **Latent Subwords**
  $$
  \hat{X}_i = \sum_b^mP_{b,i}X_{b,i}
  $$

  - 직관적으로는 모델이 각 position에 해당하는 가장 이상적인 subword block을 학습하길 기대한다.

- **Downsampling**

  - 마지막에는 $d_s$ factor를 통해서 downsampling을 진행한다(차원축소)
    $$
    F_D : \mathbb{R}^{L\times d}\rightarrow \mathbb{R}^{\frac{L}{d_s}\times d}
    $$

### Transformer Stack

- 기본적인 `Transformer` encoder-decoder 모델을 사용한다.  
  - `ByT5` 와 유사하게 encoder에 더 많은 parameter를 배치하였다. 
  - encoder layer : 24
  - encoder layer : 24
  - $134M$ parameters

## Experiments

### English Dataset

- GLUE, document classification(sentiment, news)

![image-1](/_posts/img/charformer-3.png)

### Non-standard English Dataset

- [hatespeech] Civil comments, Wikipedia comments

![image-2](/_posts/img/charformer-4.png)

### Multilingual

- Cross-lingual 

![image-3](/_posts/img/charformer-5.png)

## Conclusion

- 빠르고 범용적인 문자단위<sup>character</sup> 토크나이징 기법
