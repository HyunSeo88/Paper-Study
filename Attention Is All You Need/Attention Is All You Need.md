# Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

---

> [!INFO] BLEU
> 모델이 생성한 문장이 사람이 직접 번역한 문장(정답)과 얼마나 유사한지를 측정하는 자연어 생성 모델의 성능평가 지표로, **n-gram 정확률**을 기반으로 평가함.
> - n-gram: 연속된 n개의 단어 단위로 비교
> - 정확도(precision): 모델이 생성한 번역에서 정답 문장에서 등장한 단어들이 차지하는 비율
> - Brevity Penalty: 너무 짧은 문장이 높은 점수를 받지 않도록 조정
> ###### 한계점
> - 문장이 의미론적 유사도를 직접 반영하지 못함
> - 문장의 유창성을 평가하지 않음
> - 다중 정답이 있을 경우 하나의 정답만 비교하면 한계를 가질 수 있음
>

---
# 1 Introduction
----
Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures $[31,21,13]$.

순환 신경망(Recurrent Neural Networks, RNN), 특히 장단기 기억 네트워크(Long Short-Term Memory, LSTM) [12] 및 게이트 순환 신경망(Gated Recurrent Neural Networks, GRU[^1]) [7]은 언어 모델링(Language Modeling) 및 기계 번역(Machine Translation) [29, 2, 5]과 같은 시퀀스 모델링(Sequence Modeling) 및 변환(Transduction) 문제에서 최첨단(State-of-the-art) 접근법으로 확립되어 왔다.
이후로도 많은 연구들이 순환 언어 모델(Recurrent Language Models) 및 인코더-디코더(Encoder-Decoder) 아키텍처의 한계를 확장하는 데 지속적인 노력을 기울여 왔다

---
[^1]: GRU: LSTM의 경량화 버전으로, 리셋 게이트, 업데이트 게이트 두 개로 구성됨



---


Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_{t}$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

순환 모델(Recurrent models)은 일반적으로 입력 및 출력 시퀀스의 기호 위치(symbol positions)를 따라 연산을 수행한다. 연산 시간에서 위치를 정렬하여, 이전 은닉 상태 $h_{t−1}$ 및 위치 $t$의 입력을 기반으로 새로운 은닉 상태 $h_t$​의 시퀀스를 생성한다. 이러한 본질적인 **순차적 특성**[^2]으로 인해 훈련 예제 내에서 병렬 처리가 불가능하며, 이는 시퀀스 길이가 길어질수록 더욱 중요한 문제가 된다. 특히, 메모리 제약으로 인해 예제 간 배칭(batch processing)이 제한되기 때문이다. 최근 연구에서는 인수 분해 기법(factorization tricks) [18] 및 조건부 연산(conditional computation) [26]을 통해 계산 효율성을 크게 향상시키면서도, 후자의 경우 모델 성능 또한 개선하는 성과를 거두었다. 그러나 순차적 연산이라는 근본적인 제약은 여전히 존재한다.
주의(attention) 메커니즘은 다양한 작업에서 강력한 시퀀스 모델링 및 변환(sequence transduction) 모델의 핵심 요소로 자리 잡았으며, 입력 또는 출력 시퀀스 내 거리와 상관없이 의존성을 모델링할 수 있도록 한다 [2, 16]. 하지만 일부 예외적인 경우 [22]를 제외하면, 이러한 주의 메커니즘은 대부분 순환 신경망과 함께 사용된다.
 

---
[^2]: 순차적 특성: 이전 단계의 연산 결과가 완료되어야 다음 단계의 연산을 수행할 수 있는 구조

> [!NOTE] RNN vs CNN & Attention
> #### RNN의 연산 방식
> - 순차적 특성
> 	- RNN은 시퀀스 데이터를 한 단어씩 순서대로 처리한다.
> 	- 시점 $t$에서의 계산이 끝나야만 시점 $t+1$의 계산을 시작할 수 있다.
> 	- 순차적 종속성 존재, 장기 의존성 문제 발생 가능
> #### CNN의 연산 방식
> - 병렬 연산 가능
> 	- 입력의 여러 위치에 대해 동시에 필터 연산을 수행함, 하나의 계층에서 모든 위치를 병렬로 계산할 수 있음
> #### Attention 모듈
> - 기존 순환 신경망 기반 모델의 한계(장기 의존성, 병렬화 어려움)를 해결하기 위해 Attention 메커니즘이 도입되었다.
> - 하지만, RNN 자체의 한계를 해결하지는 못하기 때문에, 이 논문에서는 RNN을 완전히 제거한 Self-Attention을 제안한다.

---
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

---

---

# 2 Background 
---
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

순차적 연산을 줄이려는 목표는 Extended Neural GPU [20], ByteNet [15], ConvS2S [8]의 기반이 되며, 이들 모델은 합성곱 신경망(CNN)을 기본 구성 요소로 사용하여 모든 입력 및 출력 위치에 대한 은닉 표현(hidden representation)을 병렬로 계산한다. **이러한 모델에서는 두 임의의 입력 또는 출력 위치 간의 신호를 연결하는 데 필요한 연산 횟수가 위치 간 거리에 따라 증가하며**, ConvS2S의 경우 선형적으로, ByteNet의 경우 로그 스케일로 증가한다. 이로 인해 먼 위치 간의 의존성을 학습하는 것이 어려워진다 [11].
Transformer에서는 이러한 연산을 상수 개수로 줄일 수 있으나, 이는 주의(attention) 가중치를 적용한 위치들의 평균화로 인해 효과적인 해상도(effective resolution)가 낮아지는 문제를 초래할 수 있다. 이에 대한 해결책으로, 섹션 3.2에서 설명하는 다중 헤드 주의(Multi-Head Attention)를 도입하여 이 효과를 보완한다.

---

> [!NOTE] CNN이 먼위치의 단어 관계를 학습하기 위해 여러 계층을 거쳐야 하는 이유
> - 한번의 합성곱 연산에서 볼 수 있는 단어의 개수 = 필터의 개수
> - CNN은 local 정보를 먼저 학습하고, 그 후 계층을 쌓아가며 점점 더 넓은 범위를 학습함.
> - 즉, CNN은 바로 먼 위치의 단어를 연결할 수 없고, 여러 층을 거쳐야 점진적으로 먼 단어와의 관계를 학습할 수 있음.

---

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].
End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].
To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].

자기 주의(Self-attention), 또는 내부 주의(intra-attention)는 하나의 시퀀스 내에서 서로 다른 위치를 연관시켜 해당 시퀀스의 표현을 계산하는 주의 메커니즘이다. 자기 주의는 독해(reading comprehension), 추상적 요약(abstractive summarization), 텍스트 함의(textual entailment), 그리고 작업에 독립적인 문장 표현 학습(task-independent sentence representations) 등 다양한 작업에서 성공적으로 활용되었다 [4, 22, 23, 19].

**엔드투엔드 메모리 네트워크(end-to-end memory networks**[^3])는 순차적으로 정렬된 순환 연산(sequence-aligned recurrence) 대신 순환적 주의 메커니즘(recurrent attention mechanism)을 기반으로 하며, 단순한 언어(question answering, language modeling) 작업에서 우수한 성능을 보이는 것으로 확인되었다 [28].

그러나, 우리가 아는 한 Transformer는 시퀀스에 정렬된 RNN이나 합성곱을 사용하지 않고, 오직 자기 주의만을 활용하여 입력 및 출력의 표현을 계산하는 최초의 변환(transduction) 모델이다. 이후 섹션에서는 Transformer를 설명하고, 자기 주의의 필요성을 논의하며, 이를 기존 모델 [14, 15, 8]과 비교하여 가지는 장점에 대해 다룰 것이다.

---
[^3]: 메모리라는 구조를 활용하여 입력에서 출력까지 중간 단계 없이 신경망으로 한 번에 처리한다. 또한, 네트워크가 출력을 하기 전에 동일한 입력 시퀀스를 여러 번 읽고 각 단계에서 메모리 내용을 업데이트할 수 있어, 복잡한 추론 작업을 수행할 수 있다.
---

# 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 29]. Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $\mathbf{z}=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathbf{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next.
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

대부분의 경쟁력 있는 신경망 기반 시퀀스 변환(sequence transduction) 모델은 인코더-디코더(encoder-decoder) 구조를 갖는다 [5, 2, 29]. 이 구조에서 인코더는 입력 시퀀스 $(x_{1}, \ldots, x_{n})$을 연속적인 표현의 시퀀스 $z=(z_{1}, \ldots, z_{n})$로 매핑한다. 이후 디코더는 $z$를 기반으로 심볼(symbol)들의 출력 시퀀스 $(y_{1}, \ldots, y_{m})$를 한 요소씩 생성한다. 디코더는 자가 회귀(auto-regressive) 방식으로 동작하며, 다음 출력을 생성할 때 이전에 생성된 심볼들을 추가 입력으로 사용한다 [9].

Transformer는 이러한 전체적인 구조를 따르되, 인코더와 디코더 모두에서 쌓인(self-attention) 자기 주의 메커니즘과 포인트와이즈(point-wise) 완전 연결층(fully connected layers)을 활용한다. 이 구조는 그림 1의 좌측과 우측 절반에서 각각 나타낸다.

---

> [!info] Auto-regressive
>- 과거의 자신의 값을 참조하여 현재의 값을 예측하는 모델링 기법
>	- 순차적 예측: 이전 시점의 데이터를 사용하여 다음 시점의 데이터 예측
>	- 시계열 데이터 처리
>	- 문맥 이해: 자연어 처리에서 이전 단어들을 기반으로 다음 단어를 예측하는데 활용됨

---


## 3.1 Encoder and Decoder Stacks


![img-0.jpeg](img-0.jpeg)

				Figure 1: The Transformer - model architecture.

__Encoder__: The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is $\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$, where $\operatorname{Sublayer}(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.

Decoder: The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

## 3.2 Attention 

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$. We compute the dot products of the

![img-1.jpeg](img-1.jpeg)

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_{k}}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of $d_{k}$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_{k}$ [3]. We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients ${ }^{4}$. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_{k}}}$.

### 3.2.2 Multi-Head Attention 

Instead of performing a single attention function with $d_{\text {model }}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_{k}, d_{k}$ and $d_{v}$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_{v}$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

[^0]
[^0]:    ${ }^{4}$ To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1 . Then their dot product, $q \cdot k=\sum_{i=1}^{d_{k}} q_{i} k_{i}$, has mean 0 and variance $d_{k}$.

$$
\begin{aligned}
\operatorname{MultiHead}(Q, K, V) & =\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \operatorname{head}_{\mathrm{h}}\right) W^{O} \\
\text { where } \operatorname{head}_{\mathrm{i}} & =\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
$$

Where the projections are parameter matrices $W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}$ and $W^{O} \in \mathbb{R}^{h d_{v} \times d_{\text {model }}}$.

In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_{k}=d_{v}=d_{\text {model }} / h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

### 3.2.3 Applications of Attention in our Model 

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as $[31,2,8]$.
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$ ) all values in the input of the softmax which correspond to illegal connections. See Figure 2.


## 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{\text {model }}=512$, and the inner-layer has dimensionality $d_{f f}=2048$.

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text {model }}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [24]. In the embedding layers, we multiply those weights by $\sqrt{d_{\text {model }}}$.

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.

| Layer Type | Complexity per Layer | Sequential <br> Operations | Maximum Path Length |
| :-- | :--: | :--: | :--: |
| Self-Attention | $O\left(n^{2} \cdot d\right)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O\left(n \cdot d^{2}\right)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O\left(k \cdot n \cdot d^{2}\right)$ | $O(1)$ | $O\left(\log _{k}(n)\right)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n / r)$ |

bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{\text {model }}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [8].
In this work, we use sine and cosine functions of different frequencies:

$$
\begin{aligned}
P E_{(\text {pos }, 2 i)} & =\sin \left(\operatorname{pos} / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(\text {pos }, 2 i+1)} & =\cos \left(\operatorname{pos} / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$

where pos is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2 \pi$ to $10000 \cdot 2 \pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k, P E_{p o s+k}$ can be represented as a linear function of $P E_{\text {pos }}$.
We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

# 4 Why Self-Attention 

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to another sequence of equal length $\left(z_{1}, \ldots, z_{n}\right)$, with $x_{i}, z_{i} \in \mathbb{R}^{d}$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.
One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.
As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [31] and byte-pair [25] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in

the input sequence centered around the respective output position. This would increase the maximum path length to $O(n / r)$. We plan to investigate this approach further in future work.
A single convolutional layer with kernel width $k<n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n / k)$ convolutional layers in the case of contiguous kernels, or $O\left(\log _{k}(n)\right)$ in the case of dilated convolutions [15], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$. Separable convolutions [6], however, decrease the complexity considerably, to $O\left(k \cdot n \cdot d+n \cdot d^{2}\right)$. Even with $k=n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

# 5 Training 

This section describes the training regime for our models.

### 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

### 5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps ( 3.5 days).

### 5.3 Optimizer

We used the Adam optimizer [17] with $\beta_{1}=0.9, \beta_{2}=0.98$ and $\epsilon=10^{-9}$. We varied the learning rate over the course of training, according to the formula:

$$
\text { lrate }=d_{\text {model }}^{-0.5} \cdot \min \left(\text { step_num }^{-0.5}, \text { step_num } \cdot \text { warmup_steps }^{-1.5}\right)
$$

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps $=4000$.

### 5.4 Regularization

We employ three types of regularization during training:
Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{\text {drop }}=0.1$.

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

| Model | BLEU |  | Training Cost (FLOPs) |  |
| :-- | :--: | :--: | :--: | :--: |
|  | EN-DE | EN-FR | EN-DE | EN-FR |
| ByteNet [15] | 23.75 |  |  |  |
| Deep-Att + PosUnk [32] |  | 39.2 |  | $1.0 \cdot 10^{20}$ |
| GNMT + RL [31] | 24.6 | 39.92 | $2.3 \cdot 10^{19}$ | $1.4 \cdot 10^{20}$ |
| ConvS2S [8] | 25.16 | 40.46 | $9.6 \cdot 10^{18}$ | $1.5 \cdot 10^{20}$ |
| MoE [26] | 26.03 | 40.56 | $2.0 \cdot 10^{19}$ | $1.2 \cdot 10^{20}$ |
| Deep-Att + PosUnk Ensemble [32] |  | 40.4 |  | $8.0 \cdot 10^{20}$ |
| GNMT + RL Ensemble [31] | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ | $1.1 \cdot 10^{21}$ |
| ConvS2S Ensemble [8] | 26.36 | $\mathbf{4 1 . 2 9}$ | $7.7 \cdot 10^{19}$ | $1.2 \cdot 10^{21}$ |
| Transformer (base model) | 27.3 | 38.1 | $\mathbf{3 . 3} \cdot \mathbf{1 0}^{18}$ |  |
| Transformer (big) | $\mathbf{2 8 . 4}$ | $\mathbf{4 1 . 0}$ | $2.3 \cdot 10^{19}$ |  |

Label Smoothing During training, we employed label smoothing of value $\epsilon_{l s}=0.1$ [30]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

# 6 Results 

## 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than $1 / 4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{\text {drop }}=0.1$, instead of 0.3 .
For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha=0.6$ [31]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length +50 , but terminate early when possible [31].
Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU ${ }^{5}$.

## 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

[^0]
[^0]:    ${ }^{5}$ We used values of $2.8,3.7,6.0$ and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

|  | N | $d_{\text {model }}$ | $d_{\text {ff }}$ | h | $d_{k}$ | $d_{v}$ | $P_{\text {drop }}$ | $\epsilon_{l s}$ | train <br> steps | PPL <br> (dev) | BLEU <br> (dev) | params <br> $\times 10^{6}$ |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100 K | 4.92 | 25.8 | 65 |
| (A) |  |  |  | 1 | 512 | 512 |  |  |  | 5.29 | 24.9 |  |
|  |  |  |  | 4 | 128 | 128 |  |  |  | 5.00 | 25.5 |  |
|  |  |  |  | 16 | 32 | 32 |  |  |  | 4.91 | 25.8 |  |
|  |  |  |  | 32 | 16 | 16 |  |  |  | 5.01 | 25.4 |  |
| (B) |  |  |  |  | 16 |  |  |  |  | 5.16 | 25.1 | 58 |
|  |  |  |  |  | 32 |  |  |  |  | 5.01 | 25.4 | 60 |
| (C) | 2 |  |  |  |  |  |  |  |  | 6.11 | 23.7 | 36 |
|  | 4 |  |  |  |  |  |  |  |  | 5.19 | 25.3 | 50 |
|  | 8 |  |  |  |  |  |  |  |  | 4.88 | 25.5 | 80 |
|  |  | 256 |  |  | 32 | 32 |  |  |  | 5.75 | 24.5 | 28 |
|  |  | 1024 |  |  | 128 | 128 |  |  |  | 4.66 | 26.0 | 168 |
|  |  |  | 1024 |  |  |  |  |  |  | 5.12 | 25.4 | 53 |
|  |  |  | 4096 |  |  |  |  |  |  | 4.75 | 26.2 | 90 |
| (D) |  |  |  |  |  | 0.0 |  |  |  | 5.77 | 24.6 |  |
|  |  |  |  |  |  | 0.2 |  |  |  | 4.95 | 25.5 |  |
|  |  |  |  |  |  |  | 0.0 |  |  | 4.67 | 25.3 |  |
|  |  |  |  |  |  |  | 0.2 |  |  | 5.47 | 25.7 |  |
| (E) |  | positional embedding instead of sinusoids |  |  |  |  |  |  |  | 4.92 | 25.7 |  |
| big | 6 | 1024 | 4096 | 16 |  | 0.3 |  | 300 K |  | 4.33 | 26.4 | 213 |

In Table 3 rows (B), we observe that reducing the attention key size $d_{k}$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.

# 7 Conclusion 

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

# References 

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.
[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.
[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.
[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
[9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.
[11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
[12] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, $9(8): 1735-1780,1997$.
[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
[14] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.
[15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.
[16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
[17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
[18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.
[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.
[20] Samy Bengio Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[21] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.
[23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.
[24] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.
[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
[26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
[27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.
[28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.
[29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.
[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.
[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

[^3]: ㅇ
	
