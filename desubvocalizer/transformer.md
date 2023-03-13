# Abstract

Natural language understanding comprises a wide range of diverse tasks such
as textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for
discriminatively trained models to perform adequately. We demonstrate that large
gains on these tasks can be realized by generative pre-training of a language model
on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each
specific task. In contrast to previous approaches, we make use of task-aware input
transformations during fine-tuning to achieve effective transfer while requiring
minimal changes to the model architecture. We demonstrate the effectiveness of
our approach on a wide range of benchmarks for natural language understanding.
Our general task-agnostic model outperforms discriminatively trained models that
use architectures specifically crafted for each task, significantly improving upon the
state of the art in 9 out of the 12 tasks studied. For instance, we achieve absolute
improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on
question answering (RACE), and 1.5% on textual entailment (MultiNLI).

# 1 Introduction

The ability to learn effectively from raw text is crucial to alleviating the dependence on supervised
learning in natural language processing (NLP). Most deep learning methods require substantial
amounts of manually labeled data, which restricts their applicability in many domains that suffer
from a dearth of annotated resources [61]. In these situations, models that can leverage linguistic
information from unlabeled data provide a valuable alternative to gathering more annotation, which
can be time-consuming and expensive. Further, even in cases where considerable supervision
is available, learning good representations in an unsupervised fashion can provide a significant
performance boost. The most compelling evidence for this so far has been the extensive use of pretrained word embeddings [10, 39, 42] to improve performance on a range of NLP tasks [8, 11, 26, 45].

Leveraging more than word-level information from unlabeled text, however, is challenging for two
main reasons. First, it is unclear what type of optimization objectives are most effective at learning
text representations that are useful for transfer. Recent research has looked at various objectives
such as language modeling [44], machine translation [38], and discourse coherence [22], with each
method outperforming the others on different tasks.1 Second, there is no consensus on the most
effective way to transfer these learned representations to the target task. Existing techniques involve
a combination of making task-specific changes to the model architecture [43, 44], using intricate
learning schemes [21] and adding auxiliary learning objectives [50]. These uncertainties have made
it difficult to develop effective semi-supervised learning approaches for language processing.

1
https://gluebenchmark.com/leaderboard
Preprint. Work in progress.

In this paper, we explore a semi-supervised approach for language understanding tasks using a
combination of unsupervised pre-training and supervised fine-tuning. Our goal is to learn a universal
representation that transfers with little adaptation to a wide range of tasks. We assume access to
a large corpus of unlabeled text and several datasets with manually annotated training examples
(target tasks). Our setup does not require these target tasks to be in the same domain as the unlabeled
corpus. We employ a two-stage training procedure. First, we use a language modeling objective on
the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt
these parameters to a target task using the corresponding supervised objective.

For our model architecture, we use the Transformer [62], which has been shown to perform strongly on
various tasks such as machine translation [62], document generation [34], and syntactic parsing [29].
This model choice provides us with a more structured memory for handling long-term dependencies in
text, compared to alternatives like recurrent networks, resulting in robust transfer performance across
diverse tasks. During transfer, we utilize task-specific input adaptations derived from traversal-style
approaches [52], which process structured text input as a single contiguous sequence of tokens. As
we demonstrate in our experiments, these adaptations enable us to fine-tune effectively with minimal
changes to the architecture of the pre-trained model.

We evaluate our approach on four types of language understanding tasks – natural language inference,
question answering, semantic similarity, and text classification. Our general task-agnostic model
outperforms discriminatively trained models that employ architectures specifically crafted for each
task, significantly improving upon the state of the art in 9 out of the 12 tasks studied. For instance,
we achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test) [40],
5.7% on question answering (RACE) [30], 1.5% on textual entailment (MultiNLI) [66] and 5.5% on
the recently introduced GLUE multi-task benchmark [64]. We also analyzed zero-shot behaviors
of the pre-trained model on four different settings and demonstrate that it acquires useful linguistic
knowledge for downstream tasks.

#2 Related Work

Semi-supervised learning for NLP Our work broadly falls under the category of semi-supervised
learning for natural language. This paradigm has attracted significant interest, with applications to
tasks like sequence labeling [24, 33, 57] or text classification [41, 70]. The earliest approaches used
unlabeled data to compute word-level or phrase-level statistics, which were then used as features in a
supervised model [33]. Over the last few years, researchers have demonstrated the benefits of using
word embeddings [11, 39, 42], which are trained on unlabeled corpora, to improve performance on a
variety of tasks [8, 11, 26, 45]. These approaches, however, mainly transfer word-level information,
whereas we aim to capture higher-level semantics.
Recent approaches have investigated learning and utilizing more than word-level semantics from
unlabeled data. Phrase-level or sentence-level embeddings, which can be trained using an unlabeled
corpus, have been used to encode text into suitable vector representations for various target tasks [28,
32, 1, 36, 22, 12, 56, 31].

Unsupervised pre-training Unsupervised pre-training is a special case of semi-supervised learning
where the goal is to find a good initialization point instead of modifying the supervised learning
objective. Early works explored the use of the technique in image classification [20, 49, 63] and
regression tasks [3]. Subsequent research [15] demonstrated that pre-training acts as a regularization
scheme, enabling better generalization in deep neural networks. In recent work, the method has
been used to help train deep neural networks on various tasks like image classification [69], speech
recognition [68], entity disambiguation [17] and machine translation [48].

The closest line of work to ours involves pre-training a neural network using a language modeling
objective and then fine-tuning it on a target task with supervision. Dai et al. [13] and Howard and
Ruder [21] follow this method to improve text classification. However, although the pre-training
phase helps capture some linguistic information, their usage of LSTM models restricts their prediction
ability to a short range. In contrast, our choice of transformer networks allows us to capture longerrange linguistic structure, as demonstrated in our experiments. Further, we also demonstrate the

effectiveness of our model on a wider range of tasks including natural language inference, paraphrase
detection and story completion. Other approaches [43, 44, 38] use hidden representations from a
2
pre-trained language or machine translation model as auxiliary features while training a supervised
model on the target task. This involves a substantial amount of new parameters for each separate
target task, whereas we require minimal changes to our model architecture during transfer.
Auxiliary training objectives Adding auxiliary unsupervised training objectives is an alternative
form of semi-supervised learning. Early work by Collobert and Weston [10] used a wide variety of
auxiliary NLP tasks such as POS tagging, chunking, named entity recognition, and language modeling
to improve semantic role labeling. More recently, Rei [50] added an auxiliary language modeling
objective to their target task objective and demonstrated performance gains on sequence labeling
tasks. Our experiments also use an auxiliary objective, but as we show, unsupervised pre-training
already learns several linguistic aspects relevant to target tasks.
# 3 Framework
Our training procedure consists of two stages. The first stage is learning a high-capacity language
model on a large corpus of text. This is followed by a fine-tuning stage, where we adapt the model to
a discriminative task with labeled data.
3.1 Unsupervised pre-training
Given an unsupervised corpus of tokens U = {u1, . . . , un}, we use a standard language modeling
objective to maximize the following likelihood:
L1(U) = X
i
log P(ui
|ui−k, . . . , ui−1; Θ) (1)
where k is the size of the context window, and the conditional probability P is modeled using a neural
network with parameters Θ. These parameters are trained using stochastic gradient descent [51].
In our experiments, we use a multi-layer Transformer decoder [34] for the language model, which is
a variant of the transformer [62]. This model applies a multi-headed self-attention operation over the
input context tokens followed by position-wise feedforward layers to produce an output distribution
over target tokens:
h0 = UWe + Wp
hl = transformer_block(hl−1)∀i ∈ [1, n]
P(u) = softmax(hnWT
e
)
(2)
where U = (u−k, . . . , u−1) is the context vector of tokens, n is the number of layers, We is the token
embedding matrix, and Wp is the position embedding matrix.


# 3.2 Supervised fine-tuning
After training the model with the objective in Eq. 1, we adapt the parameters to the supervised target
task. We assume a labeled dataset C, where each instance consists of a sequence of input tokens,
x
1
, . . . , xm, along with a label y. The inputs are passed through our pre-trained model to obtain
the final transformer block’s activation h
m
l
, which is then fed into an added linear output layer with
parameters Wy to predict y:
P(y|x
1
, . . . , xm) = softmax(h
m
l Wy). (3)
This gives us the following objective to maximize:
L2(C) = X
(x,y)
log P(y|x
1
, . . . , xm). (4)
We additionally found that including language modeling as an auxiliary objective to the fine-tuning
helped learning by (a) improving generalization of the supervised model, and (b) accelerating
convergence. This is in line with prior work [50, 43], who also observed improved performance with
such an auxiliary objective. Specifically, we optimize the following objective (with weight λ):
L3(C) = L2(C) + λ ∗ L1(C) (5)
Overall, the only extra parameters we require during fine-tuning are Wy, and embeddings for delimiter
tokens (described below in Section 3.3).

3

Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input
transformations for fine-tuning on different tasks. We convert all structured inputs into token
sequences to be processed by our pre-trained model, followed by a linear+softmax layer.
3.3 Task-specific input transformations
For some tasks, like text classification, we can directly fine-tune our model as described above.
Certain other tasks, like question answering or textual entailment, have structured inputs such as
ordered sentence pairs, or triplets of document, question, and answers. Since our pre-trained model
was trained on contiguous sequences of text, we require some modifications to apply it to these tasks.
Previous work proposed learning task specific architectures on top of transferred representations [44].
Such an approach re-introduces a significant amount of task-specific customization and does not
use transfer learning for these additional architectural components. Instead, we use a traversal-style
approach [52], where we convert structured inputs into an ordered sequence that our pre-trained
model can process. These input transformations allow us to avoid making extensive changes to the
architecture across tasks. We provide a brief description of these input transformations below and
Figure 1 provides a visual illustration. All transformations include adding randomly initialized start
and end tokens (hsi, hei).
Textual entailment For entailment tasks, we concatenate the premise p and hypothesis h token
sequences, with a delimiter token ($) in between.
Similarity For similarity tasks, there is no inherent ordering of the two sentences being compared.
To reflect this, we modify the input sequence to contain both possible sentence orderings (with a
delimiter in between) and process each independently to produce two sequence representations h
m
l
which are added element-wise before being fed into the linear output layer.
Question Answering and Commonsense Reasoning For these tasks, we are given a context
document z, a question q, and a set of possible answers {ak}. We concatenate the document context
and question with each possible answer, adding a delimiter token in between to get [z; q; $; ak]. Each
of these sequences are processed independently with our model and then normalized via a softmax
layer to produce an output distribution over possible answers.
# 4 Experiments
# 4.1 Setup
Unsupervised pre-training We use the BooksCorpus dataset [71] for training the language model.
It contains over 7,000 unique unpublished books from a variety of genres including Adventure,
Fantasy, and Romance. Crucially, it contains long stretches of contiguous text, which allows the
generative model to learn to condition on long-range information. An alternative dataset, the 1B
Word Benchmark, which is used by a similar approach, ELMo [44], is approximately the same size
4

Table 1: A list of the different tasks and datasets used in our experiments.
Task Datasets
Natural language inference SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25]
Question Answering RACE [30], Story Cloze [40]
Sentence similarity MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6]
Classification Stanford Sentiment Treebank-2 [54], CoLA [65]
but is shuffled at a sentence level - destroying long-range structure. Our language model achieves a
very low token level perplexity of 18.4 on this corpus.
Model specifications Our model largely follows the original transformer work [62]. We trained a
12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12
attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.
We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. The learning rate
was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
Since layernorm [2] is used extensively throughout the model, a simple weight initialization of
N(0, 0.02) was sufficient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53]
and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. We also
employed a modified version of L2 regularization proposed in [37], with w = 0.01 on all non bias or
gain weights. For the activation function, we used the Gaussian Error Linear Unit (GELU) [18]. We
used learned position embeddings instead of the sinusoidal version proposed in the original work.
We use the ftfy library2
to clean the raw text in BooksCorpus, standardize some punctuation and
whitespace, and use the spaCy tokenizer.3
Fine-tuning details Unless specified, we reuse the hyperparameter settings from unsupervised
pre-training. We add dropout to the classifier with a rate of 0.1. For most tasks, we use a learning rate
of 6.25e-5 and a batchsize of 32. Our model finetunes quickly and 3 epochs of training was sufficient
for most cases. We use a linear learning rate decay schedule with warmup over 0.2% of training. λ
was set to 0.5.
# 4.2 Supervised fine-tuning
We perform experiments on a variety of supervised tasks including natural language inference,
question answering, semantic similarity, and text classification. Some of these tasks are available
as part of the recently released GLUE multi-task benchmark [64], which we make use of. Figure 1
provides an overview of all the tasks and datasets.
Natural Language Inference The task of natural language inference (NLI), also known as recognizing textual entailment, involves reading a pair of sentences and judging the relationship between
them from one of entailment, contradiction or neutral. Although there has been a lot of
recent interest [58, 35, 44], the task remains challenging due to the presence of a wide variety of
phenomena like lexical entailment, coreference, and lexical and syntactic ambiguity. We evaluate
on five datasets with diverse sources, including image captions (SNLI), transcribed speech, popular
fiction, and government reports (MNLI), Wikipedia articles (QNLI), science exams (SciTail) or news
articles (RTE).
Table 2 details various results on the different NLI tasks for our model and previous state-of-the-art
approaches. Our method significantly outperforms the baselines on four of the five datasets, achieving
absolute improvements of upto 1.5% on MNLI, 5% on SciTail, 5.8% on QNLI and 0.6% on SNLI
over the previous best results. This demonstrates our model’s ability to better reason over multiple
sentences, and handle aspects of linguistic ambiguity. On RTE, one of the smaller datasets we
evaluate on (2490 examples), we achieve an accuracy of 56%, which is below the 61.7% reported by a
multi-task biLSTM model. Given the strong performance of our approach on larger NLI datasets, it is
likely our model will benefit from multi-task training as well but we have not explored this currently.
2
https://ftfy.readthedocs.io/en/latest/
3
https://spacy.io/
5
# Table 2: Experimental results on natural language inference tasks, comparing our model with current
state-of-the-art methods. 5x indicates an ensemble of 5 models. All datasets use accuracy as the
evaluation metric.
Method MNLI-m MNLI-mm SNLI SciTail QNLI RTE
ESIM + ELMo [44] (5x) - - 89.3 - - -
CAFE [58] (5x) 80.2 79.0 89.3 - - -
Stochastic Answer Network [35] (3x) 80.6 80.1 - - - -
CAFE [58] 78.7 77.9 88.5 83.3
GenSen [64] 71.4 71.3 - - 82.3 59.2
Multi-task BiLSTM + Attn [64] 72.2 72.1 - - 82.1 61.7
Finetuned Transformer LM (ours) 82.1 81.4 89.9 88.3 88.1 56.0
Table 3: Results on question answering and commonsense reasoning, comparing our model with
current state-of-the-art methods.. 9x means an ensemble of 9 models.
Method Story Cloze RACE-m RACE-h RACE
val-LS-skip [55] 76.5 - - -
Hidden Coherence Model [7] 77.6 - - -
Dynamic Fusion Net [67] (9x) - 55.6 49.4 51.2
BiAttention MRU [59] (9x) - 60.2 50.3 53.3
Finetuned Transformer LM (ours) 86.5 62.9 57.4 59.0
Question answering and commonsense reasoning Another task that requires aspects of single
and multi-sentence reasoning is question answering. We use the recently released RACE dataset [30],
consisting of English passages with associated questions from middle and high school exams. This
corpus has been shown to contain more reasoning type questions that other datasets like CNN [19] or
SQuaD [47], providing the perfect evaluation for our model which is trained to handle long-range
contexts. In addition, we evaluate on the Story Cloze Test [40], which involves selecting the correct
ending to multi-sentence stories from two options. On these tasks, our model again outperforms the
previous best results by significant margins - up to 8.9% on Story Cloze, and 5.7% overall on RACE.
This demonstrates the ability of our model to handle long-range contexts effectively.
Semantic Similarity Semantic similarity (or paraphrase detection) tasks involve predicting whether
two sentences are semantically equivalent or not. The challenges lie in recognizing rephrasing of
concepts, understanding negation, and handling syntactic ambiguity. We use three datasets for this
task – the Microsoft Paraphrase corpus (MRPC) [14] (collected from news sources), the Quora
Question Pairs (QQP) dataset [9], and the Semantic Textual Similarity benchmark (STS-B) [6].
We obtain state-of-the-art results on two of the three semantic similarity tasks (Table 4) with a 1
point absolute gain on STS-B. The performance delta on QQP is significant, with a 4.2% absolute
improvement over Single-task BiLSTM + ELMo + Attn.
Classification Finally, we also evaluate on two different text classification tasks. The Corpus
of Linguistic Acceptability (CoLA) [65] contains expert judgements on whether a sentence is
grammatical or not, and tests the innate linguistic bias of trained models. The Stanford Sentiment
Treebank (SST-2) [54], on the other hand, is a standard binary classification task. Our model obtains
an score of 45.4 on CoLA, which is an especially big jump over the previous best result of 35.0,
showcasing the innate linguistic bias learned by our model. The model also achieves 91.3% accuracy
on SST-2, which is competitive with the state-of-the-art results. We also achieve an overall score of
72.8 on the GLUE benchmark, which is significantly better than the previous best of 68.9.
6
Table 4: Semantic similarity and classification results, comparing our model with current state-of-theart methods. All task evaluations in this table were done using the GLUE benchmark. (mc= Mathews
correlation, acc=Accuracy, pc=Pearson correlation)
Method Classification Semantic Similarity GLUE
CoLA SST2 MRPC STSB QQP
(mc) (acc) (F1) (pc) (F1)
Sparse byte mLSTM [16] - 93.2 - - - -
TF-KLD [23] - - 86.0 - - -
ECNU (mixed ensemble) [60] - - - 81.0 - -
Single-task BiLSTM + ELMo + Attn [64] 35.0 90.2 80.2 55.5 66.1 64.8
Multi-task BiLSTM + ELMo + Attn [64] 18.9 91.6 83.5 72.8 63.3 68.9
Finetuned Transformer LM (ours) 45.4 91.3 82.3 82.0 70.3 72.8
Overall, our approach achieves new state-of-the-art results in 9 out of the 12 datasets we evaluate
on, outperforming ensembles in many cases. Our results also indicate that our approach works well
across datasets of different sizes, from smaller datasets such as STS-B (≈5.7k training examples) –
to the largest one – SNLI (≈550k training examples).
5 Analysis
Impact of number of layers transferred We observed the impact of transferring a variable number
of layers from unsupervised pre-training to the supervised target task. Figure 2(left) illustrates the
performance of our approach on MultiNLI and RACE as a function of the number of layers transferred.
We observe the standard result that transferring embeddings improves performance and that each
transformer layer provides further benefits up to 9% for full transfer on MultiNLI. This indicates that
each layer in the pre-trained model contains useful functionality for solving target tasks.
Figure 2: (left) Effect of transferring increasing number of layers from the pre-trained language
model on RACE and MultiNLI. (right) Plot showing the evolution of zero-shot performance on
different tasks as a function of LM pre-training updates. Performance per task is normalized between
a random guess baseline and the current state-of-the-art with a single model.
Zero-shot Behaviors We’d like to better understand why language model pre-training of transformers is effective. A hypothesis is that the underlying generative model learns to perform many of the
tasks we evaluate on in order to improve its language modeling capability and that the more structured
7
Table 5: Analysis of various model ablations on different tasks. Avg. score is a unweighted average
of all the results. (mc= Mathews correlation, acc=Accuracy, pc=Pearson correlation)
Method Avg. Score CoLA SST2 MRPC STSB QQP MNLI QNLI RTE
(mc) (acc) (F1) (pc) (F1) (acc) (acc) (acc)
Transformer w/ aux LM (full) 74.7 45.4 91.3 82.3 82.0 70.3 81.8 88.1 56.0
Transformer w/o pre-training 59.9 18.9 84.0 79.4 30.9 65.5 75.7 71.2 53.8
Transformer w/o aux LM 75.0 47.9 92.0 84.9 83.2 69.8 81.1 86.9 54.4
LSTM w/ aux LM 69.1 30.3 90.5 83.2 71.8 68.1 73.7 81.1 54.6
attentional memory of the transformer assists in transfer compared to LSTMs. We designed a series
of heuristic solutions that use the underlying generative model to perform tasks without supervised
finetuning. We visualize the effectiveness of these heuristic solutions over the course of generative
pre-training in Fig 2(right). We observe the performance of these heuristics is stable and steadily
increases over training suggesting that generative pretraining supports the learning of a wide variety
of task relevant functionality. We also observe the LSTM exhibits higher variance in its zero-shot
performance suggesting that the inductive bias of the Transformer architecture assists in transfer.
For CoLA (linguistic acceptability), examples are scored as the average token log-probability the
generative model assigns and predictions are made by thresholding. For SST-2 (sentiment analysis),
we append the token very to each example and restrict the language model’s output distribution to only
the words positive and negative and guess the token it assigns higher probability to as the prediction.
For RACE (question answering), we pick the answer the generative model assigns the highest average
token log-probability when conditioned on the document and question. For DPRD [46] (winograd
schemas), we replace the definite pronoun with the two possible referrents and predict the resolution
that the generative model assigns higher average token log-probability to the rest of the sequence
after the substitution.
Ablation studies We perform three different ablation studies (Table 5). First, we examine the
performance of our method without the auxiliary LM objective during fine-tuning. We observe that
the auxiliary objective helps on the NLI tasks and QQP. Overall, the trend suggests that larger datasets
benefit from the auxiliary objective but smaller datasets do not. Second, we analyze the effect of the
Transformer by comparing it with a single layer 2048 unit LSTM using the same framework. We
observe a 5.6 average score drop when using the LSTM instead of the Transformer. The LSTM only
outperforms the Transformer on one dataset – MRPC. Finally, we also compare with our transformer
architecture directly trained on supervised target tasks, without pre-training. We observe that the lack
of pre-training hurts performance across all the tasks, resulting in a 14.8% decrease compared to our
full model.
# 6 Conclusion
We introduced a framework for achieving strong natural language understanding with a single
task-agnostic model through generative pre-training and discriminative fine-tuning. By pre-training
on a diverse corpus with long stretches of contiguous text our model acquires significant world
knowledge and ability to process long-range dependencies which are then successfully transferred to
solving discriminative tasks such as question answering, semantic similarity assessment, entailment
determination, and text classification, improving the state of the art on 9 of the 12 datasets we
study. Using unsupervised (pre-)training to boost performance on discriminative tasks has long
been an important goal of Machine Learning research. Our work suggests that achieving significant
performance gains is indeed possible, and offers hints as to what models (Transformers) and data sets
(text with long range dependencies) work best with this approach. We hope that this will help enable
new research into unsupervised learning, for both natural language understanding and other domains,
further improving our understanding of how and when unsupervised learning works