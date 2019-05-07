# 1. Introduction

Active Learning - Query Learning, Optimal Experimental Design

## 1.1 What is Active Learning?

Active learning is a subfield of Machine Learning (ML). The basic idea here is that if a learning algorithms is allowed to choose what data it learns from (somewhat similar to "being curious") it will have better performance with less training.

The main reason behind this idea is that, for many supervised learning tasks, labeled instances are very scarce and/or difficult/expensive to acquire and for any supervised learning algorithm to perform well, a fair amount (in order of hundreds or thousands) of labeled instances are required for training.

Giving that, active learning systems attempt to mitigate this bottleneck by asking _queries_, whenever the algorithm deems necessary, to an oracle - usually a human - throughout the learning process, so that the necessity of annotated data is decreased as the system improves its accuracy, therefore minimizing the cost associated with acquiring labels.

## 1.2 Active Learning Examples

There exists several scenarios in which a query may be posed and also several manners to which a query strategy might be applied.

#### Pool-based Active Learning

Queries are selected from a large pool of unlabeled instances *U* using an *uncertainty sampling* query strategy, which selects the instance in the pool about which the model is least certain how to label.

![Basic cycle involving active learning systems.](C:\Users\willi\Documents\Machine Learning 2019\active_learning_basic_cycle.png)

<center> Figure 1 - Basic cycle involving active learning systems. </center>

Figure 1 shows the basic cycle of this strategy. The available data for training the learner might have only a small percentage of labeled instances (*L*) at the start of the process. After creating an initial hypothesis, the learner may request labels for carefully selected instances using a confidence metric, then it incorporates this new knowledge to improve its model and, after that, the learner proceeds in a standard supervised manner.

#### Learning Curves

Active learning algorithms are generally evaluated by constructing *learning curves*, which plot the evaluation measure of interest (accuracy, for instance) as a function of the number of new instance queries that are labeled and added to *L*. Using these curves, we can infer that an active learning algorithm is superior to some other approach if it dominates the other for most of the points along their learning curves.



# 2. Scenarios

There are three main scenarios that have been considered in the literature where the learner will query the labels of instances: (i) membership query synthesis, (ii) stream-based selective sampling, and (iii) pool-based active learning.

## 2.1 Membership Query Synthesis

This big simply term simply means that the learner generates/constructs an "synthetic" instance - form some underlying natural distribution. For example, if the data is pictures of digits, then the learner would create an image that is similar to a digit - by possibly applying some kind of rotation to or cropping an existing digit picture - and send it to the oracle to label. 

![](C:\Users\willi\Documents\Machine Learning 2019\membership_query_synthesis.png)

This setting is usually tractable and efficient for many finite problem domains, but labeling this arbitrary artificial instances can be troublesome if the oracle is a human annotator. Taking the digit example given previously, Baum and Lang (1992) employed this active learning setting to the task of handwritten characters recognition with human oracles and noticed a small setback: many of the query images generated only hybrid characters that had no natural semantical meaning. To address this limitations the stream-based and pool-based scenarios have been proposed.

## 2.2 Stream-Based Selective Sampling

In this setting, it is made the assumption that obtaining an unlabeled instance is free (or inexpensive). Based on this, each instance is selected one at time and the learner is allowed to decide whether it wants to query the label of the instance or reject it based on its informativeness, which is decided using a query strategy.

![](C:\Users\willi\Documents\Machine Learning 2019\stream-based_selective_sampling.png)

Following the example of the digits dataset, the learner would select one image from the set of unlabelled image, determine whether it need to be labelled or discarded, and then repeat with the next image.

## 2.3 Pool-Based Active Learning

The assumption that large unlabeled datasets for real-world learning problems can be easily obtained motivates the *pool-based active learning* scenario, which assumes that within (usually closed - non-changing) a dataset (pool), there are only a few labeled instances *L* and a large amount of unlabeled data *U*. Queries are usually made in a greedy fashion, meaning that some informativeness measure is used to evaluate all data in the pool. Figure 1 shows the cycle of this scenario, which have been applied for many real-word problem domains in machine learning.

It is interesting to notice that the difference between pool-based and stream-based scenarios is that the former deals with the data sequentially, making local decisions, while the later scans through the entire dataset, rating and ranking instances, in order to build a query.

# 3. Query Strategy Frameworks

There have been proposed many ways for deciding how to evaluate the informativeness of unlabeled instances, which we will refer to as *query strategies*.

## 3.1 Uncertainty Sampling

I this framework, which is the simplest and most commonly used, a learner queries an instance that it seems to be most least certain about how to label. When it comes to probabilistic models, this approach is often the most straightforward one.

A more general uncertainty sampling strategy uses *entropy* as an uncertainty measure:

** FUNCTION HERE **

where y_i ranges over all possible labelings. Entropy in information theory represents the average rate at which information is produced by a stochastic source of data. Thus, it is often thought of as a measure of uncertainty or impurity in machine learning.

For binary classification, entropy-based uncertainty sampling is identical to choosing the instance with posterior closest to 0.5. Although the entropy based approach can be easily generalized to accommodate other types of classifiers, an alternative in these more complex settings - that have already shown to work well in some information extraction tasks - involves querying the instance whose best labeling is the *least confident*:

** FUNCTION HERE **

where y^* = argmax_y P(y|x:teta) is the most likely class labeling.

It is interesting to point out that sampling strategies may also be employed with non-probabilistic models, such as the work of (Lewis and Catlett, 1994) that explored uncertainty sampling in a decision tree classifier - by modifying it to have probabilistic output.

# 3.2 Query-By-Committee

Also known as (QBC) algorithm (Seung et al, 1992), this approach involves maintaining a committee *C = {teta^(1),...,teta^(C)}* of models which are all trained on the current labeled set *L*, but represent competing hypothesis. Each committee member is then allowed to vote on the labelings of query candidates. **The most informative query is considered to be the instance about which they most disagree**.

If we view machine learning as a search for the best model within the version space, which is the set of hypotheses that are consistent with the current labeled training data *L*, then our goal in active learning is to reduce the size of this search space as much as possible with as few labeled instances as possible, and this is what the QBC framework does: it minimizes version space that is consistent with the current labeled training data *L* by querying controversial regions of the input space.

In order to implement a QBC selection algorithm, there must be:

- a way to construct a committee of model that represent different regions of the version space
- a measure of disagreement among committee members

For measuring the level of disagreement, two main approaches have been proposed:

- Vote Entropy (Dagan and Engelson, 1995)
- Average Kullback-Leibler (KL) Divergence (McCallumn and Nigam, 1998)

Besides the QBC framework, there are other query strategies that attempt to minimize the version space as well.



## 3.3 Expected Model Change

Another general active learning framework is to query the instance that would impart the greatest change to the current model *if we knew its label*. An example query strategy in this framework is the "expected gradient length" (EGL) approach for discriminative probabilistic model classes.

Since discriminative probabilistic models are usually trained using gradient based optimization, the "change" imparted to the model can be measured by the length of the training gradient (i.e., the vector used to re-estimate parameter values). In other words, the learner should query the instance *x* which, if labeled and added to *L*, would result in the new training gradient of the largest magnitude.

** [maybe check math better here] **

The intuition behind this framework is that it prefers instances that are likely to most influence the model (i.e., have greatest impact on its parameters), regardless of the resulting query label.

This approach has been shown to work well in empirical studies, but can get computationally expensive if both the feature space and set of labelings are very large.

## 3.4 Variance Reduction and Fisher Information Ratio

#### Variance Reduction

Cohn et al. (1996) propose one of the first statistical analyses of active learning, demonstrating how to synthesize queries that minimize the leaner's future error by minimizing its **variance**. They describe a query strategy for *regression* learning problems, in which the output label is a real-valued number. They take advantage of the result by Geman et al. (1992), that shows that selecting points that minimize the *future* expected error is **equivalent to reducing output variance**.

Cohn et al. (1996) then use the estimated distribution of the model's output to estimate the variance of the learner after some new instance *x^~* has been labeled and added to *L*, and then query the instance resulting in the greatest future *variance reduction*.

However, this approach applies only to regression tasks, and synthesizes new queries *de novo*. For many learning problems like text classification, this technique cannot be used.

#### Fisher Information Ratio

Formally, Fisher information *I(teta)* is the variance of the *score*, which is the partial derivative of the log-likelihood function with respect to model parameters *teta*. The Fisher Information can be interpreted as the overall uncertainty about an input distribution *P(x)* with respect to the estimated model parameters. The optimal instance  to query, then, is the one which minimizes the *Fisher information ratio*.

The key idea behind the Fisher information ratio is that fisher information matrix for an unlabeled query candidate will tell us not only how uncertain the model is about the query candidate, but also tells us which model parameters are most responsible for this uncertainty. By minimizing this ratio for the entire unlabeled-pool , the learner will tend to query the instance whose model variance is most similar to the overall input distribution approximated by *U*. 

## 3.5 Estimated Error Reduction

The algorithms in the previous section minimize error indirectly by reducing model variance. However, there have been proposed some query strategies to minimize error directly. The idea is that we want to query for minimal expected future error, a opposed to maximal expected model change.

The estimated error reduction framework was proposed in 2001 and it was the first framework that directly estimated error reduction using Naive Bayes. Semi-supervised learning approaches have been combined with this framework and resulted in a dramatic improvement over random or uncertainty sampling.

Unfortunately, estimated error reduction may also be the most prohibitively expensive query selection framework. Not only does it require estimating the expected future error over U for each query, but a new model must be incrementally re-trained for each possible query labeling, which in turn iterates over the entire pool. This leads to a drastic increase in computational cost.

# Interesting Links

- Active (Machine) Learning - Computerphile (<https://www.youtube.com/watch?v=ANIw1Mz1SRI>)
- Active Learning: Curious AI Algorithms (<https://www.datacamp.com/community/tutorials/active-learning>)

# Article Topics

- Uncertainty Sampling and Fuzzy Logic







