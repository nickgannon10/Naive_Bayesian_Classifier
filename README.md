# Naive Bayesian Recursive Classifier 
There are 3 parts that comprise the Naive Recursive Bayesian Classifier: <br />
	&nbsp;&nbsp;1. Train Data Cleaning: PDF Data cleaning was the reason that I struggled the most with this homework assignment. Figuring out how to put convert the PDF into an efficacious training set was executed by placing data points into buckets. Although, the buckets do not represent the PDF in an extremely high degree of granularity, it does effectively contextualize the data.  <br />
	&nbsp;&nbsp;2. Establishment of priors, gaussian_liklihood, and naive_bayes: <br />
		&nbsp;&nbsp;&nbsp;&nbsp;a. Priors concern the calculation of P(Y=y) for all possible y <br />
		&nbsp;&nbsp;&nbsp;&nbsp;b. To calculate the posterior, P(Y=y|X=x), calculating P(X=x|Y=y) using Gaussian dist is necessary, but must be generalized <br />
		&nbsp;&nbsp;&nbsp;&nbsp;c. Generalizing from part b), calculating P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * 
		P(Y=y) for all y provides the posterior distribution <br />
	&nbsp;&nbsp;3. Test Data Cleaning: Converting the Test set, data.txt, into something that conextualizable, provides the predicted results. <br />

We train the Naive Recursive Bayesian Classifier on the PDF and then test it on the data.txt set. Ultimately, this classifier is a derivative of the Bayes Theorem, in which, there are a set of priors, then a data point is integrated into the theorem and then a posterior distribution is calculated, and finally the process repeats. # Naive_Bayesian_Classifier
