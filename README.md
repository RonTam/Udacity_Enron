# Enron Person Of Interest Prediction
*author: Ron Tam*

*contact: ron.ty.tam@gmail.com*

### Introduction
Once a darling of Wall Street, Enron is better known today as a symbol of corporate fraud and corruption.  The scandal highlighted the dangers of creative accounting practices and pointed a spotlight at regulators to find ways to avoid future catastrophies from corporate misdeeds.

As a consequence of the investigations into Enron, federal regulators released mountains of information (typically confidential) regarding the activities of company executives prior to the firm's dissolution.  In this project, we will study a subset of these documents, to see whether we can identify a "person of interest" based on data gathered on a number of company executives.  We will build a number of machine learning models to predict whether an individual is considered a person of interest, and analyze the accuracy of our results.  

### Data Introduction
The data that we will explore in this project is constructed as a dictionary of feature values, where each key of the dictionary is represented by one of 146 Enron executives.  The initial feature set consists of 20 independent variables, comprising of both financial data as well as email statistics of the individuals named.  A boolean valued variable: "poi", captures the true labeled status of each person.

We observe that only 18 of the 146 data points contain a person of interest, meaning that a generic model where all points are set to False will achieve an accuracy of 87.67%.  As such, our work will focus more on Precision and Recall as metrics instead of the simpler scores of Accuracy.

Below is a sample row of the dictionary of features: 

	[{'ALLEN PHILLIP K': {'bonus': 4175000,
  						'deferral_payments': 2869717,
  						'deferred_income': -3081055,
  						'director_fees': 'NaN',
  						'email_address': 'phillip.allen@enron.com',
  						'exercised_stock_options': 1729541,
  						'expenses': 13868,
  						'from_messages': 2195,
  						'from_poi_to_this_person': 47,
  						'from_this_person_to_poi': 65,
  						'loan_advances': 'NaN',
  						'long_term_incentive': 304805,
  						'other': 152,
  						'poi': False,
  						'restricted_stock': 126027,
  						'restricted_stock_deferred': -126027,
  						'salary': 201955,
  						'shared_receipt_with_poi': 1407,
  						'to_messages': 2902,
  						'total_payments': 4484442,
  						'total_stock_value': 1729541}]

### Exploratory Data Analysis
Given no domain knowledge on the Enron scandal (beyond information that is readily available on the web), it is difficult to choose an optimal subset of features.  However, as our dataset is relatively small, we can take advantage of some brute force techniques in finding optimal variable.  Before we engage in this endeavor, a perfunctory scan of our dataset revealed some outliers and questionable data points.  

#### Addressing Outliers and Bad Features
The keys `TOTAL` and `THE TRAVEL AGENCY IN THE PARK` do not appear to be actual people (scatterplots of values from `TOTAL` can confirm it as an aberration), and are thus removed from our data.  Additionally, an executive named `LOCKHART EUGENE E` contains an empty dictionary with no usable information, and as such that point is also removed.

Next, we observe that our dataset contains many `NaN` values, which will have to be addressed prior to modeling.  For the sake of simplicity, we replace all `NaN` with `-999`, while acknowledging that more sophisticated methods of imputation might have resulted in accuracy gains in our modeling.

We made a judgement to remove `email_address` from our feature set, as a cursory review showed that the field largely represent some permutation of the executive's name.  Given that we are not using NLP as a part of this exercise, we have deleted that feature from the data.

#### Feature Creation

To give a better semblence of normalization between some of these numbers, I created a number of additional features:

* `deferred_ratio` : ratio of `deferral_payments` to `total_payments`
* `message_ratio` : ratio of outgoing messages `to_messages' to incoming messages `from_messages`.
* `poi_from_ratio` : ratio of incoming messages from a person of interest, or `from_poi_to_this_person` divided by `from_messages`.
*  `poi_to_ratio` : ratio of outgoing messages to a person of interest, or `from_this_person_to_poi` divided by `to_messages`.
