# Enron Person Of Interest Prediction
*author: Ron Tam*

*contact: ron.ty.tam@gmail.com*

*Note: For the jupyter notebook, see [here](https://github.com/RonTam/Udacity_Enron/blob/master/final_project/Enron_Project.ipynb)*

*To generate the pickled files of the model, dataset, and features used, run [poi_id.py](https://github.com/RonTam/Udacity_Enron/blob/master/final_project/poi_id.py)*

### Introduction
Once a darling of Wall Street, Enron is better known today as a symbol of corporate fraud and corruption.  The [scandal](https://en.wikipedia.org/wiki/Enron_scandal) highlighted the dangers of creative accounting practices and pointed a spotlight at regulators to find ways to avoid future catastrophies from corporate misdeeds.

As a consequence of the investigations into Enron, federal regulators released mountains of information (typically confidential) regarding the activities of company executives prior to the firm's dissolution.  In this project, we will study a subset of these documents, to see whether we can identify a "person of interest" based on data gathered on a number of company executives.  We will build a number of machine learning models to predict whether an individual is considered a person of interest, and analyze the accuracy of our results.  Having a good model may pave the way of identifying potential "Enrons" in the future, as corporate accounting continue to get increasingly more complex in our evolving financial landscape.  Additionally, machine learning can help us effectively communicate an idea or prediction to an audience, regardless of the audience's level of exposure to advanced mathematics or computer science.

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

Next, we observe that our dataset contains many `NaN` values, which will have to be addressed prior to modeling.  We first list out the number of missing values below:  

| Feature_Names             | Count of Nulls    | Percentage Null |
| ------------------------- |-------------------| ----------------|
| salary                    |   49              | 0.3426          |
| to_messages               |   57              | 0.3986          |
| deferral_payments         |   105             | 0.7342          |
| total_payments            |   20              | 0.1398          |
| exercised_stock_options   |   42              | 0.2937          |
| bonus                     |   62              | 0.4335          |
| restricted_stock          |   34              | 0.2377          |
| shared_receipt_with_poi   |   57              | 0.3986          |
| restricted_stock_deferred |   126             | 0.8811          |
| total_stock_value         |   18              | 0.1258          |
| expenses                  |   49              | 0.3426          |
| loan_advances             |   140             | 0.9790          |
| from_messages             |   57              | 0.3986          |
| other                     |   52              | 0.3636          |
| from_this_person_to_poi   |   57              | 0.3986          |
| poi                       |   0               | 0.0000          |
| director_fees             |   127             | 0.8881          |
| deferred_income           |   95              | 0.6643          |
| long_term_incentive       |   78              | 0.5454          |
| email_address             |   32              | 0.2237          |
| from_poi_to_this_person   |   57              | 0.3986          |

We see that loan advances represent the field with the highest number of nulls.  We can conceivably believe that many high paying executives may not require a loan advance from the company.  Similarly, only a subset of executives should be expected to receved stock deferrals with restrictions (another field representing high null count).  For the sake of simplicity, we replace all `NaN` with `0`, while acknowledging that more sophisticated methods of imputation might have resulted in accuracy gains in our modeling.

We made a judgement to remove `email_address` from our feature set, as a cursory review showed that the field largely represent some permutation of the executive's name.  Given that we are not using NLP as a part of this exercise, we have deleted that feature from the data.

#### Feature Creation

To give a better semblence of normalization between some of these numbers, I created a number of additional features:

* `deferred_ratio` : ratio of `deferral_payments` to `total_payments`
* `message_ratio` : ratio of outgoing messages `to_messages' to incoming messages `from_messages`.
* `poi_from_ratio` : ratio of incoming messages from a person of interest, or `from_poi_to_this_person` divided by `from_messages`.
*  `poi_to_ratio` : ratio of outgoing messages to a person of interest, or `from_this_person_to_poi` divided by `to_messages`.

These created features comprise of ratios of existing features already in the data.  Instead of studying the amount of deferral payments, we felt that the proportion of total payments deferred is more important, as executives are compensated differently.  Along those same notes, the number of incoming messages are less informative than understanding the ratio of incoming to outgoing messages.  Finally, instead of looking at the actual numbers of incoming and outgoing messages from POIs, it may be more informative to understand the proportion of total messages that interact with POIs, whether incoming or outgoing.  We will discuss in our analysis how these new features affected our final results.

#### Feature Selection

A number of feature selection techniques were considered for this project.  First, we ran a Random Forest Classifier over our data and used the resulting model to discover feature importance.  As a robust emsemble method, Random Forest does not require feature scaling and is very flexible in dealing with potentially messay feature relationships, as such it is often a good plug-and-play model for feature selection or performance benchmarking.  Limiting each node to a square root of params, the resulting order of important features are listed below:


| Feature_Names        		| Feature_Importance    |
| -----------------------	|:---------------------:|
| other      				         | 0.107346				  |
| total_stock_value    		   | 0.090469				  |
| expenses	 		     	       | 0.080059	 			  |
| message_ratio				       | 0.072827				  | 		
| bonus						           | 0.072018				  |
| exercised_stock_options	   | 0.060140				  |
| salary					           | 0.058832				  |
| shared_receipt_with_poi	   | 0.056311				  |
| deferred_income			       | 0.051368				  |
| restricted_stock 			     | 0.049847				  |	
| total_payments 			       | 0.048884				  |
| to_messages 				       | 0.045440				  |
| deferred_ratio			       | 0.038757				  |
| long_term_incentive		     | 0.033897				  |
| from_poi_to_this_person    | 0.028768				  |
| poi_to_ratio				       | 0.028362				  |
| poi_from_ratio			       | 0.024700				  |
| from_messages				       | 0.022333				  |  		
| from_this_person_to_poi	   | 0.019868				  |
| deferral_payments			     | 0.009680				  |
| director_fees				       | 0.000092				  |
| restricted_stock_deferred  | 0.000000				  |
| loan_advances				       | 0.000000				  |

Taking the ordered list, we randomly chose a subset of the most important features for our model build.  This process is often more art than science, and numerous attempts were made to find some optimal dividing point for features.  While the ensuing performance was relatively decent, we could not find a suitable combination of models using this feature ordering that achieved a recall and precision above 0.3 (a goal that we had set as part of this project).  As such, we could not use the method for choosing the optimal features.

We use a number of more sophisticated techniques to find better features for our model, the details of which can be found in the accompanying jupyter notebook.  These included using SelectKBest algorithms, feature transformations using PCA, and a combination of SelectKBest and PCA using FeatureUnion, with a pipeline that exhustively searched for the best features via Grid Search Cross Validation.  While considerably more time consuming, the overall precisions continued to drift slightly below 0.3  (it is very well feasible that these features would work with more sophisticated models like XGBoost, but for the sake of feature selection the resulting precisions/recalls were based off of simpler classification models).

Finally, we split the features up to consider messages versus financials.  We wanted to better understand whether value was coming from the email information, or the financial information.  We hoped that this would give us more information on the overall class of features that can make the best predictions.

Interestingly, both sets of features tested well using basic ML models.   When we ran the features using only financials information, the numbers were also strong but still below the 0.30 threshold needed for Precision and Recall.  However, we found very strong performance using only the features from email messages, while completely ignoring the numbers from the financials. These features were able to exceed the 0.30 threshold on both Precision and Recall using simple ML models.

As such, the resulting feature set that I used for model building contained the following features:

* `to_messages`
* `shared_receipt_with_poi`
* `from_messages`
* `from_this_person_to_poi`
* `from_poi_to_this_person`
* `message_ratio`
* `poi_from_ratio`
* `poi_to_ratio`

Before we detailed modeling and analysis, note that `message_ratio`, `poi_from_ratio`, and `poi_to_ratio`, were features that we created in an earlier section.  It turns out that these new features did not drastically improve our output.  In fact, had we removed these three features, we would have gotten a higher Recall Value on our Logistic Regression and a similar Precision Value.  As such, we have ommitted these features from our model. 

We do no feature scaling for this dataset.  As will be demonstrated later, feature scaling is not required for Logistic Regression.  Furthermore, the added ratios give more normalized depth to the absolute messages data.

### Model Building, Validation, And Analysis

A number of algorithms were attempted in the search for the optimal model.  To parameterize our model, we use Grid Search Cross Validation to exhaustively search through possible hyperparameters.  This method of model tuning allowed us to test numerous values on the parameters and select the one that yielded the best result.  For example, we tested our Random Forest with different Max Depths, as well as the number of samples at each split, and the number of samples at each leaf.  We also tested the scoring criterion at each node split by training on both the Gini Index as well as Entropy for information gain.  Likewise, we tuned our Logistic Regression against different C values, penalty metrics, and class weights in order to find the model that gave the best performance.


The results of our testing is presented below:

| Algorithm        		  | Accuracy      |	Precision 	  | Recall   	|
| ----------------------|---------------|---------------|-----------|
| Random Forest			    | 0.86300 		  | 0.16619     	| 0.05800	  |		
| Logistic Regression	  | 0.79367    	  | 0.31490		    | 0.72900   |
| Naive Bayes			      | 0.79311 		  | 0.01023		    | 0.00900	  |
| AdaBoost				      | 0.81122 		  | 0.21973 		  | 0.27400 	|



Before we go into more detail about these numbers, it's first prudent to explain how the numbers were achieved.  We relied on the algorithm given by tester.py, which uses a Stratified Shuffle Split for Cross Validation.  The folds (1000) are made by preserving the percentage of samples for each class, meaning that we can ensure a small number of actual persons of interest in each fold we use for training and testing.  This is important in cases where there exist a large imbalance between the classified labels in the data (as we have in this project). In addition, the splitting factor uses a randomized permutation across the folds.  

For example, given a 5-Fold Stratified Shuffle Split, the data will be trained on 4 randomized permutation folds, in which each fold randomly contains one fifth of the data, while preserving as much as possible the poi to non-poi ratio within that fold.  Those 4 folds will then be used to fit our modeled algorithm, and the fifth fold to be used for prediction.  This process iterates through with each fold representing the test set once, and the resulting accuracies are averaged into the scores we see above.  

As mentioned before, accuracy score does not represent a suitable metric for measuring the performance of our models.  As such we rely on Precision and Recall.  Precision is the number of True Positives (the number of POIs correctly identified) divided by the total number of POIs labeled from our model (TP + FP).  Alternatively, recall represents the number of True Positives divided by the number of actual POIs from our data (TP + FN).  Together, these often give a better depiction on the strength of a model when the classes are strongly imbalanced.

Now let's finally look at our results.  While Random Forest yielded the best total accuracy, it's recall metric was very poor.  Alternatively, Logistic Regression performed very well with our data.  We ran a number of parameter permutations through Grid Search and found an optimal model using a C = 100000000000, class_weight = balanced, and a penalty measured by l2 norm. 

 		     	

### Conclusion

In this project, we explored email statistics and financial compensations for a number of Enron executives, in order to predict "persons of interests" that may be associated with the famous Enron Scandal.  Our data exploration involved basic exploratory data analysis, including feature selection and outlier detection.  We then fit a number of classic machine learning models to our data, and found one that performed optimally given our environment set up. 

This project was interesting as it applied classic machine learning techniques to a small yet relevant dataset from the real world.  We were able to work through dubious features and data and apply tried and true algorithms to our dataset, yielding interesting and valuable results. 

### References

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

* [sklearn documentation](http://scikit-learn.org/stable/documentation.html)
* [Enron Scandal](https://en.wikipedia.org/wiki/Enron_scandal)