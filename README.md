# Enron Person Of Interest Prediction
*author: Ron Tam*

*contact: ron.ty.tam@gmail.com*

### Introduction
Once a darling of Wall Street, Enron is better known today as a symbol of corporate fraud and corruption.  The scandal highlighted the dangers of creative accounting practices and pointed a spotlight at regulators to find ways to avoid future catastrophies from corporate misdeeds.

As a consequence of the investigations into Enron, federal regulators released mountains of information (typically confidential) regarding the activities of company executives prior to the firm's dissolution.  In this project, we will study a subset of these documents, to see whether we can identify a "person of interest" based on data gathered on a number of company executives.  We will build a number of machine learning models to predict whether an individual is considered a person of interest, and analyze the accuracy of our results.  

### Data Introduction
The data that we will explore in this project is constructed as a dictionary of feature values, where each key of the dictionary is represented by one of 146 Enron executives.  The initial feature set consists of 20 independent variables, comprising of both financial data as well as email statistics of the individuals named.  A boolean valued variable: "poi", captures the true labeled status of each person.

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

