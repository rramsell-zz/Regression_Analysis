# Regression Analysis
The analysis was accomplished by completion of the following processes and sub-processes: research question, data collection, data extraction and preparation, analysis, and data summary and implications.

Research Question

Can booking cancellation be predicted on variables found in the data?

Research Question Summary and Context

This study will contribute to the field of data analytics and the MSDA program by providing further insight into approach and method within regression analysis in solving binary classification problems. The aim of this analysis is to predict customer booking cancellation using relevant predictor variables from the dataset. The target variable is denoted as “is_canceled”, while the predictor variables (prior to dimensional reduction) are all other variables. 
There is also practical application for this study, as it serves the industry from an operational standpoint.  There is utility for a hospitality business to know the likelihood of a customers’ cancellation. This allows for better planning for logistical operations and proper asset disbursement. For example, if a hotel knew how many of its guests were likely to cancel, it could better staff the cafeterias and maids for certain time periods. The study of this dataset is important, practical, and applicable. This machine learning project made possible the realization of value through analysis for the betterment of the business.

Research Null and Alternative Hypothesis

H0 = Cancellation of hotel booking cannot be predicted with certainty from variables within the dataset.
H1¬ = Cancellation of a hotel booking can be predicted with certainty from variables within the dataset. 

Data Collection

Data Description

The volume of the data is 119,390 rows, and 32 columns. The dataset used is reliable and accurate modelling has come forward because of its availability. 
There are several limitations placed on the researcher, thereby hurting results of the analysis. The first is that the data is not current dating from years 2015 to 2017. The sparsity is 3.93% and most of the data missing is found in the country variable. This could be a heavy limitation depending on the goals of the project. However, economies of scale are not going to be considered here and the data may be imputed using other features. Columns company, country, and agent contain too many null values. Even with imputation, the results are questionable.  

These features are not to be included in the analysis. Delimitations of the study mainly surround the timeframe of the data. Since the data ranges from year 2015 to 2017 any attempt at forecasting would be unrealistic. The delimitation being, forecasts will not be attempted, but rather proscriptive explanation offered. Another delimitation lies in the dataset’s variables. There are too many obviously unrelated variables. Due to this delimitation, dimensionality reduction will be used to address the concern.
Link to the Dataset: https://www.kaggle.com/jessemostipak/hotel-booking-demand?select=hotel_bookings.csv
Below is a table of the variables within the dataset. Each is sorted as either categorical or quantitative and labeled as dependent variable (DV) or independent variable (IV).

Categorical	Continuous
Hotel IV	Lead Time IV
Is Cancelled (target variable) DV	Stays in Weekend Nights IV
Arrival Date Year IV	Stays in Weeknights IV
Arrival Date Month IV	Adults IV
Arrival Date Week Number IV	Children IV
Arrival Date Day of the Month IV	Babies IV
Meal IV	Repeat Guest IV
Country IV	Previous Cancellations IV
Market-Segment IV	Previous Bookings not Cancelled IV
Distribution Channel IV	Booking Changes IV
Reserved Room Type IV	Days on Waitlist IV
Assigned Room Type IV	ADR IV
Deposit Type IV	Required Car Parking Spaces IV
Agent IV	Total Number of Special Requests IV
Company IV	
Customer Type IV	
Reservation Status IV	
Reservation Status Date IV	

Data Gathering

Methodology. The data was procured via csv download from Kaggle. Through data wrangling, outliers will be controlled, all variables will be one-hot-encoded, correlation matrices will be plotted as they correlate to the target variable, and a fraction of a decimal point will be added to each one-hot-encoding to provide noise for the fitting of the model. A Logistic Regression model will be fitted in this phase of the analysis while also providing a statistical report on the model’s functionality through statsmodels.api in python. Through a regression analysis, confusion matrix, and precision matrix, model useability will be determined. Upon completion of the model’s evaluation, recommendations and key observations will be noted in the analysis. The Pandas package will be used in Python to change all column entries in both company and agent with any value to 1 and null values to 0. The variable country will be dropped as it is not essential or relevant to the target variable customer cancelation. Pandas will be used. The variable children will need assessment as there is a possibility of determining more-than-likely values for imputation. There are only 4 null values of the 112,593 total values. This renders mean imputation appropriate as it would not disrupt the variables distribution to any large effect. However, to be sure, the statistical distributions for variables babies and adults will be inspected with respect to children.  Those customers who fell in the group of two and three adults, returned a distribution of 75% of the entries showed zero children. Those who had zero babies (such as those with null child entries) had the same distribution. Thus, the result shows a reasonable mean imputation that could very well be the true value itself.

Advantages. Using python and the above processes yields certain advantages. Python offers a variety of packages which ease the mathematical burden placed upon statisticians for complex computation. What could have taken weeks, instead took hours. Python allows for quick interactions with data via Pandas, Numpy, statsmodels.api, and scikit learn. Furthermore, the math has little to no room for human error as the algorithms are assessed via programmatic function. 

Disadvantages. It is arguable that the allowance of computer programs to perform mathematical computations for us, has inhibited subject matter experts making them incapable of complete analysis within the mathematical field. This is possibly addressed by the democratization of complex scientific fields. The barrier to entry for statistical modelling has lowered enabling a societal movement akin to a modern industrial revolution. This has allowed for the progression of humanity and the efficiency of industry. This efficiency may lead to the elimination of absolute poverty allowing for economies to progress towards higher and higher living circumstances for the common man. Python is justified by its efficiency. 

The methodology used is justified by this very point. That the process enables the efficiency of its own qualifying standard, which is utility.
Data Extraction, Preparation, and Analysis with Calculations

The below section will be organized into steps containing a technique, advantage, and disadvantage for the technique used.

Step 1
	Package Import. The packages used in the project are: Pandas, Numpy, Sklearn, Statsmodels, Matplotlib, Seaborn, base64, and IPython. Pandas and Numpy were used in the data cleaning, gathering and exploratory data analysis of the project. Sklearn was used in the label encoding, standardizing, model creation and evaluation part of the process. Statsmodels was also used in the evaluation piece of the process. Matplotlib and Seaborn were used to create visualizations throughout. Lastly, base64 and IPython were used to download a copy of the transformed and cleaned dataset. 
	
Advantage. The above packages were used to simplify and expedite the exploratory analysis. 
Disadvantage. The packages contain code which is written by third parties. The accuracy of these packages is supported by and updated by these creators. If the packages lose support, they will no longer be useful in this analysis.

Step 2

Dimensional and Statistical Exploration. Using Pandas, Numpy, and Sklearn’s Label Encoder, the data was explored for outliers, null values, and nonsensical entries. Null entries were determined via Pandas function df.info() which returns each variables data type, non-null count, column name, column index, and entry count. Pandas df.shape function provides the dimensions of the dataset which are 119,390 by 32. This means there are 32 variables and 119,390 features. Pandas df.describe() function provides statistical distributions for each numerical variable. This is extremely useful for outlier detection and multivariate imputation. 

Advantage. The above process allows for the preparation necessary for knowing what needs to be cleaned in the data and how. This allows visual and analytical insights into the patterns found within the data. 
Disadvantage. The process provides big picture exploration. This makes is difficult for locating specific features needing removal or editing. 

Step 3

Multivariate Imputation. The above step provided insight into necessary cleaning procedures prior to regression fitting. Variable’s agent, country and company had a significant number of null entries. Children only had four total null entries. Company and agent were variables which contained specific identifiers for companies or agents within the feature. Most features contained no company or agent listed, meaning the booking was done by a private party without an agent. The logical conclusion was to impute the missing values with 0, and those with values other than 0 to 1. This allowed for a binary classification splitting those individual customers from those group customers. Multivariate imputation was used for the null entries found in the children column. More than 75% of the statistical distribution for those features containing two or more adults with zero babies had zero children. Furthermore, the median and average for the column children is zero. It is safe to assume that the four features with null entries for children had zero children. Thus, using multivariate imputation, zero was used to replace the null values. 

Advantage. The advantage using multivariate imputation is that the values replacing outliers and nulls are essentially the most likely candidates. The conclusion that the value being imputed is reasonably the correct replacement.
Disadvantage. There is still a possibility the wrong value is being imputed. Due to the large sparsity of 93% found in the country column, there is too much risk in multivariate imputing as it may sway the result of the regression model. The column is to be dropped altogether as it will corrupt the result of the study. 

Step 4

Label Encoding and Boolean Mapping. Logistic regression models assume that the input data is numeric, and the output is a binary classification. There are several variables found in the dataset that are strings and categorical. This methodology allows for the numerical representation of categorical variables so the algorithm may comprehend its input. 

Advantage. This method decomposes the categorical data into an understandable format for the computer. Label encoding (verses one-hot-encoding) allows for an accurate representation of variance throughout categorical variables. It is efficient, and accurate.
Disadvantage. The disadvantage here is the interpretation of results. Outputs must be interpreted back to their categorical inputs.

Step 5

Train Test Split. This step samples the original dataset at any percentage n. There are several pitfalls to avoid in this process, namely bootlegging and improper sample size. The sample size chosen is 20% for an industry standard 80/20 split for sampling. Furthermore, the train test split function from Sklearn provides the ability to avoid bootlegging with a random state. 

Advantage. The advantage to using this process is, as stated above, it easily samples a dataset and can easily avoid bootlegging. 
Disadvantage. The pitfalls here fall within the hyperparameters. There are many hyperparameters within the package provided by Sklearn and the documentation for the package must be upkept by the platform. If the package is not kept up to date, and the hyperparameters are used incorrectly, it will result in a bad model. 

Step 6

Compile and Fit the Model. The data needs to be split into X and Y features to create a predictive algorithm for answering the research question. Using the train test split, the data was split into predictors and target variables. Then the model was fit using Sklearn’s logistic regression function. 

Advantage. Sklearn provides a simple and easy way to fit numerical data to a machine learning model. 
Disadvantage. The disadvantages once again lie in hyperparameters and the continual need for updating of the platform.

Step 7

Evaluate the Model. Because the data has not undergone dimensionality reduction, the model had ludicrous accuracy and inaccurate AUC/ROC results. Furthermore, the model failed in its line search due to the incomparable predictors found in the data. The data needs to undergo dimensionality reduction and further preparation to be a useful model. The methods used here were AUC, confusion matrix, precision recall curve, and accuracy score.

Advantage. The advantage to fitting and evaluating a model prior to dimensionality reduction is that it provides analytical comparison for the future models. It provides comparable insights for model improvement and degradation.
Disadvantage. The initial model can sometimes be unrealistic as well as inaccurate. Again, it is necessary in offering comparisons for future models, but is not the final step to any regression analysis. 

Step 8

One-way ANOVA Test. Anova tests are used in the comparison of categorical variables in relation to their target. For those variables with p-values less than .05 we reject the null hypothesis and explore alternatives. This test was not extremely helpful as it only eliminated two categorical variables from the analysis. Big picture, it aided in the overall dimensionality reduction of the dataset. 

Advantage. One-way ANOVA testing allows for an analytical approach to the inclusion of categorical variables. This enables the objective measure of a variables value.
Disadvantage. This form of post-hoc testing alone is not enough to evaluate a dataset. Without more complete evaluation (PCA, or CAPCA) variance throughout the data will not be accurately represented in the model. 

Step 9

Principal Component Analysis (PCA), Scree Plot, and Predictor/Target Correlation. First, to aid in the exploration of variable relevance, correlation matrixes were displayed. This was accomplished via Pandas df.corr() function. Each predictor variable was mapped via its correlation pertaining to the target variable (customer cancellation).  Then (after standardizing the numerical data) variance and covariance were determined throughout. This covariance matrix was then used to provide Eigenvalues and Eigenvectors. The Kaiser Criterion states that all variables with an Eigenvalue less than one must be reduced. Coupling this rule with the Scree Plot method (while still obtaining a variance accounting of .8) principal components were determined. 

Advantage. This methodology allows for dimensionality reduction within a dataset while still representing the data wholistically. For example, if there is a group of ten people, and they are posed with a dichotic question, eight of those ten can be used to represent the total opinion of the ten without limiting the accuracy of those opinions represented while increasing processing efficiency. 
Disadvantage. It is arguable that PCA does not best represent variance with respect to the target variable, but rather dataset variance. This leads to the question, is the dimension reduction helping or hurting the model. It is quite easy to address this concern with model evaluation metrics.

Step 10

Recreate the Reduced Model. Steps four through nine were repeated but with the reduced dataset. Those variables selected for the reduced model were total special requests, parking spots, booking changes, previous bookings not cancelled, stays in weekend nights, ADR, market segment, days on waiting list, agent, hotel, previous cancellations, distribution channel, lead time, deposit type, arrival date (year, month, week number, and day), and the target (cancelled). 

Advantage. The advantages to creating a reduced model are numerous. One being less algorithmic input for prediction. Another is fewer future data collection. Finally, the largest advantage to recreating a final model is, if successfully implemented, the model can tell the story more accurately for stakeholders providing insight into the desired research question.
Disadvantage. The only disadvantages to creating, recreating, and editing a model is the possible loss of model useability/accuracy. This can be remedied by saving previous models and creating recall paths for future use.

Step 11

Evaluate the Final Model. Step seven was repeated for this model. The following were metrics used to evaluate the usability of the model: ROC, AUC score, confusion matrix, and accuracy score. 
	
Advantage. The advantage to evaluating the final model is the comparability to the previous model. This allows for insights into usability and improvements made. The models accuracy improved to the point where it no longer failed to line search within the algorithm. It also has an accuracy of 80%, and an AUC score of .83. This is a huge success for model useability as it provides insight into patterns within the data. Stakeholders may now use the model to accurately address their business environment for preferable outcomes.
Disadvantage. There are no disadvantages to evaluating a statistical model.

Data Summary and Implications

The results of the analysis are many and have yielded a useful method for stakeholders to affect their business environment. The key outcome here is a response to the proposed research question. By addressing this issue, the project will provide strategic advantages by allowing insight into customer cancellation habits. Below, the analysis outcomes are broken into results and limitations, recommendations, and proposals for further study.
Results of the Analysis and Limitations. The results of the analysis are a logistic regression model, cleaned past data for the business, confident predictions based on historical data (an answer to the research question), and most importantly the story has been told regarding customer booking cancellations. A limitation of this dataset is its age, new data must be sought after to procure ongoing research. The sparsity was a limitation as well, being about 3%. This was dealt with by multivariate imputation and dimension reduction. The below formula is the answer to the research question and can provide 80% accuracy in predicting customer booking cancelation.

P = (e(.02302322 + .00534902a + .1513049b -.03329736c+.10117951d+2.24217341e-.15767195f-.01397723g+.28435439h+.14678861i+.04311929j+.03700853k+.83892636l-.49625927m-.1393198n-.02094149o+.11631507q-4.28835152r-.25561069s)  /  (1+ e(.02302322 + .00534902a + .1513049b -.03329736c+.10117951d+2.24217341e-.15767195f-.01397723g+.28435439h+.14678861i+.04311929j+.03700853k+.83892636l-.49625927m-.1393198n-.02094149o+.11631507q-4.28835152r-.25561069s)  ))

Recommended Course of Action. 

Based on the results found in the analysis, it is apparent that deposit type, lead time, previous cancellations, and distribution channel are most associated with the target variable. Each of the principal components listed in the reduced dataset ought to be carefully considered. However, these four principal components are those which are most heavily associated with the target. This is proven via correlation and association within the analysis. The eigenvalues show that these four represent a greater majority of the variance within the data and the principal component analysis located these as key players in cancellation. 

Each of these four predictors make sense as contributors to cancellation. Those customers who are less sure of their booking are likely to pay via different means. Those with large lead times are the ones cancelling. This makes sense because the more time the customer must change their mind or seek out a competitor is longer. Previous cancellations are also indicative of the likelihood to cancel. This is likely because they are fickle in decision making, or have a large group and need to cater to everyone needs etc. 

Understanding that these variables are key indicators, marketing and operations needs to cater to these individuals to not absorb the loss. This means that strategic marketing initiatives and targeted marking are required to prevent booking cancellations in the future. Further, these customers will likely need second follow-ups for booking as well as increased customer service to retain the sale. 

Proposals for Future Study of the Dataset. There are two proposals for future study of this dataset as well as a proposal for future studies applicable to the hotel company. The data amazingly has low sparsity which is extremely useful in the building of accurate models. However, the data is outdated. The future studies should evaluate your customer’s data constantly and consistently for accurate insights for strategic management. Those studies to be done on this dataset ought to now use unsupervised random forests classification techniques for improved model accuracy. Lastly, it is suggested that the data be studied, and modeled, in respect to private and group reservations. Companies should be kept separate from families in their analysis as their likely to have different habits regarding booking.
