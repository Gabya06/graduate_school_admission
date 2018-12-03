# Graduate school admission prediction

## Project Overview
The goal of this project is to explore the data and predict whether a student will get admitted into graduate school or not.

## Data Processing and Cleaning
This was a rather small and clean dataset which I downloaded from Kaggle.

Almost all features are numeric - except for Research which is boolean. Universities, letters of recommendation and statements of purpose are all rated from 1 to 5. The target variable is Chance of Admittance which is the probability of getting admitted into graduate school. This what we will try and predict.

## Data Exploration & Visualization

##### Correlations between GRE, TOEFL Scores and Chance Of Admittance
From the pairplot we can see that there is a positive correlation between GRE Scores and chance of being admitted, and the same can be said about TOEFL Scores. GRE scores & TOEFL scores are positively correlated, and chance of being admitted seems evenly distributed.

![corrmat](/corrmat.png)

<b>Highest Correlations found with target Chance of Admit:</b>

<li>CPGA</li>
<li>GRE Score</li>
<li>TOEFL score</li>
<li>university_rating</li>

#### What if a student with 50% or more chance of admittance got admitted into grad school?

<em>Spoiler Alert</em> - this is most likely <b>NOT</b> how students are admitted into graduate school. The admittance rate would be greater than 90% in this dataset, so almost anyone would get in...

```python
data['admit'] = [1 if x>=.5 else 0 for x in data.chance_of_admit]
temp = data.admit.value_counts().reset_index()
temp['threshold'] =.5
fig, ax = plt.subplots()
sns.barplot(ax=ax, data=temp, y='admit', x='index')
plt.show(fig)

```
![admit_prob](/admit_prob.png)

#### Cumulative GPA and University Rating

1) Plotting the data using distplot

The first plot shows the distribution of the cumulative gpa feature - these scores look normally distributed.
We that as university rating increases so does the cumulative GPA from the scatterplot. There are some students attending universityes with ratings 4 with higher cumulative GPA than those attending the highest rated schools. But in general there is a linear relationship that can be seen as the university rating increases.

2) Plotting the data using qqplot 

This plot generates its own sample of the idealized distribution that we are comparing with, in this case the Gaussian distribution. The idealized normal samples are divided into 5 quantiles, and each data point is paired with a similar sample from the idealized distribution. The X-axis in the plot shows the idealized samples and the Y-axis has the data samples.
Our samples closely fit the expected diagonal pattern for a sample from a Gaussian distribution.

3) Statistical Normality tests

Taking a closer look at this - our null hypothesis H0 states that our sample is normally distributed. <br>We set our alpha to 0.05 and <br>
* if p<= alpha: we reject H0 - not normal
* if p> alpha: fail to reject - normal

a) Shapiro-Wilk test <br>
Evaluates a data sample and quantifies how likely it is to be drawn from a Gaussian distribution.

b) D'Agostino test (stats.normaltest) <br>
This calculates summary statistics on the data: <em>skewness</em> - a measure of asymmetry of the data which quantifies how far left/right the data is pushed. <em>Kurtosis</em> quantifies how much of the data is in the tail of the distribution.

In both cases, from  the given evidence that we have with this dataset, it looks like our null hypothesis is very likely true - our Cumulative GPA scores are <b>normally distributed</b>. We can not reject the null hypothese that the data is normall distributed because the p-values are greater than the chosen alpha<br>

```python
# 1) general plot of data distribution
fig, ax = plt.subplots()
_ = sns.distplot(data.cgpa, kde=True, kde_kws={'color':'red', 'shade':True, 'lw':1}, ax=ax)
ax.set_xlabel('')
ax.set_title("Distribution of CGPA Scores")
plt.show(fig)
```

```python
# 2) qqplot
from statsmodels.graphics.gofplots import qqplot
fig, ax = plt.subplots()
_ = qqplot(data.cgpa, line = 's', ax=ax)
ax.set_title("QQplot of CGPA")
plt.show(fig)
```

![qqplot](/qqplot.png)

```python
# Shapiro normality test
statistic, p = stats.shapiro(x=data.cgpa)
print('Statistics=%.3f, p=%.3f' % (statistic, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Our data sample looks Gaussian (fail to reject H0)')
else:
    print('Our data sample does not look Gaussian (reject H0)')
```

Statistics=0.993, p=0.072
Our data sample looks Gaussian (fail to reject H0)

#### GRE & TOEFL Scores and Research 

Overall GRE score percentiles are higher for those students with research than that of students who do not have research. 
From the boxplot we can see that the highest GRE scores belong to students with research experience and they also are highest in chance of getting admitted. We see a similar situation with TOEFL scores.

```python
'''
    Box plots of GRE & TOEFL Scores vs chance of admittance
'''
var = ['gre_score','toefl_score']
fig, axes = plt.subplots(figsize=(18, 8),nrows=2)
for i, v in enumerate(var):
    sns.boxplot(y = 'chance_of_admit', x =v, ax = axes[i], data=data[[v, 'chance_of_admit']], palette='Set1')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_title(v)
plt.show(fig)
```

![gre_scores](/gre_scores.png)

## Modeling - Linear Regression

#### Model 1 - Linear Regression with training and test set
<br>
Using 360 rows of data for training and 40 for testing, we scale the data using the standardscaler (mean = 0 and variance = 1) and then fit a linear regression model to the data. 

The first model gets an R^2 score is 0.78, which means that about 78% of the variability in Y can be explained using X. <br>
When looking at RMSE, we are an average of 6% chance of admittance away from the ground truth when making predictions on our test set. 

The features with the highest coefficients are <em>Cumulative GPA & TOEFL Score</em> which indicates that they have the most influence on the chance of getting admitted to graduate school in this model.

```python

# shuffle data
X, y = shuffle(data.drop(['serial_no.','chance_of_admit'], axis=1), data.chance_of_admit, random_state=23)
# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
# use Linear regression to predict target
linreg = LinearRegression()
scaler = preprocessing.StandardScaler()
# fit scaler on training data
scaler = scaler.fit(X_train)
# transform training data using standard scaler
X_train_transformed = scaler.transform(X_train)

# transform test data fit scaler
X_test_transformed = scaler.transform(X_test)

# fit model to training data
linreg = linreg.fit(X_train_transformed, y_train)
# take a look at R^2 score
linreg_score = linreg.score(X_test_transformed, y_test)
print("Linear Regression R^2 score on training set %.4f" %linreg.score(X_train_transformed,y_train))
print("Linear Regression R^2 score on test set     %.4f" %linreg_score)

pred = linreg.predict(X_test_transformed)
linreg_mse = mean_squared_error(y_pred=pred, y_true=y_test)
print("Linear Regression MSE score on training %.4f" %linreg_mse)
print("Linear Regression RMSE %.4f" %math.sqrt(linreg_mse))

# look at coefficients - which variables are most important in model?
linreg_coefs = linreg.coef_
coef_df = pd.DataFrame(data = list(zip(X_train.columns,linreg_coefs)), columns=['feature','coefficient'])
coef_df.sort_values(by = 'coefficient', ascending=False)
```

Linear Regression R^2 score on training set 0.8542
Linear Regression R^2 score on test set     0.8503
Linear Regression MSE score on training 0.0028
Linear Regression RMSE 0.0530

| Feature	        | Coefficient    |
| ------------------|:--------------:|
| Cumulative GPA	| 0.053796       |
| GRE 	            | 0.017675       |
| TOEFL             | 0.014883       |
| Research	        | 0.014161       |
| Letter of Rec	    | 0.011608       | 
| Univ Rating       | 0.010832       |
| Stat.of purpose	| 0.003996       |

#### Model 2 - Linear Regression using 10-fold cross validation
In the second model I created a pipeline to scale the date, fit it and perform cross validation. 
The average training and test R^2 score remained quite the same at 80% and 78%.

```python
# build pipeline to combine standard scaler and Linear Regression
scaler = preprocessing.StandardScaler()
linreg_pipe = Pipeline(steps=[('standardscaler', scaler ),('linear', LinearRegression())])
scores = cross_validate(return_train_score=True, error_score=True,
                        estimator=linreg_pipe, X=X ,y=y, cv=10)
                        print("Average score for train: %.4f" %scores['train_score'].mean())
print("Average score for test:  %.4f" %scores['test_score'].mean())
```

#### Model 3 - Ridge Regression
For this model I performed cross valiation in 2 ways: 

<em>1) 5-Fold cross validation by looping</em>

I looped through an array of alpha values and performed 5-fold cross validation and assigned that alpha to the Ridge regression model. I did this to then plot alpha vs the average CV score for that fold, where we can see that the optimal alpha gives the best score of 0.785.

<em>2) 5-Fold cross validation using RidgeCV</em>

Here the average R^2 score was very close at 0.788. Using the returned coefficients, the cumulative gpa is seemingly the most informative feature followed by TOEFL scores. Since GRE and TOEFL scores are highly correlated, I think the model would perform similiarly with GRE or TOEFL scores... 

```python
print("*"*30)
print("CROSS VALIDATION/K-FOLD FOR RIDGE REGRESSION")
print("*"*30)

alphas = np.logspace(-4, -.5, 30)
#alphas=[0.1, 1.0, 20.0]
scores = list()
scores_std = list()
for alpha in alphas:
    ridge.alpha = alpha
    this_scores = cross_val_score(estimator = ridge, X=X, y=y, n_jobs=1, cv=5)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

plt.figure(figsize=(6, 6))
plt.semilogx(alphas, scores)
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='red')
plt.show()
```
![cv_ridge](/cv_ridge.png)

```python
cv=5
ridge_cv = RidgeCV(alphas=alphas, cv=5)

# fit Ridge Regression using 5fold cross validation
ridge_cv.fit(X_train_transformed, y_train)
ridge_cv_score = ridge_cv.score(X_test_transformed, y_test)
print("Ridge Regression R^2 score %.4f" %ridge_cv_score)

ridge_cv_pred = ridge_cv.predict(X_test_transformed)
ridge_cv_mse = mean_squared_error(y_pred=ridge_cv_pred, y_true=y_test)

print("Ridge Regression MSE score %.4f" %ridge_cv_mse)
print("Ridge Regression RMSE %.4f" %math.sqrt(ridge_cv_mse))
```


#### Ridge Regression with best features: CGPA, TOEFL & GRE Scores
Lastly, I fit ridge regression models for Cumulative GPA, TOEFL & Score scores and then separately using Cumulative GPA and one of the two scores since these are highly correlative. The highest scoring model in the case of GRE vs TOEFL was GRE at 79% R^2.

```python
# fit scaler on training data - CPGA, GRE & TOEFL
scaler_2 = preprocessing.StandardScaler()
scaler_2.fit(X_train[['cgpa','gre_score','toefl_score']])

# transform training data using standard scaler
X_train_transformed_2 = scaler_2.transform(X_train[['cgpa','gre_score','toefl_score']])
X_test_transformed_2 = scaler_2.transform(X_test[['cgpa','gre_score','toefl_score']])
# fit Ridge Regression using 5-fold cross validation
ridge_2 = RidgeCV(alphas=alphas, cv=5)
ridge_2.fit(X_train_transformed_2, y_train)
ridge_2_score = ridge_2.score(X_test_transformed_2, y_test)
print("Ridge Regression R^2 score %.4f" % ridge_2_score)

ridge_2_pred = ridge_2.predict(X_test_transformed_2)
ridge_2_mse = mean_squared_error(y_pred=ridge_2_pred, y_true=y_test)

print("Ridge Regression MSE score %.4f" %ridge_cv_mse)
print("Ridge Regression RMSE %.4f" %math.sqrt(ridge_cv_mse))
```

#### Model with GRE Score and with TOEFL Score - which is better?
```python
features = ['gre_score','toefl_score']

new_scaler = preprocessing.StandardScaler()


for feat in features:
    feature_list = ['cgpa']
    feature_list.append(feat)
    new_scaler.fit(X_train[feature_list])
    X_train_scaled = new_scaler.transform(X_train[feature_list])
    X_test_scaled = new_scaler.transform(X_test[feature_list])
    
    # fit Ridge Regression using 5-fold cross validation - CPGA & GRE
    ridge_new = RidgeCV(alphas=alphas, cv=5)
    ridge_new.fit(X_train_scaled, y_train)
    score_new = ridge_new.score(X_test_scaled, y_test)
    print("Ridge Regression model using %s is: R^2 score %.4f" %(feat,score_new))  
     ```
