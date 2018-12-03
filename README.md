# Graduate school admission prediction

## Project Overview
The goal of this project is to explore the data and predict whether a student will get admitted into graduate school or not.

## Data Processing and Cleaning
This was a rather small and clean dataset which I downloaded from Kaggle.

Almost all features are numeric - except for Research which is boolean. Universities, letters of recommendation and statements of purpose are all rated from 1 to 5. The target variable is Chance of Admittance which is the probability of getting admitted into graduate school. This what we will try and predict.

## Data Exploration & Visualization

##### Correlations between GRE, TOEFL Scores and Chance Of Admittance
From the pairplot we can see that there is a positive correlation between GRE Scores and chance of being admitted, and the same can be said about TOEFL Scores. GRE scores & TOEFL scores are positively correlated, and chance of being admitted seems evenly distributed.

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
