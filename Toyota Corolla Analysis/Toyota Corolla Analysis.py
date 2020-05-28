"""
Created on Thu Mar 26 20:27:46 2020
@author: DESHMUKH
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pylab
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
pd.set_option('display.max_columns', 30)

# =============================================================================
# Business Problem :- Prepare a prediction model for Predicting Price.
# =============================================================================

toyota = pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
toyota = toyota[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
toyota.head()
toyota.isnull().sum()
toyota.shape

######################## - Exploratory Data Analysis - #########################

# Measure of central tendacy / first moment business decision
toyota.mean()
toyota.mode()
toyota.median()

# Measure of Dispersion / Second moment business decision
toyota.var()
toyota.std()

# Skewness / Thired moment business decision
toyota.skew()

# Kurtosis / forth moment business decision
toyota.kurt()

# Graphical Representation
## Histogram
toyota.hist() # Data is not normal distributed

toyota.describe()

## Boxplot
plt.boxplot(toyota.Price) # outliers
plt.boxplot(toyota.Age_08_04)  # outliers
plt.boxplot(toyota.KM) # outliers
plt.boxplot(toyota.HP) # outliers
plt.boxplot(toyota.cc) # outliers
plt.boxplot(toyota.Doors) # no outliers
plt.boxplot(toyota.Quarterly_Tax) # outliers
plt.boxplot(toyota.Weight) # outliers
 
# Normal Quantile Quantile plot
stats.probplot(toyota.Price, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.Age_08_04, dist ='norm', plot=pylab) # Not normal
stats.probplot(toyota.KM, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.HP, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.cc, dist='norm', plot=pylab) # not normal
stats.probplot(toyota.Doors, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.Gears, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.Quarterly_Tax, dist='norm', plot=pylab) # Not normal
stats.probplot(toyota.Weight, dist='norm', plot=pylab) # normal

# counts values
pd.value_counts(toyota['cc'].values,sort=True)

# From above we can see that in cc column there is value 16000 at index 80 which is outlier may due to wrong entry 
# Removing Outlier in cc column

toyota = toyota.drop([80],axis = 0)

# Replaceing Outlier in cc column by median
##toyota = np.where(toyota.['cc']>2500 ,np.median(toyota.cc), toyota.cc)

######################## - Scatter plot & Correlation - ########################

# Pair plot
sns.pairplot(toyota)

# Coifficient of Correlation
toyota.corr()

# Heat map
sns.heatmap(toyota.corr(),annot = True)

######################## - Splitting data in X and Y - #########################

X = toyota[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
y = toyota[["Price"]]

##################### - Spliting Data in Train and Test - ######################

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

################## - Multilinear regression model Building - ###################

model1 = smf.ols('y_train ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data = X_train).fit()
model1.summary()
# R2 = 0.87 but Pvalue of Doors is not significant
# Findiing out problem in Door data

modeldoor = smf.ols('y_train ~ Doors',data = X_train).fit()
modeldoor.summary() # Significant
# as Door is significant we have to find out influencer in data

# Cooks influencer plot
sm.graphics.influence_plot(model1)
# form this we can observe that values at index 221 ,991 is high influence

# Removing observation 221 , 960
X_train_new = X_train.drop([221,960],axis = 0)
y_train_new = y_train.drop([221,960],axis = 0)

# Buliding model after removing influencer
model2 = smf.ols('y_train_new ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data = X_train_new).fit()
model2.summary()
# R2 = 0.88 and Pvalue is significant 

# Added variable plot
sm.graphics.plot_partregress_grid(model2)

# model2 prediction
model2_predict = model2.predict(X_train_new)

# Errors
model2_error = model2_predict - y_train_new.Price

# Check distribution of error
model2_error.describe() # mean = 0 , error are right skewed 
plt.hist(model2_error)
plt.boxplot(model2_error)


plt.scatter(model2_predict,model2_error) # not showing any pattern

# RMSE
model2rmse = np.sqrt(np.mean(model2_error**2))
model2rmse # 1256

# Accuarace of model 
np.corrcoef(y_train_new.Price,model2_predict)
# 94 % accurate

################# - Improving model accuracy by transformation - ##################
# Applying variance transaformation induvidualy to improve correlation between X and Y

# Age_08_04
plt.scatter(np.cbrt(X_train_new.Age_08_04),y_train_new.Price)
np.corrcoef(np.cbrt(X_train_new.Age_08_04),y_train_new.Price) # -0.88 to -0.90

# KM
plt.scatter(np.cbrt(X_train_new.KM),y_train_new.Price)
np.corrcoef(np.cbrt(X_train_new.KM),y_train_new.Price) # -0.58 to -0.66

# HP
plt.scatter((X_train_new.HP),y_train_new.Price)
np.corrcoef((X_train_new.HP),y_train_new.Price)  # 0.26

# cc
plt.scatter(np.log(X_train_new.cc),y_train_new.Price)
np.corrcoef(np.log(X_train_new.cc),y_train_new.Price)  # 0.14 to 0.15

# Doors
plt.scatter((X_train_new.Doors),y_train_new.Price)
np.corrcoef((X_train_new.Doors),y_train_new.Price)  # 0.1788

# Gears
plt.scatter((X_train_new.Gears),y_train_new.Price)
np.corrcoef((X_train_new.Gears),y_train_new.Price)  # 0.030 

# Quarterly_Tax
plt.scatter(np.square(X_train_new.Quarterly_Tax),y_train_new.Price)
np.corrcoef(np.square(X_train_new.Quarterly_Tax),y_train_new.Price) # 0.23 to 0.25

# Weight
plt.scatter((X_train_new.Weight),y_train_new.Price)
np.corrcoef((X_train_new.Weight),y_train_new.Price) # 0.608

### Buliding model by considering all transformation
modeltrans = smf.ols('y_train_new ~ np.cbrt(Age_08_04) + np.cbrt(KM) + HP + np.log(cc) + Doors + Gears + np.square(Quarterly_Tax) + Weight',data = X_train_new).fit()
modeltrans.summary()
# after tansformation we can see that there is not any change in R value and also in this model p value is not significant
# so we use over first model as Final model

############################## - Final Model - ###############################

modelfinal = smf.ols('y_train_new ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data = X_train_new).fit()
modelfinal.summary() #RMSE = 1256 

########################## - Testing Final Model - ###########################

# TEST prediction
test_predict = modelfinal.predict(X_test)

# TEST Errors
test_error = test_predict - y_test.Price

# Check distribution of error
test_error.describe() # mean = 0 
plt.hist(test_error)
plt.boxplot(test_error)

plt.scatter(test_predict,test_error) # not showing any pattern

# TEST RMSE
test_rmse = np.sqrt(np.mean(test_error**2))
test_rmse # 1172

np.corrcoef(y_test.Price,test_predict)
# 94 % accurate
# After checking with all other model we observe that this model give us most accurate result.

"""
Final Equation -
Price  = -12660 - 113.46(Age_08_04) - 0.0183(KM) + 35.52(HP) - 3.60(cc) - 105.47(Doors) + 455.7(Gears) + 7.95(Quarterly_Tax) + 28.40(Weight)

With Confidance interval 95% 
Price  = -16100 - 119.42(Age_08_04) - 0.021(KM) + 28.6(HP) - 4.334(cc) - 194.8(Doors) + 26.558(Gears) + 4.735(Quarterly_Tax) + 25.35(Weight)
Price  = -9195 - 107.51(Age_08_04) - 0.015(KM) + 42.45(HP) - 2.865(cc) - 16.153(Doors) + 884.82(Gears) + 11.173(Quarterly_Tax) + 31.45(Weight) """






















