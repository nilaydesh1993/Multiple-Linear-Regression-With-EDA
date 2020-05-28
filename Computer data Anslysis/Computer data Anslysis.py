
"""
Created on Tue Mar 24 11:48:10 2020
@author: DESHMUKH
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pylab
import seaborn as sns
from sklearn.model_selection import train_test_split

# =============================================================================
# Business Problem :- Prepare a prediction model sales Price of the computer.
# =============================================================================

computerdata = pd.read_csv("Computer_Data.csv",index_col = 0) 
computerdata.shape
computerdata.isnull().sum()
computerdata.head()

######################## - Exploratory Data Analysis - #########################

# Mesures of Central Tendancy / First moment business decision
computerdata.mean()
computerdata.median()
computerdata.mode()

# Measure of Dispersion / Second moment of business decision
computerdata.var()
computerdata.std()

# Skewness / Thired moment business decision
computerdata.skew()

# Kurtosis / Forth moment business decision
computerdata.kurt() 

# Graphical Representation
plt.hist(computerdata.price) # Rigth Skewed
plt.hist(computerdata.speed) # Near Normal
plt.hist(computerdata.hd) # Rigth Skewed
plt.hist(computerdata.ram) # Rigth Skewed
plt.hist(computerdata.screen) # Rigth Skewed
plt.hist(computerdata.ads) # Left Skewed
plt.hist(computerdata.trend) # Normal

# Box plot
plt.boxplot(computerdata.price) # Outliers
plt.boxplot(computerdata.speed) # No Outliners
plt.boxplot(computerdata.hd) # Outliers
plt.boxplot(computerdata.ads) # No Outliers
plt.boxplot(computerdata.trend) # No Outliers
 
# Normal Quantile Quantile plot
stats.probplot(computerdata.price,dist = 'norm',plot = pylab) # Not normal - (log) sutaible 
stats.probplot(computerdata.speed,dist = 'norm',plot = pylab) # Not normal
stats.probplot(computerdata.hd,dist = 'norm',plot = pylab) # Not normal - (log) sutaible 
stats.probplot(computerdata.ram,dist = 'norm',plot = pylab) # Not normal
stats.probplot(computerdata.screen,dist = 'norm',plot = pylab) # Not normal 
stats.probplot(computerdata.ads,dist = 'norm',plot = pylab) # Not normal 
stats.probplot(computerdata.trend,dist = 'norm',plot = pylab) # Normal 

###################### - Scatter plot & Correlation - #######################

# Pair plot
sns.pairplot(computerdata,size = 1)

# correlation coiffient 
computerdata.corr()

# Heat map
sns.heatmap(computerdata.corr(),annot = True)

###################### - Splitting data in X and Y - #######################

X = computerdata.iloc[:,1:]
y = computerdata.iloc[:,0:1]

###################### - Converting Dummy variable - #######################

cd = pd.get_dummies(computerdata.cd,drop_first = True,prefix = 'cd')
multi = pd.get_dummies(computerdata.multi,drop_first = True,prefix = 'multi')
premium = pd.get_dummies(computerdata.premium,drop_first = True,prefix = 'premium')

# Droping nomial data columns
X = X.drop(['cd','multi','premium'],axis = 1) 

# Concating dummy variable state coulmns with X
X = pd.concat([X,cd,multi,premium],axis = 1)

################### - Spliting Data in Train and Test - ####################

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3 , random_state = 0)

################ - Multilinear regression model Building - #################

model1 = smf.ols('y_train ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes',data = X_train).fit()
model1.summary()
#R2 = 0.776 , Aic = 61680 , p value  = significant

# After remove influser by using influence plot we can observe there is not any changes in accuracy of model so we use some transformarion to improve accuracy
import statsmodels.api as sm
sm.graphics.influence_plot(model1)

# Added variable plot
sm.graphics.plot_partregress_grid(model1)

# model1 prediction
model1_predict = model1.predict(X_train)

# Errors
model1_error = model1_predict - y_train.price
model1_error.describe() # mean = 0 , error are right skewed 

# RMSE
model1rmse = np.sqrt(np.mean(model1_error**2))
model1rmse # 275.4

################ - Improving model accuracy by transformation - #################

# Applying variance transaformation induvidualy to improve correlation between X and Y
#speed
plt.scatter(np.log(X_train.speed) , y_train.price)
np.corrcoef(np.log(X_train.speed) , y_train.price) # converted 0.29 to 0.31

# hd
plt.scatter((X_train.hd) , y_train.price)
np.corrcoef((X_train.hd) , y_train.price) # 42 not change because causes hetroscedasticity 

# ram
plt.scatter(np.sqrt(X_train.ram) , y_train.price)
np.corrcoef(np.sqrt(X_train.ram) , y_train.price) # converted 62 to  64

# screen
plt.scatter((X_train.screen) , y_train.price)
np.corrcoef((X_train.screen) , y_train.price) # 28 not change

# ads
plt.scatter((X_train.ads) , y_train.price)
np.corrcoef((X_train.ads) , y_train.price) # 0.055 not change

# trend
plt.scatter((X_train.trend) , y_train.price)
np.corrcoef((X_train.trend) , y_train.price) #-0.20 not change

### Buliding model by considering all transformation
model2 = smf.ols('y_train ~ np.log(speed) + (hd) + np.sqrt(ram) + screen + ads + trend + cd_yes + multi_yes + premium_yes' , data = X_train).fit()
model2.summary()
#R2 = 0.79 , Aic = 61440 , p value  = significant

# Added variable plot
#sm.graphics.plot_partregress_grid(model2)

# model2 prediction
model2_predict = model2.predict(X_train)

# Errors
model2_error = model2_predict - y_train.price

# Check distribution of error
model2_error.describe() # mean = 0 , error are right skewed 
plt.hist(model2_error)
plt.boxplot(model2_error)

# Residual plot
plt.scatter(model2_predict,model2_error)

# RMSE
model2rmse = np.sqrt(np.mean(model2_error**2))
model2rmse # 266

# Accuarace of model 
np.corrcoef(y_train.price,model2_predict)
# 89 % accurate

#########################- Testing Final Model -##########################

# model2 prediction
test_predict = model2.predict(X_test)

# Errors
test_error = test_predict - y_test.price

# Check distribution of error
test_error.describe() # mean = 0 , error are right skewed 
plt.hist(test_error)
plt.boxplot(test_error)

# Residual plot
plt.scatter(test_predict,y_test.price)

# RMSE
test_rmse = np.sqrt(np.mean(test_error**2))
test_rmse # 265

# Accuarace of model 
np.corrcoef(y_test.price,test_predict)
# 89 % accurate


"""
Final Equation -
Price = -1803+484.55log(speed)+0.8(hd)+669.8(ram)+110.2(screen)+ 0.7(ads)-51(trend)+36.35(cd)+105.26(multi)-514.5(premium)

With Confidance interval 95% 
Price = -1957.9+426.74log(speed)+0.73(hd)+636.75(ram)+101(screen)+0.6(ads)-52(trend)+14.4(cd)+79.23(multi)-542.837(premium)
Price = -1648,16+506.36log(speed)+0.863(hd)+702.84(ram)+119.34(screen)+0.81(ads)-49.6(trend)+58.3(cd)+131.3(multi)-486.17(premium)
"""

