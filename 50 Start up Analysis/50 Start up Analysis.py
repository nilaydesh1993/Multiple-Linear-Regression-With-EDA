
"""
Created on Sat Mar 21 21:07:33 2020
@author: DESHMUKH
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
import pylab
import statsmodels.formula.api as smf
import seaborn as sns
# =============================================================================
# Business Problem :- Prepare a prediction model for profit of 50_startups.
# =============================================================================

startups = pd.read_csv("50_Startups.csv")
startups.info()
startups.isnull().sum()
startups.columns = "RandD","admin","marketing","state","profit"
startups = startups.replace(" ","_",regex = True)

# Measure of central tendancy / 1st Moment business decision.
startups.mean()
startups.median()
startups.mode()

# Measure of Dispersion / 2nd Moment of business decision
startups.var()
startups.std()

# Skewnees / Third moment business decision
startups.skew()

# Kurtosis / Fourth moment business decision 
startups.kurt()

startups.columns
# Graphical Representation
## Histogram
plt.hist(startups.RandD) # Nearly normal distribution
plt.hist(startups.admin) # Nearly normal distribution
plt.hist(startups.marketing) # Right skewed
plt.hist(startups.profit) #Left skewed 

## Boxplot
plt.boxplot(startups.RandD) # No outlier
plt.boxplot(startups.admin) # No outlier
plt.boxplot(startups.marketing) # No outlier
plt.boxplot(startups.profit) # outlier > 25000

# Treating outliers with median
#startups.profit = np.where(startups["profit"] < 25000,np.median(startups.profit),startups.profit)

plt.boxplot(startups.profit) # after removing outlier 

# Normal Quantile-Quantile plot
stats.probplot(startups.RandD,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.admin,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.marketing,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.profit,dist = 'norm',plot = pylab) # Normal distribution

##################################- Scatter plot -##################################

# Scatter plot
sns.pairplot(startups, size = 1, corner = True, diag_kind="kde")

# Correlation Coifficient 
startups.corr()

# Heat map
sns.heatmap(startups.corr(),annot = True)
# As we see in heat map Admin and stats have vary less correlation with profit we supposed to drop them for better model building

#############################- Converting Dummy variable -##########################

state = pd.get_dummies(startups.state,drop_first = True)

# Droping old state from startups
startups = startups.drop('state',axis = 1)

# Concating dummy variable state coulmns with X
startups = pd.concat([startups,state],axis = 1)

#########################- Spliting Data in Train and Test -########################

from sklearn.model_selection import train_test_split
startups_train, startups_test = train_test_split(startups,test_size = 0.2,random_state=0)

##########################- Multilinear regression model -##########################

model1 = smf.ols('profit ~ RandD + admin + marketing + Florida + New_York',data = startups_train).fit()
model1.summary()
# Marketing,admin,stats are insignificant

# Testing each insignificant variable with output by bulding model
modelmr = smf.ols('profit ~ marketing',data = startups_train).fit()
modelmr.summary() # significant 

modeladst = smf.ols('profit ~ Florida + New_York',data = startups_train).fit()
modeladst.summary() # not significant 

# As above we can canculed that stats are not significant 
# Building Model without stats
model2 = smf.ols('profit ~ RandD + admin + marketing',data = startups_train).fit()
model2.summary()

# Testing each insignificant variable with output by bulding model
modelad = smf.ols('profit ~ admin',data = startups_train).fit()
modelad.summary() # significant 

modeladrd = smf.ols('profit ~ RandD',data = startups_train).fit()
modeladrd.summary() # significant 

# Check data having any influential values
import statsmodels.api as sm
sm.graphics.influence_plot(model2)

# index 49 are showing inlfuence so we can exclude entire row
startupstrain_new = startups_train.drop([49],axis = 0)

model3 = smf.ols('profit ~ RandD + admin + marketing',data = startupstrain_new).fit()
model3.summary() # As we see that in this model marketing in now significant after droping 49
# as we see that admin is still not significant we try added variable plot to see depedance

# Added variable plot
sm.graphics.plot_partregress_grid(model3)

############################# Final Model Building ##################################

modelfinal = smf.ols('profit ~ RandD + marketing',data = startupstrain_new).fit()
modelfinal.summary() # p value is significant 
# R2 = 0.96 , B0 = 4980 , B1 = 0.763 , B2 = 0.030

# train_data prediction
modelfinaltrain_pred = modelfinal.predict(startupstrain_new)

# train residual values 
modelfinaltrain_resid  = modelfinaltrain_pred - startupstrain_new.profit
modelfinaltrain_resid.describe()
plt.boxplot(modelfinaltrain_pred)

# Residual plot
plt.scatter(modelfinaltrain_pred,modelfinaltrain_resid) # there is no pattern so error is independent

# RMSE value for train data 
modelfinaltrain_rmse = np.sqrt(np.mean(modelfinaltrain_resid**2)) # 7329 
modelfinaltrain_rmse 

# Accuarace of model 
np.corrcoef(startupstrain_new.profit,modelfinaltrain_pred)
# 98 % accurate


# After checking with all other model we observe that this model give us most accurate result.

"""
Final Equation -
Profit  = 49580 + 0.7626(R&D) + 0.0303(Marketing)

With Confidance interval 95% 
Profit  = 44500 + 0.685(R&D) + 0.002(Marketing)
Profit  = 54700 + 0.840(R&D) + 0.059(Marketing) """



