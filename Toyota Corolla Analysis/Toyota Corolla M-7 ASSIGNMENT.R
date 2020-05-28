# =============================================================================
# Business Problem :- Prepare a prediction model for Predicting Price.
# =============================================================================

library(readr)
toyota = read.csv(file.choose())
toyota <- toyota[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
View(toyota)

########### Exploratory Data Analysis ###########

# Measures of Central Tendency / First moment business decision
attach(toyota) 
# Mean
mean(Price)
mean(Age_08_04)
mean(KM)
mean(HP)
mean(cc)
mean(Gears)
mean(Quarterly_Tax)
mean(Weight)

# Median
median(Price)
median(Age_08_04)
median(KM)
median(HP)
median(cc)
median(Gears)
median(Quarterly_Tax)
median(Weight)

# Mode
table1 = table(Price)
table1
table1[table1 == max(table1)]

table2 = table(Age_08_04)
table2
table2[table2 == max(table2)]

table3 = table(KM)
table3
table3[table3 == max(table3)]

table4 = table(HP)
table4
table4[table4 == max(table4)]

table5 = table(cc)
table5
table5[table5 == max(table5)]

table6 = table(Gears)
table6
table6[table6 == max(table6)]

table7 = table(Quarterly_Tax)
table7
table7[table7 == max(table7)]

table8 = table(Weight)
table8
table8[table8 == max(table8)]

# Measures of Dispersion / Second moment business decision
# variance
var(Price)
var(Age_08_04)
var(KM)
var(HP)
var(cc)
var(Gears)
var(Quarterly_Tax)
var(Weight)

# standard deviation
sd(Price)
sd(Age_08_04)
sd(KM)
sd(HP)
sd(cc)
sd(Gears)
sd(Quarterly_Tax)
sd(Weight)

library(moments)
# Third moment business decision
skewness(Price)
skewness(Age_08_04)
skewness(KM)
skewness(HP)
skewness(cc)
skewness(Gears)
skewness(Quarterly_Tax)
skewness(Weight)

# Fourth moment business decision
kurtosis(Price)
kurtosis(Age_08_04)
kurtosis(KM)
kurtosis(HP)
kurtosis(cc)
kurtosis(Gears)
kurtosis(Quarterly_Tax)
kurtosis(Weight)

# Graphical Representation
# histogram
hist(Price)
hist(Age_08_04)
hist(KM)
hist(HP)
hist(cc)
hist(Gears)
hist(Quarterly_Tax)
hist(Weight)

# boxplot
boxplot(Price)
boxplot(Age_08_04)
boxplot(KM)
boxplot(HP)
boxplot(cc)
boxplot(Gears)
boxplot(Quarterly_Tax)
boxplot(Weight)
# Normal Quantile-Quantile Plot
qqnorm(Price)
qqline(Price)
qqnorm(Age_08_04)
qqline(Age_08_04)
qqnorm(KM)
qqline(KM)
qqnorm(HP)
qqline(HP)
qqnorm(cc)
qqline(cc)
qqnorm(Gears)
qqline(Gears)
qqnorm(Quarterly_Tax)
qqline(Quarterly_Tax)
qqnorm(Weight)
qqline(Weight)

# summary
summary(toyota)

# Pair plot
library(GGally)
ggpairs(toyota)
pairs(toyota)

# Correlation
cor(toyota)

# Finding outliers by bulding MLR
model1 <-lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , toyota)
summary(model1) # insignificant

modelcc <-lm(Price ~ cc ,  toyota)
summary(modelcc)# significant

modeld <-lm(Price ~ Doors ,  toyota)
summary(modeld)# significant

library(car)
# Check data having any influential values
influenceIndexPlot(model1, id.n=3) # Index Plots of the influence measures
influencePlot(model1, id.n=3) # A user friendly representation of the above

# index 81 are showing inlfuence so we can exclude entire row
toyota1 = toyota[-c(81,961,222,602),]

model2 <-lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , toyota1)
summary(model2) # significant

####################- Spliting Data in Train and Test -#####################
set.seed(111)
# Random sample indexes
train_index <- sample(1:nrow(toyota1), 0.7 * nrow(toyota1))
test_index <- setdiff(1:nrow(toyota1), train_index)

# Build X_train, y_train, X_test, y_test
X_train <- toyota1[train_index, -1]
y_train <- toyota1[train_index, "Price"]

X_test <- toyota1[test_index, -1]
y_test <- toyota1[test_index, "Price"]

####################- Multilinear regression model Building -#####################

model_train <-lm(y_train ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight , X_train)
summary(model_train) # significant

# Distribution of Error
summary(model_train$residuals)

## calclutaing RMSE
sqrt(mean(model_train$residuals^2)) # 1176

# 95% confidance interval
confint(model_train,level=0.95) #final model
predict(model_train, interval = "confidence")

# different plots
plot(model_train)
avPlots(model_train, id.n=2, id.cex=0.8, col="red")

# Prediction 
predicty = predict(model_train,X_train)

# Accuaracy of model
cor(y_train,predicty) # 94%

####################-  Testing Multilinear regression model -#####################
# Prediction on test data
predict_test = predict(model_train,X_test)

# Error
test_error = predict_test - y_test

##calclutaing RMSE
sqrt(mean(test_error^2)) # 1275

# Accuaracy of model
cor(y_test,predict_test) # 94%

