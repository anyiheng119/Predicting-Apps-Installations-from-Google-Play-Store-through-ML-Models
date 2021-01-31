# UCLA Extension Data Science Intensive 2020 Summer
# Homework: Capstone Project
# Part C: Predictive Models
# Student: Yiheng An

library(e1071)
library(qdap)
library(reprex)
library(data.table)
library(magrittr)
library(GGally)
library(tidyverse)
library(xts)
library(lubridate)
library(ggcorrplot)
library(corrplot)
library(ggmosaic)
library(cowplot)
library(reshape2)
library(caret)
library(lattice)
library(ModelMetrics)

# Import clean store data
setwd("F:/UCLAX Data Science Intensive/Capstone")
Sys.setlocale("LC_ALL", "English") # prevent loading error
store <- read.csv("store_03.csv")
store = store[,c(-1,-2)]

# Data preparation for modeling
glimpse(store)
store$Size.varies = as.factor(store$Size.varies)
store$Android.varies = as.factor(store$Android.varies)
store$Type = as.factor(store$Type)
store$Category = as.factor(store$Category)
store$Category = factor(store$Category, levels=names(sort(table(store$Category), decreasing=F)))
store$Genres = as.factor(store$Genres)
store$Content.Rating = as.factor(store$Content.Rating)
store$Installs.Group = as.factor(store$Installs.Group)

# Grouping installs into ordinal factors
table(store$Installs.Group)
store$Installs.level = store$Installs.Group
store$Installs.level = ifelse(store$Installs.Group=="(-Inf,0]","1",store$Installs.level)
store$Installs.level = ifelse(store$Installs.Group=="(0,1e+03]","1",store$Installs.level)  
store$Installs.level = ifelse(store$Installs.Group=="(1e+03,1e+04]","2",store$Installs.level)
store$Installs.level = ifelse(store$Installs.Group=="(1e+04,1e+05]","3",store$Installs.level) 
store$Installs.level = ifelse(store$Installs.Group=="(1e+05,1e+06]","4",store$Installs.level) 
store$Installs.level = ifelse(store$Installs.Group=="(1e+06,1e+07]","5",store$Installs.level) 
store$Installs.level = ifelse(store$Installs.Group=="(1e+07,1e+08]","6",store$Installs.level) 
store$Installs.level = ifelse(store$Installs.Group=="(1e+08,1e+09]","7",store$Installs.level) 
store$Installs.level = factor(store$Installs.level,levels = c("1","2","3","4","5","6","7"), ordered = TRUE)

store = store[,-15]
glimpse(store)
table(store$Installs)

## if we Use log(Installs)
store.log = filter(store,!store$Installs==0)
store.log = filter(store.log,!store.log$Content.Rating=='Unrated') #remove outliers
store.log$Category = factor(store.log$Category, levels=names(sort(table(store.log$Category), decreasing=F)))

## One-Hot Encoding£¨improve the compute speed£©
library(MASS)
store.hot = model.matrix(Installs.level ~Rating+Type+Size+Size.varies+
                          Android.Ver+Android.varies+Category+Last.Update-1, store.log) # Select "Installs.level" as the dependent variable and create a X matrix

store.hot = cbind(store.log[,15],store.hot)     # add the dependent variable
store.hot = as.data.frame(store.hot)
colnames(store.hot)[1] = 'Installs.level'
store.hot$Installs.level=factor(store.hot$Installs.level,levels = c("1","2","3","4","5","6","7"), ordered = TRUE)
glimpse(store.hot)

# Create trainset and testset
## Formal type
set.seed(99)
fullset = createDataPartition(store.log$Installs.level, p=0.80, list=FALSE)
trainset = store.log[fullset,]
testset = store.log[-fullset,]

## One-hot type
set.seed(99)
fullset2 = createDataPartition(store.hot$Installs.level, p=0.80, list=FALSE)
trainset2 = store.hot[fullset2,]
testset2 = store.hot[-fullset2,]

# Linear Regression
fit01 = lm(log(Installs)~Rating+Price+Type+Content.Rating+Size+Size.varies+
             Android.Ver+Android.varies+Category+Last.Update, data = trainset)

summary(fit01) # R2 = 0.2972  RMSE = 3.639053

library(MASS)
fit03 = stepAIC(fit01, direction="both")
summary(fit03)  #R2 = 0.9324

# Prediction analysis
pred01=predict(fit01,testset)
pred02=predict(fit02,testset)

compare = as.data.frame(pred01)
colnames(compare)[1] = 'liner_pred01'
compare$liner_pred02 = as.numeric(pred02)

compare$true.num = log(testset$Installs)
rmse(compare$true.num, compare$liner_pred01)
ggplot(aes(x=liner_pred01, y = true.num), data = compare)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  stat_smooth(method = 'lm', aes(colour = 'red'), se = F) +
  xlab("Predicted log(Installs)") + ylab("Actual log(Installs)") + ggtitle("Predicted vs. Actual log(Installs)")

plot(predict(fit01), residuals(fit01))
hist(fit01$residuals)
qqnorm(fit01$residuals);qqline(fit01$residuals)

chisq.test(store.log$Size.varies,store.log$Android.varies)

# Ordered Logistic Regression
library(MASS)
order01a = polr(Installs.level~Rating+Type+Size+Size.varies+
                  Android.Ver+Android.varies+Category+Last.Update, data = store.log, Hess=TRUE)
summary(order01a) #AIC: 29514.77

order01b = stepAIC(order01a, direction="both")
summary(order01b) #AIC: 29399.47 

# Odds Ratios-1
exp(coef(order01a))-1

# Revision
order02 = stepAIC(order01a, direction="both")
summary(order02) #AIC: 9616.963 drop Android.varies

# 10-fold Cross Validation
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# Ordered Logistic Regression (cannot run in one-hot data set)
set.seed(99)
fit.OLR = train(Installs.level~Rating+Type+Size+Size.varies+Android.Ver+Android.varies+
                 Category+Last.Update, data=trainset2, method="polr", metric=metric, trControl=control)
fit.OLR

# Linear Discriminant Analysis (LDA)
set.seed(99)
fit.LDA = train(Installs.level~., data=trainset2, method="lda", metric=metric, trControl=control)
fit.LDA

# Classification and Regression Trees (CART)
set.seed(99)
fit.tree <- train(Installs.level~., data=trainset2, method="rpart", metric=metric, trControl=control)
fit.tree

# k-Nearest Neighbors (KNN)
set.seed(99)
fit.knn <- train(Installs.level~., data=trainset2, method="knn", metric=metric, trControl=control)
fit.knn

# Support Vector Machines (SVM)
set.seed(99)
fit.svm <- train(Installs.level~., data=trainset2, method="svmRadial", metric=metric, trControl=control)
fit.svm

# Random Forest
set.seed(99)
fit.rf <- train(Installs.level~., data=trainset2, method="rf", metric=metric, trControl=control)
fit.rf

# Bayesian Generalized Linear Model
set.seed(99)
fit.logi = train(Installs.level~., data=trainset2,method="bayesglm", metric=metric, trControl=control)
fit.logi

# XGBoosting(factor)
set.seed(99)
fit.xgb = train(Installs.level~., data=trainset2, method="xgbLinear", metric=metric, trControl=control)
fit.xgb

results = resamples(list(orderlog=fit.OLR,lda=fit.LDA, cart=fit.tree, knn=fit.knn, svm=fit.svm, 
                         logi=fit.logi,rf=fit.rf, xgb=fit.xgb))
summary(results)


# Make Predictions
predict = predict(fit.xgb, testset2)
predict = as.data.frame(predict)

glimpse(store.log)
store.log = store.log[,-8]
