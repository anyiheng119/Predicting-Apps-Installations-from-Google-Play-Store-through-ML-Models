# UCLA Extension Data Science Intensive 2020 Summer
# Homework: Capstone Project
# Part D: Sentiment Analysis and Model Refinement
# Student: Yiheng An

require(devtools)
library(sentimentr)
library(sentiment)
library(stringr)
library(tidyverse)
library(tidytext)
library(tm)
library(gmodels)
library(qdap)
library(wordcloud)
library(jiebaRD)
library(jiebaR)
library(RColorBrewer)

# Import clean reviews text data
setwd("F:/UCLAX Data Science Intensive/Capstone")
Sys.setlocale("LC_ALL", "English") # prevent loading error
reviews <- read.csv("reviews.csv")


glimpse(reviews)
length(unique(reviews$App)) #863

# Data cleaning (for reviews text)
reviews = reviews %>% distinct()
reviews = reviews %>% distinct(Translated_Review, .keep_all = TRUE)
reviews = reviews[,-1]
reviews = filter(reviews,!reviews$Translated_Review=='nan')

# Checking NAs
missing_data = reviews %>% summarise_all(funs(sum(is.na(.))/n()))
missing_data = gather(missing_data, key = "variables", value = "percent_missing")
ggplot(missing_data, aes(x = reorder(variables, percent_missing), y = percent_missing)) +
  geom_bar(stat = "identity", fill = "darkgreen", aes(color = I('white')), size = 0.3)+
  xlab('variables')+
  coord_flip()+ 
  theme_bw()

# Calculate sentiment coefficients
out = polarity(reviews$Translated_Review, reviews$App)
out.data = out$all # put the details in a data frame

# Reviews text analysis
## Negative.top10
out.data = out.data[order(out.data$polarity),]
negative.top10 = out.data %>% head(10)
negative.top10$text.var

## Positive.top10
out.data = out.data[order(out.data$polarity, decreasing = T),]
positive.top10 = out.data %>% head(10)

# Frequency analysis
## Covert word lists to data frame
library (plyr)
neg.words <- ldply (out.data$neg.words, data.frame)
colnames(neg.words)[1] = 'neg.words'

## Count and filter the words
seg<-qseg[neg.words$neg.words]  # use qseg split words
seg<-seg[nchar(neg.words)>1]    # remove words whose length shorter than 1
seg<-table(seg)
seg<-seg[!grepl('[0-9]+',names(seg))]     # remove numbers
seg <- sort(seg, decreasing = TRUE)[1:50] # keep top50

## Draw a word could
wordcloud(names(seg), seg, col = brewer.pal(8, "Set1"), min.freq = min(seg),
          random.color = T, max.words = max(seg), random.order = T,  scale = c(4, 1))

# Get aggregate data
out.agg = out$group
plot(out)

# Merge the data
newdata = left_join(store.log,out.agg, by = "App")
newdata = newdata[,-c(16,17)]

# Checking NAs
missing_data = newdata %>% summarise_all(funs(sum(is.na(.))/n()))
missing_data = gather(missing_data, key = "variables", value = "percent_missing")
ggplot(missing_data, aes(x = reorder(variables, percent_missing), y = percent_missing)) +
  geom_bar(stat = "identity", fill = "darkgreen", aes(color = I('white')), size = 0.3)+
  xlab('variables')+
  coord_flip()+ 
  theme_bw()

# Data imputation
library(Hmisc)
newdata$ave.polarity=impute(newdata$ave.polarity, 0) 

library(mice)
sd = newdata[,c(1,17)]
mice = mice(sd)
mice.clean = complete(mice)

# Visualization
ggplot(newdata,aes(x = sd.polarity,fill = Installs.level))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+
  ggtitle('Average polarity Distribution')

ggplot(aes(x=ave.polarity, y = log(Installs)), data = newdata)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  ggtitle('Rating vs. Installs')

ggplot(newdata, aes(x=ave.polarity, y=Installs.level)) + geom_violin(aes(fill = Installs.level))+
  geom_boxplot(width=0.1, fill="white") + theme_bw()+ 
  labs(title="Average polarity vs. Installs")

# Data for OLR Model
newdata2 = cbind(newdata[,-c(17,18)],mice.clean[,-1])
colnames(newdata2)[17] = "sd.polarity"
glimpse(newdata2)

library(MASS)
order.new= polr(Installs.level~Rating+Type+Size+Size.varies+Android.Ver+Android.varies+
                  Category+Last.Update+sd.polarity+ave.polarity, data = newdata2, Hess=TRUE)

summary(order.new)# AIC: 29306.31
summary(order01a) # AIC: 29514.77

# Odds Ratios-1
exp(coef(order.new))-1

order.new2 = stepAIC(order.new, direction="both")
summary(order.new2) # AIC: 9616.963 drop Android.varies


## One-Hot Encoding(improve the compute speed)
library(MASS)
store.hot = model.matrix(Installs ~Rating+Type+Size+Size.varies+sd.polarity+ave.polarity+Reviews+
                           Android.Ver+Android.varies+Category+Last.Update-1, newdata2) # Select "Installs.level" as the dependent variable and create a X matrix

store.hot = cbind(newdata2[,4],store.hot)     # add the dependent variable
store.hot = as.data.frame(store.hot)
colnames(store.hot)[1] = 'Installs'
store.hot$Installs.level=factor(store.hot$Installs.level,levels = c("1","2","3","4","5","6","7"), ordered = TRUE)
glimpse(newdata2)
write.csv(store.hot,'hot_new_num.csv')

## Create train and test set
store.hot = read.csv('hot_new.csv')
store.hot = store.hot[,-1]
glimpse(store.hot)
store.hot.b = store.hot[,-c(7,8)]
## One-hot format
set.seed(99)
fullset.new = createDataPartition(store.hot$Installs.level, p=0.80, list=FALSE)
trainset.new = store.hot[fullset.new,]
testset.new = store.hot[-fullset.new,]

## Formal format
set.seed(99)
fullset.new.f = createDataPartition(newdata$Installs.level, p=0.80, list=FALSE)
trainset.new.f = newdata[fullset.new.f,]
testset.new.f = newdata[-fullset.new.f,]

# ML Model refinement
# 10-fold Cross Validation
library(caret)
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# Ordered Logistic Regression (cannot run in one-hot data set)
set.seed(99)
fit.OLR.b = train(Installs.level~Rating+Type+Size+Size.varies+Android.Ver+Android.varies+
                  Category+Last.Update, data=trainset.new.f, method="polr", metric=metric, trControl=control)
fit.OLR.b

# Linear Discriminant Analysis (LDA)
set.seed(99)
fit.LDA.b = train(Installs.level~., data=trainset.old, method="lda", metric=metric, trControl=control)
fit.LDA.b

# Classification and Regression Trees (CART)
set.seed(99)
fit.tree.b <- train(Installs.level~., data=trainset.old, method="rpart", metric=metric, trControl=control)
fit.tree.b

# k-Nearest Neighbors (KNN)
set.seed(99)
fit.knn.b <- train(Installs.level~., data=trainset.old, method="knn", metric=metric, trControl=control)
fit.knn.b

# Support Vector Machines (SVM)
set.seed(99)
fit.svm.b <- train(Installs.level~., data=trainset.old, method="svmRadial", metric=metric, trControl=control)
fit.svm.b

# Random Forest
set.seed(99)
fit.rf.b <- train(Installs.level~., data=trainset.old, method="rf", metric=metric, trControl=control)
fit.rf.b

# Bayesian Generalized Linear Model
set.seed(99)
fit.logi.b = train(Installs.level~., data=trainset.old,method="bayesglm", metric=metric, trControl=control)
fit.logi.b

# XGBoosting(factor)
set.seed(99)
fit.xgb.b = train(Installs.level~., data=trainset.old, method="xgbLinear", metric=metric, trControl=control)
fit.xgb.b


results.b = resamples(list(orderlog=fit.OLR.b,lda=fit.LDA.b, cart=fit.tree.b, knn=fit.knn.b, svm=fit.svm.b, 
                         logi=fit.logi.b,rf=fit.rf.b, xgb=fit.xgb.b))
summary(results.b)


# Make Predictions
library(caret)
predict.new.rf = predict(fit.rf, testset.new)
confusionMatrix(predict.new.rf, testset.new$Installs.level, positive = "Yes")
table(predict.new.rf,testset.new$Installs.level)

predict.new.xgb = predict(fit.xgb, testset.new)
confusionMatrix(predict.new.xgb, testset.new$Installs.level, positive = "Yes")
table(predict.new.xgb,testset.new$Installs.level)


# Explore ROC (not applicable to multi-group response here)
library(pROC)
xgb.roc = roc(response = testset.new$Installs.level, predictor = predict.new.xgb,smooth = F) 
rf.roc = roc(response = testset.new$Installs.level, predictor = predict.new.rf,smooth = F) 

xgb.roc.m = multiclass.roc(response = testset.new$Installs.level, predictor = predict.new.xgb,levels = 7)
plot(xgb.roc.m$response,xgb.roc.m$predictor)

plot.roc(rf.roc,legacy.axes = TRUE, print.auc.y = 1.0, print.auc = TRUE) # AUC = 0.881
plot.roc(xgb.roc,legacy.axes = TRUE, print.auc.y = 1.0, print.auc = TRUE)# AUC = 0.894

coords(rf.roc, "best", "threshold",transpose = TRUE) # the optimal threshold is 0.33
coords(xgb.roc, "best", "threshold",transpose = TRUE)


# Linear Regression
fit.new = lm(log(Installs)~Rating+Price+Type+Content.Rating+Size+Size.varies+ave.polarity+sd.polarity+
             Android.Ver+Android.varies+Category+Last.Update, data = trainset.new.f)

summary(fit.new) # R2 = 0.3131   RMSE = 3.590902
summary(fit01)   # R2 = 0.2972   RMSE = 3.639053


library(MASS)
fit03 = stepAIC(fit.new, direction="both")
summary(fit03)  #R2 = 0.9324

# Prediction analysis(Linear)
testset.new.f = filter(testset.new.f,!testset.new.f$Installs==0)
predict.new.lm=predict(fit.new,testset.new.f)

compare = as.data.frame(predict.new.lm)
colnames(compare)[1] = 'liner_pred01'

compare$true.num = log(testset.new.f$Installs)
rmse(compare$true.num, compare$liner_pred01)
ggplot(aes(x=liner_pred01, y = true.num), data = compare)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  stat_smooth(method = 'lm', aes(colour = 'red'), se = F) +
  xlab("Predicted log(Installs)") + ylab("Actual log(Installs)") + ggtitle("Predicted vs. Actual log(Installs)")

plot(predict(fit.new), residuals(fit.new))
hist(fit.new$residuals)
qqnorm(fit.new$residuals);qqline(fit.new$residuals)

