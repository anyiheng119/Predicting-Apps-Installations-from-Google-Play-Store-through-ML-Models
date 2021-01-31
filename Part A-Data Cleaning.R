# UCLA Extension Data Science Intensive 2020 Summer
# Homework: Capstone Project
# Part A: Data Cleaning
# Student: Yiheng An

setwd("F:/UCLAX Data Science Intensive/Capstone")
install.packages("xts")
install.packages("ggpubr")
library(ggpubr)
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

# Import data
Sys.setlocale("LC_ALL", "English") # prevent loading error
store <- read.csv("store.csv")
reviews <- read.csv("reviews.csv")

# Checks size of db
print(object.size(store), units = 'Mb')
print(object.size(reviews), units = 'Mb')

# Checks dimensions of db (number of rows and columns)
dim(store)
dim(reviews)

# Get a quick overview of the data structure
glimpse(store)
glimpse(reviews)
summary(store)

# Check first 10 rows 
store %>% head(10)
reviews %>% head(10) 
  
# Data cleaning (for store)
store = store %>% distinct()
store = store %>% distinct(App, .keep_all = TRUE)

table(store$Rating)
store$Rating = as.numeric(store$Rating)
store$Rating = ifelse(store$Rating > 5, NA, store$Rating)

table(store$Size)
store$Size = gsub("M", "000", store$Size)
store$Size = gsub("k", "", store$Size)
store$Size = as.numeric(store$Size)/1000 # make sure in Mb scale

table(store$Installs)
store$Installs = gsub("\\+", "", store$Installs)
store$Installs = gsub(",", "", store$Installs)
store$Installs = as.numeric(store$Installs)

table(store$Price)
store$Price = gsub("\\$", "", store$Price)
store$Price = as.numeric(store$Price)

table(store$Type)
store$Type = as.factor(store$Type)
store$Type = gsub("NaN", NA, store$Type)
store = filter(store,!is.na(Type))
store$Type = as.factor(store$Type)

table(store$Last.Updated)
store$Last.Updated = dmy(store$Last.Updated)
store$year = year(store$Last.Updated)
store$month = month(store$Last.Updated)

table(store$Android.Ver)
store$Android.Ver = substr(store$Android.Ver, start = 1, stop = 3) # keep first 2 digits
store$Android.Ver = as.numeric(store$Android.Ver)

# Export the data of store for document (store_01)
str(store)
write.csv(store, "store_01.csv")

# Check missing values
store <- read.csv("store_01.csv")
store = store[,-1] 
sum(is.na(store))

missing_data = store %>% summarise_all(funs(sum(is.na(.))/n()))
missing_data = gather(missing_data, key = "variables", value = "percent_missing")
ggplot(missing_data, aes(x = reorder(variables, percent_missing), y = percent_missing)) +
  geom_bar(stat = "identity", fill = "darkgreen", aes(color = I('white')), size = 0.3)+
  xlab('variables')+
  coord_flip()+ 
  theme_bw()

# Fix1: Create Dummy Variables 
store$Size.varies = ifelse(is.na(store$Size)==T,1,0)
store$Size.varies = as.factor(store$Size.varies)
store$Android.varies = ifelse(is.na(store$Android.Ver)==T,1,0)
store$Android.varies = as.factor(store$Android.varies)
str(store)
table(store$Size.varies)
summary(store)


# Fix2: Data Imputation
library(mice)
?mice
md.pattern(store)      # A tabular form of missing value present in each variable
store.missing = store[,c(3,5,12)]
mice = mice(store.missing)
mice.clean = complete(mice)
mice.clean = as.table(mice.clean)
stripplot(mice, col=c("grey",mdc(2)),pch=c(1,20))
densityplot(mice)

# Rating
before.Rating = as.data.frame(store[,3])
colnames(before.Rating) = c("Rating") 
before.Rating$imputation = "before"

after.Rating = as.data.frame(mice.clean[,1])
colnames(after.Rating) = c("Rating") 
after.Rating$imputation = "after"
Rating = rbind(before.Rating,after.Rating)

ggplot(Rating,aes(Rating,colour = imputation))+geom_boxplot()

ggplot(Rating,aes(Rating,fill = imputation))+
  geom_bar(stat="count" ,position="dodge")

# Size
before.Size = as.data.frame(store[,5])
colnames(before.Size) = c("Size") 
before.Size$imputation = "before"

after.Size = as.data.frame(mice.clean[,2])
colnames(after.Size) = c("Size") 
after.Size$imputation = "after"
Size = rbind(before.Size,after.Size)

ggplot(Size,aes(Size,colour = imputation))+geom_boxplot()

ggplot(Size,aes(Size,fill = imputation))+
  geom_bar(stat="density" ,position="dodge",alpha = 0.8)
  
  
# Android.Ver
before.Ver = as.data.frame(store[,12])
colnames(before.Ver) = c("Ver") 
before.Ver$imputation = "before"

after.Ver = as.data.frame(mice.clean[,3])
colnames(after.Ver) = c("Ver") 
after.Ver$imputation = "after"
Ver = rbind(before.Ver,after.Ver)

ggplot(Ver,aes(Ver,colour = imputation))+geom_boxplot()

ggplot(Ver,aes(Ver,fill = imputation))+
  geom_bar(stat="count" ,position="dodge",alpha = 0.8)

# Merge for clean store data
clean.store = store[,c(-3,-5,-12)]
clean.store = cbind(clean.store, mice.clean)
md.pattern(clean.store)

write.csv(clean.store, "store_02.csv")

