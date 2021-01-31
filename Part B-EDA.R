# UCLA Extension Data Science Intensive 2020 Summer
# Homework: Capstone Project
# Part B: EDA
# Student: Yiheng An

library(data.table)
library(magrittr)
library(GGally)
library(tidyverse)
library(xts)
library(lubridate)
library(ggcorrplot)
library(corrplot)
library(ggmosaic)
library(caret)

# Import clean store data
setwd("F:/UCLAX Data Science Intensive/Capstone")
Sys.setlocale("LC_ALL", "English") # prevent loading error
clean.store <- read.csv("store_02.csv")

glimpse(clean.store)
clean.store$Last.Updated = ymd(clean.store$Last.Updated)
clean.store$Size.varies = as.factor(clean.store$Size.varies)
clean.store$Android.varies = as.factor(clean.store$Android.varies)
clean.store$Installs = as.integer(clean.store$Installs)

# change date to time interval
clean.store$Last.Updated = ymd(clean.store$Last.Updated)
latest <- ymd_hms("2018-08-08 0:00:00")
clean.store$Updated.interval <- interval(clean.store$Last.Updated,latest)
clean.store$Updated.interval = time_length(clean.store$Updated.interval,'day')
clean.store = clean.store[,-c(10,11,12)]
colnames(clean.store)[15] = 'Last.Update'

# Installs grouping
store.Group = clean.store %>% 
  group_by(Installs.Group = cut(Installs, c(-Inf,0,1000,10000,100000,1000000,10000000,100000000,1000000000)))

# EDA: Categorical variables(binary)
## Size: Varies with device or not
table(clean.store$Size.varies)
clean.store %>% 
  group_by(Size.varies) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(Size.varies, -percent), percent), fill = Size.varies)+
  geom_col(fill = c("grey", "light blue"))+
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = 0.2, vjust = 2, size = 5)+ 
  theme_bw()+  
  xlab("Size.varies") + ylab("Percent") + ggtitle("Size: Varies with device or not")

## Size.varies VS. Installs (percent)
ggplot(store.Group,aes(Size.varies,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="count" ,position="fill",alpha = 0.8)+
  ylab("Percent") + ggtitle("Size.varies distribution by Installs")

ggplot(store.Group,aes(Size.varies,fill = Android.varies))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="count" ,alpha = 0.8)+
  ylab("Percent") + ggtitle("Size.varies distribution by Installs")

## Android.var: Varies with device or not
table(clean.store$Android.varies)
clean.store %>% 
  group_by(Android.varies) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(Android.varies, -percent), percent), fill = Android.varies)+
  geom_col(fill = c("grey", "light blue"))+
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = 0.2, vjust = 2, size = 5)+ 
  theme_bw()+  
  xlab("Android.varies") + ylab("Percent") + ggtitle("Android.var: Varies with device or not")

## Android.var VS. Installs (percent)
ggplot(store.Group,aes(Android.varies,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="count" ,position="fill",alpha = 0.8)+
  ylab("Percent") + ggtitle("Android.varies distribution by Installs")

## Type: Free or not
table(clean.store$Type)
clean.store %>% 
  group_by(Type) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(Type, -percent), percent), fill = Type)+
  geom_col(fill = c("grey", "light blue"))+
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = 0.2, vjust = 2, size = 5)+ 
  theme_bw()+  
  xlab("Type") + ylab("Percent") + ggtitle("Type: Free or not")

## Type VS. Installs (percent)
ggplot(store.Group,aes(Type,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="count" ,position="fill",alpha = 0.8)+
  ylab("Percent") + ggtitle("Type distribution by Installs")

# EDA: Categorical variables(multi-valued)

##Installs.Group
table(store.Group$Installs.Group)
ggplot(store.Group,aes(log(Installs),fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="count",position="stack",alpha = 0.8)+ ylab("count")+ coord_flip()

store.Group %>% 
  group_by(Installs.Group) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(Installs.Group, -percent), percent), fill = Installs.Group)+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = 0.2, vjust = 2, size = 5)+ 
  theme_bw()+  
  xlab("Type") + ylab("Percent") + ggtitle("Type: Free or not")

## Content.Rating
table(clean.store$Content.Rating)
clean.store$Content.Rating = factor(clean.store$Content.Rating, levels=names(sort(table(clean.store$Content.Rating), decreasing=T))) # ordering
ggplot(clean.store,aes(Content.Rating))+
  geom_bar(stat="count" ,position="stack",alpha = 0.8)

## Content.Rating VS. Installs (percent)
store.Group$Content.Rating = factor(store.Group$Content.Rating, levels=names(sort(table(store.Group$Content.Rating), decreasing=T))) # ordering
ggplot(store.Group,aes(Content.Rating,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="count",position="fill",alpha = 0.8)+ ylab("Percent")

## Category
length(unique(clean.store$Category))
clean.store$Category = factor(clean.store$Category, levels=names(sort(table(clean.store$Category), decreasing=F))) # ordering
ggplot(clean.store,aes(Category,fill = Size.varies))+
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+ coord_flip()

store.Group$Category = factor(store.Group$Category, levels=names(sort(table(store.Group$Category), decreasing=F))) # ordering
ggplot(store.Group,aes(Category,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="count",position="fill",alpha = 0.8)+ ylab("Percent")+ coord_flip()


ggplot(aes(x=log(Reviews), y = Rating), data = store.Group)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  ggtitle('Rating vs. Installs')

## Genres
length(unique(clean.store$Genres))
clean.store$Genres = factor(clean.store$Genres, levels=names(sort(table(clean.store$Genres), decreasing=F))) # ordering
ggplot(clean.store,aes(Genres))+
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+ coord_flip()

## Genras(count>100)
Genres = clean.store %>% group_by(Genres) %>% summarise(Count = n())
Genres.top = filter(Genres, Count >= 100)
top.names = select(Genres.top, Genres)

clean.store$Genres = factor(clean.store$Genres, levels=names(sort(table(clean.store$Genres), decreasing=F))) # ordering
clean.store %>% filter(Genres %in% top.names$Genres) %>%
  ggplot(aes(Genres))+
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+ coord_flip()

store.Group$Genres = factor(store.Group$Genres, levels=names(sort(table(store.Group$Genres), decreasing=F))) # ordering
store.Group %>% filter(Genres %in% top.names$Genres) %>%
  ggplot(aes(Genres,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="count",position="fill",alpha = 0.8)+ ylab("Percent")+ coord_flip()

## Last.Update (days)
ggplot(store.Group,aes(Last.Update))+
  geom_bar(stat="count" ,position="stack",alpha = 0.8)

ggplot(store.Group,aes(Last.Update,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="density",alpha = 0.8)+ ylab("Percent")

ggplot(store.Group,aes(factor(month),fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="count",position="fill",alpha = 0.8)+ ylab("Percent")

ggplot(aes(x=Last.Update, y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  ggtitle('Updated.interval vs. Installs')

ggplot(store.Group, aes(x=Last.Update, y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + theme_bw()+ 
  labs(title="Rating vs. Installs")

## Rating
ggplot(store.Group,aes(x = Rating,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+
  ggtitle('Rating Distribution')

ggplot(aes(x=Rating, y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.5, color = 'violetred3', size=0.8)+theme_bw()+ 
  ggtitle('Rating vs. Installs')

ggplot(store.Group, aes(x=Rating, y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + theme_bw()+ 
  labs(title="Rating vs. Installs")

## reviews
table(store.Group$Reviews)
ggplot(store.Group,aes(x = log(Reviews),fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+theme_bw()+ 
  ggtitle('log(Reviews) Distribution')

ggplot(aes(x=log(Reviews), y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.2, color = 'violetred3')+
  ggtitle('log(Reviews) vs. Installs')

ggplot(store.Group, aes(x=log(Reviews), y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + theme_bw()+ 
  labs(title="log(Reviews) vs. Installs")

## Price
table(store.Group$Price)
ggplot(store.Group,aes(x = log(Price),fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+
  ggtitle('log(Price) Distribution')

ggplot(aes(x=log(Price), y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.2, color = 'violetred3')+theme_bw()+
  ggtitle('Install vs. Rating')

ggplot(store.Group, aes(x=log(Price), y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + theme_bw()+
  labs(title="log(Price) vs. Installs")

## Size
ggplot(store.Group,aes(x = log(Size),fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+
  ggtitle('log(Size) Distribution')

ggplot(aes(x=Size, y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.5, color = 'violetred3',size=1)+theme_bw()+ 
  ggtitle('Size vs. Installs')

ggplot(store.Group, aes(x=Size, y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + 
  labs(title="Size vs. Installs")

## Android.Ver
ggplot(store.Group,aes(x = Android.Ver,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="density" ,position="stack",alpha = 0.8)+
  ggtitle('Android.Ver Distribution')

ggplot(store.Group,aes(x = Android.Ver,fill = Installs.Group))+
  scale_fill_brewer(palette = "RdYlGn")+theme_bw()+ 
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+
  ggtitle('Android.Ver Distribution')

ggplot(aes(x=Android.Ver, y = Installs.Group), data = store.Group)+
  geom_jitter(alpha = 0.5, color = 'violetred3',size=1)+theme_bw()+ 
  ggtitle('Android.Ver vs. Installs')

ggplot(store.Group, aes(x=Android.Ver, y=Installs.Group)) + geom_violin(aes(fill = Installs.Group))+
  geom_boxplot(width=0.1, fill="white") + 
  labs(title="Android.Ver vs. Installs")

## Correlation Matrix for Numerical variables
store.log2 = filter(clean.store,!clean.store$Reviews==0) ## if we Use log(Reviews)
store.log2 = filter(store.log2,!store.log2$Installs==0) ## if we Use log(Installs)
store.log2$log_reviews =log(store.log2$Reviews)
store.log2$log_installs = log(store.log2$Installs)

corrplot(cor(store.log2[,c(7,12,13,14,15,16,17)]), type="lower", method="number")

write.csv(store.Group,"store_03.csv")





