library(dplyr)
library(ggplot2)
library(lubridate)
library(gridExtra)
library(pROC)
library(gbm)
library(randomForest)


df_train = read.csv('F:/Analytics/AV - American Express/train_amex/train.csv')
df_hist = read.csv('F:/Analytics/AV - American Express/train_amex/historical_user_logs.csv')
df_test = read.csv('F:/Analytics/AV - American Express/test_LNMuIYp/test.csv')


df_test$DateTime <- ymd_hm(df_test$DateTime)
df_train$DateTime <- ymd_hm(df_train$DateTime)
df_hist$DateTime <- ymd_hm(df_hist$DateTime)

df_hist$month <- month(df_hist$DateTime)
df_hist$interest <- ifelse(df_hist$action=='interest',1,0)
df_hist$view <- ifelse(df_hist$action=='interest',0,1)


mean_interest <- df_hist %>%
    group_by(user_id,product) %>%
    summarise(mean_interest=mean(interest))

mean_view <- df_hist %>%
  group_by(user_id,product) %>%
  summarise(mean_view=mean(view))


df_train$product_category_2[is.na(df_train$product_category_2)] <- 442025
df_train$user_group_id[is.na(df_train$user_group_id)] <- 23927
df_train$gender[is.na(df_train$gender)] <- 'Male'
df_train$age_level[is.na(df_train$age_level)] <- 6
df_train$user_depth[is.na(df_train$user_depth)] <- 3
df_train$city_development_index[is.na(df_train$city_development_index)] <- 4


df_test$product_category_2[is.na(df_test$product_category_2)] <- 442025
df_test$user_group_id[is.na(df_test$user_group_id)] <- 23927
df_test$gender[is.na(df_test$gender)] <- 'Male'
df_test$age_level[is.na(df_test$age_level)] <- 6
df_test$user_depth[is.na(df_test$user_depth)] <- 3
df_test$city_development_index[is.na(df_test$city_development_index)] <- 4

#df_train$DateTime <- ymd_hm(df_train$DateTime)
df_train$wday <- wday(df_train$DateTime)
df_train$hour <- hour(df_train$DateTime)

#df_test$DateTime <- ymd_hm(df_test$DateTime)
df_test$wday <- wday(df_test$DateTime)
df_test$hour <- hour(df_test$DateTime)


#df_train$row <- NULL

df_train$t <- 1
df_test$t <- 0

df_test$is_click <- 0

df <- rbind(df_train,df_test)

df2 <- df %>%
  group_by(user_id,product) %>%
  mutate(p1_rank=rank(product,ties.method = 'first'))  


df3 <- df2 %>%
  group_by(user_id,webpage_id) %>%
  mutate(w_rank=rank(webpage_id,ties.method = 'first'))  

df4 <- df3 %>%
  group_by(user_id,campaign_id) %>%
  mutate(c_rank=rank(campaign_id,ties.method = 'first'))  


df5 <- df4 %>%
  group_by(user_id,product,webpage_id) %>%
  mutate(pw_rank=rank(webpage_id,ties.method = 'first'))

df6 <-  df5 %>%
  group_by(user_id) %>%
  mutate(lag.is_click = dplyr::lag(is_click, n = 1, default = -1))

df7 <-  df6 %>%
  group_by(user_id) %>%
  mutate(lag.is_click_2 = dplyr::lag(is_click, n = 2, default = -1))

df8 <-  df7 %>%
  group_by(user_id) %>%
  mutate(lag.is_click_3 = dplyr::lag(is_click, n = 3, default = -1))


user_click_count <- df8 %>%
  group_by(user_id) %>%
  summarise(user_click_count=n())

user_product_click_count <- df3 %>%
  group_by(user_id,product) %>%
  summarise(user_product_click_count=n())


df9 <- left_join(df8,user_click_count,by='user_id')
df10 <- left_join(df9,user_product_click_count,by=c('user_id','product'))
df11 <- left_join(df10, mean_interest,by=c('user_id','product'))
df12 <- left_join(df11, mean_view,by=c('user_id','product'))

df12$mean_interest[is.na(df12$mean_interest)] <- 0
df12$mean_view[is.na(df12$mean_view)] <- 0


train <- filter(df12,t==1)
test <- filter(df12,t==0)

dim(test)


colnames(train)

set.seed(100)
gbm_model <- gbm(is_click~.,data = train[,-c(1,3,2,7,8,10,11,12,16,21,28,13,17,18)],n.trees = 2000,verbose = TRUE, distribution = "bernoulli")

summary(gbm_model)
g <- predict(gbm_model,test,n.trees = 2000,type = 'response')

results <- data.frame(session_id=df_test$session_id,is_click=g)


write.csv(results,'GBM_model_11.csv',row.names = F)



log_model <- glm(is_click~.,data = train[,-c(1,3,2,7,8,10,11,12,16,21,28,13,17,18)],family = binomial(link = "logit"))

l <- predict(log_model,test,type = 'response')

results <- data.frame(session_id=df_test$session_id,is_click=l)


write.csv(results,'log_model_9.csv',row.names = F)




























