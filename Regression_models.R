library(readxl)
library(tidyverse)
library(ggpubr)
library(lmtest)
library(jtools) 
library(car)

setwd("~/UNI/ECMT3150/Group Assignment")

theme_set(theme_pubr())
data <- read_excel("Elon_Dogecoin_Output.xlsx", sheet='Elon Tweets and CAR')

car = log(1+data$CAR_5)

ggplot(data, aes(Tweet_Count, CAR_60)) +
  geom_point() +
  stat_smooth(method = lm)

interaction = data$Tweet_Count*log(data$Market_Cap)

ggplot(data, aes(Months, CAR_5)) +
  geom_point() +
  stat_smooth(method = lm)

ggplot(data, aes(log(hour_vol), CAR_15)) +
  geom_point() +
  stat_smooth(method = lm)


model1 <- lm(CAR_1 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + CAR_1_1 + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model1)
durbinWatsonTest(model1)
bptest(model1)

model2 <- lm(CAR_2 ~ Tweet_Count + Market_Cap + Months + interaction + CAR_2_1 + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model2)

durbinWatsonTest(model2)
bptest(model2)

model5 <- lm(CAR_5 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + CAR_5_1 + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model5)
bptest(model5)

model10 <- lm(CAR_10 ~ Tweet_Count + Market_Cap + Market_Cap*Tweet_Count + Months + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model10)
bptest(model10)

model15 <- lm(CAR_15 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model15)
bptest(model15)

model30 <- lm(CAR_30 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model30)
bptest(model30)

model60 <- lm(CAR_60 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model60)

model90 <- lm(CAR_90 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model90)

model120 <- lm(CAR_120 ~ Tweet_Count + Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data)
summary(model120)


anova(model, model_full)
#There is strong evidence that the full model is not better

plot(model)
bptest(model)
#Fail to reject null hypothesis - model is sufficiently homoskedastic

summ(model, robust="HC1")

dust(model) %>% 
  sprinkle(cols = c("estimate", "std.error", "statistic"), round = 2) %>%
  sprinkle(cols = "p.value", fn = quote(pvalString(value))) %>% 
  sprinkle_colnames("Term", "Coefficient", "SE", "T-statistic", 
                    "P-value")

wt <- 1 / lm(abs(model10$residuals) ~ model10$fitted.values)$fitted.values^2
wls_model1 <- lm(CAR_10 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data, weights=wt)
summary(wls_model1)

wt2 <- 1 / lm(abs(model15$residuals) ~ model15$fitted.values)$fitted.values^2
wls_model2 <- lm(CAR_15 ~ Tweet_Count + Market_Cap + Months + Tweet_Count*Market_Cap + Picture + Video + Link + Recognisability + hour_ret + hour_vol, data=data, weights=wt)
summary(wls_model2)

