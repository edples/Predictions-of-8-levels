


setwd("C:/comply")

m_data <- read.csv("M.csv")
profile_data <- read.csv("ProfileMetadata.csv")
library(tidyverse)

class(m_data)
class(profile_data)
str(m_data)
str(profile_data)
glimpse(m_data)
glimpse(profile_data)
head(m_data)
colSums(is.na(m_data)) #no NA's, the n.a.'s  are characters
colSums(is.na(profile_data)) # the same as above
head(profile_data)
summary(m_data)
summary(profile_data)
# The problem is clasification type , the target variable m_data$M having 8 levels.

library(stringr)

sum(str_count(profile_data$Number.of.Years.of.Births.in.profile , "n.a.")) #count the number of n.a.'s
sum(str_count(profile_data$Difference.in.Years.between.Max.Min.Year.of.Birth , "n.a."))
# as the last 2 columns of the profile_data have 84025 of "n.a." from 96737 observations, they are no longer relevant to the case.
profile_data <- profile_data[, 1:9] #slice the data, dropping the above 2 columns
glimpse(profile_data)
identical(m_data$Profile_id, profile_data$Profile_id)#check if the"Profile_id" columns of the both data sets are identical, 
# meaning that the order is the same in both datasets.
profile_data$m_var <-m_data$M #addin the M column to the relevant dataset, which the model will train and predict on.

na_profile_data <-subset(profile_data, m_var == "n.a.") # choose only the n.a. rows,this will be dataset that the model will make predictions on.
glimpse(na_profile_data)
profile_data<- subset(profile_data, !m_var == "n.a.")# choose only the rows without n.a.,this will be the training dataset 
glimpse(profile_data)

profile_data$has_year_of_birth<- as.factor(profile_data$has_year_of_birth)#convert the relevant columns of the datasets to factors
profile_data$has_country <- as.factor(profile_data$has_country)
profile_data$is_sanction <- as.factor(profile_data$is_sanction)
profile_data$is_pep <- as.factor(profile_data$is_pep)
profile_data$is_adverse_media <- as.factor(profile_data$is_adverse_media)
profile_data$Number.of.Source.Docs.for.Profile <- as.factor(profile_data$Number.of.Source.Docs.for.Profile)


na_profile_data$has_year_of_birth<- as.factor(na_profile_data$has_year_of_birth)
na_profile_data$has_country <- as.factor(na_profile_data$has_country)
na_profile_data$is_sanction <- as.factor(na_profile_data$is_sanction)
na_profile_data$is_pep <- as.factor(na_profile_data$is_pep)
na_profile_data$is_adverse_media <- as.factor(na_profile_data$is_adverse_media)
na_profile_data$Number.of.Source.Docs.for.Profile <- as.factor(na_profile_data$Number.of.Source.Docs.for.Profile)
profile_data$m_var <- as.numeric(profile_data$m_var)

library(vtreat)
library(dplyr)
library(magrittr)
# I consider the below variables  significant as predictors.
vars_test <- c("has_year_of_birth", " has_country ", "is_sanction ", "is_pep", "is_adverse_media", "Number.of.Source.Docs.for.Profile")
treatplan <- designTreatmentsZ(profile_data,vars_test)#design a treatment plan for the variables which handles the missing values, to be used in XGBoost
(scoreFrame <- treatplan %>% 
    use_series(scoreFrame) %>%
    select(varName, origName, code))

(newvars <- scoreFrame %>%
    filter(code %in% c("clean", "lev")) %>%
    use_series(varName))
trainingframe.treat <- prepare(treatplan, profile_data, varRestriction = newvars) # this makes the data compatible to XgBoost
testframe.treat <-prepare(treatplan, na_profile_data, varRestriction = newvars)


library(xgboost)
set.seed(123) # for reproducibility
model <- xgb.cv(data = as.matrix(trainingframe.treat), #perform cross-validation to find the optimal number of trees
             label = profile_data$m_var,
             nrounds = 100,
             num_class = 8,
             nfold = 5,
             objective = "multi:softmax",# multiclass target
             eta = 0.3,
             max_depth = 6,
             early_stopping_rounds = 10,
             verbose = 0    # silent
)
(evlog <- model$evaluation_log)
evlog %>% 
  summarize(ntrees.train = which.min(train_merror_mean),   # find the index of min(train_merror_mean)
            ntrees.test  = which.min(test_merror_mean))   # find the index of min(test_merror_mean)
#from the above lines, optimal ntrees= 1
ntrees <- 1
m_var_xgb <- xgboost(data = as.matrix(trainingframe.treat), # training data as matrix
                          label = profile_data$m_var,  # column of outcomes
                          nrounds = ntrees,# number of trees to build
                          num_class = 8,
                          objective = "multi:softmax", # objective
                          eta = 0.3,
                          depth = 6,
                          verbose = 0  # silent
)
na_profile_data$predicted <- predict(m_var_xgb, as.matrix(testframe.treat)) #predict on the test set and add the predictions column
# to the na_profile_data set
head(na_profile_data) #check the structure of the data set
summary(na_profile_data$predicted)

unique(na_profile_data$predicted) # to see which distinct values were predicted
unique(profile_data$m_var)
library(funModeling)
df_status(na_profile_data)




  
