list.of.packages <- c("randomGLM", "tidyverse", "data.table", "plyr", "dummies", 
                      "caret", "rstudioapi", "installr")
new.packages <- list.of.packages[!(list.of.packages %in% 
                                     installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# library(installr)
# updateR()

library(rstudioapi)
library(randomGLM)
library(tidyverse)
library(data.table)
library(dummies)
library(caret)
library(plyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# attach("./Code/supporting_functions.rda")
source("./Code/supporting_functions_source.R")
# source("./Code/RGLM_package_functions.R")

## may need to fix this and get both test and train to load in separately
set.seed(123)
setwd('C:/Users/Lance/Dropbox/Programming/R working directory/')
data.in <- fread('./Data/fire_contents.csv', na.strings=c(''), stringsAsFactors = T)
emb.score <- as.data.frame(fread('./Data/fire_cso_size_predval.csv', na.strings=c(''), stringsAsFactors = T))
data.in <- as.data.frame(data.in)

## process data
# Clean out extra columns
target.ind <- grep("target", colnames(data.in))
data.in <- data.in[1:target.ind] # With the assumption that target is the last column

# Feature selection
var.drop.list <- c("datatype", "AREA")
data.in <- data.in[, !(names(data.in) %in% var.drop.list)]
# DF[ , !(names(DF) %in% drops)]



var.list <- c('company', 'channel', 'yearsInsured', 'stateRisk', 'ContStockOtherSI', 'suncorp_household_score', 'target')
cat.var.list <- c('company', 'channel', 'stateRisk')

data.in$company <- mapvalues(data.in[["company"]], 
                               from = c("3","6","12","13","17"), 
                               to   = c("VERO","AAMI","Resilium","GIO","GIO"))

data.subset <- data.in[, var.list]
                
data.subset <- data.subset %>% 
  mutate_at(cat.var.list, funs(as.factor(.)))                       

## one-hot encoding of categorical variables
encode.list <- list()
for (var in cat.var.list) {
  var.encoding <- paste(var, ".encoding", sep = "")
  assign(var.encoding, as.data.frame(dummy(var, data = data.subset, sep = "", drop = TRUE, fun = as.integer, verbose = FALSE)))
  indx <- grep(var, cat.var.list)
  encode.list[[indx]] <- eval(parse(text = var.encoding))
}

data.subset <- data.subset[, !(names(data.subset) %in% cat.var.list)]

data.full <- data.frame(data.subset, encode.list)

# company.encoding <- as.data.frame(dummy('company', data = data.subset, sep = "", drop = TRUE, fun = as.integer, verbose = FALSE))
# channel.encoding <- as.data.frame(dummy('channel', data = data.subset, sep = "", drop = TRUE, fun = as.integer, verbose = FALSE))
# stateRisk.encoding <- as.data.frame(dummy('stateRisk', data = data.subset, sep = "", drop = TRUE, fun = as.integer, verbose = FALSE))
# 
# data.full <- data.subset[, -cat.var.list]
# 
# data.full <- data.frame(select(data.subset,-c(company,channel,stateRisk)),company.encoding, channel.encoding, stateRisk.encoding)

endoftrain.indx <- 799
train.prop <- 0.8
target.loc <- grep("target", colnames(data.full))

## split data into train, CV and test 
xTrain <- data.full[1:endoftrain.indx,-target.loc][1:floor(train.prop*endoftrain.indx),]
yTrain <- data.full[1:endoftrain.indx, target.loc][1:floor(train.prop*endoftrain.indx)]

xCV <- data.full[1:endoftrain.indx,-target.loc][-(1:floor(train.prop*endoftrain.indx)),]
yCV <- data.full[1:endoftrain.indx, target.loc][-(1:floor(train.prop*endoftrain.indx))]

xTest <- data.full[-(1:endoftrain.indx),-target.loc]
yTest <- data.full[-(1:endoftrain.indx), target.loc]

## fit model
# simple version
# RGLM <- randomGLM(xTrain,
#                   yTrain,
#                   classify=FALSE, 
#                   nBags =100,
#                   randomSeed = 123,
#                   nFeaturesInBag = 20,
#                   keepModels=TRUE)

## parameter tuning for RGLM
nbag <- seq(25, 150, 25)
model.list <- list()
CV.list <- list()
CV <- data.frame(cbind(xCV, yCV))

for (bag.cnt in nbag) {
  paste("now fitting model with number of bags =", bag.cnt, sep = " ")  
  RGLM.CV <- randomGLM(xTrain,
                    yTrain,
                    classify = F, 
                    nBags = bag.cnt,
                    randomSeed = 123,
                    nFeaturesInBag = 20,
                    nCandidateCovariates = 20,
                    mandatoryCovariates = 2,
                    keepModels=TRUE)
  

  # Store model outputs and gains
  model.list[[toString(bag.cnt)]] <- RGLM.CV  
  CV$model_score_test <- predict(model.list[[toString(bag.cnt)]], xCV)
  CV.list[toString(bag.cnt)] <- CalculateGains(CV, "yCV", "model_score_test")[3]
  
  CV <- subset(CV, select = -model_score_test)
    # Test[, !(names(Test) == "model_score_test")]
}

CV.list
# OUTPUT
# $`25`
# [1] 0.4122131
# 
# $`50`
# [1] 0.4389428
# 
# $`75`
# [1] 0.4479827
# 
# $`100`
# [1] 0.4473153
# 
# $`125`
# [1] 0.4195315
# 
# $`150`
# [1] 0.417666
best.bag.cnt <- 75 # manually adjust this after CV
best.model <- model.list[[toString(best.bag.cnt)]]

## Feature selection
varImp = best.model$timesSelectedByForwardRegression
sum(varImp>0)
varImp
table(varImp)

## select most important features and perform thinning
impF = colnames(xTrain)[varImp>=5]
impF
# TODO: Add in thinning

## Coefficients
coef(best.model$models[[30]])

nBags = length(best.model$featuresInForwardRegression)
coefMat = matrix(0, nBags, best.model$nFeatures)
for (i in 1:nBags) {
  coefMat[i, best.model$featuresInForwardRegression[[i]]] = best.model$coefOfForwardRegression[[i]]
}

coefMean = apply(coefMat, 2, mean)
names(coefMean) = colnames(xTrain)
summary(coefMean)
coefMean[impF]

## Single tree randomGLM with no replacement
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("./Code/RGLM_package_functions.R")
Singl_RGLM <- randomGLM_singl(xTrain,
                            yTrain,
                            classify=F,
                            nBags =1,
                            replace = FALSE,
                            nObsInBag = floor(train.prop*endoftrain.indx),
                            randomSeed = 123,
                            nFeaturesInBag = 28,
                            nCandidateCovariates = 28,
                            verbose = 1,
                            nThreads = 1,
                            keepModels=TRUE)

# Get predictors
Singl_RGLM$interceptOfForwardRegression
Singl_RGLM$coefOfForwardRegression


## Score test data
results.df <- xTest
results.df$model.score <- predict(best.model, xTest, type="response")

best.model.thin <- thinRandomGLM(best.model, 9)
results.df$thin.model.score <- predict(best.model.thin, xTest, type = "response")


#manual score
results.df <- mutate(results.df, singl.score = 39397.65 + 0.103942*ContStockOtherSI + 79827.191805*channelBISA - 3016.404353*yearsInsured
                                     -32020.491317*channelBROKER - 71658.286064*stateRiskACT)
# xTest$singl_score <- predict.randomGLM_mod(Singl_RGLM, xTest, type="response")
Test <- data.frame(cbind(results.df, yTest, emb.score))


## Assess model performance

CalculateGains(Test, "yTest", "model.score") # randomGLM with CVed trees
# Model gains   Max gains   Model/max 
# 0.1316291   0.3651968   0.3604334
CalculateGains(Test, "yTest", "thin.model.score") # best model thinned
# Model gains   Max gains   Model/max 
# 0.1375204   0.3651968   0.3765651 
CalculateGains(Test, "yTest", "singl.score") # randomGLM with one tree
# Model gains   Max gains   Model/max 
# 0.1207904   0.3651968   0.3307543 
CalculateGains(Test, "yTest", "EmblemPred")
# Model gains   Max gains   Model/max 
# 0.1524236   0.3651968   0.4173737

# setwd('C:/Users/Lance/Dropbox/Programming/R working directory/')















### JUNK
# data_subset <- transmute(data_subset, data_subset[[x]] <- ifelse(x %in% cat_var_list, as.factor(data_subset[[x]]))
# if (x %in% cat_var_list) as.factor(datasubset[[x]]))
for (x in cat_var_list) {
  assign(paste('onehot_', gsub('"','',x), sep = '_'), x) <- dummy(x, data = data_subset, sep = "", drop = TRUE, fun = as.integer, verbose = FALSE)
}



num_format <- function(df, in_list) {
  for (i in in_list){
    df[i] <- as.factor(df[[i]])
  }
}

num_format(data_subset, cat_var_list)


data_frame["company"] <- as.factor(data_frame$"company")



if (!is.numeric(df[i])){
  df[i] <- as.numeric(as.factor(df$i))
}





DF_list = list(select(data_subset,-c(company,channel,stateRisk)), company_encoding, channel_encoding, stateRisk_encoding)
temp = Reduce(function(...) merge(..., all=T), DF_list)
temp <- select(temp, -c(Row.names, Row.names.x, Row.names.y))





sample <- sample(seq_len(nrow(data_subset)), size = floor(.75*nrow(data_subset)))

train_set <- data_subset[sample, ]
test_set <- data_subset[-sample, ]




#temp <- merge(merge(select(data_subset,c(company,channel)), company_encoding, by=0), 
#              merge(channel_encoding, stateRisk_encoding, by=0), by=0)

















