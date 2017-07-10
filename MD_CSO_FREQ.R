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
source("./Utility Code/supporting_functions_source.R")
set.seed(123)

### Set parameters
model.choice <- 'MD_CSO_FREQ'
var.list <- c('company', 'channel', 'yearsInsured', 'stateRisk', 'ContStockOtherSI', 
              'suncorp_household_score', 'policyPremiscnt', 'occ_cso_maldam1', 'locality', 
              'building_maldamage_freq1', 'target')
cat.var.list <- c('company', 'channel', 'stateRisk', 'occ_cso_maldam1', 'locality', 'building_maldamage_freq1')
train.prop <- 0.8


## load in data
setwd('C:/Users/Lance/Dropbox/Programming/R working directory/')
data.dir <- paste('./Data/', model.choice, '/', sep = '')
data.in.m <- read.csv(paste(data.dir, 'data_m.csv', sep = ''), na.strings=c(''), stringsAsFactors = T)
data.in.h <- read.csv(paste(data.dir, 'data_h.csv', sep = ''), na.strings=c(''), stringsAsFactors = T)
emb.score <- read.csv(paste(data.dir, model.choice, '_predval.csv', sep = ''), na.strings=c(''), stringsAsFactors = T)
endoftrain.indx <- nrow(data.in.m)
data.in <- rbind(data.in.m, data.in.h)
rm(data.in.m, data.in.h)
## process data
# Clean out extra columns
# target.ind <- grep("target", colnames(data.in))
# data.in <- data.in[1:target.ind] # With the assumption that target is the last column

# Feature cleaning
# var.drop.list <- c("datatype", "AREA")
# data.in <- data.in[, !(names(data.in) %in% var.drop.list)]

data.in$company <- as.factor(mapvalues(data.in[["company"]], 
                               from = c("3","6","12","13","17"), 
                               to   = c("VERO","AAMI","Resilium","GIO","GIO")))
data.in <- as.data.table(data.in)
data.subset <- as.data.frame(data.in[,..var.list])

rm(data.in)

# ensure all categorical variables are factors                
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

# Remove initial categorical variables and replace with one-hot encoded variables
data.subset <- data.subset[, !(names(data.subset) %in% cat.var.list)]
data.full <- data.frame(data.subset, encode.list)
rm(encode.list)

for (var in cat.var.list) {
  var.encoding <- paste(var, ".encoding", sep = "")
  rm(list = c(var.encoding))
  print(paste('removing', var.encoding, sep = " "))
}

## split data into train, CV and test 
target.indx <- grep("target", colnames(data.full))
xTrain <- data.full[1:endoftrain.indx,-target.indx][1:floor(train.prop*endoftrain.indx),]
yTrain <- data.full[1:endoftrain.indx, target.indx][1:floor(train.prop*endoftrain.indx)]

xCV <- data.full[1:endoftrain.indx,-target.indx][-(1:floor(train.prop*endoftrain.indx)),]
yCV <- data.full[1:endoftrain.indx, target.indx][-(1:floor(train.prop*endoftrain.indx))]

xTest <- data.full[-(1:endoftrain.indx),-target.indx]
yTest <- data.full[-(1:endoftrain.indx), target.indx]

rm(data.full, data.subset)

nbag <- seq(25, 150, 25)
model.list <- list()
CV.list <- list()
CV <- data.frame(cbind(xCV, yCV))

# check memory usage before fitting multiple models
sort( sapply(ls(),function(x){object.size(get(x))})) 

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
# Thinning
nbag <- seq(25, 150, 25)
model.list <- list()
CV.list <- list()
best.model.thin <- thinRandomGLM(best.model, 8)


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
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("C:/Users/Lance/Documents/GitHub/GLM_bag/Utility Code/RGLM_package_functions.R")
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
results.df$thin.model.score <- predict(best.model.thin, xTest, type = "response")


#manual score
results.df <- mutate(results.df, singl.score = 39397.65 + 0.103942*ContStockOtherSI + 79827.191805*channelBISA - 3016.404353*yearsInsured
                                     -32020.491317*channelBROKER - 71658.286064*stateRiskACT)
# xTest$singl_score <- predict.randomGLM_mod(Singl_RGLM, xTest, type="response")
Test <- data.frame(cbind(results.df, yTest, emb.score))


## Assess model performance

RGLM.CV <- CalculateGains(Test, "yTest", "model.score", plot = T) # randomGLM with CVed trees
RGLM.CV <- as.data.frame(RGLM.CV)
# Model gains   Max gains   Model/max 
# 0.1316291   0.3651968   0.3604334
RGLM.CV.thin <- CalculateGains(Test, "yTest", "thin.model.score") # best model thinned
RGLM.CV.thin <- as.data.frame(RGLM.CV.thin)
# Model gains   Max gains   Model/max 
# 0.1375204   0.3651968   0.3765651 
Singl.GLM <- CalculateGains(Test, "yTest", "singl.score") # randomGLM with one tree
Singl.GLM <- as.data.frame(Singl.GLM)
# Model gains   Max gains   Model/max 
# 0.1207904   0.3651968   0.3307543 
Manual.GLM <- CalculateGains(Test, "yTest", "EmblemPred")
Manual.GLM <- as.data.frame(Manual.GLM)
# Model gains   Max gains   Model/max 
# 0.1524236   0.3651968   0.4173737
all.gains <- cbind(RGLM.CV,
RGLM.CV.thin,
Singl.GLM,
Manual.GLM)

all.gains



























