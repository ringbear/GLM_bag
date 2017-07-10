randomGLM_singl <- function(x, 
                      y, 
                      xtest = NULL, 
                      maxInteractionOrder = 1, 
                      classify = is.factor(y) | length(unique(y)) < 4, 
                      multiClass.global = TRUE, 
                      multiClass.pairwise = FALSE, 
                      multiClass.minObs = 1, 
                      multiClass.ignoreLevels = NULL, 
                      nBags = 100, 
                      replace = TRUE, 
                      sampleWeight = NULL, 
                      nObsInBag = nrow(x), # if (replace) nrow(x) else as.integer(0.632 * nrow(x))
                      nFeaturesInBag = ceiling(ifelse(ncol(x) <= 10, ncol(x), ifelse(ncol(x) <= 300, (1.0276 - 0.00276 *ncol(x)) * ncol(x), ncol(x)/5))), 
                      minInBagObs = min(max(nrow(x)/2, 5), 2 * nrow(x)/3), 
                      nCandidateCovariates = 50, 
                      corFncForCandidateCovariates = cor, 
                      corOptionsForCandidateCovariates = list(method = "pearson",  use = "p"), 
                      mandatoryCovariates = NULL, 
                      interactionsMandatory = FALSE, 
                      keepModels = is.null(xtest), 
                      thresholdClassProb = 0.5, 
                      interactionSeparatorForCoefNames = ".times.", 
                      randomSeed = 12345, 
                      nThreads = NULL, 
                      family = fam.var,
                      link = link.var,
                      verbose = 0) {
  ySaved = y
  xSaved = x
  if (classify) {
    originalYLevels = sort(unique(y))
    if (length(originalYLevels) > 2) {
      if (is.na(multiClass.minObs)) 
        multiClass.minObs = 0
      if (multiClass.minObs < 1) {
        .cat.nl("Warning: invalid input of 'multiClass.nimObs' changed to 1.")
        multiClass.minObs = 1
      }
      if (length(originalYLevels) > length(y)/multiClass.minObs | 
          length(originalYLevels) == length(y)) {
        stop("The response 'y' has too many levels for classification.\n", 
             "   Perhaps you should set 'classify = FALSE'?")
      }
      else {
        .cat.nl("randomGLM: transforming multi-level response to a series of binary variables.")
      }
      yBin = .binarizeCategoricalVar(as.character(y), 
                                     minCount = multiClass.minObs, val1 = 0, val2 = 1, 
                                     nameSep = ".vs.", namePrefix = "", ignore = multiClass.ignoreLevels, 
                                     includePairwise = multiClass.pairwise, includeLevelVsAll = multiClass.global, 
                                     levelOrder = NULL)
      nY = ncol(yBin)
      yBinNames = colnames(yBin)
      yBinLevels = attr(yBin, "includedLevels")
      out = list(binaryPredictors = list())
      for (iy in 1:nY) {
        if (verbose > 0) 
          .cat.nl("..Working on binary variable ", yBinNames[iy], 
                  " (", iy, " of ", nY, ")")
        out$binaryPredictors[[iy]] = randomGLM_mod(x = x, 
                                               y = yBin[, iy], xtest = xtest, maxInteractionOrder = maxInteractionOrder, 
                                               classify = classify, nBags = nBags, replace = FALSE, 
                                               sampleWeight = sampleWeight, nObsInBag = nObsInBag, 
                                               nFeaturesInBag = nFeaturesInBag, nCandidateCovariates = nCandidateCovariates, 
                                               corFncForCandidateCovariates = corFncForCandidateCovariates, 
                                               corOptionsForCandidateCovariates = corOptionsForCandidateCovariates, 
                                               mandatoryCovariates = mandatoryCovariates, 
                                               interactionsMandatory = interactionsMandatory, 
                                               keepModels = keepModels, thresholdClassProb = thresholdClassProb, 
                                               interactionSeparatorForCoefNames = interactionSeparatorForCoefNames, 
                                               randomSeed = randomSeed, nThreads = nThreads, 
                                               verbose = verbose - 1)
      }
      names(out$binaryPredictors) = yBinNames
      out$predictedOOB = as.matrix(sapply(out$binaryPredictors, 
                                          getElement, "predictedOOB"))
      colnames(out$predictedOOB) = .spaste("PredictionFor.", 
                                           yBinNames)
      out$predictedOOB.response = do.call(cbind, lapply(out$binaryPredictors, 
                                                        getElement, "predictedOOB.response"))
      responseNames = .spaste(rep(yBinNames, rep(2, nY)), 
                              ".ProbabilityOfClass.", as.vector(yBinLevels))
      colnames(out$predictedOOB.response) = responseNames
      rownames(out$predictedOOB.response) = rownames(out$predictedOOB) = rownames(x)
      if (!is.null(xtest)) {
        out$predictedTest = sapply(out$binaryPredictors, 
                                   getElement, "predictedTest")
        colnames(out$predictedTest) = yBinNames
        out$predictedTest.response = do.call(cbind, 
                                             lapply(out$binaryPredictors, getElement, "predictedTest.response"))
        colnames(out$predictedOOB.response) = responseNames
        rownames(out$predictedTest) = rownames(out$predictedTest.response) = rownames(xtest)
      }
      out$levelMatrix = yBinLevels
      out$thresholdClassProb = thresholdClassProb
      class(out) = c("randomGLM_mod", class(out))
      return(out)
    }
    y = as.numeric(as.factor(y)) - 1
    numYLevels = sort(unique(y))
    minY = min(y, na.rm = TRUE)
    maxY = max(y, na.rm = TRUE)
  }
  else {
    if (!is.numeric(y)) 
      stop("For quantitative prediction, the response must be numeric.")
  }
  x = as.matrix(x)
  featureNames.original = colnames(x)
  nVars = ncol(x)
  if (is.null(featureNames.original)) {
    featureNames.original = featureNames = .spaste("F", 
                                                   .prependZeros(c(1:nVars)))
    colnames(x) = featureNames.original
    namesChanged = FALSE
  }
  else {
    featureNames = make.names(featureNames.original, unique = TRUE)
    if (isTRUE(all.equal(featureNames, featureNames.original))) {
      namesChanged = FALSE
    }
    else {
      namesChanged = TRUE
      nameTranslationTable = data.frame(Column = c(1:nVars), 
                                        OriginalName = featureNames.original, CoefficientName = featureNames)
      colnames(x) = featureNames
    }
  }
  if (length(y) != nrow(x)) {
    stop("x and y must have the same number of observations.")
  }
  nSamples = length(y)
  if (nSamples < 8) {
    .cat.nl("*****************************************************************\n", 
            "* Warning in randomGLM: there are 7 or fewer observations.\n", 
            "*   This may be too few to perform meaningful model selection\n", 
            "*   on in-bag (i.e., even fewer) samples.\n", "*   Model selection algorithm will likely output additional warnings.\n", 
            "*   The resulting predictor should be used with caution.\n", 
            "*****************************************************************")
  }
  nonMandatCovars = setdiff(c(1:nVars), mandatoryCovariates)
  nMandatoryCovariates = length(mandatoryCovariates)
  if (nVars < nFeaturesInBag) {
    nFeaturesInBag = ncol(x)
    .cat.nl("Warning in randomGLM: nFeaturesInBag is larger than number of features (ncol(x)).\n", 
            "   Will use nFeaturesInBag equal to the number of features.")
  }
  if (nCandidateCovariates > nFeaturesInBag) {
    .cat.nl("Warning in randomGLM: nCandidateCovariates is larger than nFeaturesInBag.\n", 
            "  Will use nCandidateCovariates=nFeaturesInBag")
    nCandidateCovariates = nFeaturesInBag
  }
  mandatoryCovarsGiven = !is.null(mandatoryCovariates)
  if (mandatoryCovarsGiven & nCandidateCovariates < length(mandatoryCovariates)) {
    stop("Error: number of mandatoryCovariates >= nCandidateCovariates")
  }
  if (thresholdClassProb < 0 | thresholdClassProb > 1) 
    stop("Error: thresholdClassProb takes values between 0  and 1.")
  xSD = apply(x, 2, sd, na.rm = TRUE)
  validFeatures = xSD > 0
  x[, !validFeatures] = 0
  doTest = !is.null(xtest)
  if (doTest) {
    xtestSaved = xtest
    xtest = as.matrix(xtest)
    if (ncol(x) != ncol(xtest)) 
      stop("Number of learning and testing predictors (columns of x, xtest) must equal.")
    if (!is.null(colnames(xtest))) {
      if (!isTRUE(all.equal(colnames(xtest), featureNames.original))) 
        stop("Column names of 'x' and 'xtest' disagree.")
    }
    colnames(xtest) = colnames(x)
    nTestSamples = nrow(xtest)
    xtest[, !validFeatures] = 0
    predictedTestMat = matrix(NA, nTestSamples, nBags)
  }
  if (!is.null(randomSeed)) {
    if (exists(".Random.seed")) {
      saved.seed = .Random.seed
      seedSaved = TRUE
    }
    else seedSaved = FALSE
    set.seed(randomSeed)
  }
  on.exit(.disableThreads())
  nThreads = .enableThreads(nThreads, verbose)
  combinePredictors = function(...) {
    preds = list(...)
    out = list()
    out$predictedMat = sapply(preds, getElement, "predicted")
    if (doTest) 
      out$predictedTestMat = sapply(preds, getElement, 
                                    "predictedTest")
    out$candidateFeatures = lapply(preds, getElement, "candidateFeatures")
    out$featuresInForwardRegression = lapply(preds, getElement, 
                                             "featuresInForwardRegression")
    out$coefOfForwardRegression = lapply(preds, getElement, 
                                         "coefOfForwardRegression")
    out$interceptOfForwardRegression = sapply(preds, getElement, 
                                              "interceptOfForwardRegression")
    out$models = lapply(preds, getElement, "model")
    out
  }
  bagFeatures = matrix(NA, nFeaturesInBag, nBags)
  bagObsIndx = matrix(NA, nObsInBag, nBags)
  for (bag in 1:nBags) {
    yBagVar = 0
    while (yBagVar == 0) {
      bagSamples = seq(1:nObsInBag) # sample(nSamples, nObsInBag, replace = replace, prob = sampleWeight)
      yBag = y[bagSamples]
      yBagVar = var(yBag, na.rm = TRUE)
      nUniqueInBag = length(unique(bagSamples))
      if (nUniqueInBag == nSamples & nUniqueInBag < minInBagObs) 
        yBagVar = 0
    }
    bagObsIndx[, bag] = bagSamples
    bagFeatures[, bag] = c(mandatoryCovariates, sample((1:nVars)[nonMandatCovars], 
                                                       nFeaturesInBag - nMandatoryCovariates))
  }
  singleBagIteration = function(bag, verbose) {
    if (verbose > 0) {
      .cat.nl("..bag ", bag)
    }
    out = list()
    bagSamples = bagObsIndx[, bag]
    oob = c(1:nSamples)
    nOOB = length(oob)
    features = bagFeatures[, bag]
    xBag = x[bagSamples, features, drop = FALSE]
    yBag = y[bagSamples]
    if (doTest) {
      xTestBag = rbind(x[oob, features, drop = FALSE], 
                       xtest[, features, drop = FALSE])
    }
    else {
      xTestBag = x[oob, features, drop = FALSE]
    }
    pr = .forwardSelection(xBag, yBag, xTestBag, classify = classify, 
                           maxInteractionOrder = maxInteractionOrder, nCandidateCovariates = nCandidateCovariates, 
                           corFncForCandidateCovariates = corFncForCandidateCovariates, 
                           corOptionsForCandidateCovariates = corOptionsForCandidateCovariates, 
                           NmandatoryCovariates = nMandatoryCovariates, interactionsMandatory = interactionsMandatory, 
                           keepModel = TRUE, interactionSeparatorForCoefNames = interactionSeparatorForCoefNames)
    out$predicted = rep(NA, nSamples)
    out$predicted[oob] = pr$predicted[1:nOOB]
    if (doTest) {
      out$predictedTest = pr$predicted[(nOOB + 1):(nOOB + 
                                                     nTestSamples)]
    }
    dictionary = rbind(c(0, 0), cbind(1:nFeaturesInBag, 
                                      features))
    out$candidateFeatures = .translate(pr$candidateFeatures, 
                                       dictionary)
    out$featuresInForwardRegression = .translate(pr$featuresInForwardRegression, 
                                                 dictionary)
    out$coefOfForwardRegression = pr$coefOfForwardRegression
    out$interceptOfForwardRegression = pr$interceptOfForwardRegression
    if (keepModels) 
      out$model = pr$model
    rm(xBag, pr, xTestBag, features, bagSamples, oob, yBag)
    out
  }
  if (nThreads > 1) {
    ensemble = foreach(bag = 1:nBags, .combine = combinePredictors, 
                       .multicombine = TRUE, .maxcombine = nBags) %dopar% 
      singleBagIteration(bag, verbose = verbose)
  }
  else {
    bagRes = list()
    for (bag in 1:nBags) {
      bagRes[[bag]] = singleBagIteration(bag, verbose = verbose)
    }
    ensemble = do.call(combinePredictors, bagRes)
  }
  featuresInForwardRegression.all = do.call(cbind, ensemble$featuresInForwardRegression)
  timesSelectedByForwardRegression = .countsInInteractionMatrix(featuresInForwardRegression.all, 
                                                                nVars)
  colnames(bagObsIndx) = names(ensemble$candidateFeatures) = names(ensemble$featuresInForwardRegression) = names(ensemble$coefOfForwardRegression) = names(ensemble$interceptOfForwardRegression) = .spaste("Bag", 
                                                                                                                                                                                                            1:nBags)
  if (!is.null(colnames(xSaved))) {
    colnames(timesSelectedByForwardRegression) = colnames(xSaved)
  }
  predictedOOB.response1 = rowMeans(ensemble$predictedMat, 
                                    na.rm = TRUE)
  predictedOOB.response1[rowSums(!is.na(ensemble$predictedMat)) == 
                           0] = NA
  if (!is.null(rownames(xSaved))) {
    names(predictedOOB.response1) = rownames(xSaved)
  }
  out = list(predictedOOB.response = predictedOOB.response1, 
             predictedOOB = predictedOOB.response1, candidateFeatures = ensemble$candidateFeatures, 
             featuresInForwardRegression = ensemble$featuresInForwardRegression, 
             coefOfForwardRegression = ensemble$coefOfForwardRegression, 
             interceptOfForwardRegression = ensemble$interceptOfForwardRegression, 
             bagObsIndx = bagObsIndx, timesSelectedByForwardRegression = timesSelectedByForwardRegression, 
             models = if (keepModels) ensemble$models else NULL, 
             featureNamesChanged = namesChanged, nameTranslationTable = if (namesChanged) nameTranslationTable else NULL, 
             classify = classify, nFeatures = nVars, maxInteractionOrder = maxInteractionOrder, 
             yLevels = if (classify) originalYLevels else NULL, x.original = xSaved, 
             y.original = ySaved, x = x, y = y, thresholdClassProb = thresholdClassProb)
  if (doTest) {
    predictedTest.response1 = rowMeans(ensemble$predictedTestMat, 
                                       na.rm = TRUE)
    predictedTest.response1[rowSums(!is.na(ensemble$predictedTestMat)) == 
                              0] = NA
    if (!is.null(rownames(xtestSaved))) {
      names(predictedTest.response1) = rownames(xtestSaved)
    }
    out$predictedTest.response = predictedTest.response1
    out$predictedTest = predictedTest.response1
  }
  if (classify) {
    predictedOOB = ifelse(predictedOOB.response1 > thresholdClassProb, 
                          1, 0)
    predictedOOB = originalYLevels[predictedOOB + 1]
    predictedOOB.response = cbind(1 - predictedOOB.response1, 
                                  predictedOOB.response1)
    colnames(predictedOOB.response) = as.character(originalYLevels)
    out$predictedOOB = predictedOOB
    out$predictedOOB.response = predictedOOB.response
    if (doTest) {
      predictedTest = ifelse(predictedTest.response1 > 
                               thresholdClassProb, 1, 0)
      predictedTest = originalYLevels[predictedTest + 
                                        1]
      predictedTest.response = cbind(1 - predictedTest.response1, 
                                     predictedTest.response1)
      colnames(predictedTest.response) = as.character(originalYLevels)
      out$predictedTest.response = predictedTest.response
      out$predictedTest = predictedTest
    }
  }
  class(out) = c("randomGLM_mod", class(out))
  out
}


######### forwardSelection #########
.forwardSelection <- function (xBag, yBag, xTestBag, classify, maxInteractionOrder, 
          nCandidateCovariates, corFncForCandidateCovariates, corOptionsForCandidateCovariates, 
          NmandatoryCovariates, interactionsMandatory, keepModel, 
          interactionSeparatorForCoefNames) 
{
  indxMiss = apply(is.na(xBag), 2, sum) > 0
  indxVar = apply(xBag, 2, var, na.rm = T) == 0
  removeFeatures = indxMiss | indxVar
  xBag = xBag[, !removeFeatures, drop = F]
  xTestBag = xTestBag[, !removeFeatures, drop = F]
  nFeatures = ncol(xBag)
  if (NmandatoryCovariates > 0) {
    mandatCovars = c(1:NmandatoryCovariates)
    mandatCovars = mandatCovars[!removeFeatures[mandatCovars]]
    NmandatoryCovariates = length(mandatCovars)
  }
  else mandatCovars = numeric(0)
  nonMandatCovars = setdiff(c(1:nFeatures), mandatCovars)
  interactionMatrix = .interactionMatrix(nFeatures, maxInteractionOrder, 
                                         originalNames = colnames(xBag), setColNames = TRUE, 
                                         featureSeparator = interactionSeparatorForCoefNames)
  if (interactionsMandatory) {
    mandatoryInteractions = apply(interactionMatrix, 2, 
                                  function(x) {
                                    any(x %in% mandatCovars)
                                  })
  }
  else mandatoryInteractions = mandatCovars
  nMandatoryInteractions = length(mandatoryInteractions)
  if (nMandatoryInteractions > nCandidateCovariates) 
    stop("Number of mandatory interactions is larger than number of candidate covariates.")
  nInteractions = ncol(interactionMatrix)
  nonMandatInteractions = setdiff(c(1:nInteractions), mandatoryInteractions)
  x.int = .generateInteractions(xBag, maxInteractionOrder, 
                                interactionMatrix = interactionMatrix, setColNames = TRUE)
  xTest.int = .generateInteractions(xTestBag, maxInteractionOrder, 
                                    interactionMatrix = interactionMatrix, setColNames = TRUE)
  corOptionsForCandidateCovariates$x = x.int
  corOptionsForCandidateCovariates$y = yBag
  absGS = abs(do.call(corFncForCandidateCovariates, corOptionsForCandidateCovariates))
  nCandidateCovariates = min(nCandidateCovariates, ncol(x.int))
  rank = rank(-absGS[nonMandatInteractions], ties.method = "f")
  indx = c(mandatoryInteractions, nonMandatInteractions[rank <= 
                                                          (nCandidateCovariates - nMandatoryInteractions)])
  x.int = x.int[, indx, drop = F]
  xTest.int = xTest.int[, indx, drop = F]
  absGS = absGS[indx]
  candidateFeatures = interactionMatrix[, indx, drop = FALSE]
  featureMax = which.max(absGS)
  modelData = data.frame(x.int)
  rm(xBag, xTestBag, x.int, corOptionsForCandidateCovariates, 
     interactionMatrix)
  initialFormula = paste("yBag~", paste(colnames(modelData)[if (nMandatoryInteractions > 
                                                                0) 
    mandatoryInteractions
    else featureMax], collapse = " + "))
  if (classify) {
    lmInit = glm(formula(initialFormula), data = modelData, 
                 family = binomial(link = "logit"))
    lmUpper = glm(yBag ~ ., data = modelData, family = binomial(link = "logit"))
  }
  else {
    lmInit = lm(formula(initialFormula), data = modelData)
    lmUpper = lm(yBag ~ ., data = modelData)
  }
  model = stepAIC(lmInit, scope = list(upper = lmUpper), direction = "forward", 
                  trace = FALSE)
  sum = summary(model)
  selected = rownames(sum$coef)[-1]
  featuresInForwardRegression = candidateFeatures[, match(selected, 
                                                          colnames(candidateFeatures)), drop = FALSE]
  coefOfForwardRegression = sum$coef[-1, 1]
  interceptOfForwardRegression = sum$coef[1, 1]
  outHat = stats::predict(model, newdata = as.data.frame(xTest.int), 
                   type = "response")
  out = list(predicted = outHat, candidateFeatures = candidateFeatures, 
             featuresInForwardRegression = featuresInForwardRegression, 
             coefOfForwardRegression = coefOfForwardRegression, interceptOfForwardRegression = interceptOfForwardRegression, 
             model = if (keepModel) model else NULL)
  varList = ls(all.names = TRUE)
  rm(list = setdiff(varList, c("out")))
  out
}


predict.randomGLM_mod <- function (object, 
                     newdata, 
                     type = c("response", "class"), 
                     thresholdClassProb = object$thresholdClassProb, 
                     ...) 
{
  type = match.arg(type)
  if (!is.null(object$binaryPredictors)) {
    predictions = do.call(cbind, lapply(object$binaryPredictors, 
                                        predict.randomGLM_mod, newdata = newdata, type = type, 
                                        thresholdClassProb = thresholdClassProb, ...))
    if (type == "response") {
      colnames(predictions) = colnames(object$predictedOOB.response)
    }
    else colnames(predictions) = colnames(object$predictedOOB)
    return(predictions)
  }
  if (is.null(object$models)) 
    stop("The 'object' object must contain the undelying models for prediction.\n", 
         "   Please re-run the randomGLM function with argument 'keepModels = TRUE' and try again.")
  if (missing(newdata)) {
    stop("valid 'newdata' must be given.")
  }
  if (ncol(newdata) != object$nFeatures) 
    stop("Number of columns in 'newdata' differs from the number of features\n", 
         "     in the original training data.")
  if (type == "class" & !object$classify) 
    stop("type='class' is only valid in classification.")
  if (thresholdClassProb < 0 | thresholdClassProb > 1) 
    stop("Error: thresholdClassProb takes values between 0  and 1.")
  .predict.internal(object = object, newdata = newdata, type = type, 
                    thresholdClassProb = thresholdClassProb, returnBothTypes = FALSE)
}

.enableThreads <- function (nThreads, verbose) 
{
  .disableThreads()
  if (is.null(nThreads)) 
    nThreads = max(ceiling(detectCores() * 3/4), detectCores() - 
                     1)
  if (is.na(nThreads)) 
    nThreads = 1
  if (nThreads < 1) {
    warning("In function randomGLM: 'nThreads' is below 1. Will use serial execution.")
    nThreads = 1
  }
  if (nThreads > 1) {
    if (verbose > 1) 
      .cat.nl("Will use parallel calculation with ", nThreads, 
              " workers.")
    if (.Platform$OS.type == "windows") {
      cluster = makePSOCKcluster(nThreads, outfile = "")
      assign(".randomGLMparallelCluster", cluster, pos = ".GlobalEnv")
      clusterExport(cluster, varlist = ls(envir = parent.env(environment()), 
                                          all.names = TRUE), envir = parent.env(environment()))
      clusterCall(cluster, library, package = "MASS", 
                  character.only = TRUE)
      clusterCall(cluster, library, package = "gtools", 
                  character.only = TRUE)
      registerDoParallel(cluster)
    }
    else {
      registerDoParallel(nThreads)
    }
  }
  else {
    if (verbose > 1) 
      .cat.nl("Will use serial calculation with a single worker process.")
    registerDoSEQ()
  }
  nThreads
}


.disableThreads <- function () 
{
  try(stopCluster(get(".randomGLMparallelCluster", pos = ".GlobalEnv")), 
      silent = TRUE)
}

## interactionMatrix
.interactionMatrix <- function (n, maxOrder, setColNames = TRUE, originalNames = c(1:n), 
          featureSeparator = ".") 
{
  out = NULL
  for (o in 1:maxOrder) {
    combs = .combinations(n, o)
    nameMatrix = array(originalNames[combs], dim = dim(combs))
    colnames = apply(nameMatrix, 2, paste, collapse = featureSeparator)
    if (o < maxOrder) 
      combs = rbind(combs, matrix(0, maxOrder - o, ncol(combs)))
    if (setColNames) 
      colnames(combs) = colnames
    out = cbind(out, combs)
  }
  rownames(out) = .spaste("Feature.", c(1:maxOrder))
  out
}

.combinations <- function (n, order) 
{
  if (order == 1) 
    return(matrix(c(1:n), 1, n))
  nOut = choose(n + order - 1, order)
  out = matrix(0, order, nOut)
  sub = .combinations(n, order - 1)
  index = 1
  for (i in 1:n) {
    n1 = ncol(sub)
    out[, index:(index + n1 - 1)] = rbind(rep(i, ncol(sub)), 
                                          sub)
    index = index + n1
    sub = sub[, colSums(sub == i) == 0, drop = FALSE]
  }
  out
}

.spaste <- function (...) 
{
  paste(..., sep = "")
}

.generateInteractions <- function (x, maxOrder, interactionMatrix = NULL, x1 = NULL, 
                                   setColNames = FALSE, originalNames = c(1:ncol(x))) 
{
  n = ncol(x)
  if (is.null(interactionMatrix)) 
    interactionMatrix = .interactionMatrix(n, maxOrder, 
                                           setColNames = setColNames, originalNames = originalNames)
  if (nrow(interactionMatrix) != maxOrder) 
    stop("Internal error: nrow(interactionMatrix)!=maxOrder.")
  if (maxOrder == 1) {
    x.int = x[, as.vector(interactionMatrix), drop = FALSE]
  }
  else {
    if (is.null(x1)) 
      x1 = cbind(x, rep(1, nrow(x)))
    interactionMatrix[interactionMatrix == 0] = n + 1
    if (ncol(interactionMatrix) == 1) {
      x.int = as.matrix(apply(x1, 1, .generateInteractions.1row, 
                              interactionMatrix))
    }
    else x.int = t(apply(x1, 1, .generateInteractions.1row, 
                         interactionMatrix))
  }
  if (setColNames) 
    colnames(x.int) = colnames(interactionMatrix)
  x.int
}


.cat.nl <- function (...) 
{
  cat(.spaste(..., "\n"))
}

.translate <- function (data, dictionary) 
{
  translated = dictionary[match(data, dictionary[, 1]), 2]
  attributes(translated) = attributes(data)
  translated
}

.countsInInteractionMatrix <- function (im, nFeatures) 
{
  maxLevel = nrow(im)
  counts = matrix(0, maxLevel, nFeatures)
  level = maxLevel - colSums(im == 0)
  for (l in 1:maxLevel) {
    mat1 = im[1:l, level == l, drop = FALSE]
    mat1.unique = sapply(as.data.frame(mat1), unique)
    counts1 = table(unlist(mat1.unique))
    where = as.numeric(names(counts1))
    counts[l, where] = as.numeric(counts1)
  }
  rownames(counts) = .spaste("Level.", c(1:maxLevel))
  counts
}

.predict.internal <- function (object, newdata, type, thresholdClassProb, returnBothTypes = FALSE) 
{
  if (!is.null(newdata)) {
    nSamples = nrow(newdata)
    newdata.1 = cbind(newdata, rep(1, nSamples))
    colnames(newdata) = make.names(colnames(newdata), unique = TRUE)
  }
  else {
    nSamples = length(object$y)
    x.1 = cbind(object$x, rep(1, nSamples))
  }
  nBags = length(object$models)
  predictedMat = matrix(NA, nSamples, nBags)
  for (b in 1:nBags) if (inherits(object$models[[b]], "lm")) {
    bagIM = object$featuresInForwardRegression[[b]]
    if (!is.null(newdata)) {
      bagNewData = .generateInteractions(x = newdata, x1 = newdata.1, 
                                         interactionMatrix = bagIM, maxOrder = object$maxInteractionOrder, 
                                         setColNames = TRUE)
      predictedMat[, b] = predict(object$models[[b]], newdata = as.data.frame(bagNewData), 
                                  type = "response")
    }
    else {
      oob = c(1:nSamples)[-unique(object$bagObsIndx[, b])]
      bagNewData = .generateInteractions(x = object$x[oob, 
                                                      ], x1 = x.1[oob, ], interactionMatrix = bagIM, 
                                         maxOrder = object$maxInteractionOrder, setColNames = TRUE)
      predictedMat[oob, b] = predict(object$models[[b]], 
                                     newdata = as.data.frame(bagNewData), type = "response")
    }
  }
  predicted.response = rowMeans(predictedMat, na.rm = TRUE)
  predicted.response[rowSums(!is.na(predictedMat)) == 0] = NA
  names(predicted.response) = if (is.null(newdata)) {
    if (is.null(names(object$y.original))) 
      rownames(object$x.original)
    else names(object$y.original)
  }
  else rownames(newdata)
  if (type == "response" | returnBothTypes) {
    if (object$classify) {
      out.response = cbind(1 - predicted.response, predicted.response)
      colnames(out.response) = as.character(object$yLevels)
    }
    else {
      out.response = predicted.response
    }
  }
  if (type == "class" | returnBothTypes) {
    if (object$classify) {
      predicted.round = ifelse(predicted.response > thresholdClassProb, 
                               1, 0)
      out.class = object$yLevels[predicted.round + 1]
    }
    else out.class = predicted.response
  }
  if (returnBothTypes) {
    return(list(response = out.response, class = out.class))
  }
  else if (type == "class") 
    return(out.class)
  out.response
}




