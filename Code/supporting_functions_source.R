# trace(AvsE, edit = T)
# trace(AvsEPred, edit = T)
# trace(CalculateGains, edit = T)
# trace(PartialPlot, edit = T)
# trace(score.model, edit = T)
# trace(sourcePartial, edit = T)
# trace(squid_plot, edit = T)
# trace(view_tree, edit = T)

### Code for common utility functions

AvsE <- function(indata, 
                 target_name, 
                 pred_name, 
                 predictor_name, 
                 cts_or_cat = "cts", 
                 graph_groups = 20, 
                 exposure_weights = "", 
                 table_output = F, 
                 frequency_variable = "", 
                 where_condition = NULL) {
  
  # Grab the original margin parameters
  cur.mar <- par()$mar
  
  # Cast indata as a data table
  indata <- as.data.table(indata)
  
  # Check cts variable is actually cts, otherwise force to cat
  if(!class(indata[[predictor_name]]) %in% c("numeric", "integer")){
    cts_or_cat = "cat"
  }
  
  # Where condition (have to do this before changing column names)
  if (!is.null(where_condition)) {
    indata <- indata[eval(parse(text = where_condition)), ]
  }
  # Change column names of prediction, target and predictor variables. This makes column referencing less verbose.
  setnames(indata, c(pred_name, target_name, predictor_name), c("pred", "target", "predictor"))
  
  # Remove missings from the target and predicted values (I THINK WE ONLY NEED TO DO THIS IF NOT BINNING)
  indata <- indata[!is.na(target) & !is.na(pred), ]
  
  # Determine weights
  if (exposure_weights %in% colnames(indata)) {
    eweights <- indata[[exposure_weights]]
  } else {
    eweights <- 1
  }
  
  if (frequency_variable %in% colnames(indata)) {
    fvar <- indata[[frequency_variable]]
  } else {
    fvar <- 1
  }
  indata[, weights := eweights * fvar]
  
  # Categorical Predictor
  if (tolower(cts_or_cat) == "cat") {
    
    if(class(indata$predictor) != "factor") {
      indata[["predictor"]] <- as.factor(indata[["predictor"]])
    }
    
    # Calculate plot data
    plot.data <- indata[, .(target = sum(target*weights, na.rm = T)/sum(weights, na.rm = T),
                            pred = sum(pred*weights, na.rm = T)/sum(weights, na.rm = T), 
                            exposure = sum(weights, na.rm = T)), by = predictor]
    
    ## Order by predictor
    plot.data <- plot.data[order(plot.data$predictor), ]
    
    level <- as.character(unique(plot.data$predictor))
    numlevel <- length(level)
    
    ## Plot
    layout(rbind(1,2), heights=c(9,1))  # put legend on bottom 1/10th of the chart
    par(mar = c(4, 4.4, 2, 4.4))
    plot(seq_len(numlevel), plot.data$exposure, col = "grey", type = "l", lty = 2, xaxt = "n", 
         lwd=2, axes=F, xlab = NA, ylab = NA, main = paste0("AvsE by ", predictor_name))
    axis(1, at=seq_len(numlevel), labels = level)
    axis(side = 4, las=1)
    mtext(side=4, line=3.5, 'Exposure')
    par(new = T)
    plot(seq_len(numlevel), plot.data$target, type = "o", xlab = predictor_name,
         main = paste0("AvsE by ", predictor_name), xaxt = "n", 
         ylab = "Average Response", lwd = 1, col = "dodgerblue2")
    axis(1, at=seq_len(numlevel), labels = level)
    points(plot.data$predictor, plot.data$pred, col = "cyan3", type = "o", lwd = 2)
    par(mar=c(0, 0, 0, 0))
    plot.new()
    legend("center", "groups", inset = 0, legend = c("Actual target", "Predicted target", "Exposure"),
           lty = c(1, 1, 2), lwd = 2, col = c("dodgerblue2", "cyan3", "grey"), cex = 0.9, bty = "n", horiz = T, 
           text.width = max(sapply(c("Actual target", "Predicted target", "Exposure"), strwidth)))
  }
  
  #################################################
  
  # Numeric Predictor
  if (tolower(cts_or_cat) == "cts") {
    
    ## Percentile bands
    percentiles <- seq(from = 0, to = 1, length.out = graph_groups)
    perc.bands <- quantile(indata$predictor, percentiles)
    perc.bands[1] <- -Inf
    perc.bands[length(perc.bands)] <- Inf
    indata$levels <- cut(indata$predictor, unique(perc.bands), include.lowest=TRUE)
    DF <- indata[, .(avg_predictor=sum(predictor*weights, na.rm = T)/sum(weights, na.rm = T)), by = levels]
    
    indata <- merge(indata, DF, by="levels", all.x = T)
    
    plot.data <- indata[, .(target = sum(target*weights, na.rm = T)/sum(weights, na.rm = T),
                            pred = sum(pred*weights, na.rm = T)/sum(weights, na.rm = T), 
                            exposure = sum(weights, na.rm = T)), by = avg_predictor]
    
    ## Order by predictor
    plot.data <- plot.data[order(plot.data$avg_predictor), ]
    
    ## Plot
    layout(rbind(1,2), heights=c(9,1))  # put legend on bottom 1/10th of the chart
    par(mar = c(4, 4.4, 2, 4.4))
    plot(plot.data$avg_predictor, plot.data$exposure, col = "grey", type = "l", lty = 2, 
         lwd=2, axes=F, xlab = NA, ylab = NA, main = paste0("AvsE by ", predictor_name))
    axis(side = 4, las=1)
    mtext(side=4, line=3, 'Exposure')
    par(new = T)
    plot(plot.data$avg_predictor, plot.data$target, type = "o", xlab = predictor_name,
         ylab = "Average Response", col = "dodgerblue2", lwd = 2)
    points(plot.data$avg_predictor, plot.data$pred, col = "cyan3", type = "o", lwd = 2)
    par(mar=c(0, 0, 0, 0))
    plot.new()
    legend("center", "groups", c("Actual target", "Predicted target", "Exposure"), inset = 0, 
           lty = c(1, 1, 2), lwd = 2, col = c("dodgerblue2", "cyan3", "grey"), cex = 0.9, bty = "n", horiz = T, 
           text.width = max(sapply(c("Actual target", "Predicted target", "Exposure"), strwidth)))
    
    if (table_output) 
      print(plot.data)
  }
  
  # Return plot parameters to inital settings
  par(mar = cur.mar)
  layout(1)
  
}


AvsEPred <- function(indata, 
                     target_name, 
                     pred_name, 
                     exposure_weights = "",
                     frequency_variable = "", 
                     graph_groups = 20, 
                     where_condition = NULL, 
                     size_model = F,
                     table_output = F) {
  
  localenv <- environment()
  
  # Grab the original margin parameters
  cur.mar <- par()$mar
  
  
  ## where_condition
  if (!is.null(where_condition)) {
    indata <- subset(indata, eval(parse(text=where_condition)))
  }
  
  ## determine weights
  if (exposure_weights %in% colnames(indata)){
    exposure_w <- indata[[exposure_weights]]
  }
  else {
    exposure_w <- rep(1,nrow(indata))
  }
  
  if (frequency_variable %in% colnames(indata)){
    frequency_w <- indata[[frequency_variable]]
  }
  else{
    frequency_w <- rep(1,nrow(indata))
  }
  
  weights <- exposure_w*frequency_w
  target <- indata[[target_name]]
  pred <- indata[[pred_name]]
  
  ## Remove records with missing targets
  target.orig <- target
  pred.orig <- pred
  weights.orig <- weights
  
  target <- target.orig[!is.na(target.orig)]
  pred <- pred.orig[!is.na(target.orig)]
  weights <- weights.orig[!is.na(target.orig)]
  
  ## Size model
  if (size_model) {
    pred <- pred[target > 0]
    target <- target[target > 0]
    weights <- rep(1,length(target))
  }
  
  ## Order by predicted
  target.ord <- target[order(pred, decreasing = "F")]
  pred.ord <- pred[order(pred, decreasing = "F")]
  weights.ord <- weights[order(pred, decreasing = "F")]
  
  target.ord.w <- target.ord * weights.ord
  pred.ord.w <- pred.ord * weights.ord
  
  ## Percentile bands
  percentiles <- seq(from = 1/graph_groups, to = 1, length.out = graph_groups)
  perc.bands <- quantile(pred.ord, percentiles)
  
  ## Percentile bands
  percentile.size <- length(target.ord)/graph_groups
  rank <- seq_along(target.ord)
  perc.bands <- ceiling(rank/percentile.size)/graph_groups
  
  target.av <- tapply(target.ord.w, perc.bands, sum)
  pred.av <- tapply(pred.ord.w, perc.bands, sum)
  weights.av <- tapply(weights.ord, perc.bands, sum)
  
  target.plot <- target.av/weights.av
  pred.plot <- pred.av/weights.av
  
  ## Plot
  layout(rbind(1,2), heights=c(9,1))  # put legend on bottom 1/10th of the chart
  par(mar = c(4, 4, 2, 2))
  plot(percentiles, target.plot, type = "o", xlab = "Percentile", 
       main = "Actual vs Expected by Predicted Band",
       ylab = "Target", col = "dodgerblue2", lwd = 2)
  points(percentiles, pred.plot, col = "cyan3", type = "o", lwd = 2)
  par(mar=c(0, 0, 0, 0))
  plot.new()
  legend("center", "groups", inset = 0, legend = c("Actual target", "Predicted target"),
         lty = c(1, 1), lwd = 2, col = c("dodgerblue2", "cyan3"), cex = 0.9, bty = "n", horiz = T, 
         text.width = max(sapply(c("Actual target", "Predicted target"), strwidth)))
  
  # Actual / Expected
  ae <- sum(target.ord.w)/sum(pred.ord.w)
  ae.pred <- target.plot/pred.plot
  
  ## Return
  result <- rbind(cbind(percentiles, ae.pred), cbind(100, ae))
  rownames(result) <- NULL
  colnames(result) <- c("Percentile", "A/E")
  
  result2 <- cbind(target.plot, pred.plot)
  
  mad <- mean(abs(result2[,1]-result2[,2]))
  names(mad) <- c("Mean Absolute Deviation of the AvsEPred")
  print(mad)
  
  if (table_output) {
    #print(result)
    print(result2)
  }
  
  # Return plot parameters to inital settings
  par(mar = cur.mar)
  layout(1)
  
}


CalculateGains <- function(indata, 
                           target_name, 
                           pred_name, 
                           exposure_weights = "", 
                           frequency_variable = "", 
                           where_condition = NULL, 
                           size_model = F, 
                           plot = F) {
  
  localenv <- environment()
  
  
  ## where_condition
  if (!is.null(where_condition)) {
    indata <- subset(indata, eval(parse(text=where_condition)))
  }
  
  ## determine weights
  if (exposure_weights %in% colnames(indata)){
    exposure_w <- indata[[exposure_weights]]
  }
  else {
    exposure_w <- rep(1,nrow(indata))
  }
  
  if (frequency_variable %in% colnames(indata)){
    frequency_w <- indata[[frequency_variable]]
  }
  else{
    frequency_w <- rep(1,nrow(indata))
  }
  
  weights <- exposure_w*frequency_w
  target <- indata[[target_name]]
  pred <- indata[[pred_name]]
  
  ## Remove records with missing targets
  target.orig <- target
  pred.orig <- pred
  weights.orig <- weights
  
  target <- target.orig[!is.na(target.orig)]
  pred <- pred.orig[!is.na(target.orig)]
  weights <- weights.orig[!is.na(target.orig)]
  
  ## Size model
  if (size_model) {
    pred <- pred[target > 0]
    target <- target[target > 0]
    weights <- rep(1,length(target))
  }
  
  ## Totals
  target.total <- sum(target)
  pred.total <- sum(pred)
  
  nobs <- length(target)
  
  ## Sort the data
  target.ord <- target[order(target, decreasing = "T")]
  target.pred.ord <- target[order(pred, decreasing = "T")]
  
  ## Sort the weights
  weights.ord <- weights[order(target, decreasing = "T")]
  weights.pred.ord <- weights[order(pred, decreasing = "T")]
  
  perc <- cumsum(weights.ord)/sum(weights)
  perc.pred <- cumsum(weights.pred.ord)/sum(weights)
  
  ## Next get cumulative proportion of total
  target.cumsum <- cumsum(target.ord * weights.ord)/sum(target.ord *
                                                          weights.ord)
  target.pred.cumsum <- cumsum(target.pred.ord * weights.pred.ord)/
    sum(target.pred.ord * weights.pred.ord)
  
  ## Create lagged versions for differencing
  target.cumsum.lag <- c(0, target.cumsum[1:nobs - 1])
  target.pred.cumsum.lag <- c(0, target.pred.cumsum[1:nobs - 1])
  
  perc.lag <- c(0, perc[1:nobs - 1])
  perc.pred.lag <- c(0, perc.pred[1:nobs - 1])
  
  ## Calculate area under gains curve
  max_gains <- sum((perc - perc.lag) * (target.cumsum + target.cumsum.lag)/2) -
    0.5
  model_gains <- sum((perc.pred - perc.pred.lag) * 
                       (target.pred.cumsum +target.pred.cumsum.lag)/2) - 0.5
  
  ## Percentage of theoretical max gains
  pct_max_gains = model_gains/max_gains
  
  ## Generate plot if requested
  if (plot) {
    
    # Grab the original plot margin
    cur.mar <- par()$mar
    
    layout(rbind(1,2), heights=c(9,1))  # put legend on bottom 1/10th of the chart
    par(mar = c(4,4,2,2))
    plot(c(0, 1), c(0, 1), type = "l", 
         xlab = "Cumulative proportion of population",
         main = "Gains chart", ylab = "Gains", col = "dark green", lwd = 2)
    points(c(0, perc.pred), c(0, target.pred.cumsum), type = "l", col = "blue",
           lwd = 2)
    points(c(0, perc), c(0, target.cumsum), type = "l", col = "red",
           lwd = 2)
    par(mar=c(0, 0, 0, 0))
    plot.new()
    legend("center", "groups", inset = 0, legend = c("Random gains", "Model gains", "Theoretical max gains"),
           lty = c(1, 1, 2), lwd = 2, col = c("dark green", "blue","red"), cex = 0.9, bty = "n", horiz = T, 
           text.width = max(sapply(c("Random gains", "Model gains", "Theoretical max gains"), strwidth)))
    
    par(mar = cur.mar)
    layout(1)
  }
  
  ## Return
  result <- c(model_gains, max_gains, pct_max_gains)
  names(result) <- c("Model gains", "Max gains", "Model/max")
  
  result
  
}



PartialPlot <- function(indata, 
                        predictor_name, 
                        cts_or_cat = "cts", 
                        package = NULL, 
                        model = NULL, 
                        graph_groups = 20, 
                        exposure_weights = "", 
                        frequency_variable = "", 
                        idx = 100) {
  
  if (is.null(package))
    stop("Error: package name must be specified. No default value.")
  
  # Grab the original margin parameters
  cur.mar <- par()$mar
  
  # Cast indata as a data table
  indata <- as.data.table(indata)
  
  if ("pred_val" %in% colnames(indata)) {
    indata$pred_val <- NULL
  }
  
  # Get a sample of the data
  n <- nrow(indata)
  DF <- indata[sample(n, idx), ]
  
  # Determine weights
  if (exposure_weights %in% colnames(DF)) {
    eweights <- DF[[exposure_weights]]
  } else {
    eweights <- 1
  }
  
  if (frequency_variable %in% colnames(DF)) {
    fvar <- DF[[frequency_variable]]
  } else {
    fvar <- 1
  }
  DF[, weights := eweights * fvar]
  
  # Categorical Predictor
  if (tolower(cts_or_cat) == "cat") {
    DF[[predictor_name]] <- as.character(DF[[predictor_name]])
    level <-  as.character(unique(DF[[predictor_name]]))
    numlevel <- length(level)
    
    newvec <- rep(level, idx)[order(rep(level, idx))]
    DT <- do.call("rbind", replicate(numlevel, DF, simplify = FALSE))
    DT[[predictor_name]] <- newvec
    
    DT[[predictor_name]] <- as.factor(DT[[predictor_name]])
    levels(DT[[predictor_name]]) <- levels(indata[[predictor_name]])
    
    # Score new data
    DT$pred <- score.model(indata = DT, model = model, package = package)[["pred"]]
    
    # Calculate plot data
    plot.data <- DT[, .(pred = sum(pred*weights, na.rm = T)/sum(weights, na.rm = T), 
                        exposure = sum(weights, na.rm = T)), by = get(predictor_name)]
    
    ## Plot
    plot(seq_len(numlevel), plot.data$pred, type = "o", xlab = predictor_name,
         main = paste0("Partial Dependency Plot: ", predictor_name),
         ylab = "Average Response", lwd = 2, xaxt = "n", col = "dodgerblue2")
    axis(1, at=seq_len(numlevel), labels = level)
  }
  
  #######################
  
  # Numeric Predictor
  if (tolower(cts_or_cat) == "cts") {
    
    ## Percentile bands
    percentiles <- seq(from = 0, to = 1, length.out = graph_groups)
    perc.bands <- quantile(DF[[predictor_name]], percentiles)
    DF$levels <- cut(DF[[predictor_name]], unique(perc.bands), include.lowest=TRUE)
    DT <- DF[, .(avg_predictor=sum(get(predictor_name)*weights, na.rm = T)/sum(weights, na.rm = T)), by = levels]
    
    DF <- merge(DF, DT, by="levels", all.x = T)
    level <-  unique(DF[["avg_predictor"]])
    numlevel <- length(level)
    newvec <- rep(level, idx)[order(rep(level, idx))]
    DT <- do.call("rbind", replicate(numlevel, DF, simplify = FALSE))
    DT[[predictor_name]] <- newvec
    
    # Score new data
    DT$pred <- score.model(indata = DT, model = model, package = package)[["pred"]]
    
    plot.data <- DT[, .(pred = sum(pred*weights, na.rm = T)/sum(weights, na.rm = T), 
                        exposure = sum(weights, na.rm = T)), by = get(predictor_name)]
    
    # ## Order by predictor
    # plot.data <- plot.data[order(plot.data[[get]]), ]
    
    ## Plot
    par(mar = c(5, 4, 4, 4) + 0.4)
    plot(plot.data$get, plot.data$pred, type = "o", xlab = predictor_name,
         main = paste0("Partial Dependency Plot: ", predictor_name),
         ylab = "Average Response", col = "dodgerblue2", lwd = 2)
  }
  
  # Return plot parameters to inital settings
  par(mar = cur.mar)
  
}


score.model <- function(indata=NULL, 
                        model=NULL, 
                        package=NULL, 
                        target = "renew"){
  if (is.null(indata) | is.null(model) | is.null(package)){
    stop("Error: indata, model, and package 
         name must be specified. No default value.")
  }
  if (tolower(package)=="h2o"){
    temp <- as.h2o(indata)
    score <- data.frame(renew=as.vector(temp[[target]]), 
                        pred = as.vector(h2o.predict(model, temp)['p1']))
  }
  if (tolower(package)=="xgboost"){
    temp <- data.matrix(indata[,variable.list, with=FALSE])
    score <- data.frame(renew = indata[[target]],
                        pred = predict(model,temp))
  } 
  if (tolower(package)=="rpart"){
    score <- data.frame(renew = indata[[target]],
                        pred = predict(model,indata)[, 2])
  }
  return (score)
}



sourcePartial <- function(startTag='#from here',
                          endTag='#to here',
                          fn="./r_code/xx_h20_data_prep_code.R") {
  lines <- scan(fn, what=character(), sep="\n", quiet=TRUE)
  st<-grep(startTag,lines)
  en<-grep(endTag,lines)
  tc <- textConnection(lines[(st+1):(en-1)])
  source(tc)
  close(tc)
}



squid_plot <- function(model, 
                       print = TRUE){
  
  regpath <- h2o.getGLMFullRegularizationPath(model)
  DF <- data.frame(lambda = regpath$lambdas, regpath$coefficients)
  
  # Drops all columns with 0s
  DF <- DF[, !apply(DF==0, 2, all)]
  
  # Drop 'intercept' column
  DF$Intercept <- NULL
  
  DF <- DF[order(DF$lambda, decreasing = T), ]
  
  minval <- min(apply(DF[, !(colnames(DF) %in% "lambda")], MARGIN = 2, min))
  maxval <- max(apply(DF[, !(colnames(DF) %in% "lambda")], MARGIN = 2, max))
  
  layout(rbind(1,2), heights=c(7,3))  # put legend on bottom 1/10th of the chart
  par(mar=c(1, 4, 2, 4))
  matplot(x = DF[["lambda"]], as.matrix(DF[, !(colnames(DF) %in% "lambda")]), type = c("l"), pch=1, lty = 1,
          lwd = 2, col = 1:(ncol(DF)-1), xlab = "Lambda", ylab = expression(paste(beta,"-Coefficients")), 
          main = "Lambda Tuning", ylim = c(minval, maxval))
  lines(x = DF[["lambda"]], rep(0, nrow(DF)), lty = 2)
  par(mar=c(0, 6, 0, 0))
  plot.new()
  legend("center", colnames(DF)[!(colnames(DF) %in% "lambda")], lty = 1, lwd = 2, inset = 0.01, 
         col = 1:(ncol(DF)-1), ncol = 4, cex = 0.8, bty = "n")
  
  if (print == TRUE) { return(DF) }
}



view_tree <- function(fit) {
  
  require(rpart.plot)
  
  only_count <<- function(x, labs, digits, varlen)
  {
    paste(x$frame$n)
  }
  
  # Greab current plot parameters
  cur.xpd <- par()$xpd
  
  par(xpd=TRUE)
  prp(fit, faclen = 0, cex = 0.8, node.fun=only_count)
  
  # Return plot parameters to inital settings
  par(xpd = cur.xpd)
}


















