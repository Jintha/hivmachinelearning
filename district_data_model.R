#' ---
#' title: Machine Learning on HIV outcomes data
#' output: ioslides_presentation
#' ---

#+ include=FALSE
knitr::opts_chunk$set(warning = FALSE, message = FALSE)
extract_help <- function(pkg, fn = NULL, to = c("txt", "html", "latex", "ex"))
{
  to <- match.arg(to)
  rdbfile <- file.path(find.package(pkg), "help", pkg)
  rdb <- tools:::fetchRdDB(rdbfile, key = fn)
  convertor <- switch(to, 
                      txt   = tools::Rd2txt, 
                      html  = tools::Rd2HTML, 
                      latex = tools::Rd2latex, 
                      ex    = tools::Rd2ex
  )
  f <- function(x) capture.output(convertor(x))
  if(is.null(fn)) lapply(rdb, f) else f(rdb)
}

#' ## mlr tutorial
#' https://mlr-org.github.io/mlr-tutorial/release/html/
#'mlr is deprecated and mlr3 is the new library. using mlr here because this 
#'code was originally written for that.

#' ## Load tidyverse, mlr
#+ warning=FALSE, message=FALSE
library(tidyverse)
library(mlr)
library(ggplot2)
library(rpart.plot)

#Read in the dataset 

district_data = read.csv("~/myprojects/hivmachinelearning/district_hospital.csv")

#' ## Let's look inside the District data file.
str(district_data)

# Can we predict the patient outcomes
#' Yes, but using which models?

#' ## Which models may we use? The models below are supervised learning models. We already
#' know the outcome. In this case patients with death and alive outcomes.
#' 
#' - Decision trees
#' -  Logistic regression
#' - Support vector machine
#' -

#' ## These other  models which are examples of unsupervised learning

#' - Naive bayes
#'' K-nearest neighbors
#' Perceptron

#' ## First, let's start with decision trees 
#' - Decision trees are often developed by humans and serve as useful "rules of thumb" for classifying outcomes
#' - But ... if decision trees had to be built by humans every time, that would be quite time-consuming
#' - We can let the the machine build the decision tree for us. That's machine learning.
#'
#' ## But if we had to build a decision tree...
#' Which variables do we think are going to be most important to predict our outcomes?
#+ echo=FALSE
#How do you decide which model to use?
# Principal component analysis
# Pair the dimensions that make sense
# Usually start with a random forest
#
district_data %>%  
  filter(current.Outcome == "Died" | current.Outcome =="On ART") %>% 
  dplyr::select(Current.Weight,Start.Reason,Current.Regimen,current.Outcome) %>%
  mutate(Current.Weight = cut(Current.Weight,5,include.lowest=TRUE)) %>% 
  gather(variable,value,Current.Weight:Current.Regimen) %>%
#  mutate(value=as.numeric(value)) %>% 
  ggplot(aes(x=value, fill=value,color=current.Outcome)) +
  geom_density(alpha=1/3) + 
  facet_wrap(~variable, scales='free')
#  scale_fill_brewer(palette = 'Set3')

district_data %>%
  dplyr::select(Start.Reason,Current.Regimen,current.Outcome) %>% 
  filter(current.Outcome == "Died" | current.Outcome =="On ART") %>% 
  gather(variable,value,Start.Reason:Current.Regimen) %>% 
  ggplot(aes(x=value, fill=current.Outcome)) +
  geom_density(alpha=1/3) + 
  facet_wrap(~variable, scales='free') + 
  scale_fill_brewer(palette = 'Set3')


district_data %>%
  dplyr::select(Current.Regimen,current.Outcome) %>% 
  filter(current.Outcome == "Died" | current.Outcome =="On ART") %>% 
  gather(variable,value,Current.Regimen) %>% 
  ggplot(aes(x=value, fill=current.Outcome)) +
  geom_density(alpha=1/3) + 
  scale_fill_brewer(palette = 'Set3')

district_data %>%
  dplyr::select(Current.Weight,current.Outcome) %>% 
  filter(current.Outcome == "Died" | current.Outcome =="On ART") %>% 
  gather(variable,value,Current.Weight) %>% 
  ggplot(aes(x=value, fill=current.Outcome)) +
  geom_histogram() + 
  scale_fill_brewer(palette = 'Set3')




#' ## Let's use mlr to train a decision tree model. This requires:
#' - a task
#' - a learner
#' - no need for resampling strategy or measures (since all we are doing is training the model)
# train district data on death outcome with all the above variables
# will attempt the 70% train data and the rest test data

district_train = district_data %>% dplyr::select(Current.Weight,Start.Reason,Current.Regimen,current.Outcome, Current.Age) %>% 
   mutate_if(is.factor,as.character) %>% 
   filter(current.Outcome == "Died" | current.Outcome =="On ART") %>% 
   mutate(current.Outcome = ifelse(current.Outcome=='Died','Died','On_ART')) %>% 
   mutate_if(is.character,as.factor)
   
   
# %>% sample(nrow(district_data), 0.7*(district_data)) %>% district_data[.,]
# 
# 
#indices = sample()
training_data = district_data %>% sample_frac(0.7)
holdout_data = setdiff(district_data, training_data)
  
train_task = makeClassifTask(id = 'district_outcomes',
                             data = district_train,
                             target = 'current.Outcome',
                             positive = 'Died')
#lrn = makeLearner('classif.randomForestSRC', predict.type = 'prob')
lrn = makeLearner('classif.rpart', predict.type = 'prob')
model = train(task = train_task, learner = lrn)

#' ## Let's visualize the tree

rpart.plot(model$learner.model, tweak=0.8)

#' ## How well did the decision tree perform?
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1,tpr,tnr))

#' Awesome!!! Oh wait...
#' 
#' ## How well did the decision tree perform?

fv = generateFilterValuesData(train_task,method='randomForestSRC_var.select')
fv = generateFilterValuesData(train_task,method='FSelector_chi.squared')
fv
#fv$data %>% mutate(randomForestSRC_var.select=-randomForestSRC_var.select) %>% ggplot(aes(x=reorder(name,-rf.min.depth),y=rf.min.depth)) + geom_bar(stat='identity') + coord_flip()

plotFilterValues(fv)

#' ## How do you plot an ROC curve?
#' - Generate threshold vs. performance data
#' - Feed this data to the `plotROCCurves()` function
eval=FALSE
generateThreshVsPerfData(predictions,measures=list(fpr,tpr)) %>%
plotROCCurves()
#' 
#' #' ## How well did the decision tree perform?
#' 
#' #' ## Let's train a k-nearest neighbors model



lrn = makeLearner('classif.kknn', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
#' 
#' #' ## How well does k-nearest neighbors perform?
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' #' ## How well does k-nearest neighbors perform?
echo=FALSE
generateThreshVsPerfData(predictions,measures=list(fpr,tpr)) %>% plotROCCurves()
#' 
#' #' ## Let's train a naive Bayes model
lrn = makeLearner('classif.naiveBayes', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
#' 
#' #' ## What's the "a priori" (or pretest) probability of being labeled a dead?
model$learner.model$apriori
#' 
#' #' ## How does glucose differ with presence of diabetes? 
model$learner.model$tables$Current.Weight
#' #' 1st column is the mean, 2nd column is standard deviation
#' #' 
#' 
#' #' ## How well do the following models perform on district training data (the test set)?
#' #' - naive Bayes
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' #' - support vector machine
lrn = makeLearner('classif.ksvm', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' 
#' #' - logistic regression
lrn = makeLearner('classif.logreg', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
 
#' 
#' 
#' #' - linear disciminant analysis
lrn = makeLearner('classif.lda', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' 
#' #' - neural net 
lrn = makeLearner('classif.nnet', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' 
#' 
#'  
#' #' - binomial regression
lrn = makeLearner('classif.binomial', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' 
#' 
#' #' - random forest
lrn = makeLearner('classif.cforest', predict.type = 'prob')
model = train(task = train_task, learner = lrn)
predictions = predict(model,newdata=district_train)
calculateConfusionMatrix(predictions)
performance(predictions, measures = list(timepredict,acc,auc,f1))
#' 
#' 
#' 
#' #' - Pick 2 models. Write down the accuracy, auc, and f1 for both.
#' #'
#' #' ## This is where benchmarking comes in...
#' #' Let's combine the data into one task
District.All = bind_rows(district_train,holdout_data)
overall_task = makeClassifTask(id = 'District.All',
                                data = District.All,
                                target = 'current.Outcome',
                                positive = 'Died')
 
#' #' ## Let's define a resampling strategy
 rinst = makeFixedHoldoutInstance(1:200,201:532,size=532)
#' 
#' #' ## Let's define the learners
 lrns = c('classif.rpart',
          'classif.kknn',
          'classif.naiveBayes',
          'classif.ksvm',
        'classif.logreg',
          'classif.binomial',
          'classif.gbm',
          'classif.randomForest') %>%
   lapply(makeLearner) %>% 
   lapply(. %>% setPredictType('prob'))
 
#' #' ## Let's run the benchmark
 models = benchmark(tasks = overall_task,
                    learners = lrns,
                    resamplings = rinst,
                    measures = list(acc, auc, f1))
 
#' #' ## Let's get the aggregated results
 perf = getBMRAggrPerformances(models,as.df = TRUE)
 perf
#' 
#' 
#' #' ## Best accuracy award goes to...
 plotBMRBoxplots(models,measure=acc)
#' 
#' #' ## Best area under the curve award goes to...
 plotBMRBoxplots(models,measure=auc)
#' 
#' #' ## Best F1 award goes to...
 plotBMRBoxplots(models,measure=f1)
#' 
#' #' ## Why isn't it that simple?
#' #' - Machine learning models usually require "tuning"
#' #' - Using a single holdout sample isn't representative enough of out-of-sample performance (a bad algorithm could get lucky)