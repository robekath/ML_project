# Project: Machine Learning project
# Statistician: Katie Roberts
#
# set directory paths - set session working dir to your prog (or appropriate) directory


if (Sys.info()[1] == "Windows") {
  # input directory for data sets, including raw and manually edited data
  IN_DATA_DIR <- "..\\input"
# filename separator
FILE_SEP = "\\"
} else {
  # input directory for data sets, including raw and manually edited data
  IN_DATA_DIR <- "../input"
  FILE_SEP = "/"
}
  

#
# Convenience wrapper around paste for input data files
#
inDataFile <- function(filename) {
  return (paste(c(IN_DATA_DIR,FILE_SEP,filename), collapse=""))
}

#
# Convenience wrapper around paste for output data files
#
outDataFile <- function(filename) {
  return (paste(c(OUT_DATA_DIR,FILE_SEP,filename), collapse=""))
}

#
# Convenience wrapper around paste for output data files
#
outFigure <- function(filename) {
  return (paste(c(OUT_FIGURES_DIR,FILE_SEP,filename), collapse=""))
}



# load the csv 'training' data 
csvtraining = read.csv(paste(c(IN_DATA_DIR,"/pml-training.csv"),collapse=""), header=TRUE)

#subset data into useful model set of outcome, user_name, and raw values
newdata <- csvtraining[which(csvtraining$new_window=='yes'), ]
na.strings=c("","NA","#DIV/0!")
newdata[newdata == "#DIV/0!"] <- NA
newdatanoNA <- newdata[,!sapply(newdata,function(x) any(is.na(x)))]
newdatacomp = newdatanoNA[complete.cases(newdatanoNA),]

subbed <- subset(newdatacomp, select = c(2,8:10,28:39,51:59,69:71,89:100,118:127) )




library(ggplot2); library(caret)

set.seed(16165)
#create partition for training and testing (the 'test' data we will apply the predictions to will be our 'validation' set)
inTrain <- createDataPartition(y=subbed$classe,
                               p=0.60, list=FALSE)
#60% to train, 40% to test. 
training <- subbed[inTrain,]
testing <- subbed[-inTrain,]
dim(training); dim(testing)


set.seed(16165)
#nearZeroVar: to identify variables with very little variability and likely will not make good predictors
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv
#all varaibles contain some amount of variability that could make a useful predictor


#train command from the caret package
#trying to predit "classe"
#~. means to use all the other variables in this dataframe in order to predict the "type"
#from our dataset "training"
# Random Forests ####
#extension to bagging
modelFit <- train(classe ~ ., preProcess=c("center","scale"), data=training, method="rf", prox=TRUE)


#look at the model we fit
modelFit$finalModel




#predict on new samples
predictions <- predict(modelFit, newdata=testing)
predictions #gives predictions that correspond to the responses

#compare predictions to our outcomes
confusionMatrix(predictions,testing$classe)


finMod <- modelFit$finalModel
print(modelFit)
plot(finMod, main="Final Model")

#predict new values
pred <- predict(modelFit, testing)
testing$predRight <- pred == testing$classe #set a variable to see if we got the prediction right
table(pred, testing$classe) #see that we missed several
#look at which of the predictions we missed
qplot(classe, colour=predRight, data=testing, main="Class Prediction Correct")
#the points that were misclassified were mostly in classe e
qplot(user_name, colour=predRight, data=testing, main="User Prediction Correct")




############################################################
#answers to predicted values in test data set ####
#predict on new samples
# load the csv 'testing' data 
csvtest = read.csv(paste(c(IN_DATA_DIR,"/pml-testing.csv"),collapse=""), header=TRUE)


predictions <- predict(modelFit, newdata=csvtest)
predictions #gives predictions that correspond to the responses



pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id2_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions)

