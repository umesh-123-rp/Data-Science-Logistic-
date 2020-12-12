# Q:  We need to check whether the client has subscribed a term deposit or not

# Ans :
# We need to develop a model which satisfies given dataset with input and output variables
# This model to be used to decide whether the client has subscribed a term deposit or not

# Install Required packages
install.packages("mlbench")
install.packages("psych")
library(mlbench)
library(psych)
# Loading the data set
bank.full<-read.csv(file.choose(), header=T, sep = ';')
View(bank.full)
str(bank.full)
# Exploration of data set
summary(bank.full)
describe(bank.full)
attach(bank.full)
# Defining y variable with 0 & 1
bank.full$y <- ifelse(bank.full$y == "yes", 1, 0)
summary(bank.full)
table(bank.full$y)

# Build Linear Regression Model
fit<-lm(y~.,data=bank.full)
summary(fit)
plot(fit)
# R^2 value is only 0.3 and plot indicates that it is not a normal distribution.
# It does not fit to linear regression
# There are two curves joining which indicates that the distribution is Binomial.
# Therefore, we can go for logistic regression model
# Let's convert the class variable into factor form
bank.full$y <- factor(bank.full$y, levels = c(0, 1))
str(bank.full)

# Prep Training and Test data.
library(caret)
set.seed(100)
trainDataIndex <- createDataPartition(bank.full$y, p=0.7, list = F)
trainData <- bank.full[trainDataIndex, ]
testData <- bank.full[-trainDataIndex, ]
# Class distribution of train data
table(trainData$y)
# There are 3703 data points with class variable 1 and 27946 data points with class variable 0
# It indicates that class 1 is only about 15% of the total data set.
# We need to take care the unbalanced data set to avoid biasness in the model
# Therefore,let's find out down sample/ up sample

# Down Sample
set.seed(101)
down_train <- downSample(x = trainData[, colnames(trainData) %ni% "Class"],
                         y = trainData$y)
table(down_train$Class)
# We can have same number of samples for both "1" class and "0" class

# Up Sample (optional)
set.seed(100)
up_train <- upSample(x = trainData[, colnames(trainData) %ni% "Class"],
                     y = trainData$y)
table(up_train$Class)
# Similarly # We can have same number of samples for both "1" class and "0" class
  
# Build Logistic Model by considering down sample as trained data set

logitmod <- glm(Class ~ age+factor(job)+factor(marital)+factor(education)+factor(default)+balance
                +factor(housing)+factor(loan)+factor(contact)+day+factor(month)+duration
                +campaign+pdays+previous+factor(poutcome), family = "binomial", data=down_train)

summary(logitmod)

pred <- predict(logitmod, newdata = testData, type = "response")
pred
# We have observed that there are 5 independent variables which are not significant.
# We observed Null deviance as 10266.9 and Residual deviance as 5648.6 and AIC =5734
# By taking out the insignificant factors like age, pdays,day, balance and dafault, we made another model

logitmod1 <- glm(Class ~ factor(job)+factor(marital)+factor(education)+
                +factor(housing)+factor(loan)+factor(contact)+factor(month)+duration
                +campaign+previous+factor(poutcome), family = "binomial", data=down_train)
summary(logitmod1)

# We have observed that all the variables are significant.
# We observed Null deviance as 10266.9 and Residual deviance as 5652 and AIC =5728
# There is no much difference between the two models logitmod and logitmod1
# We can consider the logitmod for next subsequent process

# Recode factors
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))

# Accuracy
pred1<-data.frame(y_pred)
y_act <-testData$y
mean(y_pred == y_act) 
y_act<-data.frame(y_act)
conf<-table(pred1$y_pred,y_act$y_act)
conf
accuracy<-sum (diag(conf))/sum(conf)
accuracy
# Accuracy of the model is found to be 0.84

# Let's plot ROC Curve to balance True Positive Rate and False Positive Rate
install.packages("ROCR")
install.packages("pROC")
library(ROCR)
library(pROC)
rocrpred<-prediction(y_pred_num,testData$y)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
auc<-auc(testData$y ~ y_pred_num)
auc
# Area under curve is 0.835 which indicates a good model of Logistic Regression
##Conclusion 
# The logistic model logitmod is suitable with Bank dataset to take a decison on term deposit.
# The accuracy of the model is 0.84 with auc = 0.835 and maximum reduction in residual deviance.

