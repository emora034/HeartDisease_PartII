
library(tidyverse)
library(skimr)
library(mice)
library(VIM)
library(GGally)
library(MASS)
library(glmnet)
library(e1071) 
library(rpart)
library(pROC)
library(class)
library(randomForest)
library(FFTrees)
library(caret)
library(forcats)
library(VIM)
library(gbm)
library(neuralnet)


#Heart Disease Data Set
#This dataset contains 14 variables, where data was collected from 303 patients who were admitted to a hospital. The "goal/target" field refers to the presence of heart disease in the patient (1=yes; 0=no). The variables' information is as follows:
#1. age: The person's age in years
#2. sex: The person's sex (1 = male, 0 = female)
#3. cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
#4. trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
#5. chol: The person's cholesterol measurement in mg/dl
#6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) 
#7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
#8. thalach: The person's maximum heart rate achieved during Stress Test (exercise)
#9. exang: Exercise induced angina (1 = yes; 0 = no)
#10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot)
#11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
#12. ca: The number of major vessels (0-3) colored by flourosopy 
#13. thal: A blood disorder called thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
#14. goal/target: Heart disease (0 = no, 1 = yes)

setwd("C:/Users/emora/OneDrive/Desktop/Statistical Learning")
Heart<-read.csv("Heart Disease Data.csv", header = TRUE)
glimpse(Heart)
skim(Heart)

# Let's check for NA values
anyNA(Heart)
colSums(is.na(Heart))
#attach(Heart)
md.pattern(Heart)

#We have some categorical values that need to be defined as factors
Heart$sex=factor(Heart$sex)
levels(Heart$sex)=c("Female", "Male")
Heart$cp=factor(Heart$cp)
levels(Heart$cp)=c("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic")
Heart$fbs=factor(Heart$fbs)
levels(Heart$fbs)=c("Below120","Above120")
Heart$restecg=factor(Heart$restecg)
levels(Heart$restecg)=c("Normal","Abnormal","Hypertrophy")
Heart$exang=factor(Heart$exang)
levels(Heart$exang)=c("No","Yes")
Heart$slope=factor(Heart$slope)
levels(Heart$slope)=c("Upslopping","Flat","Downslopping")
Heart$ca=factor(Heart$ca)
Heart$target=factor(Heart$target)
levels(Heart$target)=c("No Heart Disease","Heart Disease")
head(Heart)
glimpse(Heart)

#deleting thal variable as there's missing information on the labels
#source claims 3 factors, but there are 4. 
Heart<-Heart[,-13]

table(Heart$sex,Heart$target)
prop.table(table(Heart$sex, Heart$target))



#----------------Split Data-------------
intrain <- createDataPartition(Heart$target, p = 0.6, list = FALSE)
train <-Heart[intrain,]
test <- Heart[-intrain,]
nrow(train)
nrow(test)

#----------Data Visualizations-----------
###     PENDING   ###
#Gender vs. Diseases in this data set
table(Heart$sex,Heart$target)
table(train$sex,train$target)
par(mfrow=c(1,1))
col.target <- c("blue","red")
plot(table(train$sex,train$target),xlab="Gender",ylab="Diagnostics",col=col.target, main=" ")
summary(train)

#Heart Disease vs. Gender vs. Cholesterol
ggplot(train, aes(x=target, y=chol, fill=train$sex))+geom_boxplot( )+
  labs(title="Heart Disease, Gender and Cholesterol Levels", x="Diagnosis",
       y="Cholesterol")+scale_fill_manual(values=c("aquamarine", "pink"),labels=c("Male", "Female"), name="Gender")

# Correlations
ggcorr(train, label = T)


########################################
#######################################
#######################################
#----------------Models-----------------
# 1. Decision tree
require(tree)
attach(Heart)
tree.heart=tree(target~.,train)
summary(tree.heart)
plot(tree.heart)
text(tree.heart, pretty = 0)

#predict
tree.pred<-predict(tree.heart, test, type="class")
with(test, table(tree.pred, target))
#(38+49)/121

confusionMatrix(tree.pred, test$target)

# -> Decision tree with cross-val
cv.heart = cv.tree(tree.heart, FUN = prune.misclass)
plot(cv.heart)
prune.heart = prune.misclass(tree.heart, best = 6)
plot(prune.heart)
text(prune.heart, pretty=0)
summary(prune.heart)

tree.pred = predict(prune.heart, test, type="class")
with(test, table(ctree.pred, target))
#(34+56)/121

###########################################
#############    COST   ##################
cost.p <- c(0, 150, 500, 200)

############################################
############################################
#-------LDA Approach from previous 
# LDA approach. Is accuracy higher than with log regression? 
lda.modh <- lda(target ~ ., data=train, prior = c(.6,.4))
probs2 = predict(lda.modh, newdata=test)$posterior
threshold = 0.5
h.pred = rep("No Heart Disease", nrow(test))
h.pred[which(probs2[,2] > threshold)] = "Heart Disease"
# Produce a confusion matrix
CMlda = confusionMatrix(factor(h.pred), test$target)$table
costH = sum(as.vector(CMlda)*cost.p)/sum(CMlda)
costH
#141.32



###################################
###################################
#---------RF Tree Approach--------

EconCost <- function(data, lev = NULL, model = NULL) 
{
  y.pred = data$pred 
  y.true = data$obs
  CM = confusionMatrix(y.pred, y.true)$table
  out = sum(as.vector(CM)*cost.p)/sum(CM)
  names(out) <- c("EconCost")
  out
}

#set cross-validation control to 5-fold
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE, 
                     summaryFunction = EconCost,
                     verboseIter=T)
#100 trees
#set cutoff same as my lda
rf.train <- randomForest(target ~., data=train,
                         ntree=100,mtry=5,cutoff=c(0.6,0.4),importance=TRUE, do.trace=T)
# mtry: number of variables randomly sampled as candidates at each split
# ntree: number of trees to grow
# cutoff: cutoff probabilities in majority vote

plot(rf.train)
rf.pred <- predict(rf.train, newdata=test)
confusionMatrix(rf.pred, test$target)
EconCost(data = data.frame(pred  = rf.pred, obs = test$target))

rf.train <- train(target ~., 
                  method = "rf", 
                  data = train,
                  preProcess = c("center", "scale"),
                  ntree = 100,
                  cutoff=c(0.6,0.4),
                  tuneGrid = expand.grid(mtry=c(4,5,6)), 
                  metric = "EconCost",
                  maximize = F,
                  trControl = ctrl)

# Variable importance
rf_imp <- varImp(rf.train, scale = F)
plot(rf_imp, scales = list(y = list(cex = .95)))
text(rf_imp, pretty=0)

###############################
###############################
##############################
# Gradient Boosting
GB.train <- gbm(ifelse(train$target=="No Heart Disease",0,1) ~., data=train,
                 distribution= "bernoulli",n.trees=100,shrinkage = 0.01,interaction.depth=2,n.minobsinnode = 8)
threshold = 0.5
gbmProb = predict(GB.train, newdata=test, n.trees=100, type="response")
gbmPred = rep("No Heart Disease", nrow(test))
gbmPred[which(gbmProb > threshold)] = "Heart Disease"
CMb = confusionMatrix(factor(gbmPred), test$target)$table
cost = sum(as.vector(CMb)*cost.p)/sum(CMb)
cost

# TUNING OF BOOSTING
# Now optimize the hyper-parameters with Caret:
xgbgrid = expand.grid(
  nrounds = c(200,1000),
  eta = c(0.01, 0.001), # c(0.01,0.05,0.1)
  max_depth = c(2, 4, 6),
  gamma = 1,
  colsample_bytree = c(0.2, 0.4),
  min_child_weight = c(1,5),
  subsample = 1
)

################ xgboost package required
xgb.train = train(make.names(target)~.,  data=train,
                  trControl = ctrl,
                  metric="EconCost",
                  maximize = F,
                  tuneGrid = xgbgrid,
                  method = "xgbTree")

# Variable importance
xgb_imp <- varImp(xgb.train, scale = F)
plot(xgb_imp, scales = list(y = list(cex = .95)))

threshold = 0.5
xgbProb = predict(xgb.train, newdata=test, type="prob")
xgbPred = rep("No Heart Disease", nrow(test))
xgbPred[which(xgbProb[,2] > threshold)] = "Heart Disease"
confusionMatrix(xgbPred, test$target)
CM = confusionMatrix(factor(xgbPred), test$target)$table
cost = sum(as.vector(CM)*cost.p)/sum(CM)
cost

###################33
######################
################## NN
ctrl$sampling <- NULL
nn.train <- train(make.names(target) ~., 
                  method = "nnet", 
                  data = train,
                  preProcess = c("center", "scale"),
                  MaxNWts = 300,
                  maxit = 100,
                  tuneGrid = expand.grid(size=c(2,4,6), decay=c(0.01,0.001)), 
                  metric = "EconCost",
                  maximize = F,
                  trControl = ctrl)
plot(nn.train)
nn_imp <- varImp(nn.train, scale = F)
plot(nn_imp, scales = list(y = list(cex = .95)))

threshold = 0.5
nnProb = predict(nn.train, newdata=test, type="prob")
nnPred = rep("No Heart Disease", nrow(test))
nnPred[which(nnProb[,2] > threshold)] = "Heart Disease"
EconCost(data = data.frame(pred  = nnPred, obs = test$target))


############################
############################
###########################
# Deep Neural Network
dnn.train <- train(make.names(target) ~., 
                   method = "dnn", 
                   data = train,
                   preProcess = c("center", "scale"),
                   numepochs = 50, # number of iterations on the whole training set
                   tuneGrid = expand.grid(layer1 = 1:4,
                                          layer2 = 0:2,
                                          layer3=0:2,
                                          hidden_dropout = 0, 
                                          visible_dropout = 0),
                   metric = "EconCost",
                   maximize = F,
                   trControl = ctrl)
threshold = 0.5
dnnProb = predict(dnn.train, newdata=test, type="prob")
dnnPred = rep("No Heart Disease", nrow(test))
dnnPred[which(dnnProb[,2] > threshold)] = "Heart Disease"
EconCost(data = data.frame(pred  = dnnPred, obs = test$target))

###########################
##########################
#########################
######################## FINAL COMPARISONS
# Create mode function
mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

ensemble.pred = apply(data.frame(h.pred, xgbPred), 1, mode) 
EconCost(data = data.frame(pred  = ensemble.pred, obs = test$target))

# ensemble models work better????
# when we have less models, maybe better to combine (as in regression) the 
# posterior probabilities
data.frame(probs2[,2], xgbProb[,2]) %>%
  ggcorr(palette = "RdBu", label = TRUE) + labs(title = "Correlations between different models")
# combination is going to improve results as they're all obvs. corr.

# Combinations!
ensemble.prob = (probs2[,2]+ xgbProb[,2])/2
threshold = 0.35
ensemble.pred = rep("No Heart Disease", nrow(test))
ensemble.pred[which(ensemble.prob > threshold)] = "Heart Disease"
EconCost(data = data.frame(pred  = ensemble.pred, obs = test$target))

