#reading train data
train=read.csv("D:/Study/Case Studies/datadriven.org/Predict-Blood-Donations-by-datadriven.org/original_data/train.csv")
#reading test data
test=read.csv("D:/Study/Case Studies/datadriven.org/Predict-Blood-Donations-by-datadriven.org/original_data/test.csv")

#converting the dependent variable as factor
train[,6]=factor(train[,6],
                 levels=sort(unique(train[,6])),
                 labels=sort(unique(train[,6])));

# checking correlation among the independent variables
cor(train[,c(2:5)]) # shows that "Total.Volume.Donated..c.c.." and "Number.of.Donations" are perfectly correlated
cor.test(as.numeric(train[,5]), as.numeric(train[,3])) # not correlated

# building the model
indvar=colnames(train[c(2,3,5)]);indvar # extracting colnames of independent variables
rhs=paste(indvar, collapse = "+");rhs # preparing the RHS of the model equation
lhs=colnames(train[6]);lhs # extracting the colname of the dependent variable
model=as.formula(paste(lhs, "~", rhs));model # preparing the full model equation as a formula

# checking the statistics of the model
model_fit=glm(model,data=train, family="binomial");model_fit
summary(model_fit)

# test data predictions - log-odds ratios
log_odds=predict.glm(model_fit, newdata = test);log_odds

# converting to probabilities
pred_prob=exp(log_odds)/(1+exp(log_odds));pred_prob

# preparing submission file
result=cbind(test[,1],pred_prob);result
colnames(result)[2]="Made Donation in March 2007"
write.csv(result, "D:/Study/Case Studies/datadriven.org/Predict-Blood-Donations-by-datadriven.org/result.csv", row.names = FALSE)
