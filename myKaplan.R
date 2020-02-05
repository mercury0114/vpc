library(survival)
library(survminer)
library(dplyr)

set.seed(0)

source(file = "cutofffinder.R")

data <- read.csv("./../data/survival/survival1500.csv")
data <- data[order(data$id),]
statistics <- read.csv("./../data/survival/1500/statisticsMariausOld.txt")
statistics <- statistics[statistics$id %in% data$id,]
statistics <- statistics[order(statistics$id),]

#statistics <- statistics[statistics$rank %in% c(-1),]
#statistics <- aggregate(statistics, by=list(statistics$gID), FUN=mean)

idx<-sample(c(T, F), nrow(statistics), prob=c(0.9, 0.1), replace=T)

trainStatistics = statistics[idx,]
testStatistics = statistics[!idx,]
trainData = data[idx,]
testData = data[!idx,]

biomarkerName = "length"

biomarker = trainStatistics[,biomarkerName]
names(biomarker) = biomarkerName

print(dim(biomarker))
print(dim(trainData))
print(biomarker)
print(trainData)

get.cutoff(type=c("survival_significance"), 
           filename="./../data/survival/1500/MariausOldSplit/plot",
		biomarker=biomarker, time=trainData$ost_m, event=trainData$status01,
		plots=c("kaplanmeier"))

print(PVAL)

group = sapply(testStatistics[,biomarkerName], (function(x) x > CTF))
group = as.data.frame(group)
colnames(group) = "partition"

surv_object = Surv(time=testData$ost_m, event=!testData$status01)
fit = survfit(surv_object ~ partition, data=group)
