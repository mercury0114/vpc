library("survival")
library("survminer")

data <- read.csv("./../data/survival/survival1500.csv")
data <- data[order(data$id),]
features <- read.csv("./../data/survival/1500/statisticsMariausOld.txt")
features <- features[features$id %in% data$id,]
features <- features[order(features$id),]

length <- features[,"length"] < 23.9355576250968
length <- as.factor(length)
curvature <- features[,"curvature"] < 0.909799858901203
curvature <- as.factor(curvature)

survfit <- Surv(data$ost_m, data$status01)

d <- as.data.frame(cbind(curvature, length))
res.cox <- coxph(survfit ~ curvature + length, data = d)
print(res.cox)
summary(res.cox)


# cox2 <- coxph(survfit ~ curvature + length, data=features)
# summary(cox2)
