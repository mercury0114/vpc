library(survival)
library(survminer)
library(dplyr)

data <- read.csv("./../data/survival/survivalCRC.csv")
data <- data[order(data$gID),]
statistics <- read.csv("./../data/survival/fiber.txt")
statistics <- statistics[statistics$gID %in% data$gID,]
statistics <- statistics[order(statistics$gID),]

statistics <- statistics[statistics$rank %in% c(-4,-3,-2),]
statistics <- aggregate(statistics, by=list(statistics$gID), FUN=mean)

print(dim(data))
print(dim(statistics))

source(file = "cutofffinder.R")

for (i in seq(3, dim(statistics)[2])) {
	print(i)
	biomarker = statistics[,i]
	names(biomarker) = colnames(statistics)[i]
	get.cutoff(type=c("survival_significance"), 
                filename="./../data/survival/plotsCRCFiberRankMinus4Minus2/plot",
		biomarker=biomarker, time=data$PFS_mnth, event=data$P_fact,
		plots=c("kaplanmeier"))
}
