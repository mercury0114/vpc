library(survival)
library(survminer)
library(dplyr)

source(file = "cutofffinder.R")

getPValue = function(statistics, data) {
    biomarker = statistics[,"curvature"]
    names(biomarker) = "curvature"
    print(biomarker)
    get.cutoff(type=c("survival_significance"),
                filename="./../data/survival/1500/MariausOld/plot",
        biomarker=biomarker, time=data$ost_m, event=data$status01, plots=c("kaplanmeier"))
    return(PVAL)
}

data <- read.csv("./../data/survival/survival1500.csv")
data <- data[order(data$id),]
statistics <- read.csv("./../data/survival/1500/statisticsMariausOld.txt")
statistics <- statistics[statistics$id %in% data$id,]
statistics <- statistics[order(statistics$id),]


pValues = rep(0, dim(statistics)[1])
cutoffs = rep(0, dim(statistics)[1])
for (row in seq(1, dim(statistics)[1])) {
    s = statistics[c(-row),]
    d = data[c(-row),]
    pValues[row] = getPValue(s, d)
    cutoffs[row] = CTF
}

print("Final p values are:")
print(pValues)

