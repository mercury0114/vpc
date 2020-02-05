set.seed(0)
library(survival)
library(survminer)
library(dplyr)

getStatistics = function(biomarker, data) {
    source(file = "cutofffinder.R")
    get.cutoff(type=c("survival_significance"),
                filename="./../data/survival/1500/MariausOld/plot",
        biomarker=biomarker, time=data$ost_m, event=data$status01, plots=c("kaplanmeier"))
    l = list(as.numeric(PVAL), HzR, CTF)
    names(l) = c("p", "hazard", "cutoff")
    return(l)
}

data <- read.csv("./../data/survival/survival1500.csv")
data <- data[order(data$id),]
features <- read.csv("./../data/survival/1500/statisticsMariausOld.txt")
features <- features[features$id %in% data$id,]
features <- features[order(features$id),]

table = matrix(, nrow=dim(features)[2], ncol=3)
rownames(table) = colnames(features)
colnames(table) = c("hzr_mean", "hzr_sd", "n_obs")
for (feature in colnames(features)[-1]) {
    pValues = rep(0, dim(features)[1])
    cutoffs = rep(0, dim(features)[1])
    hazards = rep(0, dim(features)[1])
    for (row in seq(1, dim(features)[1])) {
        biomarker = features[-row,feature]
        names(biomarker) = feature
        statistics = getStatistics(biomarker, data[c(-row),])
        pValues[row] = statistics$p
        cutoffs[row] = statistics$cutoff
        hazards[row] = statistics$hazard
    }
    table[feature, "hzr_mean"] = mean(hazards)
    table[feature, "hzr_sd"] = sd(hazards)
    table[feature, "n_obs"] = sum(pValues < 0.05)
    print(table)
    Sys.sleep(5)
}

write.csv(table, file="./../data/survival/1500/table.csv")

