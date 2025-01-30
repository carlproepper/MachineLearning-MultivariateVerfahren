#Benötigte Packages laden
library(tidyverse)
library(class)
library(caret)
library(GGally)
#
#Datensatz importieren (Kaggle)
wein <- read_csv("/Users/carlpropper/ML&MV_datasets/winequality-red.csv")

#
#Qualitätsstufen
table(wein$quality)

#Datensatz anpassen
wein <- wein %>%
  rename(fixed_acidity = `fixed acidity`, 
         volatile_acidity = `volatile acidity`, 
         citric_acid = `citric acid`,
         residual_sugar = `residual sugar`,
         free_sulfur_dioxide = `free sulfur dioxide`, 
         total_sulfur_dioxide = `total sulfur dioxide`) %>% 
  mutate(quality=dplyr::recode(quality, 
                               `3`= "poor",
                               `4`= "poor",
                               `5`= "poor",
                               `6`= "medium",
                               `7`= "good",
                               `8`= "good")) %>%
  mutate_at(vars(quality), factor)
#
#Häufigkeit der Klassen
table(wein$quality)


#Funktion für die Normalisierung
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) 
}
#
#Normalisierung aller numerischen Variablen
weinNorm <- as_tibble(lapply(wein[,1:11], normalize))
weinNorm <-cbind(weinNorm, wein["quality"])
#
#Datensatz in Trainings- und Testdatensatz aufteilen
set.seed(123)
index<-sample(1:nrow(weinNorm), size = 0.75*nrow(weinNorm))
trainDat<-weinNorm[index,]
testDat<-weinNorm[-index,]
#
#Ergebnisse vergleichen
dim(trainDat)

dim(testDat)

#Summary über alle Daten
summary(wein)

#Histogramm für die kardinalskalierten Daten
gg <- ggplot(wein, aes(x = alcohol))
gg + geom_histogram(binwidth = 0.5, col="black", fill="blue")

#Histogramm für die kardinalskalierten Daten
gg <- ggplot(wein, aes(x = sulphates))
gg + geom_histogram(binwidth = 0.1, col="black", fill="blue")

#Darstellung von Streuung und Dichten
ggpairs(wein, columns=9:12, aes(colour=quality, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()


#Modellermittlung und Beurteilung

#Durchführung des kNN-Algorithmus
set.seed(123)
weinMod <- knn(train = trainDat[,1:11], test = testDat[,1:11], cl = trainDat[,12] , k=35)

#Confusion Matrix
confusionMatrix(table(weinMod,testDat[,12]))

#Modelloptimierung
#Festlegung der Kontrollparameter
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)
#
#Festlegung der Parameter für tuneGrid
grid <- expand.grid(k = seq(5, 51, by = 2))
#
#Training des Modells
set.seed(123)
weinMod2 <- train(form = quality ~., 
                  data = trainDat,
                  method = "knn",
                  preProcess = "range",
                  metric = "Accuracy",
                  trControl = ctrl,
                  tuneGrid = grid)
#
#Ergebnis
weinMod2

plot(weinMod2)


#Modellvorhersage
weinPred <- predict(weinMod2, newdata = testDat[,1:11])
#
#Konfusionsmatrix
confusionMatrix(weinPred, testDat[,12])

#
#Bedeutung der einzelnen Eigenschaften
plot(varImp(weinMod2))

#Eigenschaften des neuen Weins
weinNeu<-tibble(fixed_acidity = 9.0,
                volatile_acidity = 0.55,
                citric_acid = 0.25,
                residual_sugar = 3.2,
                chlorides = 0.09,
                free_sulfur_dioxide = 14.0,
                total_sulfur_dioxide = 42.0,
                density = 0.996,
                pH = 0.32, 
                sulphates = 0.72,
                alcohol = 11.5)
#
#Klassifizierung
predict(weinMod2, newdata = weinNeu, type = "raw")

#
#Wahrscheinlichkeit
predict(weinMod2, newdata = weinNeu, type = "prob")
