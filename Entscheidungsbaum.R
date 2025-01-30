#Entscheidungsbaeume
#Pakete laden
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)
library(pROC)
library(e1071)
library(MASS)
library(tidyverse)

#Libraries laden
library(kernlab)
library(dplyr)

#Datensatz laden und Lehrzeichen aus Namen entfernen und Kleinschreibung
telco<-read_csv("/Users/carlpropper/ML&MV_datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#Teildatensatz für Einführung auswählen und anpassen
dat<-telco %>% 
  select(SeniorCitizen, Dependents, tenure, MonthlyCharges, Churn) %>% 
  mutate(SeniorCitizen=dplyr::recode(SeniorCitizen, '1'="Ja", '0'="Nein")) %>% 
  mutate(Dependents=dplyr::recode(Dependents, "Yes"="Ja", "No"="Nein")) %>% 
  mutate(Churn=dplyr::recode(Churn, "Yes"="Ja", "No"="Nein")) %>% 
  mutate_if(is.character, as.factor) %>% 
  rename(Rentner = SeniorCitizen, 
         Angehoerige = Dependents,
         Laufzeit = tenure,
         MonatsGeb = MonthlyCharges,
         Abwanderung = Churn)
summary(dat)
colnames(dat)

#Histogramm für die kardinalskalierten Daten
gg <- ggplot(dat, aes(x = Laufzeit))
gg + geom_histogram(binwidth = 5, col="black", fill="blue")

gg <- ggplot(dat, aes(x = MonatsGeb))
gg + geom_histogram(binwidth = 5, col="black", fill="blue")


#Auswahl über set.seed um vergleichbare Ergebnisse zu bekommen
set.seed(123)
index<-sample(1:nrow(dat), size = 0.75*nrow(dat))
trainDat<-dat[index,]
testDat<-dat[-index,]
#
summary(trainDat)
summary(testDat)

#Durchführung des ML-Verfahrens
set.seed(123)
ebaum01<-rpart(Abwanderung ~ ., data = trainDat, method = "class",
               control = c(minbucket=200, cp=0.0001))
#
#Ergebnisdarstellung
print(ebaum01)

rpart.plot(ebaum01, tweak=1.2)

summary(ebaum01)

ebaum02<-rpart(Abwanderung ~ ., data=trainDat, method = "class",
               control=c(minbucket=100, cp=0.0005))
ebaum02$cptable

#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(ebaum02, newdata= testDat, type = "class")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Abwanderung)

#Berechnung der Vorhersagewahrscheinlichkeiten für Testdatensatz
testProp <- predict(ebaum02, newdata = testDat, type = "prob")
#
#Darstellung der Grafik
testErg <- ifelse(testDat$Abwanderung == "Nein", 1,0)
gg <- roc(testErg, testProp[,2], plot = TRUE, print.auc = TRUE)

varImp(ebaum02)

#Basismodell ermitteln und darstellen
ebaum03<-rpart(Abwanderung ~ ., data=trainDat, method = "class",
               control=c(cp=0.00005, minbucket=30))
ebaum03$cptable

rpart.plot(ebaum03, tweak=1.1)

#Erster Optimierungsansatz
#Modell mit minimalen Cross Validation Fehler
bestcp <- ebaum03$cptable[which.min(ebaum03$cptable[,"xerror"]),"CP"]
#
#Neuen Baum mit bestem cp schätzen
ebaum04<-rpart(Abwanderung ~ ., data=trainDat, method = "class",
               control=c(cp=bestcp, minbucket=30))
ebaum04$cptable

rpart.plot(ebaum04, tweak=1.4)

#
#Modellbeurteilung Confusion Matrix
testPred <- predict(ebaum04, newdata= testDat, type = "class")
confusionMatrix(data = testPred, reference = testDat$Abwanderung)

#Modellbeurteilung ROC
testProp <- predict(ebaum04, newdata = testDat, type = "prob")
testErg <- ifelse(testDat$Abwanderung == "Ja", 1,0)
gg <- roc(testErg, testProp[,2], plot = TRUE, print.auc = TRUE)

#Beurteilung der Bedeutung der Variablen
varImp(ebaum04)

#Grenzen für Regel 2 ermitteln. 
#Spielraum für Cross Validation Fehler
values <- ebaum03$cptable[which.min(ebaum03$cptable[,"xerror"]),
                          c("xerror","xstd")]
values[1] + values[2]

values[1] - values[2]

ebaum03$cptable

#
#Modell mit 5 splits auswählen
bestcp <- ebaum03$cptable[which(ebaum03$cptable[,"nsplit"]==5),"CP"]

#Neuen Baum mit bestem cp schätzen
ebaum05<-rpart(Abwanderung ~ ., data=trainDat, method = "class",
               control=c(cp=bestcp, minbucket=30))
ebaum05$cptable


rpart.plot(ebaum05, tweak=1.3)
#
#Modellbeurteilung Confusion Matrix
testPred <- predict(ebaum05, newdata= testDat, type = "class")
confusionMatrix(data = testPred, reference = testDat$Abwanderung)
#
#Modellbeurteilung ROC
testProp <- predict(ebaum05, newdata = testDat, type = "prob")
testErg <- ifelse(testDat$Abwanderung == "Ja", 1,0)
gg <- roc(testErg, testProp[,2], plot = TRUE, print.auc = TRUE)

#Beurteilung der Bedeutung der Variablen
varImp(ebaum04) #Baum 5?

#Daten eines zu prüfenden Kunden
newDat <- data.frame(Rentner="Ja", 
                     Angehoerige="Nein", 
                     Laufzeit=15, 
                     MonatsGeb=50)
#
#Vorhersage Klassenzugehörigkeit
predict(ebaum05, newdata = newDat, type = "class")

predict(ebaum05, newdata = newDat, type = "prob")




#Regressionsanalyse mit Entscheidungsbäumen
#Datensatz laden und Variablen auswählen
bost <- Boston %>% 
  select(crim, chas, rm, dis, ptratio, medv) %>% 
  mutate_at(vars(chas), factor)
summary(bost)
