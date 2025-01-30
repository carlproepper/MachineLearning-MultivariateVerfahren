#Pakete laden
library(tidyverse)
library(rpart)
library(caret)
library(ipred)
library(gbm)
library(randomForest)
library(themis)


#
#Datensatz laden
telco<-read_csv("/Users/carlpropper/ML&MV_datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#
#Teildatensatz für Einführung auswählen und anpassen
dat<-telco %>% 
  select(gender, SeniorCitizen, Dependents, tenure, StreamingTV, 
         StreamingMovies, Contract, MonthlyCharges, Churn) %>% 
  mutate(SeniorCitizen=dplyr::recode(SeniorCitizen, '1'="Yes", '0'="No")) %>% 
  mutate(Churn=dplyr::recode(Churn, "Yes"="Churned", "No"="Stayed")) %>% 
  mutate_if(is.character, as.factor)
#
summary(dat)
#
#Auswahl Trainings und Testdatensatz über set.seed um vergleichbare Ergebnisse zu bekommen
set.seed(1234)
index<-sample(1:nrow(dat), size = 0.75*nrow(dat))
trainDat<-dat[index,]
testDat<-dat[-index,]
#
#Ermittlung des E-Baums, Parameter-Tuning und Beurteilung
set.seed(1234)
ebaum01<-rpart(Churn ~ ., data=trainDat, method = "class", 
               control=c(cp=0.0005, minbucket=30))
ebaum01$cptable
#
#Auswahl Split Nummer 9 (min xerror).
set.seed(1234)
ebaum02<-rpart(Churn ~ ., data=trainDat, method = "class",
               control=c(cp=0.00144, minbucket=30))
#
##Beurteilung des Baums
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(ebaum02, newdata= testDat, type = "class")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)

#BAGGING
#Bagging durchführen
set.seed(1234)
ebaumBag <- bagging(Churn ~ ., data=trainDat, nbag=20,
                    control=rpart.control(cp=0.00144, minbucket=30))
#
#Ergebnisse prüfen
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(ebaumBag, newdata= testDat, type = "class")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)

#BOOSTING
#Gradient Boosting für Klassifikation durchführen
#
#Vorher Daten umcodieren, da die Zielgröße mit 0,1 codiert sein muss
trainDat2 <- trainDat %>% 
  mutate(Churn=dplyr::recode(Churn, "Churned"=0, "Stayed"=1))
#
set.seed(1234)
ebaumBoos <- gbm(Churn ~ ., data = trainDat2, distribution = "bernoulli", n.trees = 100)
#
#Ergebnisse prüfen/ vorher Daten umcodieren
testDat2 <- testDat %>% 
  mutate(Churn=dplyr::recode(Churn, "Churned"=0, "Stayed"=1))
#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- round(predict(ebaumBoos, newdata= testDat2, type = "response"),0)
#
#Erstellen der Confusions Matrix
confusionMatrix(data = factor(testPred), reference = factor(testDat2$Churn))


#Kreuzvalidierung mit Gridsuche
cvControl<-trainControl(method="cv", number=5, search="grid")
#
#Suchgrid mit weiteren Parametern
gbmGrid <-  expand.grid(interaction.depth = c(2,3,4,5), n.trees = c(1,2,3,4,5)*30, 
                        shrinkage= 0.1, n.minobsinnode = 30)
#
#Durchführung (Vorsicht Datensatz mit Churned und Stayed verwenden)
set.seed(1234)
ebaumBoos2 <- train(Churn ~ ., data = trainDat, method = "gbm", metric = "Accuracy",
                    trControl = cvControl, tuneGrid = gbmGrid, verbose=FALSE)
#
#Ergebnisse
ebaumBoos2
#
#Grafisch Darstellung
plot(ebaumBoos2)
#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(ebaumBoos2, newdata= testDat, type = "raw")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)


#RandomForest
#Erstes Modell generieren
set.seed(1234)
rf01<-randomForest(Churn ~ ., data = trainDat)
#
#Ergebnis
rf01

plot(rf01)

#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(rf01, newdata= testDat, type = "class")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)

#Model Tuning für Random Forest
set.seed(1234)
rf02 <- randomForest(Churn ~ ., data = trainDat, ntree=600, mtry=2, importance = TRUE)
#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(rf02, newdata= testDat, type = "class")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)

#
#Bedeutung der Einflussgrößen
varImpPlot(rf02)


#Grid&Control Parametertuning
#Kreuzvalierung mit Gridsuche
cvControl <- trainControl(method="repeatedcv", number=3, repeats=1, search="grid")
#
#Suchgrid für Parameteranzahl
rfGrid <- expand.grid(.mtry = (2:5)) 
#
#Modellschätzung
set.seed(1234)
rf03 <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa", 
              tuneGrid = rfGrid, trControl = cvControl)
#
#Ergebnis
rf03

plot(rf03)

#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(rf03, newdata= testDat, type = "raw")
#
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Churn)

#Unbalacierte Daten
#Kreuzvalidierung mit Down-Sampling
cvControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1, 
                          sampling = "down")
#
#Modellschätzung
set.seed(1234)
rfDown <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa",
                trControl = cvControl)
#
#Vorhersage und Confusion Matrix
testPred <- predict(rfDown, newdata= testDat, type = "raw")
confusionMatrix(data = testPred, reference = testDat$Churn)

#Kreuzvalidierung mit Down-Sampling
cvControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1, 
                          sampling = "up")
#
#Modellschätzung
set.seed(1234)
rfUp <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa",
              trControl = cvControl)
#
#Vorhersage und Confusion Matrix
testPred <- predict(rfUp, newdata= testDat, type = "raw")
confusionMatrix(data = testPred, reference = testDat$Churn)


#Kreuzvalidierung mit SMOTE
cvControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1, 
                          sampling = "smote")
#
#Modellschätzung
set.seed(1234)
rfSmote <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa",
                 trControl = cvControl)
#
#Vorhersage und Confusion Matrix
testPred <- predict(rfSmote, newdata= testDat, type = "raw")
confusionMatrix(data = testPred, reference = testDat$Churn)

#Basismodell erstellen
cvControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1)
set.seed(1234)
rfBasis <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa",
                 trControl = cvControl)
#
# Modellgewichte (in der Summe Eins)
weights <- ifelse(trainDat$Churn == "Churned",
                  (1/table(trainDat$Churn)[1]) * 0.5,
                  (1/table(trainDat$Churn)[2]) * 0.5)
#
#Gleiche Zufallszahlen des Basismodells für das neue Modell verwenden
cvControl$seeds <- rfBasis$control$seeds
#
#Gewichtetes Modell
rfGewicht <- train(Churn ~ ., data = trainDat, method = "rf", metric = "Kappa",
                   trControl = cvControl, weights = weights)
#
#Vorhersage und Confusion Matrix
testPred <- predict(rfGewicht, newdata= testDat, type = "raw")
confusionMatrix(data = testPred, reference = testDat$Churn)


#Kreuzvalidierung mit SMOTE
cvControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1, 
                          sampling = "smote", classProbs = TRUE, 
                          summaryFunction = twoClassSummary)
#
#Modellschätzung
set.seed(1234)
rfSmote <- train(Churn ~ ., data = trainDat, method = "rf", metric = "ROC",
                 trControl = cvControl)
#
#Vorhersage und Confusion Matrix für SMOTE und ROC
testPred <- predict(rfSmote, newdata= testDat, type = "prob")
testPred2 <- factor(ifelse(testPred[,1] > 0.5, "Churned", "Stayed"))
confusionMatrix(data = testPred2, reference = testDat$Churn)

#Durchführung der Churn-Wahrscheinlichkeit für einzelne Kunden
predict(rf03, newdata = dat[1:5,], type = "prob")


#
##Durchführung der Churn-Ergebnisse für einzelne Kunden
predict(rf03, newdata = dat[1:5,], type = "raw")






set.seed(123)
index<-sample(1:nrow(dat), size = 0.75*nrow(dat))
trainDat<-dat[index,]
testDat<-dat[-index,]

str(testPred)
str(testDat$Outcome)
levels(testPred) <- levels(testDat$Outcome)

ebaum01<-rpart(Outcome ~ ., data=trainDat, method = "class", 
               control=c(cp=0.0005, minbucket=30))
ebaum01$cptable


rf01<-randomForest(Outcome ~ ., data = trainDat)

#Ergebnis
rf01

plot(rf01)

#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(rf01, newdata = testDat, type = "class")
confusionMatrix(data = testPred, reference = testDat$Outcome)

testPred <- factor(testPred, levels = levels(trainDat$Outcome)) # Ensure it's a factor

testDat$Outcome <- factor(testDat$Outcome, levels = levels(trainDat$Outcome))

levels(testPred)
levels(testDat$Outcome)


#Erstellen der Confusions Matrix


#Ermittlung des E-Baums, Parameter-Tuning und Beurteilung

#Model Tuning für Random Forest
set.seed(1234)
rf02 <- randomForest(Outcome ~ ., data = trainDat, ntree=600, mtry=2, importance = TRUE)
#
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(rf02, newdata= testDat, type = "class")
testPred
#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Outcome)





#Auswahl Split Nummer 9 (min xerror).
set.seed(1234)
ebaum02<-rpart(Outcome ~ ., data=trainDat, method = "class",
               control=c(cp=0.00144, minbucket=30))
#
##Beurteilung des Baums
#Durchführung der Vorhersagen für Testdatensatz
testPred <- predict(ebaum02, newdata= testDat, type = "class")
View(testDat)
View(testPred)

#Erstellen der Confusions Matrix
confusionMatrix(data = testPred, reference = testDat$Outcome)