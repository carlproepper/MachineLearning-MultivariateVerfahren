#Naive Bayes

#Benötigte Packages laden
library(tidyverse)

library(e1071)
library(tm)

library(SnowballC)
library(wordcloud)

library(caret)

#
#Datensatz laden (http://archive.ics.uci.edu/ml/datasets.php)
spamDat <- read_csv2("/Users/carlpropper/ML&MV_datasets/SMSSpamCollection.csv", col_names = FALSE) %>% 
  rename(type = X1, text = X2) %>% 
  mutate_at(vars(type), factor)

#
#Datensatz
spamDat

#Spamhäufigkeit
table(spamDat$type)

#Korpus erstellen zur Datenbereinigung mit Package tm
spamKorp <- VCorpus(VectorSource(spamDat$text)) %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stemDocument) %>% 
  tm_map(stripWhitespace)
#
#Ergebnis anzeigen
lapply(spamKorp[1:2], as.character)

#
#In DokumentTermMatrix umwandeln
spamDTM <- DocumentTermMatrix(spamKorp)
spamDTM


#Datensatz in Trainings- und Testdatensatz aufteilen
#Grenze festlegen
n<-dim(spamDTM)[1]
grenze <- floor(0.7*n)
#
#DTM Daten aufteilen
trainDatDTM<-spamDTM[1:grenze,]
testDatDTM<-spamDTM[(grenze+1):n,]
#
#Labels (type) aus Originaldatensatz
trainDatLab<-spamDat[1:grenze,]$type
testDatLab<-spamDat[(grenze+1):n,]$type

#Wordcloud erstellen
wordcloud(spamKorp, min.freq = 100, random.order = FALSE)

#Daten nach spam und ham aufteilen
spam <- subset(spamDat, type == "spam")
ham <- subset(spamDat, type == "ham")
#
#Wordclouds erstellen
wordcloud(spam$text, max.words = 50, scale = c(3, 0.5))

wordcloud(ham$text, max.words = 50, scale = c(3, 0.5))

#Durchfuerung des Algorithmus
#Vorbereitung
#Häufigste Wörter auswählen mit minimaler Häufigkeit von 7
smsTerms <- findFreqTerms(trainDatDTM, 7)
#
#Nur Häufige Wörter auswählen
trainDatDTMFreq<-trainDatDTM[,smsTerms]
testDatDTMFreq<-testDatDTM[,smsTerms]
#
#Umwandlung numerischer Daten mit selbstgeschriebener Funktion
convCount<-function(x){
  x<-ifelse(x>0, "Yes", "No")
}
#
smsTrain<-apply(trainDatDTMFreq, MARGIN=2, convCount)
smsTest<-apply(testDatDTMFreq, MARGIN=2, convCount)


#Verfahren durchführen
smsErg <- naiveBayes(smsTrain, trainDatLab)
#
#Prognosen erstellen
smsTestPred <- predict(smsErg, smsTest, "class")
#
#Konfusionsmatrix
confusionMatrix(smsTestPred, testDatLab)


#Häufigste Wörter auswählen mit minimaler Häufigkeit von 5
smsTerms2 <- findFreqTerms(trainDatDTM, 5)
#
#Nur Häufige Wörter auswählen
trainDatDTMFreq2<-trainDatDTM[,smsTerms2]
testDatDTMFreq2<-testDatDTM[,smsTerms2]
#
#Umwandlung numerischer Daten mit selbstgeschriebener Funktion
convCount<-function(x){
  x<-ifelse(x>0, "Yes", "No")
}
#
smsTrain2<-apply(trainDatDTMFreq2, MARGIN=2, convCount)
smsTest2<-apply(testDatDTMFreq2, MARGIN=2, convCount)


#Verfahren durchführen
smsErg2 <- naiveBayes(smsTrain2, trainDatLab, laplace = 1)
#
#Prognosen erstellen
smsTestPred2 <- predict(smsErg2, smsTest2)
#
#Konfusionsmatrix
confusionMatrix(smsTestPred2, testDatLab)


#Anwendung
#Neue Email
neuEmail<-c("Dear Sirs, get free access to our offer. Just call 0711800454.")
#
#Neue Email bearbeiten analog bei Modellentwicklung
neuEmailKorp <- VCorpus(VectorSource(neuEmail)) %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stemDocument) %>% 
  tm_map(stripWhitespace)
#
##Ergebnis auf Wörter des Trainingssatzes beziehen!!!
#Ergebnis anzeigen
lapply(neuEmailKorp, as.character)


#
#In DokumentTermMatrix umwandeln
neuEmailDTM <- DocumentTermMatrix(neuEmailKorp)
#
#Umwandlung numerischer Daten in Character mit selbstgeschriebener Funktion
convCount<-function(x){
  x<-ifelse(x>0, "Yes", "No")
}
#
neuEmailChar<-apply(neuEmailDTM, MARGIN=2, convCount)
#
#Vorhersagevektor erzeugen
Jas <- which(dimnames(smsTrain)$Terms  %in% names(neuEmailChar))
anzWort <- dim(smsTrain)[2]
neuEmailChar2 <- smsTrain[1,] %>% 
  rep("No", anzWort)
neuEmailChar2[Jas]<-"Yes"
#
#Prognosen erstellen
predict(smsErg, as.data.frame(t(neuEmailChar2)))

predict(smsErg, as.data.frame(t(neuEmailChar2)), type="raw")