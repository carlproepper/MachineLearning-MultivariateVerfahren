
#Bibliotheken laden
library(tidyverse)
library(lubridate)
library(arules)

library(arulesViz)

#
#Daten laden
Brotkorb<-read_csv("/Users/carlpropper/ML&MV_datasets/Brotkorb.csv")
summary(Brotkorb)
colnames(Brotkorb)

#Brotkorb als Transaktionsdatei öffnen
tDat<- read.transactions("/Users/carlpropper/ML&MV_datasets/Brotkorb.csv", 
                         format="single",
                         cols=c(3,4),
                         sep = ",",
                         rm.duplicates=TRUE)
#
#Zusammenfassung
summary(tDat)

#
#Datenstruktur
str(tDat)

#EDA - Explorative Datenanalyse

#Häufigkeiten der Items grafisch darstellen
itemFrequencyPlot(tDat, 
                  topN=10, 
                  type="absolute", 
                  col="chocolate",
                  main="Item Häufigkeit")

#Häufigkeiten je Woche
Brotkorb %>%
  mutate(Wochentag=as.factor(weekdays(as.Date(Date)))) %>%
  group_by(Wochentag) %>%
  summarise(Transactions=n_distinct(Transaction)) %>%
  ggplot(aes(x=Wochentag, y=Transactions)) +
  geom_bar(stat="identity", fill="royalblue", 
           show.legend=FALSE, colour="black") +
  geom_label(aes(label=Transactions)) +
  labs(title="Transaktionen je Wochentag") +
  scale_x_discrete(limits=c("Montag", "Dienstag", "Mittwoch", "Donnerstag",
                            "Freitag", "Samstag", "Sonntag"))

#Transaktionen je Stunde
Brotkorb %>%
  mutate(Hour=as.factor(hour(hms(Time)))) %>%
  group_by(Hour) %>%
  summarise(Transactions=n_distinct(Transaction)) %>%
  ggplot(aes(x=Hour, y=Transactions)) +
  geom_bar(stat="identity", fill="peru", show.legend=FALSE, colour="black") +
  geom_label(aes(label=Transactions)) +
  labs(title="Transactionen je Stunde")

#Untersuchung Transaktion-Produkt-Muster
set.seed(123)
image(sample(tDat,100))


#Modellermittlung und Beurteilung
#Regeln für den Brotkorb (tDat)
regeln<-apriori(tDat, parameter=list(support=0.02, confidence=0.3, minlen=2),
                control=list(verbose = FALSE))
#
#Zusammenfassung
summary(regeln)

#
#Regeln anschauen
inspect(regeln)

#Sortieren der Regeln und Beschränkung
inspect(sort(regeln, by="lift")[1:5])

#
#Nur bestimmte Produkte auswählen 
inspect(subset(regeln, items %in% "Coffee"))

#GRAPHS
#Streuungsdiagramm für Kennzahlen
plot(regeln, measure=c("support","lift"), shading="confidence")
#Grafik für Regeln
plot(regeln, method="graph")
#Gruppierte Grafik
plot(regeln, method="grouped")


#Parametervariation
#Werte für Support und Konfidenz festlegen
suppLev <- c(0.1, 0.05, 0.01, 0.005)
confLev <- c(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
#
#Platzhalter (leere Integer) 
rulSup10 <- integer(length=9)
rulSup5 <- integer(length=9)
rulSup1 <- integer(length=9)
rulSup05 <- integer(length=9)
#
# Apriori Algorithmus für Support von 10%
for(i in 1:length(confLev)){
  rulSup10[i] <- length(apriori(tDat, parameter=list(sup=suppLev[1], 
                                                     conf=confLev[i], target="rules"),
                                control=list(verbose = FALSE)))
  
}
#
# Apriori Algorithmus für Support von 5%
for(i in 1:length(confLev)){
  rulSup5[i] <- length(apriori(tDat, parameter=list(sup=suppLev[2], 
                                                    conf=confLev[i], target="rules"),
                               control=list(verbose = FALSE)))
  
}
#
# Apriori Algorithmus für Support von 1%
for(i in 1:length(confLev)){
  rulSup1[i] <- length(apriori(tDat, parameter=list(sup=suppLev[3], 
                                                    conf=confLev[i], target="rules"),
                               control=list(verbose = FALSE)))
  
}
#
# Apriori Algorithmus für Support von 0,5%
for(i in 1:length(confLev)){
  rulSup05[i] <- length(apriori(tDat, parameter=list(sup=suppLev[4], 
                                                     conf=confLev[i], target="rules"),
                                control=list(verbose = FALSE)))
  
}
#
#Data frame erstellen
anzRul <- data.frame(rulSup10, rulSup5, rulSup1, rulSup05, confLev)
#
#Anzahl der gefundenen Regeln für verschiedene Support Levels
ggplot(data=anzRul, aes(x=confLev)) +
  #
  #Linie und Punkte für Support 10%
  geom_line(aes(y=rulSup10, colour="Support Level von 10%")) + 
  geom_point(aes(y=rulSup10, colour="Support Level von 10%")) +
  #
  #Linie und Punkte für Support 5%
  geom_line(aes(y=rulSup5, colour="Support Level von 5%")) + 
  geom_point(aes(y=rulSup5, colour="Support Level von 5%")) +
  #
  #Linie und Punkte für Support 1%
  geom_line(aes(y=rulSup1, colour="Support Level von 1%")) + 
  geom_point(aes(y=rulSup1, colour="Support Level von 1%")) +  
  #
  #Linie und Punkte für Support 0.5%
  geom_line(aes(y=rulSup05, colour="Support Level von 0.5%")) + 
  geom_point(aes(y=rulSup05, colour="Support Level von 0.5%")) +
  #
  #Beschriftungen und Thema
  labs(x="Konfidenz Levels", y="Anzahl gefundene Regeln", 
       title="Apriori Algorithmus mit unterschiedlichem Support") +
  theme_bw() +
  theme(legend.title=element_blank())



#"Optimale" Regeln für den Brotkorb (tDat)
regeln2<-apriori(tDat, parameter=list(support=0.01, confidence=0.5, 
                                      minlen=2, target="rules"), 
                 control=list(verbose = FALSE))
#Zusammenfassung
summary(regeln2)


#
#Regeln anschauen
inspect(sort(regeln2, by="lift"))


#Ergebnis mit vielen Regeln für den Brotkorb (tDat)
regeln3<-apriori(tDat, parameter=list(support=0.005, confidence=0.2, 
                                      minlen=2, target="rules"), 
                 control=list(verbose = FALSE))
#Zusammenfassung
summary(regeln3)

#
#Redundante Regeln entfernen
summary(regeln3[!is.redundant(regeln3)])
#
#Wichtigste Regeln anschauen
inspect(sort(regeln3[!is.redundant(regeln3)], by="lift")[1:10])

