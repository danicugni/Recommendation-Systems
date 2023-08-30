# funzione lista di liste utenti con piu' di 20 recensioni, divisa in train, validation e test
import numpy as np

def listOfLists(user, item, beerAbv, overall, tresholdTrain, tresholdValidation): 
    # user, item... liste con userId, itemId, ... recensioni
    #treshold train percentuale dati da tenere nel train [0,1], treshold validation percentuale dati da tenere nel validation [0,1]
    trainLists = [] #creo una lista vuota per train
    validationLists = [] #creo una lista vuota per validation
    testLists = [] #creo lista vuota per il test
    rUsers, nOcc = np.unique(user, return_counts=True) #trovo lista user unici e numero recensioni lasciate da ciascuno
    occurences = [0]*len(rUsers) #creo lista di zeri che usero per contare numero occorrenze recensioni di uno user
    limitTrain = nOcc*tresholdTrain #fisso per ogni user soglia recensioni da tenere nel train
    limitValidation = nOcc*(tresholdTrain+tresholdValidation)
    n = len(user) #numero totale stringhe
    for i in range(n) : #per ogni stringa
            index = int(np.argwhere(rUsers==user[i])[0]) #trovo indice user che ha lasciato recensione
            occurences[index] +=1 #aggiungo uno alla lista che conta occcorrenze
            choiceTrain = (occurences[index] <= limitTrain[index]) #vedo se il contatore è minore della soglia Train
            if (beerAbv[i] != "-"): # una gradazione alcolica ha trattino, se non è quella
                Abv = float(beerAbv[i])
            else:
                Abv = None
            if (choiceTrain==True): # se contatore è minore tresholdTrain metti le liste nella lista di liste train
                trainLists.append([user[i], item[i], Abv, int(overall[i].split('/')[0])])
            elif (occurences[index] <= limitValidation[index]): #altrimenti se minore tresholdTrain+tresholdValidation metti le liste nella lista di liste validation
                validationLists.append([user[i], item[i], Abv, int(overall[i].split('/')[0])])
            else: #altrimenti metti nella lista di liste test
                testLists.append([user[i], item[i], Abv, int(overall[i].split('/')[0])])
            print(i)
    return (trainLists, validationLists ,testLists) #ritorno lista liste train e test

