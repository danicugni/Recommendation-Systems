import numpy as np
#dataset dataset sottoforma di lista di liste (user, item, Abv, rating... e le altre esplicative)
#userIdIdxMap un mappatura degli user del trainset o validation (dizionario con chiave userid)
#userIdIdxMap un mappatura degli item del trainset o validation(dizionario con chiave itemid)

#creo un dizionario che rappresenta la tabella del factorization machine in formato csr 
def dictCsr(listOfLists, userIdIdxMap, itemIdIdxMap): 
    nUsers = len(userIdIdxMap) #numero user dataset
    nItem = len(itemIdIdxMap) #numero item dataset

    ind = [] #inizializza lista che contiene indici colonna della tabella dei valori diversi da zero
    val = [] #inizializza lista che contiene valori diversi da zero della tabella
    indPr = [] #inizializza lista che contiene numero valori diversi da zero per riga tabella cumulati
    perRow=0 #contatore valori diversi da zero
    
    for review in listOfLists: #per ogni recensione nel training set
        userId = review[0] #prendo l'id dello user
        itemId = review[1] #prendo l'id della birra
        ABV = review[2] #gradazione alcolica birra
        rating = review[3] #valutazione "overall" della birra (da 1 a 20)

        ind.append(userIdIdxMap.get(userId)) #aggiungo indice di colonna corrispondente allo user (sfrutto mappatura user)
        val.append(1) #aggiungo 1 a lista dei valori diversi da zero
        ind.append(itemIdIdxMap.get(itemId)+nUsers) #aggiungo indice di colonna corrispondente all' item (sfrutto mappatura item dopo le colonne per gli user(nUsers))
        val.append(1)
        perRow+=2
        if ABV == None or ABV < 5 : #se la gradazione alcolica è sotto i 5 gradi
            ind.append(nItem+nUsers) #aggiungo indice di colonna
            val.append(1)
            perRow+=1
        elif ABV >=5 or ABV <6: #se la gradazione è tra 5 e 6 escluso
            ind.append(nItem+nUsers+1) #aggiungo indice di colonna
            val.append(1)
            perRow+=1
        elif ABV >=6 or ABV <8: #se la gradazione è tra 6 e 8 escluso
            ind.append(nItem+nUsers+2) #aggiungo indice di colonna
            val.append(1)
            perRow+=1
        else:
            pass
        ind.append(nItem+nUsers+3) #aggiungo indice di colonna
        val.append(rating) #aggiungo valore rating overall
        perRow+=1
        indPr.append(perRow) #aggiungo numero valori diversi da zero cumulato
    dictCsr = {'value':np.array(val, np.int32), 'colIndex':np.array(ind, np.int32), 'rowInd':np.array(indPr, np.int32)} #creo il dizionario con i tre array costruiti
    return dictCsr #ritorno il dizionario creato