import numpy as np

#dataset dataset sottoforma di lista di liste (user, item, Abv, rating)
#userIdIdxMap mappatura degli user del trainset (dizionario con chiave userid)
#userIdIdxMap mappatura degli item del trainset (dizionario con chiave itemid)

#crea due vettori che rappresentano la tabella del factorization machine in formato dimil csr per il test set 
#mantenendo la struttura della tabella del train
def csrTableTest(testset, userIdIdxMap, itemIdIdxMap):

    nUsers = len(userIdIdxMap)#numero user dataset
    nItem = len(itemIdIdxMap)#numero item dataset
    count=0

    ind = [] #inizializzo lista che contiene indici colonna della tabella
    val = [] #inizializzo lista che contiene valori diversi da zero tabella
    indPr = [] #inizializzo lista che contiene numero valori diversi da zero per riga tabella cumulati
    perRow=0 #contatore valori diversi da zero
    
    #ciclo per costruire tabella (struttura per matrici sparse) per factorization machine
    for review in testset: #per ogni recensione nel test set
        itemId = review[1] #prende l'itemId birra
        if itemId in itemIdIdxMap.keys(): #se birra presente nel train
            userId = review[0] #prende user id
            ABV = review[2] #gradazione alcolica
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
            print(count)
            count += 1 #incremento contatore recensioni testset
        else: #se birra non è nel train
            pass #non faccio niente
    return [np.array(val, np.int32), np.array(ind, np.int32), np.array(indPr, np.int32)]
#ritorna array con i valori diversi da zero nella tabelle, vettoren con gli indici di colonna di questi vettori