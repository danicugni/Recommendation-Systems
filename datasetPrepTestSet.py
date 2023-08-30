def userItemTestSetMap(listOfLists,itemMapTrain,userMapTrain,file):
    #userId e itemId vengono mappati da 1 al numero di user/item presente
    #creo due dizionari, mappo id originario in un nuovo id numerico
    
    ratingsFile = open(file, 'w') #apro in scrittura un file in cui costruirò la tripla (item,user,rating)
    for review in listOfLists:#scorro la lista di liste del test set ricevuta in input
        userId = review[0] #lo user corrisponde al primo elemento della lista
        itemId = review[1] #l'item corrisponde al secondo elemento della lista
        rating = review[3] #il rating corrisponde all'ottavo elemento della lista
    
        if itemId in itemMapTrain.keys(): #controllo se l'item è presente nel dizionario creato per il training test
            itemIdx = itemMapTrain.get(itemId) #se è già inserito assegna la chiave corrispondente
            userIdx = userMapTrain.get(userId)
            ratingsFile.write(','.join([str(itemIdx), str(userIdx), str(rating)])) #scrivo nel file aperto in precedenza la tripla considerata, separata da 
            ratingsFile.write('\n') #separo una tripla dall'altra con un 'a capo'
        else:
            pass #se non è inserito prosegui all'istruzione successiva
    ratingsFile.close() #chiudo il file 