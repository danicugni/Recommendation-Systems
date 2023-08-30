
def userItemMap(listOfLists,text):
    #userId e itemId vengono mappati da 1 al numero di user/item presente
    newItemIdx = 0 #variabili di appoggio con indice iniziale
    newUserIdx = 0#primo utente che trovo lo mappo in 0, indice di riga
    
    #creo due dizionari, uno per user e uno per item, mappo ID originario in un nuovo ID numerico progressivo
    itemIdxMap = {}
    userIdxMap = {}
    
    ratingsFile = open(text, 'w') #apro in scrittura un file in cui costruirò la tripla (item,user,rating)
    
    for review in listOfLists: #scorro la lista di liste del validation/train set ricevuta in input
        userId = review[0] #lo user corrisponde al primo elemento della lista
        itemId = review[1] #l'item corrisponde al secondo elemento della lista
        rating = review[3] #il rating corrisponde all'ottavo elemento della lista
    
        if itemId in itemIdxMap.keys(): #controllo se l'item è già stato inserito come chiave del dizionario
            itemIdx = itemIdxMap.get(itemId) #se è già inserito assegna la chiave corrispondente
        else:#assegno come chiave l' ID originale  dell'item e come valore quello corrispondente alla variabile di appoggio newItemIdx
            itemIdxMap[itemId] = newItemIdx
            itemIdx = newItemIdx 
            newItemIdx += 1 #incremento la variabile di appoggio di uno, che controlla il numero di item univoci presenti
        if userId in userIdxMap.keys(): #lo stesso controllo viene fatto per lo user
            userIdx = userIdxMap.get(userId)
        else:
            userIdxMap[userId] = newUserIdx
            userIdx = newUserIdx
            newUserIdx += 1
        ratingsFile.write(','.join([str(itemIdx), str(userIdx), str(rating)])) #scrivo nel file aperto in precedenza la tripla considerata, separata da 
        ratingsFile.write('\n') #separo una tripla dall'altra con un 'a capo'
    ratingsFile.close() #chiudo il file 
    return userIdxMap, itemIdxMap #ritorno i dizionari dove ho mappato user e item univoci


        
