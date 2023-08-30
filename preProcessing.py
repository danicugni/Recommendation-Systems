import re
import numpy as np

#funzione per ottenere 8 array dove in ciascun array sono contenuti i valori osservati delle variabili di interesse
def getRegex(data):
    regexItem = (r'beer/beerId:\s+(.*)') #regex per ottenere i beerId osservati
    regexBeerAbv = (r'beer/ABV:\s+(.*)') #regex per ottenere i beerABV osservati
    regexOver=(r'review/overall:\s+(.*)') #regex per otteneregli overall (rating totale) osservati
    regexUser = (r'review\/profileName:\s+(.*)') #regex per ottenere i profileName (nome degli utenti) osservati
    item= np.asarray(re.findall(regexItem,data)) #otteniamo e trasformiamo in array gli item presenti nel dataset
    beerAbv=np.asarray(re.findall(regexBeerAbv,data)) #otteniamo e trasformiamo in array i beerAbv presenti nel dataset
    overall=np.asarray(re.findall(regexOver, data)) #otteniamo e trasformiamo in array gli overall (rating totali) presenti nel dataset
    user= np.asarray(re.findall(regexUser,data)) #otteniamo e trasformiamo in array gli user presenti nel dataset
    return (item,beerAbv,overall,user) #ritorniamo tutti gli array


#funzione che ritorna la lista di coppie (user,item) unite (ovvero useritem) 
def zipped(user,item): #date la lista di user e item
    listOfCouple=zip(user,item) #crea un array in cui l'i-esimo elemento è dato dall'i-esimo elemento di user e dall'i-esimo elemento di item
    listOfCouple=list(listOfCouple) #converti in lista
    listOfCouple=np.asarray(listOfCouple) #converti in array
    listOfCoupleUnited=[] #crea una lista vuota
    for i in range(listOfCouple.shape[0]): #per ogni elemento dell'array
        listOfCoupleUnited.append("".join(listOfCouple[i])) #elimina la virgola che separa la coppia
    return listOfCoupleUnited #ritorna la lista di coppie (user,item) unite (ovvero useritem)


#funzione che ritorna l'indice in cui ciascun utente compare la prima volta        
def firstAppearanceUserItem(listOfCoupleUnited):
    mask=np.unique(listOfCoupleUnited,return_index=True)[1]
    return mask

#funzione che, date delle liste di partenza e una lista di indici, ritorna le corrispondenti
#sottoliste definite sulla base della lista di indici
def subLists(mask,item,beerAbv,overall,user):
    item=item[mask]
    beerAbv=beerAbv[mask]
    overall=overall[mask]
    user=user[mask]
    return (item,beerAbv,overall,user)

#funzione per ottenere la lista degli utenti univoci e il corrispondente numero di occorrenze per ognuno
def uniqueUsers(user):
    uUsers = np.unique(user, return_counts= True) #ottengo la lista di utenti unici con il numero di occorrenze per ognuno
    return uUsers #ritorno la lista e il numero di occorrenze

#funzione per ottenere la lista di user con più di 20 recensioni
def nameUsersOver(uUsers):
    rUsersName =uUsers[0][uUsers[1] > 20] #nomi di chi ha fatto più di 20 recensioni
    return rUsersName

#funzione che ritorna la lista di indici dell'array relativa agli user da tenere
def indexOfArray(user,rUsersName):
    mask=np.argwhere(np.in1d(user,rUsersName)).flatten() 
    return mask



