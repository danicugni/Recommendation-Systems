import time
import random as rn
import dataTable as dt
import factorizationMachine as fm
import preProcessing as prep
import listOfLists as lol
import datasetPrepTrainSet as dptrs
import numpy as np
from memory_profiler import profile

report = open("memoryLogSgdFM.log", "w+") #creo un file dove scrivere il report del memory profile
@profile(precision=2, stream=report) #fa il profile di sgd con due cifre dopo la virgola e scrivendo il report sul file memoryLogSgdFM.log
def sgdFm(T, alpha, lamb, sigma, nFact, epsilon, nIter, seed=None):
    start = time.process_time() #fissa istante inizio esecuzione
    print('inizio algoritmo')
    print('ottengo indici riga, colonna e valori')
    rInd = T.get('rowInd') #vettore array con valori diversi da zero per riga cumulati
    values = T.get('value') #vettore array con valori diversi da zero tabella
    colInd = T.get('colIndex') #vettore array con indici colonna tabella dei non zeri
    print('ottengo numero features, parametri e recensioni')
    p = max(colInd) #parametri totali 
    nReviews = len(rInd) #recensioni totali
    print('Numero parametri da stimare: ', p,'Recensioni totali: ', nReviews)
    print('inizializzo omega0 e omega')
    omega0 = 0 #bias globale
    omega = np.zeros(p) #vettore bias singola colonna tabella
    if seed != None: #se si vuole fissare il seed per replicare esperimento
        print('fisso il seed =', seed)
        np.random.seed(seed)
        rn.seed(seed)
    print('inizializzo V')
    V = np.random.normal(0, sigma, size=(p,nFact)) #matrice bias a coppie
    count = 0 #contatore iterazioni
    delta = epsilon + 0.01
    indices = np.arange(nReviews)
    y = fm.obsRatings(values, rInd) #array rating osservati
    print('vettore ratings osservati')
    msePrev=0
    while (abs(delta) > epsilon and count < nIter): #finche' si e' dentro condizioni per iterare
        np.random.shuffle(indices)
        for review in indices:
            if review !=0:
                ind = colInd[rInd[review-1]:(rInd[review]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione i
            else:
                ind = colInd[0:(rInd[0]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione 0 (prima recensione)
            y_i = values[rInd[review]-1] #si ottiene valore rating osservato della review estratta
            omega0 = fm.updOmega0(omega0, alpha, lamb[0], y_i, ind, omega, V) #aggiorna omega0
            for j in  ind: #per tutte le variabili con valore xj diverso da zero
                omega[j] = fm.updOmegaj(omega0, alpha, lamb[1], y_i, ind, omega, V, j) #aggiorna omega_j
                for f in range(nFact): # per tutte le variabili latenti
                    V[j,f] = fm.updVjf(omega0, alpha, lamb[2],  y_i, ind, omega, V, j, f) #aggiorna V_j,f
        mse=0
        sumVMatrix=fm.sumV(V, ind, f)[0] #calcola componente dentro sommatoria (parte matriciale) equazione modello
        y_hat = fm.yHat(colInd, omega0, omega, V, rInd, sumV=sumVMatrix)[0] #array previsioni rating con parametri aggiornati
        mse = ((y-y_hat)**2).sum()/nReviews #calcola mse 
        delta = (msePrev-mse)/msePrev # calcola differenza relativa con mse passo precedente
        msePrev=mse
        count += 1 #aggiorno numero iterazioni
        print('f.perdita: ',mse)
        print('delta: ', delta)
        print('iterazione numero: ', count, '\n')
    stop = time.process_time() #istante temporale finale di esecuzione fm
    executTime = stop - start #tempo di esecuzionefm
    print('istante fine esecuzione: ',stop)
    print('istante inizio esecuzione: ',start)
    print('tempo esecuzione: ',executTime)
    return [omega0, omega, V, executTime, count, mse] #stampa parametri aggiornati, teempo di esecuzione, mse finale e nIterazioni per ottenere convergenza

if __name__ == '__main__':
    file=input('Inserire il nome del file con estensione annessa (.txt): ')
    myString = open(file, "r") #leggo il file di testo test
    data = myString.read() #prendo il file letto
    item, beerAbv, overall, user = prep.getRegex(data)
    print("Ho caricato le variabile del dataset")
    listOfCoupleUnited= prep.zipped(user, item)
    mask = prep.firstAppearanceUserItem(listOfCoupleUnited)
    item, beerAbv, overall, user = prep.subLists(mask,item, beerAbv, overall, user)
    print("Eliminate le recensioni doppie")
    uUsers = prep.uniqueUsers(user)
    print("Il quantile della distribuzione del numero di recensioni lasciate dal singolo utente: ",np.quantile(uUsers[1],0.77))
    rUsersName = prep.nameUsersOver(uUsers)
    mask = prep.indexOfArray(user, rUsersName)
    item, beerAbv, overall, user = prep.subLists(mask,item, beerAbv, overall, user)
    print("Eliminati utenti con meno di 20 recensioni")
    train = lol.listOfLists(user, item, beerAbv, overall, 0.7, 0.05)[0]
    fileTrain=input('Inserire il nome del file con estensione (.txt) che conterrà le triple (item,user,rating) del training set: ')
    userMapTrain,itemMapTrain=dptrs.userItemMap(train, fileTrain)
    tableAsDict = dt.dictCsr(train, userMapTrain, itemMapTrain)
    sgdFm(tableAsDict, 0.001, [0.1, 0.4, 0.7], 0.001, 1, 0.05, 100, seed=1)