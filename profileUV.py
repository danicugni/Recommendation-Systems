from memory_profiler import profile
import time
import preProcessing as prep
import listOfLists as lol
import datasetPrepTrainSet as dptrs
import datasetUseSet as dus
import utilityMatrixCentering as ucm
import numpy as np
import sgd
import math

report =open("memoryLogSgdUV.log", "w+") #creo un file dove scrivere il report del memory profile
@profile(precision=2, stream=report) #fa il profile di sgd con due cifre dopo la virgola e scrivendo il report sul file memoryLogSgdUV.log
def sGD(matrix, alpha, regularization,latentFactors, epsilon, maxIter,seed1=None,seed2=None): #dati la matrice dei rating osservati,il learning rate alpha
    #il parametro di regolarizzazione, il numero di fattori latenti
    latentFactors=int(latentFactors)
    start = time.process_time() #istante temporale iniziale di esecuzione dello sgd
    n = matrix.shape[0] #numero di user
    p = matrix.shape[1] #numero di item
    indices = np.arange(len(matrix.data)) #lista di indici da 0 a (numero di rating osservati -1)
    mean = ucm.meanCoo(matrix) #calcola la media dei rating osservati centrati
    standardDev = np.sqrt(np.var(matrix.data)) #calcola la deviazione standard dei rating osservati
    print(standardDev)
    if(mean <= 0): #se la media empirica è minore o uguale a 0
        U = np.zeros((n, latentFactors), dtype = np.float32)  #si inizializza U con tutti 0
        V = np.zeros((latentFactors, p), dtype = np.float32) #si inizializza V con tutti 0
    else: #se la media empirica è maggiore di 0
        U = np.full((n, latentFactors), math.sqrt(mean / latentFactors), dtype=(np.float32)) #si inizializza U con la radice quadrata del
                                                    # rapporto tra la media empirica e il numero di dimensioni latenti
        V = np.full((latentFactors, p), math.sqrt(mean / latentFactors), dtype=(np.float32)) #si inizializza V con la radice quadrata del
                                                    # rapporto tra la media empirica e il numero di dimensioni latenti
    U = sgd.perturbation(U, mean, standardDev, seed1) #si perturba la matrice U generando casualmente da una Normale con media
    #la media dei rating osservati centeati e deviazione standard pari alla deviazione standard dei rating osservati
    V = sgd.perturbation(V, mean, standardDev, seed2) #si perturba la matrice V generando casualmente da una Normale con media
    #la media dei rating osservati centeati e deviazione standard pari alla deviazione standard dei rating osservati
    print("dimensioni Matrice",matrix.shape)
    print("dimensioni U:", U.shape)
    print("dimensioni V:", V.shape)
    print("Inizio a fare il prodotto tra U e V")
    print("Inizio a calcolare il primo mse")
    mse = sgd.mseCoo(matrix, U, V) #calcola il primo mse
    print("primo mse calcolato")
    print(mse)
    convergence = False #poniamo a False la variabile usata per far stoppare l'algoritmo quando quest'ultimo converge
    numIter = 0 #variabile che tiene conto del numero di iterazioni posta pari a 0
    numObs = len(matrix.data) #numero totale di osservazioni
    print(numObs)
    np.random.seed(1) #fissiamo il seed a 1
    while not convergence and numIter < maxIter: #fino a quando non si verifica la condizione di convergenza o non si
        #raggiunge il numero massimo di iterazioni ammesse
        numIter += 1 #incrementa di uno il numero di iterazioni effettuate
        print("iterazione numero: ", numIter)
        np.random.shuffle(indices) #mescola in modo casuale la lista di indici
        for a in indices: #per tutti gli indici nel nuovo ordine
            j=matrix.col[a] #colonna del rating che si sta visitando
            i=matrix.row[a] #riga del rating che si sta visitando
            error = matrix.data[a] - (np.dot(U[i, :], V[:, j])) #calcola l'errore come la differenza tra rating
            #osservato e rating stimato
            U[i, :] = (1 - alpha*regularization) * U[i, :] + (alpha * error * V[:, j]) #aggiorna la corrispondente riga di U
            V[:, j] = (1 - alpha*regularization) * V[:, j] + (alpha * error * U[i, :]) #aggiorna la corrispondente colonna di V

        newMse = sgd.mseCoo(matrix, U, V) #calcola il nuovo mse
        
        deltaMse = -(newMse - mse) / mse #calcola la differenza tra il nuovo mse ed il vecchio mse
        print("nuovo mse: ", newMse)
        print("delta: ", deltaMse)
        mse = newMse # aggiorna il valore di mse con quello corrente
        if np.abs(deltaMse) < epsilon:  #se l'incremento è sotto la soglia,
            convergence = True #aggiorno "convergenza"
            Rapprox = np.dot(U, V) #effettuo il prodotto matriciale tra U e V
    if (numIter==maxIter):
        Rapprox = np.dot(U, V) #effettuo il prodotto matriciale tra U e V
    stop = time.process_time() #istante temporale finale di esecuzione dello sgd
    executTime = stop - start #tempo di esecuzione dello sgd
    print(stop)
    print(start)
    print(executTime)
    return Rapprox, executTime,numIter,latentFactors,alpha,regularization,epsilon,newMse #ritorna la matrice delle stime e il tempo di esecuzione

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
    print('UVApprox training')
    ratingsFileTrain = open(fileTrain, 'r')
    matrixTrainUV= dus.loadDenseCsrMatrix(ratingsFileTrain)
    print("il massimo dei rating della matrice è: ",max(matrixTrainUV.data))#solo di controllo
    vectMeanRow=[]
    vectMeanCol=[]
    matrixTrainUV=ucm.centeringMatrix(matrixTrainUV,vectMeanRow)[0]
    matrixTrainUV=matrixTrainUV.tocsc()
    matrixTrainUV=ucm.centeringMatrix(matrixTrainUV,vectMeanCol)[0]
    matrixTrainUV=matrixTrainUV.tocoo()
    sGD(matrixTrainUV, 0.004, 0.8,15, 0.005, 100,seed1=123,seed2=456)