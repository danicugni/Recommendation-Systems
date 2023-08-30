import numpy as np
import math
import utilityMatrixCentering as ucm
import time

#calcolo del MSE per matrice in formato COO

def mseCoo(matrixObs, U, V): #dati la matrice dei rating osservati, le matrici U e V
    sse = 0 #si inizializza a 0 il numeratore del mse
    for i in range(len(matrixObs.data)): #per tutti i rating osservati
        sse += ((matrixObs.data[i] - np.dot(U[matrixObs.row[i], :], V[:, matrixObs.col[i]]))**2) #incrementa sse di
        #una quantità pari alla differenza, elevata al quadrato, tra rating osservato e rating stimato
    return sse/len(matrixObs.data) #ritorna il mse facendo il rapporto tra sse e il numero di rating osservati

#perturbazione di una matrice
def perturbation(matrix, mean,sigma, seed): #dati una matrice, una media, una deviazione standard e un seme
    if (seed != None):
        np.random.seed(seed)#fissiamo il seed
    else:
        pass
    mNormal = np.random.normal(mean, sigma, size=(matrix.shape)) #generiamo una matrice (di dimensione pari alle dimensioni
    #della matrice data in input) di valori casuali da una Normale con specifiche media e deviazione standard
    matrix += mNormal #aggiungiamo a ciascun elemento della matrice di partenza il corrispondente valore casuale generato
    return matrix      #ritorniamo la nuova matrice


#implementazione dello sgd
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
    U = perturbation(U, mean, standardDev, seed1) #si perturba la matrice U generando casualmente da una Normale con media
    #la media dei rating osservati centeati e deviazione standard pari alla deviazione standard dei rating osservati
    V = perturbation(V, mean, standardDev, seed2) #si perturba la matrice V generando casualmente da una Normale con media
    #la media dei rating osservati centeati e deviazione standard pari alla deviazione standard dei rating osservati
    print("dimensioni Matrice",matrix.shape)
    print("dimensioni U:", U.shape)
    print("dimensioni V:", V.shape)
    print("Inizio a fare il prodotto tra U e V")
    print("Inizio a calcolare il primo mse")
    mse = mseCoo(matrix, U, V) #calcola il primo mse
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

        newMse = mseCoo(matrix, U, V) #calcola il nuovo mse
        
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

def addRowMeans(matrixApprox, vectorMeans): #dati la matrice dei rating stimati e il vettore delle medie di riga
    for i in range(matrixApprox.shape[0]):#per ogni riga della matrice dei rating stimati
        matrixApprox[i,:] += vectorMeans[i] #aggiungi la media di riga
    return matrixApprox #ritorna la nuova matrice dei rating stimati

def addColMean(matrixApprox, vectorMeans): #dati la matrice dei rating stimati e il vettore delle medie di colonna
     matrixApprox += vectorMeans #aggiungi ad ogni colonna della matrice dei rating stimati la corrispondente media
     return matrixApprox        #ritorna la nuova matrice dei rating stimati












    


