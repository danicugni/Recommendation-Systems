#L'equazione del modello a cui si fa riferimenot è l'equazione sotto la 8.11 del libro "Recommender
#Systems" di Charu C. Aggarwal
#Per la stima dei parametri si è fatto riferimento all'equazione 7 e all'algoritmo 1 (sgd)
#del paper "Factorization Machines with {libFM}" di Steffen Rendle (2012)

#omega0 singolo parametro globale modello
#omega vettore parametri p-dimensionale 
#V matrice parametri p*k (p numero parametri modello, f numero fattori latenti)
#listOfIndex indici colonna tabella con valori diversi da zero per recensione i
#factor fattore di riferimento
#sumVM componente sommatoria (parte matriciale) equazione modello
#alpha learning rate
#lamb parametro regolarizzazione
#obsReview rating osservato
#column colonna tabella dei dati corrispondente al parametro omega j-esimo (j-esima colonna)
#values array contenente tutti i valori diversi da zero nella tabella dei dati
#nReviews numero totale recensioni
#dataset dataset sottoforma di lista di liste
#T dizionario contente tabella dei dati in formato csr (dizionario con tre np.array)
#sigma sigma utilizzato per generare da normale (0,sigma) i valori della matrice V
#nFact numero fattori latenti utilizzati nel modello
#epsilon soglia per fermare otimizzazione parametri modello tramite sgd
#nIter numero iterazioni massime
#seed seed per fissare e rendere ripetibile esperimento

import numpy as np
import time

#funzione per calcolare ls nell'equazione del modello
def lS(V, listOfIndex, factor):
    L = sum(V[listOfIndex,factor]) #somma i parametri di V per recensione i e fattore f
    return L # ritorna la somma

#funzione per calcolare componente sommatoria (parte matriciale) equazione modello
def sumV(V, listOfIndex, factor=None):
    k = V.shape[1] # numero fattori latenti
    s = 0 #inizializzo la componente della sommatoria
    L = 0 #inizializzo 
    for j in range(k): #per tutti i fattori
        s += (lS(V, listOfIndex, j)**2)-sum(np.power(V[listOfIndex,j], 2)) #aggiungo il loro contributo alla somma
        if (j == factor): #se j e' il fattore passato come argomento funzione
            L = lS(V, listOfIndex, factor) # calcola e tiene lS
    return [s,L] #ritorna s ed lS

#funzione per previsione tramite equazione modello
def factorizationMachine(listOfIndex, omega0, omega, V, sumVM=None, factor=None): 
    if sumVM == None: #se sumVM non passato come argomento
        pred = omega0 + sum(omega[listOfIndex]) + 0.5*sumV(V, listOfIndex, factor)[0] #calcola il valore dell'equazione del modello, ossia previsione rating
    else: #altrimenti
        pred = omega0 + sum(omega[listOfIndex]) + 0.5*sumVM #calcola il valore dell'equazione del modello, ossia previsione rating
    return [pred,sumV(V, listOfIndex, factor)[1]] #ritorna il valore della previsione e valore sumV

#funzione per aggiornare il valore del parametro omega0 con gradiente
def updOmega0(omega0, alpha, lamb, obsReview, listOfIndex, omega, V):
    om0 = omega0 - alpha*(2*(factorizationMachine(listOfIndex, omega0, omega, V)[0]-obsReview)+ 2*lamb*omega0) # aggiorna il valore di omega0
    return om0 #ritorna il valore di omega0 aggiornato 

#funzione per aggiornare il j-esimo parametro del vettore omega con gradiente
def updOmegaj(omega0, alpha, lamb, obsReview, listOfIndex, omega, V, column):
    omj = omega[column] - alpha*(2*(factorizationMachine(listOfIndex, omega0, omega, V)[0]-obsReview)+ 2*lamb*omega[column]) # aggiorna il valore di omegaj
    return omj #ritorna il valore di omegaj aggiornato

#funzione per aggiornare parametro nella cella [j,f] della matrice V
def updVjf(omega0, alpha, lamb,  obsReview, listOfIndex, omega, V, column, factor):
    Vjf = V[column,factor] - alpha*(2*(factorizationMachine(listOfIndex, omega0, omega, V, factor)[1]-V[column,factor])
                                    *((factorizationMachine(listOfIndex, omega0, omega, V, factor)[0]-obsReview))
                                    +2*lamb*V[column,factor]) # aggiorna il valore di vjf
    return Vjf #ritorna il valore di vjf aggiornato

#funzione per ottenere vettore con i ratings osservati nel dataset
def obsRatings(values, indPr):
    nReviews = len(indPr)
    y = [None]*nReviews #inizializza vettore per contenere i ratings osservati nel dataset
    a=0
    for i in indPr: #per tutte le recensioni
        y[a] = values[i-1] #prende l'overall rating
        a+=1
    return np.array(y) #ritorna vettore contenente gli overall rating

#funzione per ottenere vettore con i valori previsti dal modello
def yHat(colInd, omega0, omega, V, indPr, sumV=None):
    nReviews = len(indPr) #numero recensioni
    y_hat = [None]*nReviews ##inizializza vettore per contenere le previsioni
    if sumV==None: #se non si passa sumV
        for i in range(nReviews): # per tutte le recensioni
            if i==0:
                ind = colInd[0:(indPr[0]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione 0 (prima recensione)
            else :      
                 ind = colInd[indPr[i-1]:(indPr[i]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione i
            y_hat[i] = factorizationMachine(ind, omega0, omega, V)[0] #calcola previsione rating i-esima recensione
    else:
        for i in range(nReviews): # per tutte le recensioni
            if i==0:
                ind = colInd[0:(indPr[0]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione 0 (prima recensione)
            else :      
                 ind = colInd[indPr[i-1]:(indPr[i]-1)] #prende gli indici colonna tabella con valori diversi da zero per recensione i
            y_hat[i] = factorizationMachine(ind, omega0, omega, V, sumVM=sumV)[0] #calcola previsione rating i-esima recensione
    return np.array(y_hat) #ritorna array con previsioni e numero recensioni totali

#Funzione per stimare i parametri del modello tramite sgd
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
    print('inizializzo V')
    V = np.random.normal(0, sigma, size=(p,nFact)) #matrice bias a coppie
    count = 0 #contatore iterazioni
    delta = epsilon + 0.01
    indices = np.arange(nReviews)
    y = obsRatings(values, rInd) #array rating osservati
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
            omega0 = updOmega0(omega0, alpha, lamb[0], y_i, ind, omega, V) #aggiorna omega0
            for j in  ind: #per tutte le variabili con valore xj diverso da zero
                omega[j] = updOmegaj(omega0, alpha, lamb[1], y_i, ind, omega, V, j) #aggiorna omega_j
                for f in range(nFact): # per tutte le variabili latenti
                    V[j,f] = updVjf(omega0, alpha, lamb[2],  y_i, ind, omega, V, j, f) #aggiorna V_j,f
        mse=0
        sumVMatrix=sumV(V, ind, f)[0] #calcola la sommatoria (parte matriciale) equazione modello
        y_hat = yHat(colInd, omega0, omega, V, rInd, sumV=sumVMatrix)[0] #array previsioni rating con parametri aggiornati
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    