import numpy as np
import sgd 

def validationUV(matrixValidationUV):
    file=open("risultati validation.txt",'w') #apriamo in scrittura il file dove salveremo i risultati ottenuti col validation set
    file.write(','.join([str('Tempo esecuzione'),str('Numero iterazioni'),str('Fattori latenti'),str('Alpha'),str('Lambda'), str('Epsilon'),str('MSE')])) #la prima riga contiene il nome delle variabili 
    file.write('\n')
    print("Inserire i parametri necessari per la decomposizione UV del validation set")
    latentFactors=list(map(int,(input('Inserire i fattori latenti separati da virgola: ').split(","))))
    for i in latentFactors: #per tutti i fattori latenti inseriti come input in precedenza
        if (i <= 0): #se il numero di fattori latenti è minore o uguale a 0
            raise ValueError("'il numero di fattori latenti' deve essere maggiore di zero") #dare errore dicendo che il numero di fattori latenti deve essere maggiore di zero 
    ext=list(map(float,(input('Inserire estremo inferiore (maggiore o uguale a 0), superiore (minore o uguale a uno) e step per i valori di lambda che si vogliono testare separati da virgola: ').split(","))))
    if (ext[0] < 0): #se il valore minimo del parametro di regolarizzazione è minore di zero
         raise ValueError("'lambda' deve essere maggiore di zero") #dare errore dicendo che il valore minimo del parametro di regolarizzazione deve essere maggiore di zero
    if (ext[1] > 1): #se il valore massimo del parametro di regolarizzazione è maggiore di uno
        raise ValueError("'lambda' deve essere minore di uno") #dare errore dicendo che il massimo del parametro di regolarizzazione deve essere minore di uno
    if (ext[2] <0): #se lo step del parametro di regolarizzazione è minore di zero
        raise ValueError("lo step deve essere maggiore di zero") #dare errore dicendo che deve essere maggiore di zero
    regul=np.arange(ext[0],ext[1],ext[2]) #crea un array con valori che vanno da estremo inferiore (incluso) a estremo superiore (escluso) con lo step inserito
    alpha=float(input("Inserire il learning rate: "))
    if (alpha <= 0): #se il learning rate è minore o uguale a 0
        raise ValueError("'alpha' deve essere maggiore di zero") #dare errore dicendo che il learning rate deve essere maggiore di zero    
    maxIter=int(input("Inserire il numero massimo di iterazioni da considerare per lo stochastic gradient descend: "))
    if (maxIter <= 0): #se il numero massimo di iterazioni è minore o uguale a 0
        raise ValueError("Il numero massimo di iterazioni deve essere maggiore di zero") #dare errore dicendo che il numero di iterazioni deve essere maggiore di zero       
    epsilon=float(input("Inserire l'incremento infinitesimale in valore assoluto per far convergere l'algoritmo' : "))
    if (epsilon <= 0): #se epsilon è minore o uguale a 0
        raise ValueError("'Epsilon' deve essere maggiore di zero") #dare errore dicendo che il numero di iterazioni deve essere maggiore di zero
    num=int(input("Inserire il numero di volte per cui ripetere l'SGD per ogni combinazione di parametri: "))
    for j in range(len(latentFactors)): #per tutti i fattori latenti
        for i in range(len(regul)): #per tutti i possibili valori del parametro di regolarizzazione
            output=np.zeros(shape=(num,7)) #creiamo un array di zero in cui salvare momentaneamente i risultati di una combinazione di parametri
            for z in range(num): #per il numero di volte in cui ripetere l'SGD per ogni combinazione di parametri
                matrixApproxUV,time,iterazione,fattoriLat,alpha,lambd, epsilon,mse = sgd.sGD(matrixValidationUV, alpha, regul[i], latentFactors[j], epsilon, maxIter) #calcola lo SGD
                output[z,:]=np.round((time,iterazione,fattoriLat,alpha,lambd,epsilon,mse),decimals=4) #arrotonda i risultati ottenuti con lo SGD
            for k in range(output.shape[1]): #per ogni variabile
                file.write(str(np.mean(output[:,k]))) #inseriamo la corrispondente media nel file di testo
                file.write(',') #separando ciascuna variabile con una virgola
            file.write('\n')  
    file.close()
                