import numpy as np

def classification(matrixObs, matrixApprox): #matrixObs è la matrice contenente le
#osservazioni del test set; matrixApprox è la matrice R= U*V
    count05 = 0
    count2= 0
    matrixApprox=np.clip(matrixApprox, 1, 20) #modifica gli elementi della matrice che sono minori di 1 e maggiori di 20, ponendo
    #a 1 gli elementi minori di 1 e a 20 gli elementi maggiori di 20
    # matrixApprox = np.round(matrixApprox, decimals = 0) #arrotonda tutti i rating stimati all'intero più vicino
    for i in range(len(matrixObs.data)): #per tutti i rating osservati
        if(np.abs(matrixObs.data[i] - matrixApprox[matrixObs.row[i], matrixObs.col[i]]) < 0.5): #se il rating osservato è uguale al
            #rating previsto
            count05 +=1 #aumenta di uno il numero di previsioni corrette
        else:
            pass #altrimenti vai avanti nell'iterazione
        if(np.abs(matrixObs.data[i] - matrixApprox[matrixObs.row[i], matrixObs.col[i]]) < 2): #se il rating osservato è uguale al
            #rating previsto
            count2 +=1 #aumenta di uno il numero di previsioni corrette
        else:
            pass #altrimenti vai avanti nell'iterazione   
    return [count05 * 100 / len(matrixObs.data),count2 * 100 / len(matrixObs.data)]   #ritorna la percentuale di rating previsti correttamente

def rmseForecast(matrixObs, matrixApprox): #date la matrice dei rating osservati e la matrice dei rating stimati
    """Calcola l'errore quadratico medio di M-UV, dove M[n x p]"""
    sse = ((matrixObs.data - matrixApprox[matrixObs.row, matrixObs.col]) ** 2).sum() #calcola il numeratore del rmse
    return np.sqrt(sse / len(matrixObs.data)) #ritorna il rmse del test set

