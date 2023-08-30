import numpy as np

#obs rating osservati nel testset
#prev valori previsti dal modello
#nReviews recensioni nel testset

def classification(obs, pre): 
    count05 = 0
    count2=0
    nReviews=len(obs) #numero recensioni
    pre=np.clip(pre, 1, 20) #modifica le previsioni che sono minori di 1 e maggiori di 20, ponendo
    #a 1 gli elementi minori di 1 e a 20 gli elementi maggiori di 20
    for i in range(nReviews): #per tutti i rating osservati
        if(np.abs(obs[i] - pre[i]) < 0.5): #se il rating osservato si distacca poco dal previsto
            count05 +=1 #aumenta di uno il numero di previsioni corrette
        else:
            pass #altrimenti vai avanti nell'iterazione
        if(np.abs(obs[i] - pre[i]) < 2): #se il rating osservato si distacca poco dal previsto
            count2 +=1 #aumenta di uno il numero di previsioni corrette
        else:
            pass #altrimenti vai avanti nell'iterazione    
    return [count05 * 100 / nReviews,count2 * 100 / nReviews]   #ritorna la percentuale di rating previsti correttamente

def rmseForecast(obs, pre):
    nReviews=len(obs)
    sse = ((obs - pre) ** 2).sum() #calcola il numeratore del rmse
    return np.sqrt(sse / nReviews) #ritorna il rmse del test set

