#Importo i moduli con le funzioni necessarie

import numpy as np
import factorizationMachine as fm
import forecastEvaluationFM as fefm

#tableAsDict dizionario contente tabella dei dati in formato csr (dizionario con tre np.array, costruiti in base al training set)
#colIndex array contente gli indici di colonna dei valori diversi da zero nella tabella dei dati (test set)
#value array contenente tutti i valori diversi da zero nella tabella dei dati (test set)
#indPr array contente numero valori diversi da zero cumulati per riga nella tabella dei dati (test set)

def fmTesting(tableAsDict, colIndex, value, indPr):
    
    file=open("risultatiFm.txt",'w') #crea il file e lo apre in scrittura
    file.write(','.join([str('alpha'), str('tripla lambda'),str('epsilon'),str('Numero iterazioni'),str('time'), str('Mse finale'),str('RMSE'), str('percentuale classificazione 1-20'), str('percentuale classificazione 1-20')]))
    file.write('\n')
    nCombIper=int(input("numero di combinazioni di iperparametri scelte: "))
    if (nCombIper <= 0): #se il numero di combinazioni è minore o uguale a 0
        raise ValueError("Il numero di combinazioni deve essere maggiore di zero") #dare errore dicendo che il numero di combinazioni deve essere maggiore di zero                 
    combination = [None]*nCombIper #inizializzo lista che conterrà combinazione di iperparametri
    for i in range(nCombIper): #per numero di combinazioni da inserire 
        combination[i]=list(map(float,(input("Inserire alpha(learning rate), tripla di lambda (parametri regolarizzazione), epsilon in questo ordine :\n").split(",")))) #si inserisce la combinazione
        if len(combination[i]) != 5: #se i parametri inseriti non sono 5
            raise ValueError("i valori inseriti devono essere 5")
        for j in combination[i]:
            if (j < 0): #se è minore di 0
                raise ValueError("alpha, epsilon e i lambda devono essere maggiori o uguali a zero")
    maxIter=int(input("Inserire il numero massimo di iterazioni da considerare per lo stochastic gradient descend: "))
    if (maxIter <= 0): #se il numero massimo di iterazioni è minore o uguale a 0
        raise ValueError("Il numero massimo di iterazioni deve essere maggiore di zero") #dare errore dicendo che il numero di iterazioni deve essere maggiore di zero
    casualSeed=int(input("inserire seed se si vuole rendere esperimento replicabile: "))
    if (casualSeed <= 0): #se il seed è minore o uguale a 0
        raise ValueError("Il seed deve essere maggiore di zero") #dare errore dicendo che il seed deve essere maggiore di zero
    obs = fm.obsRatings(value, indPr) #ottiene array con rating osservati nel test set
    for i in combination: #per ogni combinazione di iperparametri scelta
        omega0, omega, V, timefm, nIterfm, msefm = fm.sgdFm(tableAsDict, i[0], i[1:4], 0.001, 1, i[4], maxIter, seed=casualSeed) #stima parametri 
        # modello, tempo esecuzione, numero iterazioni e mse finale
        prev = fm.yHat(colIndex, omega0, omega, V, indPr) #calcola i rating previsti dal modello
        percfm=fefm.classification(obs, prev) #calcola percentuale rating correttamente previsti
        rmsefm=fefm.rmseForecast(obs, prev) #calcola rmse
        output= np.round((i[0],i[1],i[2],i[3],i[4],nIterfm,timefm,msefm,rmsefm,percfm[0],percfm[1]),decimals=4) #crea array con gli output da inserire nel file
        for k in output: #inserisce gli output nel file separati da virgola
            file.write(str(k))
            if k != output[-1]:
                file.write(',')
            else:
                pass
        file.write('\n')#finita la combinazione di parametri va a capo nel file per la prossima
    file.close() #chiude il file

#output è un file di testo con parametri scelti, numero iterazioni, tempo esecuzione, mse finale nel training, rmse nel test, percentuale corretamente classificati
#per ogni combinazione di iperparametri scelta