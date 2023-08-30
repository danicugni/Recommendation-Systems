import numpy as np
import sgd 
import forecastEvaluationUV as fe 

def trainingUV(matrixTrainUV,matrixTest,vectMeanCol,vectMeanRow):
    print("Inserimento dei parametri necessari per la decomposizione UV del training set")
    numComb=int(input("Quale numero di combinazioni di parametri si vogliono testare? "))
    matrixComb=np.zeros(shape=(numComb,7))#viene inizializzata la matrice contenente le combinazioni di parametri da utilizzare per l'SGD
    for i in range(numComb):#ciclo per ognuna colonne della matrice
        matrixComb[i,:]=list(map(float,(input('Inserire in sequenza separando con virgola i parametri da testare per alpha,lambda, numero fattori latenti,epsilon,num iter,seed1,seed2 : ').split(","))))#inserisco nella matrice i parametri
        if (matrixComb[i,0] <= 0): #se il learning rate è minore o uguale a 0
            raise ValueError("'alpha' deve essere maggiore di zero") #dare errore dicendo che il learning rate deve essere maggiore di zero
        if (matrixComb[i,1] < 0): #se il valore del parametro di regolarizzazione è minore di zero
             raise ValueError("'lambda' deve essere maggiore di zero") #dare errore dicendo che il valore minimo del parametro di regolarizzazione deve essere maggiore di zero
        if (matrixComb[i,2] <= 0): #se il numero di fattori latenti è minore o uguale a 0
            raise ValueError("'il numero di fattori latenti' deve essere maggiore di zero") #dare errore dicendo che il numero di fattori latenti deve essere maggiore di zero
        if (matrixComb[i,3] < 0): #se epsilon è minore o uguale a 0
            raise ValueError("'Epsilon' deve essere maggiore o uguale di zero") #dare errore dicendo che il numero di epsilon deve essere maggiore di zero
        if (matrixComb[i,4] <= 0): #se il numero massimo di iterazioni è minore o uguale a 0
            raise ValueError("Il numero massimo di iterazioni deve essere maggiore di zero") #dare errore dicendo che il numero massimo di iterazioni deve essere maggiore di zero  
        if (matrixComb[i,5] <=0 or matrixComb[i,6] <= 0):#se almeno uno dei due seed aggiunti è minore di zero
            raise ValueError("Il seed deve essere maggiore di zero")#il seed deve essere maggiore di zero
    file=open("risultati train test.txt",'w')#si apre in scrittura il file di testo che conterrà i risultati
    file.write(','.join([str('Tempo esecuzione'),str('Numero iterazioni'),str('Fattori latenti'),str('Alpha'),str('Lambda'), str('Epsilon'),str('MSEtrain'),str('RMSEtest'),str('% classification test 1-20'),str('% classification test 0-5')]))#intestazione colonne del file
    file.write('\n')#stampa un 'a capo'
    for i in range(matrixComb.shape[0]):#ciclo che scorre ognuna delle righe della matrice
        listOfParam=matrixComb[i,:]#salva in una lista la i-esima riga della matrice
        matrixApproxUV,time,iterations,latentFactors,alpha,lambd,epsilon,mse = sgd.sGD(matrixTrainUV, listOfParam[0], listOfParam[1], int(listOfParam[2]), listOfParam[3],int(listOfParam[4]),123,456) #richiamo la funzione che calcola l'SGD passando come argomenti gli elementi della lista
        matrixApproxUV = sgd.addColMean(matrixApproxUV, vectMeanCol)#aggiungo la media di colonna alla matrice UV
        matrixApproxUV = sgd.addRowMeans(matrixApproxUV, vectMeanRow)#aggiungo la media di riga alla matrice UV
        rmseUV=fe.rmseForecast(matrixTest, matrixApproxUV)#calcola RMSE sul testset
        percUV=fe.classification(matrixTest, matrixApproxUV)#calcola la percentuale di previsioni correttamente classificate
        output=np.round((time,iterations,latentFactors,alpha,lambd,epsilon,mse,rmseUV,percUV[0],percUV[1]),decimals=4)#salva i risultati in un array arrontando le cifre decimali
        for k in output:#scorri l'array contente i risultati 
            file.write(str(k))#scrivi il k-esimo elemento
            if k != output[-1]:
                file.write(',')
            else:
                pass
        file.write('\n')    
    file.close()#chiudi il file
