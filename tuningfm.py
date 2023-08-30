import factorizationMachine as fm
import numpy as np

#tableAsDict dizionario contente tabella dei dati in formato csr (dizionario con tre np.array) (validation set)
     
#funzione per valutare i learning raate e gli epsilon tenendo fissi i parametri di regolarizzazione a zero   
def tuningAlphaEpsilon(tableAsDict):
    file=open("risultatiValidationAlphaEpsilonFm.txt",'w') #crea il file e lo apre in scrittura
    file.write(','.join([str('Numero iterazioni medie'),str('Alpha'), str('Epsilon'),str('MSE medio')])) 
    file.write('\n')
    listEpsilon=list(map(float,(input("Inserire gli incrementi infinitesimali in valore assoluto per far convergere l'algoritmo, separando con la virgola: ").split(","))))
    for i in listEpsilon: #per tutti gli epsilon inseriti
        if (i < 0): #se epsilon è minore di 0
            raise ValueError("incremento deve essere maggiore o uguale a zero")
    ext=list(map(float,(input('Inserire estremo inferiore (maggiore o uguale a 0), superiore (minore o uguale a uno) e step per i valori del learning rate che si vogliono testare separati da virgola: ').split(","))))
    if (ext[0] < 0): #se il valore minimo del learning rate è minore di zero
         raise ValueError("'learning rate' deve essere maggiore di zero") #dare errore dicendo che il valore minimo del parametro di regolarizzazione deve essere maggiore di zero
    if (ext[2] <0): #se lo step del learning rate è minore di zero
        raise ValueError("lo step deve essere maggiore di zero") #dare errore dicendo che deve essere maggiore di zero
    listAlpha=np.arange(ext[0],ext[1],ext[2])
    maxIter=int(input("Inserire il numero massimo di iterazioni da considerare per lo stochastic gradient descend: "))
    if (maxIter <= 0): #se il numero massimo di iterazioni è minore o uguale a 0
        raise ValueError("Il numero massimo di iterazioni deve essere maggiore di zero") #dare errore dicendo che il numero di iterazioni deve essere maggiore di zero
    nIter=int(input("Inserire il numero di esecuzioni del sgd con la combinazione di alpha e epsilon selezionata: "))
    if (nIter <= 0): #se il numero massimo di esecuzioni è minore o uguale a 0
        raise ValueError("Il numero di lanci deve essere maggiore di zero") #dare errore dicendo che il numero di lanci deve essere maggiore di zero
    
    for epsilon in listEpsilon: #per tutti gli epsilon
        for alpha in listAlpha: #per tutti gli alpha
            meanL=0
            meanC=0
            for i in range(nIter): #fa algoritmo per numero di esecuzioni inserito
                count, loss=fm.sgdFm(tableAsDict, alpha, [0, 0, 0], 0.001, 1, epsilon, maxIter)[4:6] #calcola numero iterazioni e mse finale
                meanL+=loss/nIter #aggiorna media mse finale
                meanC+=count/nIter #aggiorna media numero iterazioni
            output=np.round((meanC,alpha,epsilon,meanL),decimals=4) #crea array con gli output da inserire nel file
            for k in output: #inserisce gli output nel file separati da virgola
                file.write(str(k)) 
                if k != output[-1]:
                    file.write(',')
                else:
                    pass
            file.write('\n') #finita la cobinazione di alpha e epsilon va a capo nel file per la prossima
    file.close() #chiude il file
    
#output è un file di testo con alpha e epsilon scelti, numero iterazioni medie,  mse finale medio
#per ogni coppia alpha e epsilon scelta
                
#funzione per valutare i lambda e conseguentemente scegliere iperparametri da usare nel training
def tuningLambda(tableAsDict):
    file=open("risultatiValidationlambdaFm.txt",'w') #crea il file e lo apre in scrittura
    file.write(','.join([str('Numero iterazioni medie'),str('Alpha'), str('Epsilon'), str('tripla Lambda'),str('MSE medio')]))
    file.write('\n')
    nCombination=int(input("numero di combinazioni di learning rate (alpha) e incrementi infinitesimali in valore assoluto (epsilon) scelte: "))
    if (nCombination <= 0): #se il numero massimo di combinazioni è minore o uguale a 0
        raise ValueError("Il numero di combinazioni deve essere maggiore di zero") #dare errore dicendo che il numero di combinazioni deve essere maggiore di zero                 
    coppia = [None]*nCombination #inizializzo lista che conterrà le coppie
    for i in range(nCombination): #per numero di coppie da inserire 
        coppia[i]=list(map(float,(input("Inserire la coppia alpha/epsilon in questo ordine, separando con la virgola,: ").split(","))))  #si insersice la coppia
        if len(coppia[i]) != 2: #se i parametri inseriti non sono 2
            raise ValueError("i valori inseriti devono essere 2")
        for j in coppia[i]:
            if (j < 0): #se è minore di 0
                raise ValueError("sia alpha che epsilon devono essere maggiori o uguali a zero")
    nLamb=int(input("numero di triple di parametri di regolarizzazione (lambda) che si vogliono provare : "))
    if (nLamb <= 0): #se il numero massimo di triple è minore o uguale a 0
        raise ValueError("Il numero di triple deve essere maggiore di zero") #dare errore dicendo che il numero di triple deve essere maggiore di zero 
    tripla=[None]*nLamb #inizializzo lista che conterrà le triple
    for i in range(nLamb): #per numero di triple da inserire 
        tripla[i]=list(map(float,(input("Inserire la tripla di lambda (il primo agisce sull'aggiornamento del parametro globale, il secondo sull'aggiornamento dei parametri per l'effetto della singola feature, il terzo sull'aggiornamento dei parametri che approssimano gli effetti di interazione tra due features), separando con la virgola: ").split(",")))) #si insersice la tripla
        if len(tripla[i]) != 3: #se i lambda inseriti non sono 3
            raise ValueError("i valori inseriti devono essere 3")
        for j in tripla[i]:
            if (j < 0): #se è minore di 0
                raise ValueError("lambda deve essere maggiore o uguale a zero")
    maxIter=int(input("Inserire il numero massimo di iterazioni da considerare per lo stochastic gradient descend: "))
    if (maxIter <= 0): #se il numero massimo di iterazioni è minore o uguale a 0
        raise ValueError("Il numero massimo di iterazioni deve essere maggiore di zero") #dare errore dicendo che il numero di iterazioni deve essere maggiore di zero
    nIter=int(input("Inserire il numero di esecuzioni del sgd con la combinazione di alpha e epsilon selezionata: "))
    if (nIter <= 0): #se il numero di esecuzioni è minore o uguale a 0
        raise ValueError("Il numero di lanci deve essere maggiore di zero") #dare errore dicendo che il numero di lanci deve essere maggiore di zero
    
    for cpp in coppia: #per tutte le coppie lambda/epsilon inserite
        for lamb in tripla: #per tutte le triple di lambda
            meanL=0
            meanC=0
            for i in range(nIter): #fa algoritmo per numero di esecuzioni inserito
                count, loss=fm.sgdFm(tableAsDict, cpp[0], lamb, 0.001, 1 ,cpp[1], maxIter)[4:6] #calcola numero iterazioni e mse finale
                meanL+=loss/nIter #aggiorna media mse finale
                meanC+=count/nIter #aggiorna media numero iterazioni
            output=np.round((meanC,cpp[0],cpp[1],lamb[0], lamb[1], lamb[2],meanL),decimals=4) #crea array con gli output da inserire nel file
            for k in output: #inserisce gli output nel file separati da virgola
                file.write(str(k))
                if k != output[-1]:
                    file.write(',')
                else:
                    pass
            file.write('\n') #finita la combinazione di parametri va a capo nel file per la prossima
    file.close() #chiude il file
    
#output è un file di testo con parametri scelti, numero iterazioni medie,  mse finale medio
#per ogni combinazione di iperparametri scelta
        
    
        
        
        
        
        
        
        #     for i in range(nIter):
        #         count, loss=fm.sgdFm(tableAsDict, alphaEpsilon[0], lambdas, 0.001, 1, alphaEpsilon[1], nIterMax)[4:6]
        #         meanL+=loss/nIter
        #         meanC+=count/nIter
        #     grid[countAE,countLambda,0] = meanL
        #     grid[countAE,countLambda,1] = meanC
        #     countLambda+=1
        # countAE+=1

