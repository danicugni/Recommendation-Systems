import numpy as np

#calcolo della media di una matrice in formato coo
def meanCoo(matrix):
    return np.mean(matrix.data)

#funzione che calcola la media di riga (o colonna) di una matrice csr (csc)
def meanRowCol(matrix,index):
    pos0=matrix.indptr[index] #numero di elementi precedenti al primo elemento della riga (colonna)
    pos1=matrix.indptr[index + 1] #numero di elementi precedenti al primo elemento della riga (colonna) successiva
    mean=np.mean([matrix.data[pos0:pos1]]) #calcola la media di riga (colonna)
    return mean #ritorna la media

#funzione che effettua la centratura per riga (per colonna) di una matrice csr (csc)
def centeringMatrix(matrix,vector=[]):
    pos0=0
    pos1=0
    z=len(matrix.indptr) #numero di righe (colonne) più 1
    for index in range(z-1): #per ogni riga (colonna) 
        mean=meanRowCol(matrix,index) #calcola la media di riga (colonna)
        vector.append(mean) #aggiungi alla lista la media di riga (colonna)
        pos0=matrix.indptr[index] #numero di elementi precedenti al primo elemento della riga (colonna)
        pos1=matrix.indptr[index+1] #numero di elementi precedenti al primo elemento della riga (colonna) successiva
        for j in range(pos0,pos1): #per ogni elemento della riga (colonna)
            matrix.data[j]=float(matrix.data[j])-mean #sottrai la media di riga (colonna)
    return (matrix, vector) #ritorna la matrice centrata e il vettore delle medie







