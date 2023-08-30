from scipy.sparse import coo_matrix

def loadDenseCsrMatrix(file): 
    I = [] #lista vuota item
    U = [] #lista vuota user
    R = [] #lista vuota rating
    n = 0  #numero di righe
    p = 0  #numero di colonne
    
    for line in file: #per ogni riga del file contenente le triple (item,user,rating)
         lineEntries = line.split(",") #esegui lo split della riga alla virgola
         U.append(int(lineEntries[1])) #inserisco nella lista di user 
         I.append(int(lineEntries[0])) #inserisco nella lista di item
         R.append(int(lineEntries[2])) #inserisco nella lista di rating
         n = max(n, int(lineEntries[1]) + 1) #il numero di righe della matrice è il massimo degli indici mappati per gli user
         p = max(p, int(lineEntries[0]) + 1) #il numero di colonne della matrice è il massimo degli indici mappati per gli item
    return coo_matrix((R, (U, I)), shape=(n, p), dtype = float).tocsr() #ritorna la matrice in formato csr 

