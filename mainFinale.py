import preProcessing as prep
import listOfLists as lol
import datasetPrepTrainSet as dptrs
import datasetUseSet as dus
import utilityMatrixCentering as ucm
import validationUV as vuv
import datasetPrepTestSet as dpts
import trainTestUV as tuv
import dataTable as dt
import dataTableTest as dtt
import tuningfm as tfm
import fmTesting as fmt


def main():
    file=input('Inserire il nome del file con estensione annessa: ')
    myString = open(file, "r") #leggo il file di testo test
    data = myString.read() #prendo il file letto
    print('\n')
    print("Pre processing")
    item, beerAbv, overall, user = prep.getRegex(data)
    print("Ho caricato le variabile del dataset")
    listOfCoupleUnited= prep.zipped(user, item)
    mask = prep.firstAppearanceUserItem(listOfCoupleUnited)
    item, beerAbv, overall, user = prep.subLists(mask,item, beerAbv,overall,user)
    print("Eliminate le recensioni doppie")
    uUsers = prep.uniqueUsers(user)
    rUsersName = prep.nameUsersOver(uUsers)
    mask = prep.indexOfArray(user, rUsersName)
    item, beerAbv, overall, user = prep.subLists(mask,item, beerAbv, overall, user)
    print("Eliminati utenti con meno di 20 recensioni")
    train, validation, test = lol.listOfLists(user, item, beerAbv, overall, 0.7, 0.05)  
    print('Decompisizione UV validation')
    fileValidation=input('Inserire il nome del file con estensione (.txt) che conterrà le triple (item,user,rating) del validation set: ')
    userMapValidation,itemMapValidation=dptrs.userItemMap(validation,fileValidation)
    ratingsFileValidation = open(fileValidation, 'r')
    matrixValidationUV= dus.loadDenseCsrMatrix(ratingsFileValidation)
    print("Il rating massimo della matrice è: ",max(matrixValidationUV.data))
    vectMeanRow=[]#vettore delle medie di riga
    vectMeanCol=[]#vettore di colonna
    matrixValidationUV,vectMeanRow=ucm.centeringMatrix(matrixValidationUV,vectMeanRow)
    matrixValidationUV=matrixValidationUV.tocsc()#conversione della matrice di utilità del validation in formato csc
    matrixValidationUV,vectMeanCol=ucm.centeringMatrix(matrixValidationUV,vectMeanCol)
    matrixValidationUV=matrixValidationUV.tocoo()#conversione della matrice di utilità del validation in formato coo
    mean=ucm.meanCoo(matrixValidationUV)
    print("Media dei rating POST norm: ",mean)
    vuv.validationUV(matrixValidationUV)
    print("Consultare il file 'risultati validation.txt' per stabilire le combinazioni di parametri da utilizzare nel training set")
    print('\n')
    print('Decompisizione UV training e testing')
    fileTrain=input('Inserire il nome del file con estensione (.txt) che conterrà le triple (item,user,rating) del training set: ')
    userMapTrain,itemMapTrain=dptrs.userItemMap(train, fileTrain)
    ratingsFileTrain = open(fileTrain, 'r')#apro in lettura il file
    matrixTrainUV= dus.loadDenseCsrMatrix(ratingsFileTrain)
    print("Il rating massimo della matrice è: ",max(matrixTrainUV.data))
    vectMeanRow=[]#vettore delle medie di riga
    vectMeanCol=[]#vettore di colonna
    matrixTrainUV,vectMeanRow=ucm.centeringMatrix(matrixTrainUV,vectMeanRow)
    matrixTrainUV=matrixTrainUV.tocsc()#conversione della matrice di utilità del train in formato csc
    matrixTrainUV,vectMeanCol=ucm.centeringMatrix(matrixTrainUV,vectMeanCol)
    matrixTrainUV=matrixTrainUV.tocoo()#conversione della matrice di utilità del train in formato coo
    mean=ucm.meanCoo(matrixTrainUV)
    print("Media dei rating POST norm: ",mean)
    fileTest=input('Inserire il nome del file con estensione (.txt) che conterrà le triple (item,user,rating) del test set: ')
    dpts.userItemTestSetMap(test,itemMapTrain,userMapTrain,fileTest)
    ratingsFileTest = open(fileTest, 'r')
    matrixTest=dus.loadDenseCsrMatrix(ratingsFileTest) 
    matrixTest=matrixTest.tocoo()#conversione della matrice di utilità del train in formato coo
    tuv.trainingUV(matrixTrainUV, matrixTest, vectMeanCol, vectMeanRow)
    print("Consultare il file 'risultati train test.txt' per vedere i risultati della decomposizione UV")
    print('\n')
    print('Factorization Machine validation')
    tableAsDictVal = dt.dictCsr(validation, userMapValidation,itemMapValidation)
    tfm.tuningAlphaEpsilon(tableAsDictVal)
    print("Consultare il file 'risultatiValidationAlphaEpsilonFm.txt' per stabilire le combinazioni di alpha e epsilon da testare nel passaggio successivo")
    tfm.tuningLambda(tableAsDictVal)
    print("Consultare il file 'risultatiValidationlambdaFm.txt' per stabilire le combinazioni di alpha e epsilon e lambda da utilizzare nel training set")    
    print('\n')
    print('Factorization Machine training e testing')
    tableAsDict = dt.dictCsr(train, userMapTrain, itemMapTrain)
    value, colIndex, indPr= dtt.csrTableTest(test, userMapTrain, itemMapTrain)
    fmt.fmTesting(tableAsDict, colIndex, value, indPr)
    print("Consultare il file 'risultatiFm.txt' per vedere i risultati del factorization machine")
    


if __name__ == "__main__":
    main()
