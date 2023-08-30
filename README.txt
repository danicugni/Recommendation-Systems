Descrizione del contenuto dell'archivio midst_group02.zip:

a) MODULI IN COMUNE TRA DECOMPOSIZIONE UV E FACTORIZATION MACHINES:
	- preProcessing.py: permette di: ottenere tutte le osservazioni riguardanti le variabili utilizzate nel progetto (item, ABV, overall, user) tramite regular 
			    expressions, prendere in considerazione una sola volta le coppie (user,item) che si ripetono, selezionare gli user con più di 20 recensioni
	- listOfLists.py: permette di: dividere il dataset in validation, training e test set, creando per ognuno dei tre insiemi una lista di liste. 
			  Ciascuna sottolista è una lista contenente i valori osservati delle variabili considerate in ciascuna recensione 
	- datasetPrepTrainSet.py: permette di: creare due dizionari in cui si mappano sia item che user in ID numerici, creare un file di testo dove in ciascuna riga
				  è presente una tripla (ID item, ID user,rating), con ID item e ID user ID numerici definiti in precedenza
	- mainFinale.py: permette di: eseguire tutte le funzioni presenti nei moduli importati ed ottenere i risultati della decomposizione UV e del factorization machines 

b) MODULI RIGUARDANTI SOLO LA DECOMPOSIZIONE UV:
	- datasetUseSet.py: permette di: costruire, a partire da un file di testo in cui in ciascuna riga si ha una tripla (item ID, user ID, rating), una matrice 
			    di utilità in formato Comprassed Sparse Row (CSR)
	- utilityMatrixCentering.py: permette di: sottrarre a ciascun utente e a ciascun user la corrispondente media, calcolare la media totale di una matrice in 
				formato COOrdinate (COO)
	- sgd.py: permette di: calcolare il Mean Squared Error (MSE) di una matrice in formato COOrdinate (COO), perturbare una matrice generando casualmente da una 
		  Normale, applicare lo Stochastic Gradient Descent (SGD) nella decomposizione UV, aggiungere la media di riga e la media di colonna ad una matrice in
		  formato sparso
	- validationUV.py: permette di: applicare lo Stochastic Gradient Descent (SGD) al validation set, salvando i risultati ottenuti in un file di testo
	- datasetPrepTestSet.py: permette di: creare, a partire dal test set e dalle mappature di item e user precedentemente effettuate sul training set, 
				 un file di testo in cui in ciascuna riga sono presenti solo le triple (item ID, user ID, rating) relative agli item presenti sia
				 nel training che nel test set 
	- forecastEvaluationUV.py: permette di: calcolare il Root Mean Squared Error (RMSE) e la percentuale di osservazioni del test set classificate correttamente
	- trainTestUV.py: permette di: applicare lo Stochastic Gradient Descent (SGD) al training set e salvare i risultati ottenuti, (con il modello allenato sul training),
                          sul test set in un file di testo 
	- profileUV.py: permette di: effettuare la profilazione della memoria applicando lo Stochastic Gradient Descent (SGD) alla decomposizione UV

c) MODULI RIGUARDANTI SOLO IL FACTORIZATION MACHINES:
	- dataTable.py: permette di: costruire, a partire da una lista di liste e dai due dizionari che mappano item e user, la tabella dei dati del factorization machine
	  		in formato simil Comprassed Sparse Row (CSR)
	- factorizationMachine.py: permette di: ottenere i ratings osservati a partire dalla tabella in formato CSR, applicare lo Stochastic Gradient Descent (SGD) nel
	  			factorization machine per ottenerne i parametri stimati, ottenere i rating previsti dal factorization machine dati i parametri stimati
	- tuningfm.py: permette di: applicare lo Stochastic Gradient Descent (SGD) al validation set, valutando prima diversi learning rate ed epsilon (incremento 
		       infinitesimale in valore assoluto per far convergere SGD), poi, fissando learning rate ed epsilon si valutano i parametri di regolarizzazione.
	- dataTableTest.py: permette di:costruire, a partire dalla lista di liste del test e dai due dizionari che mappano item e user nel train, la tabella dei dati del 
          		    factorization machine in formato simil Comprassed Sparse Row (CSR) per il test set
	- forecastEvaluationFM.py: permette di: calcolare il Root Mean Squared Error (RMSE) e la percentuale di osservazioni del test set classificate correttamente
	- fmTesting.py: permette di: applicare lo Stochastic Gradient Descent (SGD) al training set e salvare i risultati ottenuti, (con il modello allenato sul training),
          		sul test set in un file di testo 
	- profileFM.py: permette di: effettuare la profilazione della memoria applicando lo Stochastic Gradient Descent (SGD) al Factorization Machines

Per eseguire il software (ad eccezione della profilazione della memoria applicando lo Stochastic Gradient Descent (SGD) alla decomposizione UV e al factorization machines):
	1) aprire il terminale nella directory in cui sono presenti tutti i moduli e il file di testo "ratebeer.txt" (*) contenente il dataset con cui si andrà a lavorare
	2) scrivere: python mainFinale.py
	3) quando viene chiesto "Inserire il nome del file con estensione annessa:", digitare: ratebeer.txt
	4) seguire le indicazioni che vengono fornite per guidare l'utente nell'esecuzione del codice

Per ottenere la profilazione della memoria e il grafico dell'utilizzo della memoria nel tempo applicando lo Stochastic Gradient Descent (SGD) alla decomposizione UV:
	1) aprire il terminale nella directory in cui sono presenti tutti i moduli e il file di testo "ratebeer.txt" (*) contenente il dataset con cui si andrà a lavorare
	2) scrivere: python -m mprof run profileUV.py
	3) quando viene chiesto "Inserire il nome del file con estensione annessa:", digitare: ratebeer.txt
	4) seguire le indicazioni che vengono fornite per guidare l'utente nell'esecuzione del codice
	5) al termine dell'esecuzione di "profileUV.py", scrivere: python -m mprof plot --flame

Per ottenere la profilazione della memoria e il grafico dell'utilizzo della memoria nel tempo applicando lo Stochastic Gradient Descent (SGD) al factorization machines:
	1) aprire il terminale nella directory in cui sono presenti tutti i moduli e il file di testo "ratebeer.txt" (*) contenente il dataset con cui si andrà a lavorare
	2) scrivere: python -m mprof run profileFM.py
	3) quando viene chiesto "Inserire il nome del file con estensione annessa:", digitare: ratebeer.txt
	4) seguire le indicazioni che vengono fornite per guidare l'utente nell'esecuzione del codice
	5) al termine dell'esecuzione di "profileFM.py", scrivere: python -m mprof plot --flame

(*) Link al drive in cui è presente il file di testo "ratebeer.txt": https://drive.google.com/file/d/1H3k7x2lGczeHauVx1hz0WJLAEQxmIlxL/view?usp=sharing
