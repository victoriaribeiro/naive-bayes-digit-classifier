from numpy import argmax
from math import log
from pandas import read_csv, DataFrame

# Adding 1 to the numerator count ensures probability value doesnot become 0. 
# Adding 2 to the denominator reflects the number of states that are possible for the attribute under consideration. 
# In this case we have binary attributes.

def binarizacao( dadosTreino ):
    return ( dadosTreino >= 150 ).astype( int )

def gerarResultado( resultados, file_name ):
    df = DataFrame( data = resultados, columns = ['ImageId', 'Label'] )
    df.to_csv( file_name, sep = ',', index=False )

def treinar( labels, dadosTreinoBin ):
    train, priors = contagem( labels, dadosTreinoBin )
    print("treinando")

    for i in range( 10 ):
        for j in range ( 784 ):
            train[i][j] = ( train[i][j] + 1 ) / ( priors[i] + 2 ) #laplace smoothing
        priors[i] = priors[i] / len( labels )

    return train, priors

def aplicar( train, priors, dadosTesteBin ):
    resultados = [[0]*2 for i in range( len( dadosTesteBin ) )]
    score = [0]*10

    print("aplicando")
    for i in range( len( dadosTesteBin ) ):
            for j in range( len( priors ) ):
                score[j] =  priors[j] 
                for k in range( 784 ):
                    score[j] *= train[j][k] if dadosTesteBin[i][k] == 1 else 1.0 - train[j][k]
            resultados[i] = [i+1, argmax( score )]
    return resultados


def contagem( labels, dadosTreinoBin ):
    train = [[0]*785 for _ in range(10)]
    priors = [0]*10

    for i in range( len( labels ) ):
        priors[labels[i]] = priors[labels[i]] + 1
        for j in range ( 784 ):
            train[labels[i]][j] = train[labels[i]][j] + dadosTreinoBin[i][j]
    return train, priors

def main():
    dadosTreino = read_csv( 'train.csv' ).as_matrix()
    dadosTeste = read_csv( 'test.csv' ).as_matrix()

    labels = dadosTreino[0:,0]
    dadosTreino = dadosTreino[0:,1:]

    dadosTreinoBin = binarizacao( dadosTreino )
    train, priors = treinar( labels, dadosTreinoBin )

    dadosTesteBin = binarizacao( dadosTeste )
    resultados = aplicar( train, priors, dadosTesteBin )

    gerarResultado( resultados, "resultado.csv" )

main()
