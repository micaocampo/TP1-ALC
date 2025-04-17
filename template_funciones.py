def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    # Completar! Have fun
    n = matriz.shape[0] #numero de filas de la matriz (mismo que columnas)
    L = np.eye(n, k=0) #crea matriz identidad
    U = matriz.copy()
    for j in range (n): #recorre las columnas
        for i in range (j+1, n): #recorre las filas
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j]*U[j,:]
    return L,U
    

def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    K = np.eye(A.shape[0], k=0) # matriz identidad
    for i in range (A.shape[0]): #recorrre filas
        cantidad_apunta = 0 #marcará la cantidad de museos a los que apunta el museo i
        for j in range (A.shape[0]): #recorre columnas
            cantidad_apunta = cantidad_apunta + A[i,j] #suma todos los elementos de la fila i (son 0, si no apunta y 1, si apunta)
        K[i,i] = cantidad_apunta

    Kinv = invertirK(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = Kinv @ A # Calcula C multiplicando Kinv y A 
    # si queremos C que sea las probabilidades, no deberia ser A^t @ Kinv?
    return C

def invertirK(K):
    # calculamos la inversa de una matriz que tiene numeros en la diagonal y ceros en todo lo demas, ya que sabemos que K tiene este formato siempre
    # la diagonal no puede tener ceros => todos los museos deben apuntar al menos a 1.
    Kinv = K.copy()
    #invierto todos los numeros de la diagonal
    for i in range (K.shape[0]):
        n = K[i,i]
        Kinv[i,i] = 1/n 
    return Kinv
    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (1 - alfa)* C 
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = (alfa/N) * np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B