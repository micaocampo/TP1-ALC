import numpy as np
import scipy

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

def calcula_K(A):
    tamaño_A = A.shape[0]
    K = np.zeros((tamaño_A,tamaño_A)) # matriz llena de ceros con el mismo tamaño de A (cuadrada)

    for i in range (tamaño_A): #recorre filas
        cantidad_apunta = 0 #marcará la cantidad de museos a los que apunta el museo i
        for j in range (tamaño_A): #recorre columnas
            cantidad_apunta = cantidad_apunta + A[i,j] #suma todos los elementos de la fila i (son 0, si no apunta y 1, si apunta)
        K[i,i] = cantidad_apunta

    return K
    
def calcula_L(A): # L = K - A
    K = calcula_K(A)
    L = K - A
    return L


def calcula_R(A): # R = A - P
    K = calcula_K(A)
    tamaño_A = A.shape[0]
    P = np.zeros((tamaño_A, tamaño_A)) #matriz cuadrada llena de ceros del tamaño de A 
    E = sum(sum(A))/2 #suma todos los elementos de la matriz A

    for i in range (tamaño_A):
        for j in range (tamaño_A):
            P[i,j] = (K[i,i]*K[j,j])/(2*E)

    R = A - P
    
    return R


def calcula_lambda(L,v): # 1/4 * st * L * s
    # Recibe L y v y retorna el corte asociado
    tamaño_s = len(v)
    s = [1] * tamaño_s

    for i in range (tamaño_s):
        if (v[i] < 0):
            s[i] = -1
    s_matriz = np.reshape(s,(tamaño_s, 1)) #crea al vector s como una matriz vertical
    s_matriz_t = np.reshape(s, (1, tamaño_s)) # crea al vector s traspuesto como una matriz (horizontal)

    lambdon = (s_matriz_t @ L @ s_matriz)[0][0]
    lambdon = lambdon/4

    return lambdon

def calcula_Q(R,v): # st * R * s
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    tamaño_s = len(v)
    s = [1] * tamaño_s

    for i in range (tamaño_s):
        if (v[i] < 0):
            s[i] = -1

    tamaño_s = len(v)
    s_matriz = np.reshape(v,(tamaño_s, 1)) #crea al vector s como una matriz vertical
    s_matriz_t = np.reshape(v, (1, tamaño_s)) # crea al vector s traspuesto como una matriz (horizontal)

    Q = (s_matriz_t @ R @ s_matriz)[0][0]

    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.reshape(np.random.uniform(-1, 1, A.shape[0]),(A.shape[0],1))# Generamos un vector de partida aleatorio, entre -1 y 1
   v = (v)/(np.linalg.norm(v)) # Lo normalizamos
   v1 = A@v # Aplicamos la matriz una vez
   v1 = (v1)/(np.linalg.norm(v1)) # normalizamos
   vt = np.reshape(v,(1,A.shape[0])) # Hacemos el v traspuesto para calcular el autovalor estimado
   v1t = np.reshape(v1,(1,A.shape[0])) # Otra vez
   l = (vt@A@v)/(vt@v) # Calculamos el autovalor estimado
   l1 = (v1t@A@v1)/(v1t@v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A@v # Calculo nuevo v1
      v1 = (v1)/(np.linalg.norm(v1)) # Normalizo
      v1t = np.reshape(v1,(1,A.shape[0])) # Hacemos el v traspuesto para calcular el autovalor estimado
      l1 = (v1t@A@v1)/(v1t@v1) # Calculo autovalor
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = l1[0][0] # Calculamos el autovalor
   v1 = np.array([elemento for sublista in v1 for elemento in sublista]) #dejamos el vector como array y no como matriz
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    v1t = np.reshape(v1,(1,A.shape[0]))
    deflA = A - (l1*((np.outer(v1,v1t)/(v1t@v1)))) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeros autovectores y autovalores de A}
   # Have fun!
   deflA = deflaciona(A,tol,maxrep)
   return metpot1(deflA,tol,maxrep)

#del template anterior
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

#del template anterior
def calculaInversa(A):
    #sabiendo que A es inversible, hacemos
    L, U = calculaLU(A)
    #sabemos que A*A^-1 = I, osea LUA^-1 = I
    #entonces para encontrar la inversa de A, buscamos plantear este sistema:
    #Ly=bi Ux=y, donde L es triangular inferior, U superior, y b es un vector canonico (los distinguimos por la i)
    #luego nos queda de la siguiente manera
    tamaño_A = A.shape[0]
    #creamos una A inversa vacía para ir guardando sus valores luego
    Ainv = np.zeros((tamaño_A,tamaño_A))
    #creamos la identidad para ir usando sus vectores
    I = np.eye(tamaño_A)
    #iteramos sobre todas las columnas:
    for i in range(0,tamaño_A):
        #tomamos el vector correspondiente de la identidad
        b = I[:,i]
        #resolvemos el sistema Ly = bi
        y = scipy.linalg.solve_triangular(L, b, lower = True) 
        #resolvemos el sistema Ux = y
        x = scipy.linalg.solve_triangular(U, y, lower = False)#del template anterior
        #este resultado es un vector, y representa la columna i en la matriz A inversa que estamos construyendo
        Ainv[:,i] = x
        #repetimos con todas las columnas
    return Ainv

def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    A_mu = A + (mu*np.eye(A.shape[0], k=0))
    A_mu_inv = calculaInversa(A_mu)
    

    return metpot1(A_mu_inv,tol=tol,maxrep=maxrep) #preguntar


def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + (mu*np.eye(A.shape[0], k=0)) # Calculamos la matriz A shifteada en mu
   iX = calculaInversa(X) # La invertimos
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu #se contradice con la consigna
   return v,l,_

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
   # Retorna una lista con conjuntos de nodos representando las comunidades.
   # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
   if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
       nombres_s = range(A.shape[0])
   if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
       return([nombres_s])
   else: # Sino:
       L = calcula_L(A) # Recalculamos el L
       v,l,_ = metpotI2(L,0.1) # Encontramos el segundo autovector más chico de L
       # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
       Ap = A[v>0,:][:,v>0]
       Am = A[v<0,:][:,v<0]
       return(
               laplaciano_iterativo(Ap,niveles-1,
                                    nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
               laplaciano_iterativo(Am,niveles-1,
                                    nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
               )        

def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return(list(nombres_s))
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Arreglar el v (?)
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[v>0,:][:,v>0] # Parte de R asociada a los valores positivos de v
            Rm = R[v<0,:][:,v<0] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                Ap = A[v>0,:][:,v>0]
                Am = A[v<0,:][:,v<0]
                nombres_separados = [[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]]
                res = modularidad_iterativo(Ap,Rp,nombres_separados[0]) + modularidad_iterativo(Am,Rm,nombres_separados[1])
                return(res)
            
