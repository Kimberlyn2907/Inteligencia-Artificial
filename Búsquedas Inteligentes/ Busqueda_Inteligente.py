import collections
import heapq
import time 

# --- 1. Definición del Grafo y Parámetros ---

#Grafo ponderado con sus respectivos costos en las aristas
grafo = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 5, 'E': 2},
    'C': {'G': 3, 'F': 2},
    'D': {'H': 1},
    'E': {'G': 2},
    'F': {'G': 1},
    'G': {},
    'H': {}
}

inicio = 'A'
objetivo = 'G'
print(f"Búsqueda en Grafo (Inicio: {inicio}, Objetivo: {objetivo})\n")

# --- 2. Búsqueda Primero en Anchura (BFS) ---

def bfs(grafo, inicio, objetivo):
    """Explora por niveles, garantiza el camino más corto en número de aristas."""
    # Cola para almacenar nodos a visitar: (nodo_actual, camino_hasta_aqui)
    cola = collections.deque([(inicio, [inicio])])
    visitados = {inicio}

    while cola:
        nodo_actual, camino = cola.popleft() # (Cola)

        if nodo_actual == objetivo:
            return camino

        # Explorar vecinos
        #Aqui se realiza la interaccion con los vecinos del nodo actual
        for vecino in grafo.get(nodo_actual, {}):
            if vecino not in visitados:
                visitados.add(vecino)
                nuevo_camino = camino + [vecino]
                cola.append((vecino, nuevo_camino))
    
    return "Camino no encontrado"

# --- 3. Búsqueda Primero en Profundidad (DFS) ---

def dfs(grafo, inicio, objetivo):
    """Explora una rama tan profundo como sea posible."""
    # Pila para almacenar nodos a visitar: (nodo_actual, camino_hasta_aqui)
    pila = [(inicio, [inicio])]

    while pila:
        # Lógica LIFO (Pila) - pop() sin índice toma el último elemento
        nodo_actual, camino = pila.pop() 

        if nodo_actual == objetivo:
            return camino

        # Explorar vecinos. orden inverso para que el orden de exploración
        # sea consistente (si A tiene vecinos B y C, primero se explorará C).
       
        for vecino in reversed(list(grafo.get(nodo_actual, {}).keys())):
            if vecino not in camino: # Evita ciclos simples en el camino actual
                nuevo_camino = camino + [vecino]
                pila.append((vecino, nuevo_camino))
    
    return "Camino no encontrado"


# --- 4. Búsqueda de Costo Uniforme (UCS) ---

def ucs(grafo, inicio, objetivo):
    """Garantiza el camino con el menor costo total."""
    # Cola de Prioridad: (costo_acumulado, nodo_actual, camino_hasta_aqui)
    # heapq ordena por el primer elemento (el costo)
    cola_prioridad = [(0, inicio, [inicio])]
    
    # Almacena el costo mínimo encontrado hasta ahora para cada nodo
    costos_minimos = {inicio: 0}

    while cola_prioridad:
        costo_actual, nodo_actual, camino = heapq.heappop(cola_prioridad)

        if nodo_actual == objetivo:
            return camino, costo_actual

        # Si ya hemos encontrado un camino más barato a este nodo, lo ignoramos 
        if costo_actual > costos_minimos.get(nodo_actual, float('inf')):
             continue
        
        # Explorar vecinos
        for vecino, costo_arista in grafo.get(nodo_actual, {}).items():
            nuevo_costo = costo_actual + costo_arista
            
            # Si hemos encontrado un camino más barato al vecino, actualizamos
            if vecino not in costos_minimos or nuevo_costo < costos_minimos[vecino]:
                costos_minimos[vecino] = nuevo_costo
                nuevo_camino = camino + [vecino]
                heapq.heappush(cola_prioridad, (nuevo_costo, vecino, nuevo_camino))
    
    return "Camino no encontrado", 0

# --- 5. Ejecución y Resultados ---

print("--- RESULTADOS DE BÚSQUEDA ---")

# 1. Ejecutar BFS
camino_bfs = bfs(grafo, inicio, objetivo)
print(f"\n1: BFS (Primero en Anchura):")
print(f"    Camino: {camino_bfs}")
if isinstance(camino_bfs, list):
    print("   * Propiedad: Encuentra el camino con la menor cantidad de pasos.")

# 2. Ejecutar DFS
camino_dfs = dfs(grafo, inicio, objetivo)
print(f"\n2: DFS (Primero en Profundidad):")
print(f"    Camino: {camino_dfs}")
if isinstance(camino_dfs, list):
    print("  *  Propiedad: Encuentra un camino, priorizando la profundidad.")

# 3. Ejecutar UCS
camino_ucs, costo_ucs = ucs(grafo, inicio, objetivo)
print(f"\n3: UCS (Costo Uniforme):")
print(f"    Camino: {camino_ucs}")
if isinstance(camino_ucs, list):
    print(f"    Costo Total: {costo_ucs}")
    print("  *  Propiedad: Encuentra el camino con el menor costo total acumulado.")
