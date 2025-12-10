// Clase Nodo: Representa cada elemento en el árbol.
class Nodo {
    String nombre;
    Nodo izquierda;
    Nodo derecha;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izquierda = null;
        this.derecha = null;
    }
}

// Clase ArbolBinario: Contiene la lógica para gestionar el árbol.
public class ArbolBinario {
    private Nodo raiz;

    public ArbolBinario() {
        this.raiz = null;
    }

    // --- Métodos del Árbol Binario ---
    
    // 1. Método vacio(): boolean
    public boolean vacio() {
        return raiz == null;
    }

    // 2. Método insertar(nombre): void
    public void insertar(String nombre) {
        this.raiz = insertarRecursivo(raiz, nombre);
    }
    
    private Nodo insertarRecursivo(Nodo actual, String nombre) {
        if (actual == null) {
            return new Nodo(nombre);
        }

        // Comparamos el nombre para decidir si va a la izquierda o derecha
        int comparacion = nombre.compareTo(actual.nombre);
        
        if (comparacion < 0) {
            actual.izquierda = insertarRecursivo(actual.izquierda, nombre);
        } else if (comparacion > 0) {
            actual.derecha = insertarRecursivo(actual.derecha, nombre);
        }
        // Si el nombre ya existe, no hacemos nada (ABB no permite duplicados)
        return actual;
    }

    // 3. Método buscarNodo(nombre): Nodo
    public Nodo buscarNodo(String nombre) {
        return buscarNodoRecursivo(raiz, nombre);
    }
    
    private Nodo buscarNodoRecursivo(Nodo actual, String nombre) {
        if (actual == null || actual.nombre.equals(nombre)) {
            return actual;
        }

        if (nombre.compareTo(actual.nombre) < 0) {
            return buscarNodoRecursivo(actual.izquierda, nombre);
        } else {
            return buscarNodoRecursivo(actual.derecha, nombre);
        }
    }

    // 4. Método ImprimirArbol(): void
    // Imprime el árbol utilizando un recorrido en orden (in-order)
    public void imprimirArbol() {
        if (vacio()) {
            System.out.println("El árbol está vacío.");
        } else {
            imprimirInOrden(raiz);
        }
    }

    private void imprimirInOrden(Nodo nodo) {
        if (nodo != null) {
            imprimirInOrden(nodo.izquierda);
            System.out.print(nodo.nombre + " ");
            imprimirInOrden(nodo.derecha);
        }
    }

    // Método main para probar la funcionalidad
    public static void main(String[] args) {
        ArbolBinario arbol = new ArbolBinario();

        // 1. Verificar si el árbol está vacío
        System.out.println("¿El árbol está vacío? " + arbol.vacio());

        // 2. Insertar nodos
        System.out.println("\nInsertando nodos en el árbol...");
        arbol.insertar("Manzana");
        arbol.insertar("Naranja");
        arbol.insertar("Banana");
        arbol.insertar("Kiwi");
        arbol.insertar("Uva");
        arbol.insertar("Pera");
        arbol.insertar("Mango");

        // 3. Imprimir el árbol
        System.out.println("Elementos en el árbol (in-order):");
        arbol.imprimirArbol();

        // 4. Buscar un nodo
        System.out.println("\n\nBuscando nodos...");
        String nombreABuscar = "Banana";
        Nodo nodoEncontrado = arbol.buscarNodo(nombreABuscar);
        if (nodoEncontrado != null) {
            System.out.println("Nodo encontrado: " + nodoEncontrado.nombre);
        } else {
            System.out.println("Nodo '" + nombreABuscar + "' no encontrado.");
        }

        nombreABuscar = "Melon";
        nodoEncontrado = arbol.buscarNodo(nombreABuscar);
        if (nodoEncontrado != null) {
            System.out.println("Nodo encontrado: " + nodoEncontrado.nombre);
        } else {
            System.out.println("Nodo '" + nombreABuscar + "' no encontrado.");
        }
    }
}
