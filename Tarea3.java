
public class Taarea3Dijkstra {

    public static Map<String, Integer> dijkstra(Map<String, Map<String, Integer>> graph, String start) {
       
        Map<String, Integer> distances = new HashMap<>();
        for (String node : graph.keySet()) {
            distances.put(node, Integer.MAX_VALUE); 
        }
        distances.put(start, 0); 
     
        PriorityQueue<Node> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(n -> n.distance));
        priorityQueue.add(new Node(start, 0));

        while (!priorityQueue.isEmpty()) {
            Node current = priorityQueue.poll();

            
            if (current.distance > distances.get(current.name)) {
                continue;
            }

         
            for (Map.Entry<String, Integer> neighborEntry : graph.get(current.name).entrySet()) {
                String neighbor = neighborEntry.getKey();
                int weight = neighborEntry.getValue();
                int distance = current.distance + weight;

            
                if (distance < distances.get(neighbor)) {
                    distances.put(neighbor, distance);
                    priorityQueue.add(new Node(neighbor, distance));
                }
            }
        }
        return distances;
    }

    
    static class Node {
        String name;
        int distance;

        Node(String name, int distance) {
            this.name = name;
            this.distance = distance;
        }
    }

  
    public static void main(String[] args) {
        
        Map<String, Map<String, Integer>> graph = new HashMap<>();

        graph.put("A", new HashMap<>());
        graph.get("A").put("B", 1);
        graph.get("A").put("C", 4);

        graph.put("B", new HashMap<>());
        graph.get("B").put("A", 1);
        graph.get("B").put("C", 2);
        graph.get("B").put("D", 5);

        graph.put("C", new HashMap<>());
        graph.get("C").put("A", 4);
        graph.get("C").put("B", 2);
        graph.get("C").put("D", 1);

        graph.put("D", new HashMap<>());
        graph.get("D").put("B", 5);
        graph.get("D").put("C", 1);

        // Nodo de inicio
        String startNode = "A";
        Map<String, Integer> shortestPaths = dijkstra(graph, startNode);

        System.out.println("Distancias más cortas desde el nodo " + startNode + ":");
        for (Map.Entry<String, Integer> entry : shortestPaths.entrySet()) {
            System.out.println("Hasta " + entry.getKey() + " = " + entry.getValue());
        }
    }
}