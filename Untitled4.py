# Converted from Untitled4.ipynb

def find_maximum_clique(graph):
    """
    Find the maximum clique in an undirected graph.
    
    Args:
        graph: A dictionary mapping vertex to its adjacent vertices
               Example: {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
               
    Returns:
        A list of vertices forming the maximum clique found
    """
    def is_clique(vertices):
        """Check if a set of vertices forms a clique"""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] not in graph[vertices[i]]:
                    return False
        return True
    
    # Initialize variables to track the maximum clique
    max_clique = []
    step_counter = [0]  # Use a list to allow modification in nested function
    
    def backtrack(potential_clique, remaining_vertices, depth):
        step_counter[0] += 1
        indent = "  " * depth
        
        print(f"{indent}Step {step_counter[0]}:")
        print(f"{indent}Current clique: {potential_clique}")
        print(f"{indent}Remaining vertices: {remaining_vertices}")
        
        # Base case: no more vertices to consider
        if not remaining_vertices:
            print(f"{indent}No more vertices to consider.")
            if len(potential_clique) > len(max_clique):
                print(f"{indent}Found new maximum clique of size {len(potential_clique)}: {potential_clique}")
                max_clique.clear()
                max_clique.extend(potential_clique)
            return
        
        # Pruning: if remaining vertices + current clique size is smaller than max_clique
        if len(potential_clique) + len(remaining_vertices) <= len(max_clique):
            print(f"{indent}Pruning: can't improve current maximum of size {len(max_clique)}")
            return
        
        # Try including vertices
        for i, vertex in enumerate(remaining_vertices):
            new_potential = potential_clique + [vertex]
            
            # Check if new_potential is still a clique
            if is_clique(new_potential):
                print(f"{indent}Adding vertex {vertex} to clique")
                # New remaining vertices are those that come after current vertex
                # and are connected to all vertices in the new potential clique
                new_remaining = []
                for next_vertex in remaining_vertices[i+1:]:
                    is_connected_to_all = True
                    for clique_vertex in new_potential:
                        if next_vertex not in graph[clique_vertex]:
                            is_connected_to_all = False
                            break
                    if is_connected_to_all:
                        new_remaining.append(next_vertex)
                
                print(f"{indent}New remaining vertices: {new_remaining}")
                # Recursively search with the new clique
                backtrack(new_potential, new_remaining, depth + 1)
            else:
                print(f"{indent}Vertex {vertex} doesn't form a clique with {potential_clique}, skipping")
    
    # Start with empty clique and all vertices
    all_vertices = list(graph.keys())
    print("Starting maximum clique search:")
    backtrack([], all_vertices, 0)
    
    print("\nFinal result:")
    print(f"Maximum clique found: {max_clique} with size {len(max_clique)}")
    return max_clique

# Example usage:
if __name__ == "__main__":
    # Example graph as adjacency list
    # You can replace this with your own graph
    example_graph = {
        1: [2, 9],
        2: [1, 3, 9],
        3: [2, 9, 6, 10, 8, 11, 4],
        4: [3, 5, 11],
        5: [4, 11],
        6: [3, 7, 9],
        7: [2, 4, 6, 8, 9, 10, 11],
        8: [3, 7, 11],
        9: [1, 2, 3, 6, 7],
        10: [3, 7],
        11: [8, 7, 3, 4, 5]
    }
    
    print("Graph adjacency list:")
    for vertex, neighbors in example_graph.items():
        print(f"Vertex {vertex} is connected to: {neighbors}")
    print()
    
    max_clique = find_maximum_clique(example_graph)

def find_maximum_independent_set(graph):
    """
    Find the maximum independent set in an undirected graph.
    
    Args:
        graph: A dictionary mapping vertex to its adjacent vertices
               Example: {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
               
    Returns:
        A list of vertices forming the maximum independent set found
    """
    # Initialize variables to track the maximum independent set
    max_independent_set = []
    step_counter = [0]  # Use a list to allow modification in nested function
    
    def is_independent(vertices):
        """Check if a set of vertices forms an independent set"""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] in graph[vertices[i]]:
                    return False
        return True
    
    def backtrack(potential_set, remaining_vertices, depth):
        step_counter[0] += 1
        indent = "  " * depth
        
        print(f"{indent}Step {step_counter[0]}:")
        print(f"{indent}Current independent set: {potential_set}")
        print(f"{indent}Remaining vertices: {remaining_vertices}")
        
        # Base case: no more vertices to consider
        if not remaining_vertices:
            print(f"{indent}No more vertices to consider.")
            if len(potential_set) > len(max_independent_set):
                print(f"{indent}Found new maximum independent set of size {len(potential_set)}: {potential_set}")
                max_independent_set.clear()
                max_independent_set.extend(potential_set)
            return
        
        # Pruning: if remaining vertices + current set size is smaller than max_independent_set
        if len(potential_set) + len(remaining_vertices) <= len(max_independent_set):
            print(f"{indent}Pruning: can't improve current maximum of size {len(max_independent_set)}")
            return
        
        # Try including the first vertex in remaining_vertices
        vertex = remaining_vertices[0]
        
        # Skip vertex's neighbors when including vertex
        new_remaining = [v for v in remaining_vertices[1:] if v not in graph[vertex]]
        print(f"{indent}Including vertex {vertex}, excluding its neighbors")
        print(f"{indent}New remaining vertices: {new_remaining}")
        
        # Include vertex in the set
        backtrack(potential_set + [vertex], new_remaining, depth + 1)
        
        # Try excluding the first vertex
        print(f"{indent}Excluding vertex {vertex}")
        backtrack(potential_set, remaining_vertices[1:], depth + 1)
    
    # Start with empty set and all vertices
    all_vertices = list(graph.keys())
    print("Starting maximum independent set search:")
    backtrack([], all_vertices, 0)
    
    print("\nFinal result:")
    print(f"Maximum independent set found: {max_independent_set} with size {len(max_independent_set)}")
    return max_independent_set

# Example usage:
if __name__ == "__main__":
    # Example graph as adjacency list
    example_graph = {
        1: [2, 9],
        2: [1, 3, 9],
        3: [2, 9, 6, 10, 8, 11, 4],
        4: [3, 5, 11],
        5: [4, 11],
        6: [3, 7, 9],
        7: [2, 4, 6, 8, 9, 10, 11],
        8: [3, 7, 11],
        9: [1, 2, 3, 6, 7],
        10: [3, 7],
        11: [8, 7, 3, 4, 5]
    }
    
    print("Graph adjacency list:")
    for vertex, neighbors in example_graph.items():
        print(f"Vertex {vertex} is connected to: {neighbors}")
    print()
    
    max_independent_set = find_maximum_independent_set(example_graph)
