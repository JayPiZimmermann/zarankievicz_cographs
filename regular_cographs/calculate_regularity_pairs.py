import json
import os
import shutil

def compute_and_save_regularities(max_n, output_dir="cograph_data"):
    """
    Computes all possible regularities (degrees) k for regular cographs 
    of vertex count n up to max_n.
    Saves the result for each n in a separate JSON file.
    """
    
    # 1. Setup Output Directory
    if os.path.exists(output_dir):
        # Optional: Clean up old data to avoid confusion
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 2. Initialize DP Table
    # realizable[n] will store a set of possible degrees k
    realizable = {1: {0}}
    
    # Save the base case
    save_to_json(1, {0}, output_dir)
    
    print(f"Computing regularities up to N = {max_n}...")

    # 3. Dynamic Programming Loop
    for n in range(2, max_n + 1):
        realizable[n] = set()
        
        # We only need to check splits up to n // 2 due to symmetry
        # n = i + j
        for i in range(1, n // 2 + 1):
            j = n - i
            
            degrees_i = realizable[i]
            degrees_j = realizable[j]
            
            # --- Operation A: Disjoint Union (Sum) ---
            # Rule: Preserves degree. 
            # If G1 has degree k and G2 has degree k, G1 + G2 has degree k.
            common_degrees = degrees_i.intersection(degrees_j)
            realizable[n].update(common_degrees)
            
            # --- Operation B: Join (Product) ---
            # Rule: Preserves co-degree.
            # Co-degree c = (num_vertices - 1) - degree  <-- WAIT, definition check
            # Standard def: codegree = |V| - 1 - deg. 
            # Let's check algebra:
            # G1 (n1, k1), G2 (n2, k2). Join G = G1 * G2.
            # v in G1 connects to all G2. New degree = k1 + n2.
            # v in G2 connects to all G1. New degree = k2 + n1.
            # For G to be regular, k1 + n2 must equal k2 + n1.
            # => k1 - n1 = k2 - n2
            # => n1 - k1 = n2 - k2
            # This quantity (n - k) is often called the "deficiency" or related to codegree.
            # Let C = n - k. If C_i == C_j, valid join.
            # New degree K = k1 + n2 = (n1 - C) + n2 = (n1 + n2) - C = n - C.
            
            # Calculate set of 'deficiencies' (n - k) for part j
            deficiencies_j = {j - k for k in degrees_j}
            
            for k_i in degrees_i:
                c_i = i - k_i # Deficiency of part i
                
                if c_i in deficiencies_j:
                    # Match found! The resulting degree is n - C
                    new_degree = n - c_i
                    realizable[n].add(new_degree)
        
        # 4. Save result for this n
        save_to_json(n, realizable[n], output_dir)

    print(f"\nDone! Results saved in '{output_dir}/'")

def save_to_json(n, degrees, output_dir):
    """Helper to write the sorted list of degrees to a JSON file."""
    filename = os.path.join(output_dir, f"n_{n}.json")
    
    data = {
        "n": n,
        "count": len(degrees),
        "regularities": sorted(list(degrees))
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # --- Configuration ---
    MAX_N = 300  # Change this to your desired limit
    
    compute_and_save_regularities(MAX_N)