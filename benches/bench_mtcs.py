import pandas as pd
import numpy as np
from mtcspy.mtcs import TradeCreditNetwork
from benches.utils import parse_csv_to_obligations, create_undirected_graph_from_viability_matrix

def bench_perturbation():
    # Initialize results storage
    results = []
    perturbation_ratios = np.arange(0, 0.11, 0.01)  # [0.00, 0.01, 0.02, ..., 0.10]
    n_iterations = 1 # TODO: Change to 10

    # For each synthetic network
    for i in range(3):
        print(f"Synthetic Network {i + 1}")
        
        # Load network once per dataset
        obligations = parse_csv_to_obligations(f"benches/data/synthethic_network_0{i + 1}.csv")
        
        # For each perturbation ratio
        for ratio in perturbation_ratios:
            print(f"Processing perturbation ratio: {ratio}")
            
            # Run multiple iterations
            for iteration in range(n_iterations):
                # Create fresh network for each iteration
                network = TradeCreditNetwork(obligations)

                print("network created")
                
                # Calculate initial degrees
                graph = create_undirected_graph_from_viability_matrix(network.viability_matrix)
                degrees_initial = pd.Series(dict(graph.degree()))

                print("degrees initial calculated")
                
                # Calculate number of edges to perturb
                viable_edges = network.viability_matrix.sum().sum()
                edges_to_perturb = int(viable_edges * ratio)

                print("edges to perturb calculated")

                # Perturb the network
                network.perturb(edges_to_perturb)
                
                print("network perturbed")
                
                # Calculate final degrees
                graph = create_undirected_graph_from_viability_matrix(network.viability_matrix)
                degrees_final = pd.Series(dict(graph.degree()))
                
                print("degrees final calculated")

                # Calculate percentage of nodes that changed degree
                degrees_initial_aligned, degrees_final_aligned = degrees_initial.align(degrees_final, fill_value=0)
                changed_nodes = (degrees_initial_aligned != degrees_final_aligned).sum()
                total_nodes = len(degrees_initial)
                pct_changed = (changed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
                
                # Store results
                results.append({
                    'network': i + 1,
                    'perturbation_ratio': ratio,
                    'iteration': iteration + 1,
                    'pct_nodes_changed': pct_changed
                })

                print(f"Iteration {iteration + 1} of {n_iterations} completed")
                print(f"Percentage of nodes that changed degree: {pct_changed}%")   
                
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('perturbation_results.csv', index=False)
    print("Results saved to perturbation_results.csv")

if __name__ == "__main__":
    bench_perturbation()