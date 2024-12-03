import pandas as pd
import numpy as np
from mtcspy.mtcs import TradeCreditNetwork
from benches.utils import parse_csv_to_obligations, create_undirected_graph_from_viability_matrix

def bench_perturbation():
    # initialize results storage
    results = []
    perturbation_ratios = np.arange(0, 0.11, 0.01)  # [0.00, 0.01, 0.02, ..., 0.10]
    n_iterations = 10

    # for each synthetic network
    for i in range(3):
        print(f"Synthetic Network {i + 1}")
        
        # load network once per dataset
        obligations = parse_csv_to_obligations(f"benches/data/synthethic_network_0{i + 1}.csv")

        # create copy network for each dataset
        network_copy = TradeCreditNetwork(obligations)
        
        # calculate initial outstanding obligations
        initial_obligations = network_copy.obligation_matrix.sum().sum()

        # perform mtcs over a copy of the network
        network_copy.mtcs()

        # calculate final outstanding obligations
        final_obligations = network_copy.obligation_matrix.sum().sum()

        optimal_cleared_amount = initial_obligations - final_obligations

        # for each perturbation ratio
        for ratio in perturbation_ratios:
            print(f"Processing perturbation ratio: {ratio}")
            
            # run multiple iterations
            for iteration in range(n_iterations):
                print(f"Iteration {iteration + 1} of {n_iterations} for dataset {i + 1} and perturbation ratio {ratio}")

                # create fresh network for each iteration
                network = TradeCreditNetwork(obligations)

                print("network created")
                
                # calculate initial degrees
                graph = create_undirected_graph_from_viability_matrix(network.viability_matrix)
                degrees_initial = pd.Series(dict(graph.degree()))

                print("degrees initial calculated")
                
                # calculate number of edges to perturb
                viable_edges = network.viability_matrix.sum().sum()
                edges_to_perturb = int(viable_edges * ratio)

                print("edges to perturb calculated")
                print(f"Edges to perturb: {edges_to_perturb}")

                # perturb the network
                network.perturb(edges_to_perturb)
                
                print("network perturbed")
                
                # calculate final degrees
                graph = create_undirected_graph_from_viability_matrix(network.viability_matrix)
                degrees_final = pd.Series(dict(graph.degree()))
                
                print("degrees final calculated")

                # calculate percentage of nodes that changed degree
                degrees_initial_aligned, degrees_final_aligned = degrees_initial.align(degrees_final, fill_value=0)
                changed_nodes = (degrees_initial_aligned != degrees_final_aligned).sum()
                total_nodes = len(degrees_initial)
                pct_changed = (changed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
                
                print(f"Percentage of nodes that changed degree: {pct_changed}%")

                # Perform mtcs over the perturbed network
                network.mtcs()

                # calculate final outstanding obligations
                final_obligations = network.obligation_matrix.sum().sum()

                # calculate cleared amount
                cleared_amount = initial_obligations - final_obligations

                # calculate percentage of optimal cleared amount
                pct_cleared = (cleared_amount / optimal_cleared_amount) * 100

                assert 0 <= pct_cleared <= 100, "Percentage of optimal cleared amount is not between 0 and 100"

                print(f"Percentage of optimal cleared amount: {pct_cleared}%")

                # store results
                results.append({
                    'network': i + 1,
                    'perturbation_ratio': ratio,
                    'iteration': iteration + 1,
                    'pct_nodes_changed': pct_changed,
                    'pct_cleared': pct_cleared
                })
                
    # convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('benches/results/perturbation_results.csv', index=False)
    print("Results saved to benches/results/perturbation_results.csv")

def bench_network_simplex():
    # Initialize results storage
    results = []
    
    # Define w values to test (max iterations for network simplex)
    w_values = [100, 500, 1000, None]  # None means unlimited iterations
    
    # For each synthetic network
    for i in range(3):
        print(f"Synthetic Network {i + 1}")
        
        # Load network once per dataset
        obligations = parse_csv_to_obligations(f"benches/data/synthethic_network_0{i + 1}.csv")
        
        # Create baseline network to get optimal solution (unlimited iterations)
        baseline_network = TradeCreditNetwork(obligations)
        initial_obligations = baseline_network.obligation_matrix.sum().sum()
        
        # Run MTCS with unlimited iterations to get optimal solution
        baseline_network.mtcs(w=None)
        optimal_final_obligations = baseline_network.obligation_matrix.sum().sum()
        optimal_cleared_amount = initial_obligations - optimal_final_obligations
        
        print(f"Optimal cleared amount: {optimal_cleared_amount}")
        
        # For each w value
        for w in w_values:
            print(f"Processing w = {w}")
            
            # Create fresh network
            network = TradeCreditNetwork(obligations)
                    
            # Run MTCS with current w
            network.mtcs(w=w)
            
            
            # Calculate final obligations and cleared amount
            final_obligations = network.obligation_matrix.sum().sum()
            cleared_amount = initial_obligations - final_obligations
            
            # Calculate percentage of optimal cleared amount
            pct_optimal = (cleared_amount / optimal_cleared_amount) * 100 if optimal_cleared_amount > 0 else 100
            
            print(f"Cleared amount: {cleared_amount} ({pct_optimal:.2f}% of optimal)")
            
            # Store results
            results.append({
                'network': i + 1,
                'w': 'unlimited' if w is None else w,
                'cleared_amount': cleared_amount,
                'pct_optimal': pct_optimal,
            })
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('benches/results/network_simplex_results.csv', index=False)
    print("Results saved to benches/results/network_simplex_results.csv")

if __name__ == "__main__":
    bench_network_simplex()
    bench_perturbation()