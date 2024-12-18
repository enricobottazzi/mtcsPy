import time
import pandas as pd
import numpy as np
from mtcspy.mtcs import TradeCreditNetwork
from benches.utils import parse_csv_to_obligations, create_undirected_graph_from_viability_matrix

def bench_perturbation():
    # initialize results storage
    results = []
    perturbation_ratios = np.arange(0.01, 0.21, 0.01)  # [0.00, 0.01, 0.02, ..., 0.10]
    n_iterations = 5

    # for each synthetic network
    for i in range(3):
        print(f"synthetic network {i + 1}")
        
        # load network once per dataset
        obligations = parse_csv_to_obligations(f"benches/data/synthethic_network_0{i + 1}.csv")

        # create copy network for each dataset
        network_copy = TradeCreditNetwork(obligations)
        
        # calculate initial outstanding obligations
        initial_obligations = network_copy.obligation_matrix.sum().sum()

        # perform mtcs over a copy of the network
        start = time.time()
        network_copy.mtcs()
        end = time.time()
        print("mtcs performed")
        print(f"time taken for mtcs: {end - start} seconds")

        # calculate final outstanding obligations
        final_obligations = network_copy.obligation_matrix.sum().sum()

        optimal_cleared_amount = initial_obligations - final_obligations

        # for each perturbation ratio
        for ratio in perturbation_ratios:
            print(f"processing perturbation ratio: {ratio}")
            
            # run multiple iterations
            for iteration in range(n_iterations):
                start = time.time()
                print(f"iteration {iteration + 1} of {n_iterations} for dataset {i + 1} and perturbation ratio {ratio}")

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
                print(f"edges to perturb: {edges_to_perturb}")

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
                
                print(f"percentage of nodes that changed degree: {pct_changed}%")

                start = time.time()
                # Perform mtcs over the perturbed network
                network.mtcs()
                end = time.time()
                print("mtcs on perturbed networkperformed")
                print(f"time taken for mtcs: {end - start} seconds")

                # calculate final outstanding obligations
                final_obligations = network.obligation_matrix.sum().sum()

                # calculate cleared amount
                cleared_amount = initial_obligations - final_obligations

                # calculate percentage of optimal cleared amount
                pct_cleared = (cleared_amount / optimal_cleared_amount) * 100

                assert 0 <= pct_cleared <= 100, "Percentage of optimal cleared amount is not between 0 and 100"

                print(f"Percentage of optimal cleared amount: {pct_cleared}%")

                end = time.time()
                print(f"time taken for iteration {iteration + 1} and perturbation ratio {ratio}: {end - start} seconds")

                # store results
                results.append({
                    'network': i + 1,
                    'perturbation_ratio': ratio,
                    'iteration': iteration + 1,
                    'pct_nodes_changed': pct_changed,
                    'pct_cleared': pct_cleared
                })
                
        # save results for this network to a separate CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'benches/results/perturbation_results_network_{i+1}.csv', index=False)
        print(f"Results saved to benches/results/perturbation_results_network_{i+1}.csv")

def bench_network_simplex():        
    # For each synthetic network
    for i in range(3):
        print(f"Synthetic Network {i + 1}")
        
        # Load network once per dataset
        obligations = parse_csv_to_obligations(f"benches/data/synthethic_network_0{i + 1}.csv")
        
        # Create baseline network to get optimal solution (unlimited iterations)
        network = TradeCreditNetwork(obligations)

        # print the number of nodes in the network
        print(f"Number of nodes in the network: {len(network.nodes)}")

        # print the number of edges in the network
        print(f"Number of edges in the network: {network.viability_matrix.sum().sum()}")

        # Run MTCS and print the number of network simplex iterations needed
        network.mtcs()


if __name__ == "__main__":
    bench_perturbation()
    bench_network_simplex()
