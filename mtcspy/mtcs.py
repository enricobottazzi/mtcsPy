import time
import pandas as pd
import networkx as nx
import numpy as np
from mtcspy.utils import shuffle_matrix, unshuffle_matrix

class Obligation:
    def __init__(self, debtor, creditor, amount):
        assert amount >= 0, "The amount should not be negative"
        assert debtor != creditor, "The debtor and creditor should be different"

        self.debtor = debtor
        self.creditor = creditor
        self.amount = amount

    def __repr__(self):
        return f"Obligation({self.debtor}, {self.creditor}, {self.amount})"
    
    def __eq__(self, other):
        return self.debtor == other.debtor and self.creditor == other.creditor and self.amount == other.amount
            
class TradeCreditNetwork:
    def __init__(self, obligations: list[Obligation]):
        """
        obligations: list of Obligation
        """  
        assert all([isinstance(o, Obligation) for o in obligations]), "All obligations should be of type Obligation"
        assert all([o.debtor != o.creditor for o in obligations]), "There shouldn't be any self obligations"

        for o in obligations:
            assert o.amount >= 0, "The amount of the obligation should be positive"
        
        # each node is a firm, then sort the nodes based on their index
        self.nodes = list(set([o.debtor for o in obligations] + [o.creditor for o in obligations]))
        self.nodes = sorted(self.nodes)

        # The obligations are stored in an obligation matrix O where O[i, j] is the amount that firm i owes to firm j
        self.obligation_matrix = pd.DataFrame(0, index=self.nodes, columns=self.nodes)
        self.viability_matrix = pd.DataFrame(0, index=self.nodes, columns=self.nodes)

        for o in obligations:
            self.obligation_matrix.at[o.debtor, o.creditor] += o.amount

        self.viability_matrix = (self.obligation_matrix > 0).astype(int)

        self.perturbed = False
    
    def get_c(self):
        """
        c is a vector representing the credit of each firm
        """
        return self.obligation_matrix.sum(axis=0)
    
    def get_d(self):
        """
        d is a vector representing the debit of each firm
        """
        return self.obligation_matrix.sum(axis=1)
    
    def get_b(self):
        """
        b is a vector representing the net balance of each firm
        """
        return self.get_c() - self.get_d()

    def mtcs(self):
        """
        Run multilateral trade credit setoff (MTCS) algorithm on the network
        """

        obligation_matrix = self.obligation_matrix
        b = self.get_b()

        if self.perturbed: 
            # as a consequence of perturbation, some edges are no longer viable, so we need to update the obligation matrix
            obligation_matrix = self.obligation_matrix * self.viability_matrix
            b = obligation_matrix.sum(axis=0) - obligation_matrix.sum(axis=1)

        G = nx.DiGraph()

        # create a node for each firm where the node demand is the net balance of the firm 
        for node, net_balance in b.items():
            G.add_node(node, demand=net_balance)
        
        # for each entry in the matrix, create an edge between the two firms
        viable_edges = zip(*np.where(self.viability_matrix == 1))
        for debtor, creditor in viable_edges:
            amount = obligation_matrix.iloc[debtor, creditor]
            if amount > 0:  # this check might be redundant depending on your data
                G.add_edge(self.nodes[debtor], self.nodes[creditor], capacity=amount, weight=1)

        # solve the MCF problem
        flow_dict, iterations = nx.min_cost_flow(G)

        print(f"MTCS converged in {iterations} iterations")

        # update the matrix with the new obligations
        obligation_matrix_output = pd.DataFrame(0, index=self.nodes, columns=self.nodes)

        for debtor, creditors in flow_dict.items():
            for creditor, amount in creditors.items():
                if amount > 0:
                    obligation_matrix_output.at[debtor, creditor] = amount

        # if perturbed, add back the obligations for the edges that were removed during perturbation (but not settled!)
        if self.perturbed:
            # Create a boolean mask for edges that are zero in the perturbed viability matrix
            zero_edges = (self.viability_matrix == 0)
            
            # Use the mask to add back the original obligations
            obligation_matrix_output = obligation_matrix_output.add(
                self.obligation_matrix.where(zero_edges, 0)
            )

        self.obligation_matrix = obligation_matrix_output

    def shuffle(self, pi):
        """
        Shuffle the obligation and viability matrices based on a random permutation `pi`
        """

        self.obligation_matrix = shuffle_matrix(self.obligation_matrix, pi)
        self.viability_matrix = shuffle_matrix(self.viability_matrix, pi)

    def unshuffle(self, pi):
        """
        Unshuffle the matrices to restore original order based on the permutation `pi` used for shuffling
        """

        self.obligation_matrix = unshuffle_matrix(self.obligation_matrix, pi)
        self.viability_matrix = unshuffle_matrix(self.viability_matrix, pi)

    def perturb(self, xi):
        """
        Perturb the viability matrix using random add/del techinque with parameter xi
        """

        n = len(self.nodes)

        # flatten the viability matrix
        viability_vector = self.viability_matrix.to_numpy().flatten()

        # get indices of 1s and 0s
        ones_indices = np.where(viability_vector == 1)[0]

        # randomly flip xi of the 1s to 0s
        flip_indices = np.random.choice(ones_indices, xi, replace=False)
        viability_vector[flip_indices] = 0

        # get indices of 0s
        zeros_indices = np.where(viability_vector == 0)[0]

        # randomly flip xi of the 0s to 1s
        flip_indices = np.random.choice(zeros_indices, xi, replace=False)
        viability_vector[flip_indices] = 1

        # reshape the viability vector to a matrix
        self.viability_matrix = pd.DataFrame(viability_vector.reshape(n, n), index=self.nodes, columns=self.nodes)
        self.perturbed = True

    # # this algorithm resembles the secure version of the algorithm described in the paper, we comment this out as it is inefficient
    # def perturb(self, xi):
        # n = len(self.nodes)
    
        # # convert viability matrix to a vector
        # viability_vector = self.viability_matrix.to_numpy().flatten()

        # # shuffle the vector based on a random permutation pi_2
        # pi_2 = np.random.permutation(n * n)
        # viability_vector = viability_vector[pi_2]

        # # Delete first ξ randomized existing edges
        # count_deleted = 0
        # for i in range(n * n):
        #     if count_deleted == xi:
        #         break
        #     if viability_vector[i] == 1:
        #         viability_vector[i] = 0
        #         count_deleted += 1

        # # shuffle the vector based on a random permutation pi_3
        # pi_3 = np.random.permutation(n * n)
        # viability_vector = viability_vector[pi_3]
        
        # # Add first ξ randomized non-existing edges
        # count_added = 0
        # for i in range(n * n):
        #     if count_added == xi:
        #         break
        #     if viability_vector[i] == 0:
        #         viability_vector[i] = 1
        #         count_added += 1
        
        # # unshuffle the vector from pi_3
        # pi_3 = np.argsort(pi_3)
        # viability_vector = viability_vector[pi_3]

        # # unshuffle the vector from pi_2
        # pi_2 = np.argsort(pi_2)
        # viability_vector = viability_vector[pi_2]

        # # reshape the vector to a matrix
        # self.viability_matrix = pd.DataFrame(viability_vector.reshape(n, n), index=self.nodes, columns=self.nodes)
        # self.perturbed = True