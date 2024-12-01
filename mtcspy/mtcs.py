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

        for i in self.nodes:
            for j in self.nodes:
                if self.obligation_matrix.at[i, j] > 0:
                    self.viability_matrix.at[i, j] = 1
    
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
        G = nx.DiGraph()

        b = self.get_b()

        # create a node for each firm where the node demand is the net balance of the firm 
        for node, net_balance in b.items():
            G.add_node(node, demand=net_balance)

        # For each entry in the matrix, create an edge between the two firms with capacity equal to the amount of the obligation and weight set to 1 by default
        for (debtor, creditor), amount in self.obligation_matrix.stack().items():
            if amount > 0:
                G.add_edge(debtor, creditor, capacity=amount, weight=1)

        # Solve the MCF problem
        flow_dict = nx.min_cost_flow(G)

        # Update the matrix with the new obligations
        self.obligation_matrix = pd.DataFrame(0, index=self.nodes, columns=self.nodes)

        for debtor, creditors in flow_dict.items():
            for creditor, amount in creditors.items():
                if amount > 0:
                    self.obligation_matrix.at[debtor, creditor] = amount

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

        # shuffle the viability vector using a random permutation
        pi_1 = np.random.permutation(len(viability_vector))
        viability_vector = viability_vector[pi_1]

        # delete the first xi viable edges from the viability vector
        del_count = 0
        for i in range (n * n):
            if del_count == xi:
                break
            if viability_vector[i] == 1:
                viability_vector[i] = 0
                del_count += 1

        # shuffle the viability vector once again using a different random permutation
        pi_2 = np.random.permutation(len(viability_vector))
        viability_vector = viability_vector[pi_2]

        # add the first xi viable edges to the partially perturbed viability vector
        add_count = 0
        for i in range (n * n):
            if add_count == xi:
                break
            if viability_vector[i] == 0:
                viability_vector[i] = 1
                add_count += 1

        # unshuffle the viability vector using the inverse permutation pi_2
        viability_vector = viability_vector[np.argsort(pi_2)]

        # unshuffle the viability vector using the inverse permutation pi_1
        viability_vector = viability_vector[np.argsort(pi_1)]

        # reshape the viability vector to a matrix
        self.viability_matrix = pd.DataFrame(viability_vector.reshape(n, n), index=self.nodes, columns=self.nodes)

        # update the obligation matrix based on the perturbed viability matrix
        self.obligation_matrix = self.obligation_matrix * self.viability_matrix

# TODO: 
# - Do I need to unshuffle the viability matrix too?