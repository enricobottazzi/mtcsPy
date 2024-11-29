import pandas as pd
import networkx as nx

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
            
class ObligationMatrix:
    def __init__(self, obligations: list[Obligation]):
        """
        obligations: list of Obligation
        Build an obligation matrix O from a list of obligations between firms such that O[i, j] is the amount that firm i owes to firm j
        """  
        assert all([isinstance(o, Obligation) for o in obligations]), "All obligations should be of type Obligation"
        assert all([o.debtor != o.creditor for o in obligations]), "There shouldn't be any self obligations"

        for o in obligations:
            assert o.amount >= 0, "The amount of the obligation should be positive"
        
        # each node is a firm, then sort the nodes based on their index
        self.nodes = list(set([o.debtor for o in obligations] + [o.creditor for o in obligations]))
        self.nodes = sorted(self.nodes)

        # The obligations are stored in an obligation matrix O where O[i, j] is the amount that firm i owes to firm j
        self.matrix = pd.DataFrame(0, index=self.nodes, columns=self.nodes)

        for o in obligations:
            self.matrix.at[o.debtor, o.creditor] = o.amount

        # Create a graph to represent the network
        self.create_graph()

    def get_c(self):
        """
        c is a vector representing the credit of each firm
        """
        return self.matrix.sum(axis=0)
    
    def get_d(self):
        """
        d is a vector representing the debit of each firm
        """
        return self.matrix.sum(axis=1)
    
    def get_b(self):
        """
        b is a vector representing the net balance of each firm
        """
        return self.get_c() - self.get_d()

    def create_graph(self):
        """
        Create a nx graph for the payment system
        """
        G = nx.DiGraph()

        b = self.get_b()

        # create a node for each firm where the node demand is the net balance of the firm 
        for node, net_balance in b.items():
            G.add_node(node, demand=net_balance)

        # For each entry in the matrix, create an edge between the two firms with capacity equal to the amount of the obligation and weight set to 1 by default
        for (debtor, creditor), amount in self.matrix.stack().items():
            if amount > 0:
                G.add_edge(debtor, creditor, capacity=amount, weight=1)

        return G


    def mtcs(self):
        """
        Run multilateral trade credit setoff (MTCS) algorithm on the network
        """

        graph = self.create_graph()

        # Solve the MCF problem
        flow_dict = nx.min_cost_flow(graph)

        # Update the matrix with the new obligations
        self.matrix = pd.DataFrame(0, index=self.nodes, columns=self.nodes)

        for debtor, creditors in flow_dict.items():
            for creditor, amount in creditors.items():
                if amount > 0:
                    self.matrix.at[debtor, creditor] = amount    