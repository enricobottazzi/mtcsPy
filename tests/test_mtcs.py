from mtcspy.mtcs import Obligation, TradeCreditNetwork
import pandas as pd
import numpy as np
import random
import pulp

def random_obligations(n, e, amount_range):
    """
    Generate `e` random obligations for `n` firms with amounts randomly sampled between amount_range[0] and amount_range[1].
    """

    obligations = []

    for _ in range(e):
        debtor = random.randint(0, n-1)
        creditor = random.randint(0, n-1)
        while creditor == debtor:
            creditor = random.randint(0, n-1)
        amount = random.randint(amount_range[0], amount_range[1])
        obligations.append(Obligation(debtor, creditor, amount))
    
    return obligations

obligations = random_obligations(100, 10000, (1, 1000))

def test_mtcs():

    trade_credit_network = TradeCreditNetwork(obligations)
    o_init = trade_credit_network.obligation_matrix.copy()
    b_init = trade_credit_network.get_b().copy()

    # perform mtcs
    trade_credit_network.mtcs()
    o_final = trade_credit_network.obligation_matrix
    b_final = trade_credit_network.get_b()
    
    # balance conservation constraint 
    assert (b_init == b_final).all()

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= o_final.at[i, j] <= o_init.at[i, j]

    # perform mtcs with LP
    o_final_lp = mtcs_LP(o_init)

    # assert that constraints are satisfied
    # balance conservation constraint
    for i in trade_credit_network.nodes:
        net_balance_lp_i = o_final_lp.sum(axis=0).at[i] - o_final_lp.sum(axis=1).at[i]
        assert b_init[i] == b_final[i] == net_balance_lp_i

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= o_final_lp.at[i, j] <= o_init.at[i, j]

    # assert that the total obligations in o_final and o_final_lp are the same
    # note that the solution might not be unique but the amount of optimal flow should be the same
    assert o_final.sum().sum() == o_final_lp.sum().sum()

def mtcs_LP(matrix: pd.DataFrame):
    """
    Solve the MTCS problem using linear programming
    """

    # Define the problem
    prob = pulp.LpProblem("Maximize_amount", pulp.LpMaximize)

    # Define the decision variables $f_{ij}$ for each cell in the matrix
    decision_variables = {}
    for i in matrix.index:
        for j in matrix.columns:
            f_ij = pulp.LpVariable(f'f_{i}_{j}', lowBound=0, upBound=1, cat='Continuous')
            decision_variables[(i, j)] = f_ij
    
    # Objective function: Maximize the total transaction amount that can be settled
    objective = pulp.lpSum(matrix.loc[i, j] * decision_variables[(i, j)] for i in matrix.index for j in matrix.columns)
    prob += objective

    # Add the positive balance constraints for each node i
    for i in matrix.index:
        prob += (
            pulp.lpSum([matrix.loc[j, i] * decision_variables[(j, i)] for j in matrix.index]) -
            pulp.lpSum([matrix.loc[i, j] * decision_variables[(i, j)] for j in matrix.columns])
            >= 0, f"Positive_balance_constraint_{i}"
        )

    prob.solve()        
    settlement_results = {(i, j): pulp.value(decision_variables[(i, j)]) for i in matrix.index for j in matrix.columns}

    # Update the liability matrix and the balances
    for (i, j), value in settlement_results.items():
        val = 0 if value==None else value
        settled_amount = val * matrix.loc[i, j]
        matrix.loc[i, j] -= int(round(settled_amount, 2))

    return matrix

def test_shuffle_unshuffle():
        
    trade_credit_network = TradeCreditNetwork(obligations)
    o_init = trade_credit_network.obligation_matrix.copy()
    viability_init = trade_credit_network.viability_matrix.copy()

    pi = np.random.permutation(len(trade_credit_network.nodes))

    # shuffle and unshuffle the network based on a random permutation
    trade_credit_network.shuffle(pi)
    trade_credit_network.unshuffle(pi)

    # assert that the matrices are the same
    assert (o_init == trade_credit_network.obligation_matrix).all().all()
    assert (viability_init == trade_credit_network.viability_matrix).all().all()

def test_shuffle_and_mtcs():
    trade_credit_network = TradeCreditNetwork(obligations)
    trade_credit_network_copy = TradeCreditNetwork(obligations)
    b_init = trade_credit_network.get_b().copy()
    o_init = trade_credit_network.obligation_matrix.copy()

    # perform mtcs over a copy of the network
    trade_credit_network_copy.mtcs()
    total_obligations = trade_credit_network_copy.obligation_matrix.sum().sum()

    # perform mtcs over a shuffled network
    pi = np.random.permutation(len(trade_credit_network.nodes))
    trade_credit_network.shuffle(pi)
    trade_credit_network.mtcs()
    total_obligations_shuffled = trade_credit_network.obligation_matrix.sum().sum()

    # assert that the total obligations left in the network are the same
    assert total_obligations == total_obligations_shuffled

    # unshuffle the network
    trade_credit_network.unshuffle(pi)
    o_final = trade_credit_network.obligation_matrix
    b_final = trade_credit_network.get_b()

    # assert that the constraints are met in the shuffled network
    # balance conservation constraint
    assert (b_init == b_final).all()

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= o_final.at[i, j] <= o_init.at[i, j]

def test_perturb():

    trade_credit_network = TradeCreditNetwork(obligations)
    o_init = trade_credit_network.obligation_matrix.copy()
    b_init = trade_credit_network.get_b().copy()
    v_init = trade_credit_network.viability_matrix.copy()
    viable_edges_init = v_init.sum().sum()

    # perturb the network using xi = 0, nothing should change
    trade_credit_network.perturb(0)
    assert (v_init == trade_credit_network.viability_matrix).all().all()

    # perturn the nextwork using a random xi between 0 and viable_edges
    xi = random.randint(0, viable_edges_init)
    trade_credit_network.perturb(xi)

    # the number of viable edges should be the same
    viable_edges_final = trade_credit_network.viability_matrix.sum().sum()
    assert viable_edges_init == viable_edges_final

    # run the mtcs on the perturbed network
    trade_credit_network.mtcs()

    o_final = trade_credit_network.obligation_matrix
    b_final = trade_credit_network.get_b()

    # assert that the constraints are met in the perturbed network
    assert (b_init == b_final).all()

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= o_final.at[i, j] <= o_init.at[i, j]

