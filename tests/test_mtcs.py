from mtcspy.mtcs import Obligation, TradeCreditNetwork
from pandas import DataFrame
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

obligations = random_obligations(100, 10000, (1, 100))

def test_mtcs():

    trade_credit_network = TradeCreditNetwork(obligations)
    o_matrix_init = trade_credit_network.matrix.copy()
    b_init = trade_credit_network.get_b()
    trade_credit_network.mtcs()
    b_final = trade_credit_network.get_b()
    
    # balance conservation constraint 
    assert (b_init == b_final).all()

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= trade_credit_network.matrix.at[i, j] <= o_matrix_init.at[i, j]

    # perform mtcs with LP
    o_matrix_final_lp = mtcs_LP(o_matrix_init)

    # fetch total obligations in lm_final_lp
    total_obligations_lp = o_matrix_final_lp.sum().sum()

    # assert that constraints are satisfied
    # balance conservation constraint
    for i in trade_credit_network.nodes:
        net_balance_lp_i = o_matrix_final_lp.sum(axis=0).at[i] - o_matrix_final_lp.sum(axis=1).at[i]
        assert b_init[i] == b_final[i] == net_balance_lp_i

    # no novel obligations constraint
    for i in trade_credit_network.nodes:
        for j in trade_credit_network.nodes:
            assert 0 <= o_matrix_final_lp.at[i, j] <= o_matrix_init.at[i, j]

    # fetch total obligations in trade_credit_network
    total_obligations = trade_credit_network.matrix.sum().sum()

    # assert that the total obligations in lm and lm_final_lp are the same
    # note that the solution might not be unique but the amount of optimal flow should be the same
    assert total_obligations == total_obligations_lp

def mtcs_LP(matrix: DataFrame):
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
