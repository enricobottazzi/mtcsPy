import pandas as pd
import networkx as nx
from mtcspy.mtcs import Obligation

def parse_csv_to_obligations(file_path: str) -> list[Obligation]:
    """
    Parse a CSV file containing trade credit obligations and convert it to a list of Obligation objects.
    
    Expected CSV format:
    debtor,creditor,amount
    0,1,100
    1,2,200
    ...
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Obligation objects
    """
    try:
        # read CSV file
        df = pd.read_csv(file_path)
        
        # validate column names
        required_columns = {'debtor', 'creditor', 'amount'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
            
        # convert DataFrame rows to Obligation objects
        obligations = []
        for _, row in df.iterrows():
            # validate amount is non-negative
            if row['amount'] < 0:
                raise ValueError(f"Found negative amount: {row['amount']}")
                
            # validate debtor != creditor
            if row['debtor'] == row['creditor']:
                raise ValueError(f"Found self-obligation: debtor={row['debtor']}, creditor={row['creditor']}")
                
            obligation = Obligation(
                debtor=row['debtor'],
                creditor=row['creditor'],
                amount=row['amount']
            )
            obligations.append(obligation)
            
        return obligations
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find CSV file: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise Exception(f"Error parsing CSV file: {str(e)}")
    

def create_undirected_graph_from_viability_matrix(viability_matrix: pd.DataFrame) -> nx.Graph:
    """
    Create an undirected graph from a viability matrix.
    """
    graph = nx.Graph()

    # find all pairs (i, j) for which a viable edge exists
    positive_edges = viability_matrix.stack().loc[lambda x: x > 0].index
    
    # add edges to the graph
    graph.add_edges_from(positive_edges)
    
    return graph