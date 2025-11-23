import pandas as pd

def ensure_dataframe(obj):
    """
    Ensure that the returned object is a pandas DataFrame.
    
    Args:
        obj: Object to check, could be DataFrame, tuple, or other
        
    Returns:
        DataFrame: Either the original DataFrame or the first element of a tuple if it's a DataFrame
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, tuple) and len(obj) > 0:
        if isinstance(obj[0], pd.DataFrame):
            print(f"Converting tuple to DataFrame (first element)")
            return obj[0]
        else:
            print(f"Unable to convert tuple to DataFrame, first element is {type(obj[0])}")
            return pd.DataFrame()  # Return empty DataFrame as fallback
    else:
        print(f"Unable to convert {type(obj)} to DataFrame")
        return pd.DataFrame()  # Return empty DataFrame as fallback
