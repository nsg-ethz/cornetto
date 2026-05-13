"""
Calculate forwarding behaviour table from Batfish analysis (bf.q.forwardingBehaviorTable().answer(snapshot=snapshot).frame() question from the modified Batfish service).
This assumes that Batfish is already running and accessible.
"""

from pybatfish.client.session import Session
import pandas as pd

def get_forwarding_behaviour_table(
    snapshot_path: str,
    snapshot_name: str,
    bf: Session = None,
    ) -> pd.DataFrame:
    """
    Get the forwarding behaviour table from Batfish analysis.
    """
    # Create session if not provided
    if bf is None:
        bf = Session(host="localhost")

    # Get the forwarding behaviour table from Batfish
    bf.init_snapshot(snapshot_path, name=snapshot_name, overwrite=True)
    bf.set_snapshot(snapshot_name)
    df = bf.q.forwardingBehaviorTable().answer(snapshot=snapshot_name).frame()
    return df
