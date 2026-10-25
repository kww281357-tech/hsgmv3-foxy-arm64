"""Trusted evaluator-only geodesic; never pass this callable or results to policy."""
import math
import numpy as np


def make_geodesic_evaluator(environment):
    """Use the loaded scene NavMesh; added-actor topology needs separate auditing.

    This is the reference scene geodesic used only for scoring, not an executable
    route or goal-directed policy action. It does not claim a new actor is absent
    from that route. Both experiment arms must share the exact same scene mesh.
    """
    from habitat_sim import ShortestPath
    pf=environment.sim.pathfinder
    def distance(start,goal):
        query=ShortestPath()
        query.requested_start=np.asarray(start,dtype=float)
        query.requested_end=np.asarray(goal,dtype=float)
        if not pf.find_path(query):return None
        value=float(query.geodesic_distance)
        return value if math.isfinite(value) and value>=0 else None
    return distance
