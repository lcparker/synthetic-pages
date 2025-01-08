import numpy as np
from typing import Tuple
from nrrd_file import Nrrd

Array3DIndex = Tuple[int | slice, int | slice, int | slice]

def create_label_mapping(giver, receiver, offset=2):
    MIN_REQUIRED_OVERLAP = 10
    counts = {}
    giver_edge, receiver_edge = _get_matching_edge(giver, receiver, offset)
    intersection = np.logical_and(giver_edge != 0, receiver_edge != 0)
    for (a,b) in zip(receiver_edge[intersection], giver_edge[intersection]):
        if a in counts.keys():
            counts[a][b] += 1
        else:
            counts[a] = {i: 0 for i in range(32)}

    correspondences = {}
    for k, v in counts.items():
        highest = 0
        if max(v.values()) > MIN_REQUIRED_OVERLAP:
            for q, p in v.items():
                if p > v[highest]:
                    highest = q
        correspondences[k] = highest
    
    matched_keys = correspondences.keys()
    free_keys = [i for i in range(1,32) if i not in matched_keys]
    for i in range(1,32):
        if i not in matched_keys:
            correspondences[i] = free_keys.pop()
    return correspondences

def _get_matching_edge(giver: Nrrd, receiver: Nrrd, edge_offset: int = 2) -> Tuple[np.ndarray,np.ndarray]:
    giver_superior = giver.metadata['space origin'] > receiver.metadata['space origin']
    giver_inferior = giver.metadata['space origin'] < receiver.metadata['space origin']
    if not np.sum(giver_superior) + np.sum(giver_inferior) == 1:
        raise ValueError("Volumes must be adjacent.")

    if np.sum(giver_superior) == 1: # giver is superior along some axis
        giver_slice = tuple(edge_offset if x else slice(None) for x in giver_superior)
        receiver_slice = tuple(-edge_offset if x else slice(None) for x in giver_superior)
    elif np.sum(giver_inferior) == 1:
        giver_slice = tuple(-edge_offset if x else slice(None) for x in giver_inferior)
        receiver_slice = tuple(edge_offset if x else slice(None) for x in giver_inferior)

    return giver.volume[giver_slice], receiver.volume[receiver_slice]

def match_labels(label_mapping: dict[int, int], volume):
    # Create lookup array with same dtype and known size
    lookup = np.zeros(33, dtype=np.uint8)  # 0-32 inclusive
    for k, v in label_mapping.items():
        lookup[k] = v
    
    # Map values using array indexing
    return lookup[volume]

def match_stitches(giver: Nrrd, receiver: Nrrd, offset: int = 2):
    correspondences = create_label_mapping(giver, receiver, offset=offset)
    result = match_labels(correspondences, receiver.volume)

    updated_receiver = Nrrd(result, receiver.metadata)
    return updated_receiver
