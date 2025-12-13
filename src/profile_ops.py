"""Optimized profile operations using NumPy."""

from __future__ import annotations
import numpy as np
from typing import Tuple
from functools import lru_cache


def sum_profile_fast(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute profile of disjoint union (sum) of two graphs.

    For sum G + H: profile_sum[i] = max(p1[i], p2[i])

    Args:
        p1: Profile array of first graph (length n1+1)
        p2: Profile array of second graph (length n2+1)

    Returns:
        Profile array of sum (length n1+n2+1)
    """
    n1 = len(p1) - 1
    n2 = len(p2) - 1
    n = n1 + n2

    result = np.zeros(n + 1, dtype=np.int32)
    result[0] = n  # K_{0,n} always exists

    # For i > 0, take max
    max_len = max(len(p1), len(p2))
    for i in range(1, n + 1):
        v1 = p1[i] if i < len(p1) else 0
        v2 = p2[i] if i < len(p2) else 0
        result[i] = max(v1, v2)

    return result


def product_profile_fast(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute profile of complete join (product) of two graphs.

    For product G × H: profile_prod[s] = max_{a+c=s} (p1[a] + p2[c])
    This is a max-convolution.

    Args:
        p1: Profile array of first graph
        p2: Profile array of second graph

    Returns:
        Profile array of product
    """
    n1 = len(p1) - 1
    n2 = len(p2) - 1
    n = n1 + n2

    result = np.zeros(n + 1, dtype=np.int32)
    result[0] = n

    # Max-convolution: result[s] = max_{a+c=s} (p1[a] + p2[c])
    for s in range(n + 1):
        a_min = max(0, s - n2)
        a_max = min(s, n1)

        best = 0
        for a in range(a_min, a_max + 1):
            c = s - a
            best = max(best, int(p1[a]) + int(p2[c]))
        result[s] = best

    return result


def product_profile_check_ktt(p1: np.ndarray, p2: np.ndarray, T: int) -> bool:
    """
    Fast check if product would contain K_{T,T}.

    Only computes profile[T], not full profile.

    Returns:
        True if product would contain K_{T,T}
    """
    n1 = len(p1) - 1
    n2 = len(p2) - 1

    a_min = max(0, T - n2)
    a_max = min(T, n1)

    for a in range(a_min, a_max + 1):
        c = T - a
        if int(p1[a]) + int(p2[c]) >= T:
            return True
    return False


def sum_profile_check_ktt(p1: np.ndarray, p2: np.ndarray, T: int) -> bool:
    """
    Fast check if sum would contain K_{T,T}.

    Returns:
        True if sum would contain K_{T,T}
    """
    v1 = p1[T] if T < len(p1) else 0
    v2 = p2[T] if T < len(p2) else 0
    return max(v1, v2) >= T


def profile_avoids_kst(profile: np.ndarray, s: int, t: int) -> bool:
    """Check if profile avoids K_{s,t}."""
    if s < len(profile) and profile[s] >= t:
        return False
    if t < len(profile) and profile[t] >= s:
        return False
    return True


def profiles_to_array(profiles: list[tuple[int, ...]]) -> np.ndarray:
    """Convert list of profile tuples to 2D numpy array (padded)."""
    if not profiles:
        return np.zeros((0, 0), dtype=np.int32)

    max_len = max(len(p) for p in profiles)
    arr = np.zeros((len(profiles), max_len), dtype=np.int32)

    for i, p in enumerate(profiles):
        arr[i, :len(p)] = p

    return arr


def batch_product_profiles(
    profiles1: np.ndarray,
    profiles2: np.ndarray,
    n1: int,
    n2: int,
    T: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute product profiles for all pairs of profiles.

    Args:
        profiles1: (num1, max_len1) array of profiles
        profiles2: (num2, max_len2) array of profiles
        n1: vertex count for profiles1
        n2: vertex count for profiles2
        T: optional pruning threshold

    Returns:
        Tuple of:
        - valid_mask: (num1, num2) bool array of valid pairs
        - result_profiles: list of result profile arrays
        - pair_indices: (K, 2) array of (i, j) indices for valid pairs
    """
    num1 = len(profiles1)
    num2 = len(profiles2)
    n = n1 + n2

    valid_pairs = []
    result_profiles = []

    for i in range(num1):
        p1 = profiles1[i]
        for j in range(num2):
            p2 = profiles2[j]

            # Fast K_{T,T} check
            if T is not None and product_profile_check_ktt(p1, p2, T):
                continue

            # Compute full profile
            result = product_profile_fast(p1[:n1+1], p2[:n2+1])

            valid_pairs.append((i, j))
            result_profiles.append(result)

    if not valid_pairs:
        return np.zeros((num1, num2), dtype=bool), [], np.zeros((0, 2), dtype=np.int32)

    valid_mask = np.zeros((num1, num2), dtype=bool)
    for i, j in valid_pairs:
        valid_mask[i, j] = True

    pair_indices = np.array(valid_pairs, dtype=np.int32)

    return valid_mask, result_profiles, pair_indices


def batch_sum_profiles(
    profiles1: np.ndarray,
    profiles2: np.ndarray,
    n1: int,
    n2: int,
    T: int | None = None
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Compute sum profiles for all pairs of profiles.

    Similar to batch_product_profiles but for disjoint union.
    """
    num1 = len(profiles1)
    num2 = len(profiles2)

    valid_pairs = []
    result_profiles = []

    for i in range(num1):
        p1 = profiles1[i]
        for j in range(num2):
            p2 = profiles2[j]

            # Fast K_{T,T} check
            if T is not None and sum_profile_check_ktt(p1, p2, T):
                continue

            # Compute full profile
            result = sum_profile_fast(p1[:n1+1], p2[:n2+1])

            valid_pairs.append((i, j))
            result_profiles.append(result)

    if not valid_pairs:
        return np.zeros((num1, num2), dtype=bool), [], np.zeros((0, 2), dtype=np.int32)

    valid_mask = np.zeros((num1, num2), dtype=bool)
    for i, j in valid_pairs:
        valid_mask[i, j] = True

    pair_indices = np.array(valid_pairs, dtype=np.int32)

    return valid_mask, result_profiles, pair_indices


def profile_to_bytes(profile: np.ndarray) -> bytes:
    """Convert profile to bytes for hashing/storage."""
    return profile.tobytes()


def bytes_to_profile(data: bytes, dtype=np.int32) -> np.ndarray:
    """Convert bytes back to profile array."""
    return np.frombuffer(data, dtype=dtype)
