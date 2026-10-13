"""
stats_utils.py
--------------
stats.json 로드 유틸리티.
순환 import 방지를 위해 precompute_stats.py에서 분리.
"""

import json


def load_stats(stats_path: str) -> dict:
    """
    저장된 통계 파일 로드.

    Returns:
        stats: {layer_idx(int): {"mu": float, "sigma": float}}
    """
    with open(stats_path, "r") as f:
        stats_json = json.load(f)
    return {int(k): v for k, v in stats_json.items()}