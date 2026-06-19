"""Multi-sensor (DASAR A/D/G) embedding fusion for the same detection event.

Each bowhead call is typically detected independently by up to three DASAR
instruments (A, D, G) on the same site. Fusing their embeddings gives the
probe more context about the event's acoustic signature from multiple
look-angles/distances.

Strategy
--------
1. Group spectrogram indices by (date, time, site) — the detection-event key.
2. For each event with ≥ 2 sensor observations, gather per-sensor embeddings.
3. Concatenate (or mean-pool) → one fused vector per event.
4. Pass the fused vectors to any probe.

If a sensor is missing for a given event, the slot is zero-filled so the fused
dimension stays constant regardless of how many sensors observed the event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bowhead.benchmark.backbones.fusion import fuse


@dataclass
class SensorGroup:
    """Indices of spectrogram images from each DASAR sensor for one event."""
    event_key: str              # e.g. "20080828_003309_3"
    sensor_indices: dict[str, int]  # sensor_id -> index into the dataset


def build_sensor_groups(
    dasar: np.ndarray,
    date: np.ndarray,
    site: np.ndarray,
    datetime_str: np.ndarray | None = None,
) -> list[SensorGroup]:
    """Group dataset indices by (date, time, site) detection-event key.

    Parameters
    ----------
    dasar : (N,) array of sensor IDs ('A', 'D', 'G')
    date  : (N,) array of date strings (YYYYMMDD)
    site  : (N,) array of site strings ('3' or '5')
    datetime_str : (N,) optional; if given, groups by exact datetime for
                   tighter event matching. If None, groups by date×site only
                   (coarser; same as the split key).

    Returns
    -------
    list of SensorGroup, one per unique (date/datetime × site) event.
    Multi-sensor events have len(sensor_indices) > 1.
    """
    if datetime_str is not None:
        keys = [f"{dt}_{s}" for dt, s in zip(datetime_str, site)]
    else:
        keys = [f"{d}_{s}" for d, s in zip(date, site)]

    event_map: dict[str, dict[str, int]] = {}
    for i, (key, das) in enumerate(zip(keys, dasar)):
        if key not in event_map:
            event_map[key] = {}
        # Last sensor wins on collision (same event, same sensor twice — rare)
        event_map[key][str(das)] = i

    return [
        SensorGroup(event_key=key, sensor_indices=sens)
        for key, sens in event_map.items()
    ]


def fuse_sensor_embeddings(
    embeddings: np.ndarray,
    sensor_groups: list[SensorGroup],
    sensors: tuple[str, ...] = ("A", "D", "G"),
    strategy: str = "concat",
) -> tuple[np.ndarray, list[str]]:
    """Build one fused embedding per detection event.

    Parameters
    ----------
    embeddings : (N, D) — one row per spectrogram
    sensor_groups : list of SensorGroup (from build_sensor_groups)
    sensors : ordered sensor IDs to include (missing → zero vector)
    strategy : "concat" | "mean"

    Returns
    -------
    fused : (M, D_fused) where M = len(sensor_groups)
    event_keys : list of M event-key strings (same order)
    """
    d = embeddings.shape[1]
    rows = []
    for group in sensor_groups:
        per_sensor: dict[str, np.ndarray] = {}
        for s in sensors:
            if s in group.sensor_indices:
                per_sensor[s] = embeddings[group.sensor_indices[s]]
            else:
                per_sensor[s] = np.zeros(d, dtype=np.float32)
        # Stack to (len(sensors), D) then fuse
        stacked = {s: per_sensor[s][None, :] for s in sensors}
        rows.append(fuse(stacked, strategy=strategy)[0])

    fused = np.stack(rows).astype(np.float32)
    event_keys = [g.event_key for g in sensor_groups]
    return fused, event_keys
