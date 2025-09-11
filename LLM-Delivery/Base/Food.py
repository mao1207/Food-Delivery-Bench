# Base/Food.py
# -*- coding: utf-8 -*-
"""
High-level food condition model (coarse categories + simple flags).

JSON only defines high-level tags per item:
- category: HOT/COLD/FROZEN/AMBIENT
- odor: none/strong
- motion_sensitive: true/false
- nonthermal_time_sensitive: true/false      # time-driven deterioration unrelated to temperature
- prep_time_s: number (optional)

Outputs you likely show in UI:
- temperature_C               (float)
- motion_damage ∈ [0,1]       (spill/deform; step-wise on overspeed events: first 0.5, then 0.8)
- odor_contamination ∈ [0,1]  (set by bag when mixed with strong-odor items)
- state ∈ [0,1]               (FROZEN -> melt_fraction; others -> nonthermal_state if enabled)

Bag-level temperature coupling / odor mixing should be handled outside.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Dict, Any
import json


# ---------------- Enums ----------------
class FoodCategory(Enum):
    HOT     = auto()
    COLD    = auto()
    FROZEN  = auto()
    AMBIENT = auto()


class OdorLevel(Enum):
    NONE   = 0
    STRONG = 1


# ---------------- Helpers ----------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------- Category presets (shared physics per category) ----------------
_CATEGORY_PRESETS: Dict[FoodCategory, Dict[str, Any]] = {
    FoodCategory.HOT: dict(
        temperature_C=65.0,
        tau_air_s=900.0,
        ideal_temp_C=(60.0, 75.0),
        base_decay_per_s=0.00012,
        k_temp_dev=0.0010,
        melt_start_C=None,
        k_melt_per_C_per_s=0.0,
    ),
    FoodCategory.COLD: dict(
        temperature_C=5.0,
        tau_air_s=600.0,
        ideal_temp_C=(2.0, 8.0),
        base_decay_per_s=0.00012,
        k_temp_dev=0.0009,
        melt_start_C=None,
        k_melt_per_C_per_s=0.0,
    ),
    FoodCategory.FROZEN: dict(
        temperature_C=-18.0,
        tau_air_s=700.0,
        ideal_temp_C=(-18.0, -5.0),
        base_decay_per_s=0.00010,
        k_temp_dev=0.0008,
        melt_start_C=-5.0,            # melting above -5°C
        k_melt_per_C_per_s=0.002,
    ),
    FoodCategory.AMBIENT: dict(
        temperature_C=23.0,
        tau_air_s=1000.0,
        ideal_temp_C=(15.0, 30.0),
        base_decay_per_s=0.00005,
        k_temp_dev=0.0005,
        melt_start_C=None,
        k_melt_per_C_per_s=0.0,
    ),
}

# Global defaults for cross-cutting features
_MOTION_SPEED_THRESHOLD = 1.7     # m/s; overspeed events cause step-like damage
_NONTHERMAL_RATE_PER_S  = 0.0015  # nonthermal_state increase rate (if enabled) per second


@dataclass
class FoodItem:
    """
    A single food item with coarse physics and simple runtime signals.
    Call step(dt, env_temp_C, speed_m_s) each tick.
    """

    # Identity
    name: str
    category: FoodCategory

    # High-level flags from JSON
    odor_level: OdorLevel = OdorLevel.NONE
    motion_sensitive: bool = False
    nonthermal_time_sensitive: bool = False

    # Core state
    temperature_C: float = 23.0
    mass_kg: float = 0.5

    # Category physics (filled from presets)
    tau_air_s: float = 800.0
    ideal_temp_C: Tuple[float, float] = (15.0, 30.0)
    base_decay_per_s: float = 1e-4
    k_temp_dev: float = 8e-4
    melt_start_C: float | None = None
    k_melt_per_C_per_s: float = 0.0

    # Motion signals (unified spill/deformation)
    motion_speed_threshold: float = _MOTION_SPEED_THRESHOLD
    motion_damage: float = 0.0
    _last_over_speed: bool = False  # for rising-edge detection

    # Non-thermal time-driven deterioration
    nonthermal_rate_per_s: float = _NONTHERMAL_RATE_PER_S
    nonthermal_state: float = 0.0  # 0..1

    # Odor mixing (set by bag)
    odor_contamination: float = 0.0  # 0..1

    # Preparation
    prep_time_s: float = 0.0
    prep_elapsed_s: float = 0.0
    is_preparing: bool = False
    prepared: bool = True

    # Bookkeeping
    age_s: float = 0.0
    quality: float = 1.0  # not required for UI, but kept for completeness

    # ---------------- Lifecycle ----------------
    def begin_preparation(self, prep_time_s: float) -> None:
        """Start or reset preparation countdown."""
        self.prep_time_s = max(0.0, float(prep_time_s))
        self.prep_elapsed_s = 0.0
        self.is_preparing = self.prep_time_s > 0.0
        self.prepared = not self.is_preparing
        # age_s starts at 0 when preparation finishes

    # ---------------- Simulation step ----------------
    def step(self, dt_s: float, env_temp_C: float, speed_m_s: float = 0.0) -> None:
        if dt_s <= 0.0:
            return

        remaining = dt_s

        # Preparation phase (no evolution)
        if self.is_preparing and not self.prepared:
            need = max(0.0, self.prep_time_s - self.prep_elapsed_s)
            if remaining <= need:
                self.prep_elapsed_s += remaining
                return
            # finish during this tick
            self.prep_elapsed_s += need
            self.is_preparing = False
            self.prepared = True
            remaining -= need
            self.age_s = 0.0

        if not self.prepared or remaining <= 0.0:
            return

        dt_s = remaining
        self.age_s += dt_s

        # 1) Temperature evolution: first-order towards env
        tau = max(1e-6, self.tau_air_s)
        self.temperature_C += (env_temp_C - self.temperature_C) * (dt_s / tau)

        # 2) Non-thermal deterioration (if enabled)
        if self.nonthermal_time_sensitive:
            self.nonthermal_state = _clamp(self.nonthermal_state + self.nonthermal_rate_per_s * dt_s, 0.0, 1.0)

        # 3) Frozen melt (thermal)
        if self.melt_start_C is not None and self.k_melt_per_C_per_s > 0.0:
            if self.temperature_C > self.melt_start_C:
                inc = (self.temperature_C - self.melt_start_C) * self.k_melt_per_C_per_s * dt_s
                self.nonthermal_state  # (kept independent)  # noqa: just to emphasize separation
                self._accumulate_melt(inc)

        # 4) Motion damage — step-wise on *rising* overspeed events
        over = (speed_m_s > self.motion_speed_threshold)
        if self.motion_sensitive:
            if over and not self._last_over_speed:
                # rising edge: bump to 0.5 on first, 0.8 on second (or higher)
                if self.motion_damage < 0.5:
                    self.motion_damage = 0.5
                elif self.motion_damage < 0.8:
                    self.motion_damage = 0.8
                # cap at 1.0 if you ever want to add a third bump later
                self.motion_damage = _clamp(self.motion_damage, 0.0, 1.0)
            # remember state for next tick
            self._last_over_speed = over
        else:
            # not sensitive; keep memory clean
            self._last_over_speed = False

        # 5) Optional overall quality (kept simple; you may ignore it in UI)
        q_drop = self.base_decay_per_s * dt_s
        low, high = self.ideal_temp_C
        if self.temperature_C < low:
            q_drop += self.k_temp_dev * (low - self.temperature_C) * dt_s
        elif self.temperature_C > high:
            q_drop += self.k_temp_dev * (self.temperature_C - high) * dt_s
        # gentle link-in (no need to display):
        q_drop += 0.2 * self.nonthermal_state * dt_s
        q_drop += 0.3 * self.melt_fraction * dt_s
        q_drop += 0.4 * self.motion_damage * dt_s
        self.quality = _clamp(self.quality - q_drop, 0.0, 1.0)

    # ---------------- Internals ----------------
    melt_fraction: float = 0.0  # keep as a dedicated thermal state for FROZEN

    def _accumulate_melt(self, delta: float) -> None:
        self.melt_fraction = _clamp(self.melt_fraction + delta, 0.0, 1.0)

    # ---------------- Builders ----------------
    @staticmethod
    def _preset_for(category: FoodCategory) -> Dict[str, Any]:
        return _CATEGORY_PRESETS[category].copy()

    @staticmethod
    def from_simple_spec(name: str, spec: Dict[str, Any]) -> "FoodItem":
        """
        SIMPLE spec (no physics numbers in JSON):
          - category: "HOT" | "COLD" | "FROZEN" | "AMBIENT"
          - odor: "none" | "strong"                    (default: "none")
          - motion_sensitive: true/false               (default: false)
          - nonthermal_time_sensitive: true/false      (default: false)
          - prep_time_s: number                        (default: 0)
        """
        cat = FoodCategory[spec["category"].upper()]
        odor = str(spec.get("odor", "none")).lower()
        odor_level = OdorLevel.STRONG if odor == "strong" else OdorLevel.NONE

        p = FoodItem._preset_for(cat)
        item = FoodItem(
            name=name,
            category=cat,
            odor_level=odor_level,
            motion_sensitive=bool(spec.get("motion_sensitive", False)),
            nonthermal_time_sensitive=bool(spec.get("nonthermal_time_sensitive", False)),
            temperature_C=float(p["temperature_C"]),
            tau_air_s=float(p["tau_air_s"]),
            ideal_temp_C=tuple(p["ideal_temp_C"]),
            base_decay_per_s=float(p["base_decay_per_s"]),
            k_temp_dev=float(p["k_temp_dev"]),
            melt_start_C=(None if p["melt_start_C"] is None else float(p["melt_start_C"])),
            k_melt_per_C_per_s=float(p["k_melt_per_C_per_s"]),
        )
        item.begin_preparation(float(spec.get("prep_time_s", 0.0)))
        return item

    @staticmethod
    def load_simple_catalog(json_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Expected JSON:
        {
          "items": [
            {"name":"Burger", "category":"HOT", "odor":"none",
             "motion_sensitive":false, "nonthermal_time_sensitive":false, "prep_time_s":120},
            ...
          ]
        }
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        catalog: Dict[str, Dict[str, Any]] = {}
        for ent in items:
            nm = ent.get("name")
            if not nm:
                continue
            catalog[nm] = {
                "category": ent["category"],
                "odor": ent.get("odor", "none"),
                "motion_sensitive": bool(ent.get("motion_sensitive", False)),
                "nonthermal_time_sensitive": bool(ent.get("nonthermal_time_sensitive", False)),
                "prep_time_s": float(ent.get("prep_time_s", 0.0)),
            }
        return catalog

    @staticmethod
    def make_from_catalog(name: str, catalog: Dict[str, Dict[str, Any]]) -> "FoodItem":
        spec = catalog.get(name)
        if spec is None:
            raise KeyError(f"FoodItem spec not found for name='{name}'")
        return FoodItem.from_simple_spec(name, spec)

    # ---------------- UI helper ----------------
    def simple_metrics(self) -> Dict[str, float]:
        """
        Return the minimal signals you asked to display:
          - temp_C          : current temperature
          - damage          : motion damage (0..1, step-wise)
          - odor_mix        : odor contamination (0..1, to be set by bag)
          - state           : FROZEN->melt_fraction, else nonthermal_state (0..1)
        """
        state = self.melt_fraction if self.category is FoodCategory.FROZEN else self.nonthermal_state
        return {
            "temp_C": float(self.temperature_C),
            "damage": float(self.motion_damage),
            "odor_mix": float(self.odor_contamination),
            "state": float(state),
        }
