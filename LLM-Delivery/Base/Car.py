# -*- coding: utf-8 -*-
# Base/Car.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

class CarState(str, Enum):
    USABLE = "usable"
    PARKED = "parked"

@dataclass
class Car:
    owner_id: str
    avg_speed_cm_s: float = 2000.0     # ~72 km/h
    rate_per_min: float = 1.0          # $/min
    state: CarState = CarState.USABLE
    park_xy: Optional[Tuple[float, float]] = None

    def park_here(self, x: float, y: float):
        self.state = CarState.PARKED
        self.park_xy = (float(x), float(y))

    def unpark(self):
        self.state = CarState.USABLE
        self.park_xy = None
