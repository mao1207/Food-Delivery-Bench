# -*- coding: utf-8 -*-
# Base/DeliveryMan.py

import time, math, random
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple, Callable

from Base.Timer import VirtualClock
from Base.EScooter import EScooter, ScooterState
from Base.Car import Car, CarState
from Base.Store import StoreManager
from Base.Insulated_bag import InsulatedBag
from Base.Food import OdorLevel  # noqa: F401
from Base.Comms import get_comms, HelpType
from Base.Settlement import compute_settlement, SettlementConfig
from Base.ActionSpace import ACTION_API_SPEC, parse_action as parse_vlm_action
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm.base_model import BaseModel

# --- 仅处理两类：bytes 和 ndarray ---
def _ensure_png_bytes(img) -> bytes:
    # 1) 已是 bytes：直返
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    # 2) ndarray：转 PNG bytes（支持灰度/BGR/BGRA/float）
    import numpy as np
    from PIL import Image
    from io import BytesIO

    arr = img
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(arr.max()) if arr.size else 1.0
            if mx <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        mode = "L"
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # BGR -> RGB
        out = arr[:, :, ::-1].copy()
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # BGRA -> RGBA（保留 alpha）
        b, g, r, a = arr.transpose(2, 0, 1)
        out = np.dstack([r, g, b, a])
        mode = "RGBA"
    else:
        raise ValueError(f"unsupported ndarray shape: {arr.shape}")

    out = np.ascontiguousarray(out)
    bio = BytesIO()
    Image.fromarray(out, mode=mode).save(bio, format="PNG")
    return bio.getvalue()


# ===== Transport Modes =====
class TransportMode(str, Enum):
    WALK          = "walk"
    SCOOTER       = "e-scooter"
    DRAG_SCOOTER  = "drag_scooter"
    CAR           = "car"
    BUS           = "bus"

# ===== Speeds (cm/s) =====
DEFAULT_AVG_SPEED_CM_S = {
    TransportMode.WALK:          1350.0,
    TransportMode.SCOOTER:       800.0,
    TransportMode.DRAG_SCOOTER:  800.0,
    TransportMode.CAR:           2000.0,
    TransportMode.BUS:           1200.0,
}

# ===== Energy/Batt =====
ENERGY_DECAY_WALK_PCT_PER_M = 0.01
ENERGY_DECAY_DRAG_PCT_PER_M = 0.10
SCOOTER_BATT_DECAY_PCT_PER_M = 0.5

DEFAULT_ENERGY_PCT_MAX = 100.0
DEFAULT_REST_RATE_PCT_PER_MIN = 8.0
DEFAULT_HOSPITAL_DURATION_S = 20.0

ITEM_ESC_BATTERY_PACK = "escooter_battery_pack"
ITEM_ENERGY_DRINK     = "energy_drink"

# ===== Actions =====
class DMActionKind(str, Enum):
    MOVE_TO              = "move_to"
    ACCEPT_ORDER         = "accept_order"
    VIEW_ORDERS          = "view_orders"
    PICKUP               = "pickup"
    CHARGE_ESCOOTER      = "charge_escooter"
    WAIT                 = "wait"
    REST                 = "rest"
    BUY                  = "buy"
    USE_BATTERY_PACK     = "use_battery_pack"
    USE_ENERGY_DRINK     = "use_energy_drink"
    POST_HELP_REQUEST    = "post_help_request"
    ACCEPT_HELP_REQUEST  = "accept_help_request"
    EDIT_HELP_REQUEST    = "edit_help_request"
    GIVE_TO_AGENT        = "give_to_agent"
    SWITCH_TRANSPORT     = "switch_transport"
    RENT_CAR             = "rent_car"
    RETURN_CAR           = "return_car"
    BOARD_BUS            = "board_bus"
    ALIGHT_BUS           = "alight_bus"
    WAIT_FOR_BUS         = "wait_for_bus"

@dataclass
class DMAction:
    kind: DMActionKind
    data: Dict[str, Any] = field(default_factory=dict)
    on_done: Optional[Callable[['DeliveryMan'], None]] = None

@dataclass
class DeliveryMan:
    agent_id: str
    city_map: Any
    world_nodes: List[Dict[str, Any]]
    x: float
    y: float
    mode: TransportMode = TransportMode.WALK
    clock: VirtualClock = field(default_factory=lambda: VirtualClock())

    # runtime
    speed_cm_s: float = field(init=False)
    energy_pct: float = field(init=False, default=DEFAULT_ENERGY_PCT_MAX)
    earnings_total: float = field(default=100.0)
    is_rescued: bool = field(default=False)

    # orders
    active_orders: List[Any] = field(default_factory=list)
    carrying: List[int] = field(default_factory=list)
    current_order: Optional[Any] = field(default=None)

    # misc
    last_updated_s: float = field(default_factory=time.time)
    name: str = "DM"

    # viewer / ue
    _viewer: Optional[Any] = field(default=None, repr=False)
    _viewer_agent_id: Optional[str] = field(default=None, repr=False)
    _ue: Optional[Any] = field(default=None, repr=False)

    # managers
    _order_manager: Optional[Any] = field(default=None, repr=False)
    _store_manager: Optional[StoreManager] = field(default=None, repr=False)
    _bus_manager: Optional[Any] = field(default=None, repr=False)

    # scheduling
    _queue: List[DMAction] = field(default_factory=list, repr=False)
    _current: Optional[DMAction] = field(default=None, repr=False)

    # history
    completed_orders: List[Dict[str, Any]] = field(default_factory=list)

    # handlers
    _action_handlers: Dict[DMActionKind, Callable[['DeliveryMan', DMAction, bool], None]] = field(init=False, repr=False)

    # equipment
    insulated_bag: Optional[InsulatedBag] = None
    e_scooter: Optional[EScooter] = None
    car: Optional[Car] = None
    inventory: Dict[str, int] = field(default_factory=dict)

    # flags
    towing_scooter: bool = False

    # contexts（虚拟时间）
    _charge_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)
    _rest_ctx:   Optional[Dict[str, float]] = field(default=None, repr=False)
    _wait_ctx:   Optional[Dict[str, float]] = field(default=None, repr=False)
    _hospital_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)

    # movement context
    _move_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)  # {"tx":float,"ty":float,"tol":float,"blocked":0/1}

    # rental billing
    _rental_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)  # {"last_tick_sim": float, "rate_per_min": float}
    
    # bus context
    _bus_ctx: Optional[Dict[str, Any]] = field(default=None, repr=False)  # {"bus_id": str, "boarding_stop": str, "target_stop": str}

    # movement interrupt
    _interrupt_move_flag: bool = field(default=False, repr=False)
    _interrupt_reason: Optional[str] = field(default=None, repr=False)

    # rest config
    rest_rate_pct_per_min: float = field(default=DEFAULT_REST_RATE_PCT_PER_MIN)
    low_energy_threshold_pct: float = field(default=30.0)

    # 强制取餐
    _pickup_ready_orders: Optional[List[Any]] = None
    _force_pickup_now: bool = False

    # 思考暂停
    _timers_paused: bool = field(default=False, repr=False)
    _orders_last_tick_sim: Optional[float] = field(default=None, repr=False)
    _last_bag_tick_sim: Optional[float] = None

    # VLM
    vlm_prompt: str = "prompt"
    vlm_past_memory: List[str] = field(default_factory=list)
    vlm_ephemeral: Dict[str, str] = field(default_factory=dict)
    vlm_errors: List[str] = field(default_factory=list)
    vlm_last_action_note: Optional[str] = None
    vlm_last_compiled_input: Optional[str] = None

    _vlm_client: Optional["BaseModel"] = field(default=None, repr=False)

    map_exportor: Optional[Any] = field(default=None, repr=False)

    # avg speeds
    avg_speed_by_mode: Dict[TransportMode, float] = field(default_factory=lambda: dict(DEFAULT_AVG_SPEED_CM_S))

    def __post_init__(self):
        self.speed_cm_s = float(self.avg_speed_by_mode.get(self.mode, DEFAULT_AVG_SPEED_CM_S[self.mode]))
        self.energy_pct = DEFAULT_ENERGY_PCT_MAX
        if self.e_scooter is None:
            self.e_scooter = EScooter()
        setattr(self.e_scooter, "owner_id", str(self.agent_id))
        if self.insulated_bag is None:
            self.insulated_bag = InsulatedBag()

        self._action_handlers = {
            DMActionKind.MOVE_TO:             self._handle_move_to,
            DMActionKind.ACCEPT_ORDER:        self._handle_accept_order,
            DMActionKind.VIEW_ORDERS:         self._handle_view_orders,
            DMActionKind.PICKUP:              self._handle_pickup_food,
            DMActionKind.CHARGE_ESCOOTER:     self._handle_charge_escooter,
            DMActionKind.WAIT:                self._handle_wait,
            DMActionKind.REST:                self._handle_rest,
            DMActionKind.BUY:                 self._handle_buy,
            DMActionKind.USE_BATTERY_PACK:    self._handle_use_battery_pack,
            DMActionKind.USE_ENERGY_DRINK:    self._handle_use_energy_drink,
            DMActionKind.POST_HELP_REQUEST:   self._handle_post_help_request,
            DMActionKind.ACCEPT_HELP_REQUEST: self._handle_accept_help_request,
            DMActionKind.EDIT_HELP_REQUEST:   self._handle_edit_help_request,
            DMActionKind.GIVE_TO_AGENT:       self._handle_give_to_agent,
            DMActionKind.SWITCH_TRANSPORT:    self._handle_switch_transport,
            DMActionKind.RENT_CAR:            self._handle_rent_car,
            DMActionKind.RETURN_CAR:          self._handle_return_car,
            DMActionKind.BOARD_BUS:           self._handle_board_bus,
            DMActionKind.ALIGHT_BUS:          self._handle_alight_bus,
            DMActionKind.WAIT_FOR_BUS:        self._handle_wait_for_bus,
        }
        self._recalc_towing()

    def _save_png(self, data: bytes, path: str) -> bool:
        with open(path, "wb") as f:
            f.write(data)
        return True

    def _export_vlm_images_debug_once(self, save_dir: str = "debug_snaps") -> List[str]:
        imgs = self.vlm_collect_images()  # 全是 PNG bytes 或 None
        os.makedirs(save_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        names = ["global", "local", "fpv"]
        saved_paths: List[str] = []

        for i, img in enumerate((imgs or [])[:3]):
            if img is None:
                continue
            path = os.path.join(save_dir, f"agent{self.agent_id}_{ts}_{names[i]}.png")
            self._save_png(img, path)
            saved_paths.append(path)

        # if saved_paths:
        #     self._log(f"debug snapshots saved: {saved_paths}")
        return saved_paths

    # ===== wiring =====
    def set_order_manager(self, om: Any): self._order_manager = om
    def set_store_manager(self, store_mgr: Any): self._store_manager = store_mgr
    def set_bus_manager(self, bus_mgr: Any): self._bus_manager = bus_mgr
    def set_ue(self, ue: Any): self._ue = ue

    def register_to_comms(self):
        comms = get_comms()
        if comms: comms.register_agent(self)

    def bind_viewer(self, viewer: Any):
        self._viewer = viewer
        self._viewer_agent_id = str(self.agent_id)
        if hasattr(viewer, "add_agent"):
            def _proxy_on_done(aid, *args):
                event = "move"
                if len(args) == 3:
                    fx, fy, event = args
                elif len(args) == 2:
                    fx, fy = args
                else:
                    fx = fy = None
                self._on_view_event(aid, event, {"x": fx, "y": fy})
            viewer.add_agent(self._viewer_agent_id, self.x, self.y,
                             speed_cm_s=self.get_current_speed_for_viewer(),
                             label_text=f"{self.agent_id}",
                             on_anim_done=_proxy_on_done)
        if hasattr(self._viewer, "register_delivery_man"):
            self._viewer.register_delivery_man(self)

    def bind_simworld(self):
        if self._ue and hasattr(self._ue, "spawn_delivery_man"):
            self._ue.spawn_delivery_man(self.agent_id, self.x, self.y)

    def set_vlm_client(self, client: "BaseModel"):
        self._vlm_client = client
        self.vlm_infer_fn = self._vlm_infer

    # DeliveryMan.vlm_collect_images
    def vlm_collect_images(self) -> List[bytes]:
        imgs = [None, None, None]

        # 0/1：两张地图（可能本来就是 bytes，也可能是 ndarray） -> 统一成 bytes
        exp = getattr(self, "map_exportor", None)
        if exp is not None:
            orders = list(self.active_orders) if self.active_orders else []
            g, l = exp.export(agent_xy=(float(self.x), float(self.y)), orders=orders)
            imgs[0] = _ensure_png_bytes(g) if g is not None else None
            imgs[1] = _ensure_png_bytes(l) if l is not None else None

        # 2：第一视角（UE 返回 ndarray 或 bytes） -> 统一成 bytes
        cam_id = int(getattr(self, "_viewer_agent_id", getattr(self, "name", "0")))
        fpv = self._ue.get_camera_observation(cam_id, viewmode="lit")
        imgs[2] = _ensure_png_bytes(fpv) if fpv is not None else None

        return imgs

    def _vlm_infer(self, compiled_prompt: str) -> str:
        if self._vlm_client is None:
            raise RuntimeError("VLM client not set. Call set_vlm_client(client) first.")
        images = self.vlm_collect_images()
        return self._vlm_client.generate(prompt=compiled_prompt, images=images)

    # ===== pause/resume =====
    def _ctx_mark_pause(self, ctx: Optional[Dict[str, float]], now: float):
        if ctx is not None and "paused_at" not in ctx:
            ctx["paused_at"] = float(now)

    def _ctx_mark_resume(self, ctx: Optional[Dict[str, float]], now: float):
        if ctx is not None and ctx.get("paused_at") is not None:
            delta = float(now) - float(ctx["paused_at"])
            ctx["start_sim"] += delta
            ctx["end_sim"]   += delta
            ctx.pop("paused_at", None)

    def timers_pause(self):
        if self._timers_paused: return
        now = self.clock.now_sim()
        self._ctx_mark_pause(self._charge_ctx, now)
        self._ctx_mark_pause(self._rest_ctx,  now)
        self._ctx_mark_pause(self._wait_ctx,  now)
        self._ctx_mark_pause(self._hospital_ctx, now)
        self._timers_paused = True

    def timers_resume(self):
        if not self._timers_paused: return
        now = self.clock.now_sim()
        self._ctx_mark_resume(self._charge_ctx, now)
        self._ctx_mark_resume(self._rest_ctx,  now)
        self._ctx_mark_resume(self._wait_ctx,  now)
        self._ctx_mark_resume(self._hospital_ctx, now)
        if self._rental_ctx is not None:
            self._rental_ctx["last_tick_sim"] = now
        self._timers_paused = False
        self._orders_last_tick_sim = now

    # ===== viewer events =====
    def _on_view_event(self, agent_id: str, event: str, payload: Dict[str, Any]):
        fx = float(payload.get("x", self.x)) if payload.get("x", self.x) is not None else self.x
        fy = float(payload.get("y", self.y)) if payload.get("y", self.y) is not None else self.y
        self.x, self.y = fx, fy

        if event == "move":
            self._auto_try_dropoff()

            tol = 300.0
            now_sim = self.clock.now_sim()
            orders_here: List[Any] = []
            for o in self.active_orders:
                if getattr(o, "has_picked_up", False): continue
                pu = self._xy_of_node(getattr(o, "pickup_node", None))
                if pu and self._is_at_xy(pu[0], pu[1], tol_cm=tol):
                    orders_here.append(o)

            if orders_here:
                ready_orders: List[Any] = []
                waiting_notes: List[str] = []
                for o in orders_here:
                    is_ready = True if not hasattr(o, "is_ready_for_pickup") else o.is_ready_for_pickup(now_sim)
                    if is_ready:
                        ready_orders.append(o)
                    else:
                        remain = float(o.remaining_prep_s(now_sim)) if hasattr(o, "remaining_prep_s") else 0.0
                        mins = int((remain + 59) // 60)
                        waiting_notes.append(f"#{getattr(o,'id',None)} ~{mins} mins")
                if waiting_notes:
                    self.vlm_ephemeral["pickup_wait_note"] = "Arrived at pickup door, not ready: " + ", ".join(waiting_notes)
                if ready_orders:
                    self.vlm_ephemeral["pickup_hint"] = self._pickup_hint_for_orders(ready_orders)
                    self._pickup_ready_orders = list(ready_orders); self._force_pickup_now = True
                else:
                    self._pickup_ready_orders = None; self._force_pickup_now = False; self.vlm_ephemeral.pop("pickup_hint", None)
            else:
                if not self._force_pickup_now:
                    self.vlm_ephemeral.pop("pickup_hint", None)
                    self.vlm_ephemeral.pop("pickup_wait_note", None)
            return

        if event == "blocked":
            if self._interrupt_reason == "escooter_depleted":
                self._log("movement blocked (ESCOOTER depleted) -> re-decide")
            else:
                self.vlm_add_error("movement blocked"); self._log("movement blocked")
            self._interrupt_reason = None
            if self._move_ctx is not None:
                self._move_ctx["blocked"] = 1.0
            return

    # ===== logging & VLM =====
    def _log(self, text: str):
        if self._viewer and hasattr(self._viewer, "log_action"):
            prefix = f"[Agent {self._viewer_agent_id or self.name}] "
            self._viewer.log_action(prefix + text)
        else:
            print(f"[DeliveryMan {self.name}] {text}")

    def vlm_add_memory(self, text: str): self.vlm_past_memory.append(str(text))
    def vlm_clear_memory(self): self.vlm_past_memory.clear()
    def vlm_add_ephemeral(self, tag: str, text: str): self.vlm_ephemeral[str(tag)] = str(text)
    def vlm_clear_ephemeral(self): self.vlm_ephemeral.clear(); self.vlm_last_action_note = None
    def vlm_add_error(self, msg: str): self.vlm_errors.append(str(msg))
    def vlm_clear_errors(self): self.vlm_errors.clear()
    def _register_success(self, note: str): self.vlm_last_action_note = note; self.vlm_clear_errors()

    # ===== state/speed =====
    def is_busy(self) -> bool:
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "is_busy"):
            return bool(self._viewer.is_busy(self._viewer_agent_id))
        return self._current is not None

    def _recalc_towing(self):
        if self.mode == TransportMode.DRAG_SCOOTER:
            self.towing_scooter = True
        elif self.e_scooter and self.e_scooter.state == ScooterState.DEPLETED and not self.e_scooter.park_xy:
            self.towing_scooter = True
        else:
            self.towing_scooter = False

    def get_current_speed_for_viewer(self) -> float:
        self._recalc_towing()
        return float(self.speed_cm_s)

    def set_mode(self, mode: TransportMode, *, override_speed_cm_s: Optional[float] = None):
        mode = TransportMode(mode)
        if mode == TransportMode.SCOOTER:
            if not self.e_scooter:
                mode = TransportMode.DRAG_SCOOTER
            else:
                owner_ok = (getattr(self.e_scooter, "owner_id", None) == str(self.agent_id))
                usable   = (self.e_scooter.state != ScooterState.DEPLETED)
                if not (owner_ok and usable):
                    mode = TransportMode.DRAG_SCOOTER
        self.mode = mode

        if self.mode == TransportMode.SCOOTER and self.e_scooter:
            base = float(self.e_scooter.avg_speed_cm_s)
            if override_speed_cm_s is not None:
                base = self.e_scooter.clamp_speed(float(override_speed_cm_s)); self.e_scooter.avg_speed_cm_s = base
            self.speed_cm_s = base
        elif self.mode == TransportMode.CAR and self.car:
            base = float(self.car.avg_speed_cm_s)
            if override_speed_cm_s is not None: base = float(override_speed_cm_s)
            self.speed_cm_s = base
        else:
            base = float(self.avg_speed_by_mode.get(self.mode, DEFAULT_AVG_SPEED_CM_S[self.mode]))
            if override_speed_cm_s is not None: base = float(override_speed_cm_s)
            self.speed_cm_s = base

        self._recalc_towing()
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "set_speed"):
            self._viewer.set_speed(self._viewer_agent_id, self.get_current_speed_for_viewer())
        if self._ue and hasattr(self._ue, "set_speed"):
            self._ue.set_speed(self._viewer_agent_id, self.get_current_speed_for_viewer())

    def set_speed_override(self, speed_cm_s: float):
        v = float(speed_cm_s)
        if self.mode == TransportMode.SCOOTER and self.e_scooter:
            v = self.e_scooter.clamp_speed(v); self.e_scooter.avg_speed_cm_s = v
        self.speed_cm_s = v
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "set_speed"):
            self._viewer.set_speed(self._viewer_agent_id, self.get_current_speed_for_viewer())
        if self._ue and hasattr(self._ue, "set_speed"):
            self._ue.set_speed(self._viewer_agent_id, self.get_current_speed_for_viewer())

    def on_move_consumed(self, distance_cm: float):
        if distance_cm <= 0.0: return
        self._consume_by_distance(distance_cm)

    def _consume_by_distance(self, distance_cm: float):
        distance_m = max(0.0, float(distance_cm) / 100.0)
        self._recalc_towing()

        if self.mode in (TransportMode.WALK, TransportMode.DRAG_SCOOTER):
            per_m = ENERGY_DECAY_DRAG_PCT_PER_M if self.mode == TransportMode.DRAG_SCOOTER else ENERGY_DECAY_WALK_PCT_PER_M
            self.energy_pct -= per_m * distance_m
            if self.energy_pct <= 0.0:
                self.energy_pct = 0.0
                self._trigger_hospital_if_needed()

        elif self.mode == TransportMode.SCOOTER and self.e_scooter:
            before = self.e_scooter.battery_pct
            delta_pct = distance_m * SCOOTER_BATT_DECAY_PCT_PER_M
            self.e_scooter.consume_pct(delta_pct)
            if self.e_scooter.battery_pct <= 0.0 and before > 0.0:
                self.e_scooter.state = ScooterState.DEPLETED
                self._interrupt_reason = "escooter_depleted"
                self._log("ESCOOTER depleted -> interrupt & switch to DRAG_SCOOTER")
                self._interrupt_move_flag = True
                self.set_mode(TransportMode.DRAG_SCOOTER)

        elif self.mode == TransportMode.CAR:
            pass

    def rescue(self):
        self.energy_pct = 100.0
        self.is_rescued = False

    # ===== map helpers =====
    def _xy_of_node(self, node: Any) -> Optional[Tuple[float, float]]:
        try:
            return float(node.position.x), float(node.position.y)
        except Exception:
            return None

    def _is_at_xy(self, x: float, y: float, tol_cm: float = 300.0) -> bool:
        return math.hypot(self.x - x, self.y - y) <= float(tol_cm)

    def _nearest_poi_xy(self, kind: str, tol_cm: float = 300.0) -> Optional[Tuple[float, float]]:
        cand = None; best_d = float("inf")
        for n in getattr(self.city_map, "nodes", []):
            if getattr(n, "type", "") != kind and getattr(self.city_map._door2poi.get(n), "type", "") != kind:
                continue
            xy = self._xy_of_node(n)
            if not xy: continue
            d = math.hypot(self.x - xy[0], self.y - xy[1])
            if d < best_d:
                best_d, cand = d, xy
        if cand is not None and best_d <= tol_cm:
            return cand
        return None

    def _closest_poi_xy(self, kind: str) -> Optional[Tuple[float, float]]:
        cand = None; best_d = float("inf")
        for n in getattr(self.city_map, "nodes", []):
            if getattr(n, "type", "") != kind and getattr(self.city_map._door2poi.get(n), "type", "") != kind:
                continue
            xy = self._xy_of_node(n)
            if not xy: continue
            d = math.hypot(self.x - xy[0], self.y - xy[1])
            if d < best_d:
                best_d, cand = d, xy
        return cand

    # ===== VLM text =====
    def _fmt_xy_m(self, x_cm: float, y_cm: float) -> str:
        return f"({x_cm/100.0:.2f}m, {y_cm/100.0:.2f}m)"

    def _fmt_xy_m_opt(self, xy: Optional[Tuple[float, float]]) -> str:
        if not xy:
            return "N/A"
        x, y = xy
        return self._fmt_xy_m(float(x), float(y))

    def _remaining_range_m(self) -> Optional[float]:
        if not self.e_scooter: return None
        return float(self.e_scooter.battery_pct) / max(1e-9, SCOOTER_BATT_DECAY_PCT_PER_M)

    def _agent_state_text(self) -> str:
        active_ids = [getattr(o, "id", None) for o in self.active_orders if getattr(o, "id", None) is not None]
        carrying_ids = list(self.carrying)
        mode_str = "towing a scooter" if self.towing_scooter else self.mode.value
        speed_kmh = self.get_current_speed_for_viewer() * 0.036
        lines = []
        lines.append(f"Your current mode is {mode_str}, at {self._fmt_xy_m(self.x, self.y)}.")
        lines.append(f"Your speed ~{speed_kmh:.1f} km/h, energy {self.energy_pct:.0f}%.")
        lines.append(f"Earnings ${self.earnings_total:.2f}.")
        if active_ids: lines.append(f"Active orders: {', '.join(map(str, active_ids))}.")
        if carrying_ids: lines.append(f"Carrying: {', '.join(map(str, carrying_ids))}.")
        lines.append(f"Rest +{self.rest_rate_pct_per_min:.1f}%/min.")
        if self.e_scooter:
            rng_m = self._remaining_range_m(); rng_km = (rng_m/1000.0) if rng_m is not None else None
            park_str = f"parked at {self._fmt_xy_m_opt(self.e_scooter.park_xy)}" if self.e_scooter.park_xy else "not parked"
            if rng_km is not None:
                lines.append(f"Scooter {self.e_scooter.state.value}, batt {self.e_scooter.battery_pct:.0f}%, range {rng_km:.1f} km.")
            else:
                lines.append(f"Scooter {self.e_scooter.state.value}, batt {self.e_scooter.battery_pct:.0f}%, range N/A.")
            lines.append(f"Charge {self.e_scooter.charge_rate_pct_per_min:.1f}%/min, {park_str}.")
        if self.car:
            lines.append(f"Car {self.car.state.value}, rate ${self.car.rate_per_min:.2f}/min, park_xy={self._fmt_xy_m_opt(self.car.park_xy) if self.car.park_xy else 'N/A'}, rental={'on' if self._rental_ctx else 'off'}.")
        return " ".join(lines)

    def build_vlm_input(self) -> str:
        parts: List[str] = ["### system_prompt\n"+self.vlm_prompt]
        if self.vlm_past_memory:
            parts.append("### past_memory"); parts += [f"- {m}" for m in self.vlm_past_memory]
        parts.append("### agent_state"); parts.append(self._agent_state_text())
        parts.append("### map_snapshot"); parts.append(self._map_brief())
        parts.append("### action_api"); parts.append(ACTION_API_SPEC.strip())
        if self.vlm_last_action_note:
            parts.append("### last_action_note"); parts.append(self.vlm_last_action_note)
        if self.vlm_errors:
            parts.append("### recent_errors"); parts += [f"- {e}" for e in self.vlm_errors]
        if self._force_pickup_now and self.vlm_ephemeral.get("pickup_hint"):
            parts.append("### ephemeral_context"); parts.append(f"[pickup_hint]\n{self.vlm_ephemeral['pickup_hint']}")
        elif self.vlm_ephemeral:
            parts.append("### ephemeral_context")
            for k, v in self.vlm_ephemeral.items():
                if k != "pickup_hint": parts.append(f"[{k}]\n{v}")
        txt = "\n".join(parts)
        self.vlm_last_compiled_input = txt
        return txt

    def _map_brief(self) -> str:
        if hasattr(self.city_map, "agent_info_package_xy"):
            pkg = self.city_map.agent_info_package_xy(float(self.x), float(self.y), include_docks=False, limit_next=20, limit_s=40, limit_poi=80)
            if isinstance(pkg, dict) and pkg.get("text"): return str(pkg["text"])
        return "map_brief: N/A"

    # ===== history =====
    def record_completed_order(self, order: Any, *, duration_s: float, rating: float):
        oid = getattr(order, "id", None)
        earn = getattr(order, "earnings", None)
        self.completed_orders.append(dict(
            id=oid, duration_s=float(duration_s), rating=float(rating), earnings=earn,
            pickup=getattr(order, "pickup_road_name", ""), dropoff=getattr(order, "dropoff_road_name", "")
        ))

    # ===== VLM decider =====
    # def _default_decider(self) -> Optional[DMAction]:
    #     if self.is_rescued or self._hospital_ctx is not None or self.energy_pct <= 0.0:
    #         return None
    #     input_prompt = self.build_vlm_input()
    #     self.vlm_last_compiled_input = input_prompt
    #     infer = getattr(self, "vlm_infer_fn", None)
    #     if not callable(infer):
    #         self.vlm_add_error("VLM not configured: call set_vlm(infer_fn) first.")
    #         return None
    #     try:
    #         raw = infer(input_prompt)
    #     except Exception as e:
    #         self.vlm_add_error(f"VLM inference error: {e}")
    #         return None
    #     try:
    #         return parse_vlm_action(raw, self)
    #     except Exception as e:
    #         self.vlm_add_error(f"VLM parse error: {e}; output sample={str(raw)[:160]}")
    #         return None

    def _default_decider(self) -> Optional[DMAction]:
        input_prompt = self.build_vlm_input()
        self.vlm_last_compiled_input = input_prompt

        # self._export_vlm_images_debug_once(save_dir="debug_snaps")
        # imgs = self.vlm_collect_images()

        # ===== 到店即取：一次性生成 bag_cmd；触发后清理标记 =====
        if self._force_pickup_now and self._pickup_ready_orders:
            def _auto_bag_cmd_for_orders(orders: List[Any]) -> str:
                # 把每单所有条目塞进第一个仓位（若无则用 "A"），确定性更利于复现
                if self.insulated_bag and getattr(self.insulated_bag, "labels", None):
                    lab = str(self.insulated_bag.labels[0]) or "A"
                else:
                    lab = "A"
                lines = []
                for o in orders:
                    oid = getattr(o, "id", None)
                    n = len(list(getattr(o, "items", []) or []))
                    if n > 0:
                        idxs = ",".join(str(i) for i in range(1, n + 1))
                        lines.append(f"order {oid}: {idxs} -> {lab}")
                    else:
                        lines.append(f"order {oid}:")
                return "\n".join(lines)

            orders_copy = list(self._pickup_ready_orders)
            cmd = _auto_bag_cmd_for_orders(orders_copy)
            self._force_pickup_now = False
            self._pickup_ready_orders = None
            self.vlm_ephemeral.pop("pickup_hint", None)
            return DMAction(DMActionKind.PICKUP, data=dict(orders=orders_copy, tol_cm=300.0, bag_cmd=cmd))

        # ===== 不可决策状态 =====
        if self.is_rescued or self._hospital_ctx is not None or self.energy_pct <= 0.0:
            return None

        def _at(x: float, y: float, tol: float = 300.0) -> bool:
            return self._is_at_xy(float(x), float(y), tol_cm=float(tol))

        # ===== 低电：去充电站 -> 到站才下发 CHARGE_ESCOOTER =====
        if self.e_scooter and self.e_scooter.battery_pct <= 10.0 and self._charge_ctx is None:
            tgt = self._closest_poi_xy("charging_station")
            if tgt:
                tx, ty = tgt
                if not _at(tx, ty):
                    return DMAction(DMActionKind.MOVE_TO, data=dict(tx=tx, ty=ty, use_route=True, snap_cm=120.0))
                return DMAction(DMActionKind.CHARGE_ESCOOTER, data=dict(target_pct=100.0))

        # ===== 低体力：去休息区 -> 到位才 REST =====
        if self.energy_pct <= self.low_energy_threshold_pct:
            tgt = self._closest_poi_xy("rest_area")
            if tgt:
                tx, ty = tgt
                if not _at(tx, ty):
                    return DMAction(DMActionKind.MOVE_TO, data=dict(tx=tx, ty=ty, use_route=True, snap_cm=120.0))
                return DMAction(DMActionKind.REST, data=dict(target_pct=100.0))

        # ===== 当前订单状态 =====
        pending_pick = [o for o in self.active_orders if not getattr(o, "has_picked_up", False)]
        delivering   = [o for o in self.active_orders if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False)]

        # 已取 -> 送（选最近送达点；若已在点位，则不下发 MOVE，交给 poll_time_events 的自动送达）
        if delivering:
            def _d_to_do(o):
                n = getattr(o, "dropoff_node", None)
                if not n: return float("inf")
                return math.hypot(self.x - float(n.position.x), self.y - float(n.position.y))
            o = min(delivering, key=_d_to_do)
            n = getattr(o, "dropoff_node", None)
            if n:
                dx, dy = float(n.position.x), float(n.position.y)
                if not _at(dx, dy):  # 避免原地 MOVE
                    return DMAction(DMActionKind.MOVE_TO, data=dict(tx=dx, ty=dy, use_route=True, snap_cm=120.0))
            return None

        # 待取 -> 去店（选最近取餐点；到店由 _on_view_event 触发“强制取餐”）
        if pending_pick:
            def _d_to_pu(o):
                n = getattr(o, "pickup_node", None)
                if not n: return float("inf")
                return math.hypot(self.x - float(n.position.x), self.y - float(n.position.y))
            o = min(pending_pick, key=_d_to_pu)
            n = getattr(o, "pickup_node", None)
            if n:
                px, py = float(n.position.x), float(n.position.y)
                if not _at(px, py):
                    return DMAction(DMActionKind.MOVE_TO, data=dict(tx=px, ty=py, use_route=True, snap_cm=120.0))

                # 已在店门口：如果此刻没 ready，就不要卡住，立刻去别处走动一下
                ready_now = bool(self._force_pickup_now and self._pickup_ready_orders)
                if not ready_now:
                    for _ in range(5):
                        tgt = self.random_target_on_roads()
                        if tgt is None: break
                        tx, ty = tgt
                        if not _at(tx, ty, tol=400.0):
                            return DMAction(DMActionKind.MOVE_TO, data=dict(tx=tx, ty=ty, use_route=True, snap_cm=120.0))
                # 若 ready_now 为真，不在这里下指令；让顶部的“到店即取”分支触发 PICKUP
            return None

        # 没单 -> 接单（选最近取餐点的订单）
        if self._order_manager and not self.active_orders:
            pool = list(getattr(self._order_manager, "_orders", []) or [])
            if pool:
                def _d_to_order(o):
                    n = getattr(o, "pickup_node", None)
                    if not n: return float("inf")
                    return math.hypot(self.x - float(n.position.x), self.y - float(n.position.y))
                order = min(pool, key=_d_to_order)
                return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(order=order))

        # 闲逛：保证目标与当前位置有最小距离，避免原地 MOVE
        for _ in range(5):
            tgt = self.random_target_on_roads()
            if tgt is None: break
            tx, ty = tgt
            if not _at(tx, ty, tol=400.0):
                return DMAction(DMActionKind.MOVE_TO, data=dict(tx=tx, ty=ty, use_route=True, snap_cm=120.0))
        return None

    # ===== loop/scheduling =====
    def kickstart(self):
        if self._current is None and not self._queue:
            self.timers_pause()
            act = self._default_decider()
            self.timers_resume()
            if act is not None: self.enqueue_action(act)

    def enqueue_action(self, action: DMAction, *, allow_interrupt: bool = False):
        if self.is_rescued or self._hospital_ctx is not None: return
        if allow_interrupt and self._current is not None:
            self._queue.clear(); self._current = None
            self._start_action(action, allow_interrupt=True); return
        self._queue.append(action); self._start_next_if_idle()

    def clear_queue(self): self._queue.clear()

    def _start_next_if_idle(self):
        if self._current is None and self._queue:
            act = self._queue.pop(0); self._start_action(act)

    def _start_action(self, act: DMAction, allow_interrupt: bool = True):
        self._current = act
        handler = self._action_handlers.get(act.kind)
        if handler is None:
            self._finish_action(success=False); return
        handler(self, act, allow_interrupt)

    def _finish_action(self, *, success: bool):
        if self._current and callable(self._current.on_done):
            self._current.on_done(self)
        if success and self._current:
            self._register_success(f"action {self._current.kind.value} succeeded")
        self._current = None

        self.timers_pause()
        next_act = self._default_decider()
        self.timers_resume()

        if next_act is not None:
            self.enqueue_action(next_act)
        self._start_next_if_idle()

    def register_action(self, kind: DMActionKind, handler: Callable[['DeliveryMan', DMAction, bool], None]):
        self._action_handlers[kind] = handler

    # ===== Handlers =====
    def _handle_move_to(self, _self, act: DMAction, allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        self._interrupt_move_flag = False
        sx, sy = float(self.x), float(self.y)
        tx, ty = float(act.data.get("tx", self.x)), float(act.data.get("ty", self.y))
        tol = float(act.data.get("arrive_tolerance_cm", 300.0))

        self._recalc_towing()
        if "expected_dist_cm" not in act.data:
            act.data["expected_dist_cm"] = self._estimate_distance_cm(sx, sy, tx, ty, bool(act.data.get("use_route", True)), float(act.data.get("snap_cm", 120.0)))

        mode_str = 'towing' if self.towing_scooter else self.mode.value
        speed_to_use = self.get_current_speed_for_viewer()
        self._log(f"move from {self._fmt_xy_m(sx, sy)} to {self._fmt_xy_m(tx, ty)} [mode={mode_str}, speed={speed_to_use:.1f} cm/s]")

        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "set_speed"): self._viewer.set_speed(self._viewer_agent_id, speed_to_use)
        if self._ue and hasattr(self._ue, "set_speed"): self._ue.set_speed(self._viewer_agent_id, speed_to_use)

        if hasattr(self.city_map, "route_xy_to_xy"):
            route = self.city_map.route_xy_to_xy(float(sx), float(sy), float(tx), float(ty), snap_cm=float(120)) or []
        else:
            route = [(sx, sy), (tx, ty)]

        # 记录移动上下文，统一在 poll_time_events 判定完成/失败
        self._move_ctx = {"tx": float(tx), "ty": float(ty), "tol": float(tol), "blocked": 0.0}

        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "go_to_xy"):
            self._viewer.go_to_xy(self._viewer_agent_id, route, allow_interrupt=allow_interrupt, show_path_ms=2000)
        if self._ue and hasattr(self._ue, "go_to_xy_async"):
            self._ue.go_to_xy_async(self._viewer_agent_id, route, speed_cm_s=self.get_current_speed_for_viewer(),
                                    accel_cm_s2=None, decel_cm_s2=None, arrive_tolerance_cm=tol)

    def _handle_accept_order(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        order = act.data.get("order"); order_manager = act.data.get("order_manager") or self._order_manager
        if self.is_rescued or order is None:
            self.vlm_add_error("accept_order failed"); self._finish_action(success=False); return
        order.is_accepted = True
        if all(o is not order for o in self.active_orders): self.active_orders.append(order)
        self.current_order = order
        order.sim_started_s = float(self.clock.now_sim()); order.sim_elapsed_active_s = 0.0; order.sim_delivered_s = None
        now = order.sim_started_s; longest = float(getattr(order, "prep_longest_s", 0.0) or 0.0)
        order.prep_started_sim = now; order.prep_ready_sim = now + longest
        if order_manager and hasattr(order_manager, "remove_order") and hasattr(order, "id"):
            order_manager.remove_order(order.id, self.city_map, self.world_nodes)
        oid = getattr(order, "id", None); self._log(f"accept order{f' #{oid}' if oid is not None else ''}")
        self._finish_action(success=True)

    def _handle_view_orders(self, _self, act: DMAction, _allow_interrupt: bool):
        om = act.data.get("order_manager") or self._order_manager
        if om and hasattr(om, "orders_text"):
            pool_text = om.orders_text(); self.vlm_last_action_note = "You just viewed the order pool."
            if pool_text: self.vlm_add_ephemeral("order_pool", pool_text); self._log("view orders")
        self._finish_action(success=False)

    def _pickup_hint_for_orders(self, orders):
        lines = ["You have ARRIVED at a pickup door. Immediately place items into bag and confirm pickup.",
                 "Output ONE combined bag_cmd per order: 'order <id>: 1,2 -> A; 3 -> B'", ""]
        for o in orders:
            items = list(getattr(o, "items", []) or []); lines.append(f"- Order #{getattr(o,'id',None)} items:")
            if items:
                for i, it in enumerate(items, start=1):
                    name = getattr(it, "name", None) or getattr(it, "title", None) or getattr(it, "label", None) or str(it)
                    lines.append(f"    {i}. {name}")
            else: lines.append("    (none)")
        lines += ["", "Current bag layout:", self.insulated_bag.list_items() if self.insulated_bag else "(no bag)", "", "Example:", "  order 12: 1,2 -> A; 3 -> B"]
        return "\n".join(lines)

    def _handle_pickup_food(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        orders = list(act.data.get("orders") or []); tol = float(act.data.get("tol_cm", 300.0)); spec_text = (act.data.get("bag_cmd") or "").strip()

        here_orders = []
        for o in orders:
            if getattr(o, "has_picked_up", False): continue
            pu_xy = self._xy_of_node(getattr(o, "pickup_node", None))
            if pu_xy and self._is_at_xy(pu_xy[0], pu_xy[1], tol_cm=tol): here_orders.append(o)

        if not here_orders: self.vlm_last_action_note = "Nothing to pick up here."; self._finish_action(success=False); return
        if not spec_text:
            self.vlm_errors.append("pickup needs a bag_cmd."); self.vlm_ephemeral["pickup_hint"] = self._pickup_hint_for_orders(here_orders); self._finish_action(success=False); return
        if not self.insulated_bag: self.insulated_bag = InsulatedBag()

        now_sim = self.clock.now_sim(); picked_ids: List[int] = []
        for o in here_orders:
            oid = getattr(o, "id", None); items = list(getattr(o, "items", []) or [])
            per_order_cmd = None
            lines = [ln.strip() for ln in spec_text.splitlines() if ln.strip()]
            for ln in lines:
                if ln.lower().startswith("order "):
                    head, tail = (ln.split(":", 1) + [""])[:2]
                    _, raw_id = head.strip().split(None, 1)
                    _oid = int(raw_id.strip())
                    if _oid == oid: per_order_cmd = tail.strip(); break
            if per_order_cmd is None: per_order_cmd = spec_text
            for it in items:
                it.temp_c = float(getattr(it, "serving_temp_c", 25.0)); it.prepared_at_sim = float(getattr(o, "prep_ready_sim", now_sim)); it.picked_at_sim = float(now_sim)
            items_map = {i+1: items[i] for i in range(len(items))}
            self.insulated_bag.move_items(per_order_cmd); self.insulated_bag.add_items(per_order_cmd, items_map)
            o.has_picked_up = True
            if oid is not None and oid not in self.carrying: self.carrying.append(oid)
            if oid is not None: picked_ids.append(oid)

        self.vlm_ephemeral["bag_after_pickup"] = self.insulated_bag.list_items(); self.vlm_last_action_note = "Pickup succeeded."
        if picked_ids: self._log(f"picked up order #{picked_ids[0]}" if len(picked_ids) == 1 else f"picked up orders {picked_ids}")
        self._finish_action(success=True)

    # ===== Charging (ESCOOTER) =====
    def _advance_charge_to_now(self):
        if self._charge_ctx and self.e_scooter:
            now = self.clock.now_sim()
            t0, t1 = self._charge_ctx["start_sim"], self._charge_ctx["end_sim"]
            p0, pt = self._charge_ctx["start_pct"], self._charge_ctx["target_pct"]
            if t1 <= t0: cur = pt
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = p0 + (pt - p0) * r
            self.e_scooter.charge_to(cur)

    def _handle_charge_escooter(self, _self, act: DMAction, _allow_interrupt: bool):
        if self._charge_ctx is not None:
            self.vlm_add_error("charge failed: already charging"); self._finish_action(success=False); return
        if not self.e_scooter:
            self.vlm_add_error("charge failed: no scooter"); self._finish_action(success=False); return
        if self._nearest_poi_xy("charging_station", tol_cm=300.0) is None:
            self.vlm_add_error("charge failed: not near a charging station"); self._finish_action(success=False); return

        # 把车就地停好；人改走路
        if self.e_scooter.state != ScooterState.PARKED or not self.e_scooter.park_xy:
            self.e_scooter.park_here(self.x, self.y)
            if self.mode == TransportMode.SCOOTER: 
                self.set_mode(TransportMode.WALK)
        else:
            px, py = self.e_scooter.park_xy
            if not self._is_at_xy(px, py, tol_cm=300):
                self.vlm_add_error("charge failed: not at parked scooter location"); self._finish_action(success=False); return

        target_pct = float(act.data.get("target_pct", 100.0))
        target_pct = max(0.0, min(100.0, target_pct))
        before = float(self.e_scooter.battery_pct)
        if target_pct <= before + 1e-6: 
            # 已经够电了：动作视为完成
            self._finish_action(success=True); 
            return
        rate_m = float(self.e_scooter.charge_rate_pct_per_min)
        if rate_m <= 0.0:
            self.vlm_add_error("charge failed: invalid rate"); self._finish_action(success=False); return

        duration_sim_s = (target_pct - before) / rate_m * 60.0
        now_sim = self.clock.now_sim()
        self._charge_ctx = dict(
            start_sim=now_sim, 
            end_sim=now_sim + duration_sim_s, 
            start_pct=before, 
            target_pct=target_pct
        )
        self._log(f"start charging scooter: {before:.0f}% -> {target_pct:.0f}% (~{duration_sim_s/60.0:.1f} min @virtual)")

        # 关键：充电开始即视为本动作完成，让 agent 立刻可以去做别的
        self._finish_action(success=True)


    # ===== WAIT / REST =====
    def _handle_wait(self, _self, act: DMAction, _allow_interrupt: bool):
        if act.data.get("until") == "charge_done" and self._charge_ctx:
            now_sim = self.clock.now_sim(); end_sim = float(self._charge_ctx["end_sim"])
            if end_sim <= now_sim: self._finish_action(success=True); return
            self._wait_ctx = dict(start_sim=now_sim, end_sim=end_sim); return
        duration_s = float(act.data.get("duration_s", 0.0))
        if duration_s <= 0.0: self._finish_action(success=True); return
        now_sim = self.clock.now_sim(); self._wait_ctx = dict(start_sim=now_sim, end_sim=now_sim + float(duration_s))

    def _handle_rest(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if self.is_rescued:
            self.vlm_add_error("rest failed: in hospital rescue"); self._finish_action(success=False); return
        if self._nearest_poi_xy("rest_area", tol_cm=300.0) is None:
            self.vlm_add_error("rest failed: not near a rest_area"); self._finish_action(success=False); return

        target = float(act.data.get("target_pct", 100.0)); target = max(0.0, min(100.0, target))
        before = float(self.energy_pct)
        if target <= before + 1e-6: self._log(f"rest skipped: already at {before:.0f}%"); self._finish_action(success=True); return
        rate = float(self.rest_rate_pct_per_min)
        if rate <= 0.0: self.vlm_add_error("rest failed: invalid rest rate"); self._finish_action(success=False); return

        duration_sim_s = (target - before) / rate * 60.0
        now_sim = self.clock.now_sim()
        self._rest_ctx = dict(start_sim=now_sim, end_sim=now_sim + duration_sim_s, start_pct=before, target_pct=target)
        self._log(f"start resting: {before:.0f}% -> {target:.0f}% (~{duration_sim_s/60.0:.1f} min @virtual)")

    # ===== Store / Consumables =====
    def _handle_buy(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if self._nearest_poi_xy("store", tol_cm=300.0) is None:
            self.vlm_add_error("buy failed: not in a store"); self._finish_action(success=False); return
        if not self._store_manager:
            self.vlm_add_error("buy failed: no store manager"); self._finish_action(success=False); return

        item_id = str(act.data.get("item_id", "") or act.data.get("name", "")).strip()
        qty = int(act.data.get("qty", 1))
        if not item_id or qty <= 0:
            self.vlm_add_error("buy failed: need item_id/name and positive qty"); self._finish_action(success=False); return

        ok, msg, cost = self._store_manager.purchase(self, item_id=item_id, qty=qty)
        if not ok:
            self.vlm_add_error(f"buy failed: {msg}"); self._finish_action(success=False); return

        self.inventory[item_id] = int(self.inventory.get(item_id, 0)) + int(qty)
        self._log(f"bought {qty} x {item_id} for ${cost:.2f}")
        self._register_success(f"bought {qty} x {item_id}")
        self._finish_action(success=True)

    def _handle_use_battery_pack(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if not self.e_scooter:
            self.vlm_add_error("use_battery_pack failed: no e-scooter"); self._finish_action(success=False); return
        item_id = act.data.get("item_id", ITEM_ESC_BATTERY_PACK)
        cnt = int(self.inventory.get(item_id, 0))
        if cnt <= 0:
            self.vlm_add_error(f"use_battery_pack failed: inventory=0 ({item_id})"); self._finish_action(success=False); return
        self.inventory[item_id] = cnt - 1
        self.e_scooter.charge_to(100.0)
        if self.e_scooter.state != ScooterState.PARKED:
            self.e_scooter.state = ScooterState.USABLE; self.set_mode(TransportMode.SCOOTER)
        self._log(f"used '{item_id}': scooter battery -> 100% (remaining {self.inventory[item_id]})")
        self._register_success("used battery pack; e-scooter fully recharged")
        self._finish_action(success=True)

    def _handle_use_energy_drink(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if self._hospital_ctx is not None:
            self.vlm_add_error("use_energy_drink failed: in hospital rescue"); self._finish_action(success=False); return
        item_id = act.data.get("item_id", ITEM_ENERGY_DRINK)
        cnt = int(self.inventory.get(item_id, 0))
        if cnt <= 0:
            self.vlm_add_error(f"use_energy_drink failed: inventory=0 ({item_id})"); self._finish_action(success=False); return
        self.inventory[item_id] = cnt - 1
        before = float(self.energy_pct); self.energy_pct = float(min(100.0, before + 50.0))
        self._log(f"used '{item_id}': energy {before:.0f}% -> {self.energy_pct:.0f}% (remaining {self.inventory[item_id]})")
        self._register_success("used energy drink; +50% energy")
        self._finish_action(success=True)

    # ===== NEW: Comms =====
    def _handle_post_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        comms = get_comms()

        help_type = act.data.get("help_type")
        if isinstance(help_type, str): help_type = HelpType[help_type]
        bounty = float(act.data.get("bounty", 0.0))
        ttl_s  = float(act.data.get("ttl_s", 0.0))
        payload = dict(act.data.get("payload") or {})

        kwargs = dict(publisher_id=str(self.agent_id), kind=help_type, reward=bounty,
                      time_limit_s=ttl_s, location_xy=(float(self.x), float(self.y)))

        if help_type in (HelpType.HELP_PICKUP, HelpType.HELP_DELIVERY):
            kwargs["order_id"] = int(payload["order_id"])
        elif help_type == HelpType.HELP_BUY:
            buy_items = {}
            for item_id, qty in payload.get("buy_list", []):
                q = int(qty)
                if q > 0: buy_items[str(item_id)] = q
            kwargs["buy_items"] = buy_items
        elif help_type == HelpType.HELP_CHARGE:
            kwargs["target_pct"] = float(payload.get("want_charge_pct", payload.get("target_pct", 100.0)))

        ok, msg, rid = comms.post_request(**kwargs)
        if not ok:
            self.vlm_add_error(f"post_help_request failed: {msg}"); self._finish_action(success=False); return
        self._log(f"posted help request #{rid} ({help_type.name}) bounty=${bounty:.2f}")
        self._register_success(f"posted help request #{rid}")
        self._finish_action(success=True)

    def _handle_accept_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        if not comms: self.vlm_add_error("accept_help_request failed: no comms"); self._finish_action(success=False); return
        req_id = int(act.data.get("req_id"))
        ok, msg = comms.accept_request(req_id=req_id, helper_id=self.agent_id)
        if not ok:
            self.vlm_add_error(f"accept_help_request failed: {msg}"); self._finish_action(success=False); return
        self._log(f"accepted help request #{req_id}"); self._register_success(f"accepted request #{req_id}")
        self._finish_action(success=True)

    def _handle_edit_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        req_id = int(act.data.get("req_id"))
        new_bounty = act.data.get("new_bounty", None)
        new_ttl_s  = act.data.get("new_ttl_s", None)
        ok, msg = comms.modify_request(publisher_id=str(self.agent_id), req_id=req_id, reward=new_bounty, time_limit_s=new_ttl_s)
        if not ok:
            self.vlm_add_error(f"edit_help_request failed: {msg}"); self._finish_action(success=False); return
        self._log(f"edited help request #{req_id}"); self._register_success(f"edited request #{req_id}")
        self._finish_action(success=True)

    def _handle_give_to_agent(self, _self, act: DMAction, _allow_interrupt: bool):
        raise NotImplementedError("give_to_agent via Comms")

    # ===== NEW: Transport switching / car rental =====
    def _handle_switch_transport(self, _self, act: DMAction, _allow_interrupt: bool):
        to = str(act.data.get("to", "")).strip().lower()
        tol = 300.0

        if self.mode == TransportMode.SCOOTER and self.e_scooter:
            self.e_scooter.park_here(self.x, self.y)
        if self.mode == TransportMode.CAR and self.car:
            self.car.park_here(self.x, self.y)

        if to in ("walk", TransportMode.WALK.value):
            self.set_mode(TransportMode.WALK)
            self._log("switch -> WALK")
            self._finish_action(success=True); return

        if to in ("e-scooter", "scooter", TransportMode.SCOOTER.value):
            if not self.e_scooter:
                self.vlm_add_error("switch failed: no scooter"); self._finish_action(success=False); return
            if self.e_scooter.park_xy:
                px, py = self.e_scooter.park_xy
                if not self._is_at_xy(px, py, tol_cm=tol):
                    self.vlm_add_error("switch failed: not near your scooter"); self._finish_action(success=False); return
                self.e_scooter.unpark()
            if self.e_scooter.state == ScooterState.DEPLETED:
                self.set_mode(TransportMode.DRAG_SCOOTER); self._log("switch -> DRAG_SCOOTER (battery depleted)")
            else:
                self.set_mode(TransportMode.SCOOTER); self._log("switch -> SCOOTER")
            self._finish_action(success=True); return

        if to in ("car", TransportMode.CAR.value):
            if not self.car:
                self.vlm_add_error("switch failed: no rented car"); self._finish_action(success=False); return
            if self.car.park_xy:
                px, py = self.car.park_xy
                if not self._is_at_xy(px, py, tol_cm=tol):
                    self.vlm_add_error("switch failed: not near your car"); self._finish_action(success=False); return
                self.car.unpark()
            self.set_mode(TransportMode.CAR); self._log("switch -> CAR")
            self._finish_action(success=True); return

        self.vlm_add_error("switch failed: invalid target"); self._finish_action(success=False)

    def _handle_rent_car(self, _self, act: DMAction, _allow_interrupt: bool):
        tol_xy = self._nearest_poi_xy("car_rental", tol_cm=300.0)
        if tol_xy is None:
            self.vlm_add_error("rent_car failed: not at car_rental"); self._finish_action(success=False); return
        if self.car is not None:
            self.vlm_add_error("rent_car failed: already have a car"); self._finish_action(success=False); return

        if self.e_scooter: self.e_scooter.park_here(self.x, self.y)

        rate = float(act.data.get("rate_per_min", 1.0))
        speed = float(act.data.get("avg_speed_cm_s", DEFAULT_AVG_SPEED_CM_S[TransportMode.CAR]))
        self.car = Car(owner_id=str(self.agent_id), avg_speed_cm_s=speed, rate_per_min=rate, state=CarState.USABLE, park_xy=None)
        self.set_mode(TransportMode.CAR)
        self._rental_ctx = {"last_tick_sim": self.clock.now_sim(), "rate_per_min": float(self.car.rate_per_min)}
        self._log(f"rent car @ ${self.car.rate_per_min:.2f}/min")
        self._finish_action(success=True)

    def _handle_return_car(self, _self, act: DMAction, _allow_interrupt: bool):
        tol_xy = self._nearest_poi_xy("car_rental", tol_cm=300.0)
        if tol_xy is None:
            self.vlm_add_error("return_car failed: not at car_rental"); self._finish_action(success=False); return
        if not self.car:
            self.vlm_add_error("return_car failed: no car"); self._finish_action(success=False); return

        if self.mode == TransportMode.CAR: self.set_mode(TransportMode.WALK)
        self._log("return car: stop billing")
        self.car = None
        self._rental_ctx = None
        self._finish_action(success=True)

    # ===== auto dropoff =====
    def _auto_try_dropoff(self):
        tol = 300.0
        delivered_any = False
        remaining: List[Any] = []
        for o in self.active_orders:
            done = False
            do_xy = self._xy_of_node(getattr(o, "dropoff_node", None))
            if do_xy and getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                if self._is_at_xy(do_xy[0], do_xy[1], tol_cm=tol):
                    oid = getattr(o, "id", None)
                    o.has_delivered = True
                    o.sim_delivered_s = float(self.clock.now_sim())
                    now_sim = self.clock.now_sim()
                    for it in getattr(o, "items", []) or []:
                        it.delivered_at_sim = float(now_sim)
                    if self.insulated_bag and hasattr(self.insulated_bag, "remove_items"):
                        self.insulated_bag.remove_items(getattr(o, "items", []) or [])
                    if oid in self.carrying:
                        try: self.carrying.remove(oid)
                        except ValueError: pass

                    duration_s = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0)
                    time_limit_s = float(getattr(o, "time_limit_s", 0.0) or 0.0)
                    base_earn = float(getattr(o, "earnings", 0.0) or 0.0)
                    items = list(getattr(o, "items", []) or [])

                    settle_res = compute_settlement(
                        order_base_earnings=base_earn,
                        duration_s=duration_s,
                        time_limit_s=time_limit_s,
                        items=items,
                        config=None
                    )

                    self.add_earnings(settle_res.total_pay)

                    oid = getattr(o, "id", None)
                    self.completed_orders.append(dict(
                        id=oid,
                        duration_s=duration_s,
                        rating=float(settle_res.stars),
                        earnings=base_earn,
                        bonus_extra=float(settle_res.extra_pay),
                        paid_total=float(settle_res.total_pay),
                        breakdown=settle_res.breakdown,
                        pickup=getattr(o, "pickup_road_name", ""),
                        dropoff=getattr(o, "dropoff_road_name", "")
                    ))

                    extra_str = f" (extra {settle_res.extra_pay:+.2f}, stars={settle_res.stars})"
                    self._log(f"dropped off order #{oid}{extra_str}" if oid is not None else f"dropped off order{extra_str}")

                    self._log(f"dropped off order #{oid}" if oid is not None else "dropped off order")
                    delivered_any = True; done = True

            if not done: remaining.append(o)
        self.active_orders = remaining
        if delivered_any: self._register_success("auto dropoff succeeded")

    # ===== distance helpers =====
    def _estimate_distance_cm(self, x0: float, y0: float, x1: float, y1: float, use_route: bool, snap_cm: float) -> float:
        if use_route and hasattr(self.city_map, "route_xy_to_xy"):
            pts = self.city_map.route_xy_to_xy(float(x0), float(y0), float(x1), float(y1), snap_cm=float(snap_cm)) or []
            if len(pts) >= 2:
                dist = 0.0
                for i in range(len(pts)-1):
                    dx = pts[i+1][0] - pts[i][0]; dy = pts[i+1][1] - pts[i][1]
                    dist += math.hypot(dx, dy)
                return float(dist)
        return float(math.hypot(x1 - x0, y1 - y0))

    def random_target_on_roads(self) -> Optional[Tuple[float, float]]:
        nodes = [n for n in getattr(self.city_map, "nodes", []) if getattr(n, "type", "") in ("normal", "intersection")]
        if not nodes: return None
        n = random.choice(nodes)
        return (float(n.position.x), float(n.position.y))

    # ===== misc =====
    def add_earnings(self, amount: float):
        self.earnings_total += float(amount)

    def to_text(self) -> str:
        active_ids = [getattr(o, "id", None) for o in self.active_orders]
        mode_str = 'towing' if self.towing_scooter else self.mode.value
        lines = [
            f"[DeliveryMan {self.name}]",
            f"  Position : {self._fmt_xy_m(self.x, self.y)}",
            f"  Mode     : {mode_str}",
            f"  Speed    : {self.get_current_speed_for_viewer():.1f} cm/s",
            f"  Energy   : {self.energy_pct:.0f}%",
            f"  Earnings : ${self.earnings_total:.2f}",
            f"  Active Orders : {active_ids}",
            f"  Carrying : {self.carrying}",
            f"  Queue    : {len(self._queue)} action(s), Busy: {self.is_busy()}",
        ]
        if self.e_scooter:
            rng_m = self._remaining_range_m(); rng_km = (rng_m/1000.0) if rng_m is not None else None
            lines += [f"  Scooter  : state={self.e_scooter.state.value}, battery={self.e_scooter.battery_pct:.0f}% "
                      f"({self.e_scooter.charge_rate_pct_per_min:.1f}%/min), avg_speed={self.e_scooter.avg_speed_cm_s:.0f} cm/s, "
                      f"park_xy={self._fmt_xy_m_opt(self.e_scooter.park_xy)}, remaining={'{:.1f} km'.format(rng_km) if rng_km is not None else 'N/A'}"]
        if self.car:
            lines += [f"  Car      : state={self.car.state.value}, rate=${self.car.rate_per_min:.2f}/min, park_xy={self._fmt_xy_m_opt(self.car.park_xy)}, rental={'on' if self._rental_ctx else 'off'}"]
        return "\n".join(lines)

    # ===== progress for UI =====
    def charging_progress(self) -> Optional[Dict[str, Any]]:
        if self._charge_ctx and self.e_scooter:
            ctx = self._charge_ctx; now = self.clock.now_sim(); t0, t1 = ctx["start_sim"], ctx["end_sim"]; p0, pt = ctx["start_pct"], ctx["target_pct"]
            if t1 <= t0: cur = pt; prog = 1.0
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = p0 + (pt - p0) * r; prog = 0.0 if pt <= p0 else (cur - p0) / max(1e-9, pt - p0)
            xy = self.e_scooter.park_xy if self.e_scooter.park_xy else (self.x, self.y)
            return dict(progress=float(max(0.0, min(1.0, prog))), current_pct=float(cur), target_pct=float(pt), xy=xy)
        return None

    def resting_progress(self) -> Optional[Dict[str, Any]]:
        if self._rest_ctx:
            ctx = self._rest_ctx; now = self.clock.now_sim(); t0, t1 = ctx["start_sim"], ctx["end_sim"]; e0, et = ctx["start_pct"], ctx["target_pct"]
            if t1 <= t0: cur = et; prog = 1.0
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = e0 + (et - e0) * r; prog = 0.0 if et <= e0 else (cur - e0) / max(1e-9, et - e0)
            return dict(progress=float(max(0.0, min(1.0, prog))), current_pct=float(cur), target_pct=float(et), xy=(self.x, self.y))
        return None

    def rescue_progress(self) -> Optional[Dict[str, Any]]:
        if self._hospital_ctx:
            ctx = self._hospital_ctx; now = self.clock.now_sim(); t0, t1 = ctx["start_sim"], ctx["end_sim"]
            r = 0.0 if t1 <= t0 else (now - t0) / (t1 - t0); r = max(0.0, min(1.0, r))
            return dict(progress=r, xy=(self.x, self.y))
        return None

    # ===== hospital =====
    def _trigger_hospital_if_needed(self):
        if self.is_rescued or self._hospital_ctx is not None: return
        self.is_rescued = True
        hxy = self._closest_poi_xy("hospital") or (self.x, self.y)
        if self._ue and hasattr(self._ue, "teleport_xy"): self._ue.teleport_xy(str(self.agent_id), float(hxy[0]), float(hxy[1]))
        self.x, self.y = float(hxy[0]), float(hxy[1])
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "set_agent_xy"): self._viewer.set_agent_xy(self._viewer_agent_id, self.x, self.y)
        self._queue.clear(); self._current = None
        now_sim = self.clock.now_sim()
        self._hospital_ctx = dict(start_sim=now_sim, end_sim=now_sim + float(DEFAULT_HOSPITAL_DURATION_S))

    # ===== tick =====
    def poll_time_events(self):
        now = self.clock.now_sim()

        # 订单活动用时
        if self._orders_last_tick_sim is None: self._orders_last_tick_sim = now
        if not self._timers_paused:
            delta = max(0.0, now - self._orders_last_tick_sim)
            if delta > 0:
                for o in self.active_orders:
                    if getattr(o, "is_accepted", False) and not getattr(o, "has_delivered", False):
                        cur = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0); o.sim_elapsed_active_s = cur + delta
            self._orders_last_tick_sim = now

        # 备好即写一次初温
        for o in list(self.active_orders):
            if hasattr(o, "is_ready_for_pickup") and o.is_ready_for_pickup(now):
                if not getattr(o, "_temps_bootstrapped", False):
                    for it in (getattr(o, "items", []) or []):
                        if hasattr(it, "serving_temp_c"): it.temp_c = float(it.serving_temp_c)
                        if hasattr(it, "prepared_at_sim"): it.prepared_at_sim = float(getattr(o, "prep_ready_sim", now))
                    o._temps_bootstrapped = True  # type: ignore[attr-defined]

        # MOVE 到达/阻塞判定
        if self._move_ctx is not None:
            if self._interrupt_move_flag:
                self._move_ctx["blocked"] = 1.0
                self._interrupt_move_flag = False

            tx = float(self._move_ctx["tx"]); ty = float(self._move_ctx["ty"]); tol = float(self._move_ctx["tol"])
            if self._move_ctx.get("blocked", 0.0) == 1.0:
                self._move_ctx = None
                if self._current and self._current.kind == DMActionKind.MOVE_TO:
                    self._finish_action(success=False)
            elif self._is_at_xy(tx, ty, tol_cm=tol):
                self._move_ctx = None
                self._auto_try_dropoff()
                if self._current and self._current.kind == DMActionKind.MOVE_TO:
                    self._finish_action(success=True)

        # WAIT
        if self._current and self._current.kind == DMActionKind.WAIT and self._wait_ctx:
            if now >= self._wait_ctx["end_sim"]:
                self._wait_ctx = None; self._finish_action(success=True)

        # CHARGE
        if self._charge_ctx and self.e_scooter:
            t0, t1 = self._charge_ctx["start_sim"], self._charge_ctx["end_sim"]; p0, pt = self._charge_ctx["start_pct"], self._charge_ctx["target_pct"]
            if t1 <= t0: cur = pt
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = p0 + (pt - p0) * r
            self.e_scooter.charge_to(cur)
            if now >= t1:
                self.e_scooter.charge_to(pt); self._charge_ctx = None; self._log(f"charging finished: {p0:.0f}% -> {pt:.0f}%")
                self.vlm_ephemeral["scooter_ready"] = "Scooter charged. You can switch back to scooter near its park spot."

        # REST
        if self._rest_ctx:
            t0, t1 = self._rest_ctx["start_sim"], self._rest_ctx["end_sim"]; e0, et = self._rest_ctx["start_pct"], self._rest_ctx["target_pct"]
            if t1 <= t0: cur = et
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = e0 + (et - e0) * r
            self.energy_pct = float(cur)
            if now >= t1:
                self.energy_pct = float(et); self._log(f"rest finished: {e0:.0f}% -> {et:.0f}%"); self._rest_ctx = None
                if self._current and self._current.kind == DMActionKind.REST: self._finish_action(success=True)

        # HOSPITAL
        if self._hospital_ctx and now >= self._hospital_ctx["end_sim"]:
            self.rescue(); self._hospital_ctx = None; self._log("rescue finished: full energy at Hospital"); self.kickstart()

        # 租车计费
        if self._rental_ctx and not self._timers_paused:
            dt = max(0.0, now - float(self._rental_ctx["last_tick_sim"]))
            if dt > 0.0:
                rate = float(self._rental_ctx["rate_per_min"])
                cost = rate * (dt / 60.0)
                if self.earnings_total - cost <= 0.0:
                    was_car = (self.mode == TransportMode.CAR)
                    self.car = None; self._rental_ctx = None
                    if was_car: self.set_mode(TransportMode.WALK)
                    self.earnings_total = max(0.0, self.earnings_total - cost)
                    self._log("rental ended (no money) -> remove car; switch to WALK")
                else:
                    self.earnings_total -= cost
                    self._rental_ctx["last_tick_sim"] = now

        # 保温袋温度推进
        if self.insulated_bag and hasattr(self.insulated_bag, "tick_temperatures"):
            if self._last_bag_tick_sim is None:
                self._last_bag_tick_sim = now
            else:
                dt = max(0.0, now - self._last_bag_tick_sim)
                if dt > 0:
                    self.insulated_bag.tick_temperatures(dt); self._last_bag_tick_sim = now

        # 公交状态推进
        if self._bus_ctx and self.mode == TransportMode.BUS:
            self._update_bus_riding(now)

    # ===== Bus handlers =====
    def _handle_board_bus(self, _self, act: DMAction, _allow_interrupt: bool):
        """上车动作"""
        self.vlm_clear_ephemeral()
        if not self._bus_manager:
            self.vlm_add_error("board_bus failed: no bus manager"); self._finish_action(success=False); return
        
        bus_id = act.data.get("bus_id")
        if not bus_id:
            self.vlm_add_error("board_bus failed: need bus_id"); self._finish_action(success=False); return
        
        bus = self._bus_manager.get_bus(bus_id)
        if not bus:
            self.vlm_add_error(f"board_bus failed: bus {bus_id} not found"); self._finish_action(success=False); return
        
        # 检查是否在公交附近
        distance = math.hypot(self.x - bus.x, self.y - bus.y)
        if distance > 500.0:  # 5米内才能上车
            self.vlm_add_error("board_bus failed: not near bus"); self._finish_action(success=False); return
        
        if not bus.can_board():
            self.vlm_add_error("board_bus failed: bus not at stop"); self._finish_action(success=False); return
        
        # 上车
        if bus.board_passenger(str(self.agent_id)):
            self._bus_ctx = {
                "bus_id": bus_id,
                "boarding_stop": bus.get_current_stop().id if bus.get_current_stop() else "",
                "target_stop": act.data.get("target_stop", ""),
                "boarded_time": self.clock.now_sim()
            }
            self.set_mode(TransportMode.BUS)
            self._log(f"boarded bus {bus_id}")
            self._register_success(f"boarded bus {bus_id}")
            self._finish_action(success=True)
        else:
            self.vlm_add_error("board_bus failed: could not board"); self._finish_action(success=False)

    def _handle_alight_bus(self, _self, act: DMAction, _allow_interrupt: bool):
        """下车动作"""
        self.vlm_clear_ephemeral()
        if not self._bus_ctx:
            self.vlm_add_error("alight_bus failed: not on bus"); self._finish_action(success=False); return
        
        bus_id = self._bus_ctx.get("bus_id")
        bus = self._bus_manager.get_bus(bus_id) if self._bus_manager else None
        if not bus:
            self.vlm_add_error("alight_bus failed: bus not found"); self._finish_action(success=False); return
        
        # 检查是否在目标站点
        target_stop_id = act.data.get("stop_id") or self._bus_ctx.get("target_stop")
        if target_stop_id:
            current_stop = bus.get_current_stop()
            if not current_stop or current_stop.id != target_stop_id:
                self.vlm_add_error(f"alight_bus failed: not at target stop {target_stop_id}"); self._finish_action(success=False); return
        
        # 下车
        if bus.alight_passenger(str(self.agent_id)):
            # 更新位置到公交当前位置
            self.x = bus.x
            self.y = bus.y
            self._bus_ctx = None
            self.set_mode(TransportMode.WALK)  # 下车后改为步行
            self._log(f"alighted from bus {bus_id}")
            self._register_success(f"alighted from bus {bus_id}")
            self._finish_action(success=True)
        else:
            self.vlm_add_error("alight_bus failed: could not alight"); self._finish_action(success=False)

    def _handle_wait_for_bus(self, _self, act: DMAction, _allow_interrupt: bool):
        """等车动作"""
        self.vlm_clear_ephemeral()
        if not self._bus_manager:
            self.vlm_add_error("wait_for_bus failed: no bus manager"); self._finish_action(success=False); return
        
        stop_id = act.data.get("stop_id")
        route_id = act.data.get("route_id")
        max_wait_s = float(act.data.get("max_wait_s", 300.0))  # 默认等5分钟
        
        if not stop_id:
            self.vlm_add_error("wait_for_bus failed: need stop_id"); self._finish_action(success=False); return
        
        # 检查是否在站点附近
        nearest_stop, distance = self._bus_manager.find_nearest_bus_stop(self.x, self.y)
        if not nearest_stop or nearest_stop.id != stop_id or distance > 500.0:
            self.vlm_add_error(f"wait_for_bus failed: not at stop {stop_id}"); self._finish_action(success=False); return
        
        # 查找该站点的公交车
        buses_at_stop = self._bus_manager.find_buses_at_stop(stop_id)
        if route_id:
            buses_at_stop = [bus for bus in buses_at_stop if bus.route.id == route_id]
        
        if buses_at_stop:
            # 有车在站，可以上车
            bus = buses_at_stop[0]  # 选择第一辆车
            self._log(f"bus {bus.id} available at stop {stop_id}")
            self._register_success(f"found bus {bus.id} at stop")
            self._finish_action(success=True)
        else:
            # 没车，开始等待
            self._wait_ctx = {
                "start_sim": self.clock.now_sim(),
                "end_sim": self.clock.now_sim() + max_wait_s,
                "stop_id": stop_id,
                "route_id": route_id
            }
            self._log(f"waiting for bus at stop {stop_id} (max {max_wait_s}s)")

    def _update_bus_riding(self, now: float):
        """更新乘坐公交状态"""
        if not self._bus_ctx or not self._bus_manager:
            return
        
        bus_id = self._bus_ctx.get("bus_id")
        bus = self._bus_manager.get_bus(bus_id)
        if not bus:
            # 公交不存在，强制下车
            self._bus_ctx = None
            self.set_mode(TransportMode.WALK)
            self._log("bus disappeared, switched to walk")
            return
        
        # 跟随公交位置
        self.x = bus.x
        self.y = bus.y
        
        # 检查是否到达目标站点
        target_stop_id = self._bus_ctx.get("target_stop")
        if target_stop_id and bus.is_at_stop():
            current_stop = bus.get_current_stop()
            if current_stop and current_stop.id == target_stop_id:
                # 到达目标站点，自动下车
                self._log(f"arrived at target stop {target_stop_id}, auto alighting")
                self.enqueue_action(DMAction(DMActionKind.ALIGHT_BUS, data={"stop_id": target_stop_id}))