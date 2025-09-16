# -*- coding: utf-8 -*-
# Base/DeliveryMan.py

import time, math, random
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple, Callable, Deque, Set
from collections import deque
from concurrent.futures import Future, Executor
import re
import copy

from Base.Timer import VirtualClock
from Base.EScooter import EScooter, ScooterState
from Base.Car import Car, CarState
from Base.Store import StoreManager
from Base.Insulated_bag import InsulatedBag, IcePack, HeatPack
from Base.Comms import get_comms, HelpType
from Base.Settlement import compute_settlement, SettlementConfig
from Base.Prompt import get_system_prompt
from Base.ActionSpace import ACTION_API_SPEC, parse_action as parse_vlm_action, action_to_text
from Base.RunRecorder import RunRecorder
from Base.BusManager import BusManager
from utils.util import _ensure_png_bytes
from utils.global_logger import get_agent_logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm.base_model import BaseModel

# ===== Transport Modes =====
class TransportMode(str, Enum):
    WALK          = "walk"
    SCOOTER       = "e-scooter"
    DRAG_SCOOTER  = "drag_scooter"
    CAR           = "car"
    BUS           = "bus"

ITEM_ESC_BATTERY_PACK = "escooter_battery_pack"
ITEM_ENERGY_DRINK     = "energy_drink"
ITEM_ICE_PACK         = "ice_pack"
ITEM_HEAT_PACK        = "heat_pack"

class DeliveryMethod(str, Enum):
    LEAVE_AT_DOOR     = "leave_at_door"     # 直接放门口
    KNOCK             = "knock"             # 敲门
    CALL              = "call"              # 打电话
    HAND_TO_CUSTOMER  = "hand_to_customer"  # 面交

VALID_DELIVERY_METHODS = {
    DeliveryMethod.LEAVE_AT_DOOR.value,
    DeliveryMethod.KNOCK.value,
    DeliveryMethod.CALL.value,
    DeliveryMethod.HAND_TO_CUSTOMER.value,
}

# ===== Actions =====
class DMActionKind(str, Enum):
    MOVE_TO              = "move_to"
    ACCEPT_ORDER         = "accept_order"
    VIEW_ORDERS          = "view_orders"
    VIEW_BAG             = "view_bag"
    PICKUP               = "pickup"
    PLACE_FOOD_IN_BAG    = "place_food_in_bag"
    CHARGE_ESCOOTER      = "charge_escooter"
    WAIT                 = "wait"
    REST                 = "rest"
    BUY                  = "buy"
    USE_BATTERY_PACK     = "use_battery_pack"
    USE_ENERGY_DRINK     = "use_energy_drink"
    USE_ICE_PACK         = "use_ice_pack"
    USE_HEAT_PACK        = "use_heat_pack"
    VIEW_HELP_BOARD      = "view_help_board"
    POST_HELP_REQUEST    = "post_help_request"
    ACCEPT_HELP_REQUEST  = "accept_help_request"
    EDIT_HELP_REQUEST    = "edit_help_request"
    SWITCH_TRANSPORT     = "switch_transport"
    RENT_CAR             = "rent_car"
    RETURN_CAR           = "return_car"
    PLACE_TEMP_BOX       = "place_temp_box"       # publisher/helper 放盒
    TAKE_FROM_TEMP_BOX   = "take_from_temp_box"   # helper 取盒
    REPORT_HELP_FINISHED = "report_help_finished" # helper 报告完成
    DROP_OFF             = "drop_off"
    SAY                  = "say"
    BOARD_BUS            = "board_bus"
    VIEW_BUS_SCHEDULE    = "view_bus_schedule"
    TURN_AROUND          = "turn_around"
    STEP_FORWARD         = "step_forward" 

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
    cfg: Dict[str, Any] = field(default_factory=dict)

    # runtime
    speed_cm_s: float = field(init=False)
    energy_pct: float = field(init=False, default=100.0)
    earnings_total: float = field(default=100.0)
    is_rescued: bool = field(default=False)
    hospital_rescue_fee: float = 0.0

    # orders
    active_orders: List[Any] = field(default_factory=list)
    carrying: List[int] = field(default_factory=list)

    # misc
    name: str = "DM"

    # viewer / ue
    _viewer: Optional[Any] = field(default=None, repr=False)
    _viewer_agent_id: Optional[str] = field(default=None, repr=False)
    _ue: Optional[Any] = field(default=None, repr=False)

    # managers
    _order_manager: Optional[Any] = field(default=None, repr=False)
    _store_manager: Optional[StoreManager] = field(default=None, repr=False)
    _bus_manager: Optional[BusManager] = field(default=None, repr=False)

    # scheduling
    _queue: List[DMAction] = field(default_factory=list, repr=False)
    _current: Optional[DMAction] = field(default=None, repr=False)

    # lifecycle / metrics
    _recorder: Optional[RunRecorder] = field(default=None, repr=False)
    _sim_active_elapsed_s: float = 0.0
    _lifecycle_done: bool = False
    
    # realtime lifecycle tracking
    _realtime_start_ts: Optional[float] = field(default=None, repr=False)
    _realtime_stop_hours: float = 0.0

    # history
    completed_orders: List[Dict[str, Any]] = field(default_factory=list)

    # ----- HELP DELIVERY 追踪 -----
    helping_order_ids: Set[int] = field(default_factory=set)              # 我作为 helper 正在帮送的 order_id
    _help_delivery_req_by_oid: Dict[int, int] = field(default_factory=dict, repr=False)  # order_id -> req_id
    help_completed_order_ids: Set[int] = field(default_factory=set)       # 我作为 publisher 已通过 Comm 收到"完成"并已结算的 order_id
    _helping_wait_ack_oids: Set[int] = field(default_factory=set, repr=False)            # 我作为 helper 已推送完成消息，等待对方结算的 order_id

    # DeliveryMan dataclass 字段里新增（放在"----- HELP DELIVERY 追踪 -----"附近即可）
    help_orders: Dict[int, Any] = field(default_factory=dict)          # 我作为 helper 正在处理的订单：oid -> order_obj
    help_orders_completed: Set[int] = field(default_factory=set)       # 我作为 helper 已送到并上报过完成（等待赏金/或已拿到）的 oid
    accepted_help: Dict[int, Any] = field(default_factory=dict)    # req_id -> HelpRequest
    completed_help: Dict[int, Any] = field(default_factory=dict)   # req_id -> HelpRequest


    # handlers
    _action_handlers: Dict[DMActionKind, Callable[['DeliveryMan', DMAction, bool], None]] = field(init=False, repr=False)

    # equipment
    insulated_bag: Optional[InsulatedBag] = None
    e_scooter: Optional[EScooter] = None
    assist_scooter: Optional[EScooter] = None # ← 新增：帮别人的车（只能拖/充，不能骑）
    car: Optional[Car] = None
    inventory: Dict[str, int] = field(default_factory=dict)

    charge_price_per_pct: float = field(default=0.1)

    # flags
    towing_scooter: bool = False

    # contexts（虚拟时间）
    _charge_ctx: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _rest_ctx:   Optional[Dict[str, float]] = field(default=None, repr=False)
    _wait_ctx:   Optional[Dict[str, float]] = field(default=None, repr=False)
    _hospital_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)

    # movement context
    _move_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)  # {"tx":float,"ty":float,"tol":float,"blocked":0/1}

    # rental billing
    _rental_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)  # {"last_tick_sim": float, "rate_per_min": float}

    # bus context
    _bus_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)  # {"bus_id": str, "boarding_stop": str, "target_stop": str}

    # movement interrupt
    _interrupt_move_flag: bool = field(default=False, repr=False)
    _interrupt_reason: Optional[str] = field(default=None, repr=False)

    # rest config
    rest_rate_pct_per_min: float = field(default=8.0)
    low_energy_threshold_pct: float = field(default=30.0)

    # 强制放置
    _pending_food_by_order: Dict[int, List[Any]] = field(default_factory=dict, repr=False)
    _force_place_food_now: bool = False

    # 思考暂停
    _timers_paused: bool = field(default=False, repr=False)
    _orders_last_tick_sim: Optional[float] = field(default=None, repr=False)
    _life_last_tick_sim: Optional[float] = field(default=None, repr=False)
    _last_bag_tick_sim: Optional[float] = None

    # VLM
    vlm_prompt: str = get_system_prompt()
    vlm_past_memory: List[str] = field(default_factory=list)
    vlm_ephemeral: Dict[str, str] = field(default_factory=dict)
    vlm_errors: Optional[str] = None
    vlm_last_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=5), repr=False)
    vlm_last_compiled_input: Optional[str] = None

    # --- VLM 异步通道 ---
    _vlm_executor: Optional[Executor] = field(default=None, repr=False)
    _vlm_future: Optional[Future] = field(default=None, repr=False)
    _vlm_results_q: Deque[Dict[str, Any]] = field(default_factory=deque, repr=False)
    _vlm_inflight_token: Optional[int] = field(default=None, repr=False)
    _vlm_token_ctr: int = field(default=0, repr=False)
    _waiting_vlm: bool = field(default=False, repr=False)


    _vlm_client: Optional["BaseModel"] = field(default=None, repr=False)

    _vlm_retry_count: int = field(default=0, repr=False)
    _vlm_retry_max: int = field(default=5, repr=False)
    _vlm_last_bad_output: Optional[str] = field(default=None, repr=False)

    map_exportor: Optional[Any] = field(default=None, repr=False)

    pace_state: str = "normal"  # "accel" / "normal" / "decel"
    pace_scales: Dict[str, float] = field(init=False, repr=False)

    logger: logging.Logger = field(init=False, repr=False)


    def __post_init__(self):
        # 使用全局logger
        self.logger = get_agent_logger(f"DeliveryMan{self.agent_id}")
        
        self.avg_speed_by_mode = {
            TransportMode.WALK:      self.cfg["avg_speed_cm_s"]["walk"],
            TransportMode.SCOOTER:   self.cfg["avg_speed_cm_s"]["e-scooter"],
            TransportMode.DRAG_SCOOTER: self.cfg["avg_speed_cm_s"]["drag_scooter"],
            TransportMode.CAR:       self.cfg["avg_speed_cm_s"]["car"],
            TransportMode.BUS:       self.cfg["avg_speed_cm_s"]["bus"],
        }

        self.energy_cost_by_mode = {
            TransportMode.WALK:      self.cfg["energy_pct_decay_per_m_by_mode"]["walk"],
            TransportMode.DRAG_SCOOTER: self.cfg["energy_pct_decay_per_m_by_mode"]["drag_scooter"],
            TransportMode.SCOOTER:   self.cfg["energy_pct_decay_per_m_by_mode"]["e-scooter"],
            TransportMode.CAR:       self.cfg["energy_pct_decay_per_m_by_mode"]["car"],
            TransportMode.BUS:       self.cfg["energy_pct_decay_per_m_by_mode"]["bus"],
        }

        self.pace_scales = dict({
            "accel":  1.25,
            "normal": 1.00,
            "decel":  0.75,
        }, **(self.cfg.get("pace_scales", {})))

        self.scooter_batt_decay_pct_per_m = self.cfg["scooter_batt_decay_pct_per_m"]
        self.hospital_duration_s = self.cfg["hospital_duration_s"]
        self.hospital_rescue_fee = float(self.cfg.get("hospital_rescue_fee", 0.0))
        self.charge_price_per_pct = self.cfg["charge_price_per_percent"]
        self.rest_rate_pct_per_min = self.cfg["rest_rate_pct_per_min"]
        self.low_energy_threshold_pct = self.cfg["low_energy_threshold_pct"]

        self.energy_pct = self.cfg["energy_pct_max"]
        self.earnings_total = self.cfg["initial_earnings"]

        self.ambient_temp_c = float(self.cfg.get("ambient_temp_c", 22.0))
        self.k_food_per_s = float(self.cfg.get("k_food_per_s", 1.0/1800.0))

        self.speed_cm_s = self.avg_speed_by_mode[self.mode]

        es_cfg = self.cfg["escooter_defaults"]
        if self.e_scooter is None:
            self.e_scooter = EScooter()
            self.e_scooter.battery_pct = es_cfg["initial_battery_pct"]
            self.e_scooter.charge_rate_pct_per_min = es_cfg["charge_rate_pct_per_min"]
            self.e_scooter.avg_speed_cm_s = self.cfg["avg_speed_cm_s"]["e-scooter"]

        setattr(self.e_scooter, "owner_id", str(self.agent_id))
        # NEW: 兼容性补丁——确保存有 with_owner 标志
        if not hasattr(self.e_scooter, "with_owner"):
            setattr(self.e_scooter, "with_owner", True)
        if self.insulated_bag is None:
            self.insulated_bag = InsulatedBag()
            self.insulated_bag.ambient_temp_c = self.ambient_temp_c


        comms = get_comms()
        if comms:
            self.e_scooter = comms.register_scooter(str(self.agent_id), self.e_scooter)

        self._action_handlers = {
            DMActionKind.MOVE_TO:             self._handle_move_to,
            DMActionKind.ACCEPT_ORDER:        self._handle_accept_order,
            DMActionKind.VIEW_ORDERS:         self._handle_view_orders,
            DMActionKind.VIEW_BAG:            self._handle_view_bag,
            DMActionKind.PICKUP:              self._handle_pickup_food,
            DMActionKind.PLACE_FOOD_IN_BAG:  self._handle_place_food_in_bag,
            DMActionKind.CHARGE_ESCOOTER:     self._handle_charge_escooter,
            DMActionKind.WAIT:                self._handle_wait,
            DMActionKind.REST:                self._handle_rest,
            DMActionKind.BUY:                 self._handle_buy,
            DMActionKind.USE_BATTERY_PACK:    self._handle_use_battery_pack,
            DMActionKind.USE_ENERGY_DRINK:    self._handle_use_energy_drink,
            DMActionKind.USE_ICE_PACK:        self._handle_use_ice_pack,
            DMActionKind.USE_HEAT_PACK:       self._handle_use_heat_pack,
            DMActionKind.VIEW_HELP_BOARD:    self._handle_view_help_board,
            DMActionKind.POST_HELP_REQUEST:   self._handle_post_help_request,
            DMActionKind.ACCEPT_HELP_REQUEST: self._handle_accept_help_request,
            DMActionKind.EDIT_HELP_REQUEST:   self._handle_edit_help_request,
            DMActionKind.SWITCH_TRANSPORT:    self._handle_switch_transport,
            DMActionKind.RENT_CAR:            self._handle_rent_car,
            DMActionKind.RETURN_CAR:          self._handle_return_car,
            DMActionKind.PLACE_TEMP_BOX:       self._handle_place_temp_box,
            DMActionKind.TAKE_FROM_TEMP_BOX:   self._handle_take_from_temp_box,
            DMActionKind.REPORT_HELP_FINISHED: self._handle_report_help_finished,
            DMActionKind.DROP_OFF:             self._handle_drop_off,
            DMActionKind.SAY:                  self._handle_say,
            DMActionKind.BOARD_BUS:            self._handle_board_bus,
            DMActionKind.VIEW_BUS_SCHEDULE:    self._handle_view_bus_schedule,
            DMActionKind.TURN_AROUND:          self._handle_turn_around,
            DMActionKind.STEP_FORWARD:         self._handle_step_forward,
        }
        self._recalc_towing()

        self._vlm_retry_max = int(self.cfg.get("vlm", {}).get("retry_max", self._vlm_retry_max))

        # --- Lifecycle & Recorder ---
        life_cfg = dict(self.cfg.get("lifecycle", {}) or {})
        life_hours = float(life_cfg.get("duration_hours", 0.0))
        life_s = life_hours * 3600.0
        export_path = os.path.join(
            str(life_cfg.get("export_path", ".")),
            f"run_report_agent{self.agent_id}.json"
        )
        # --- Realtime lifecycle tracking ---
        self._realtime_start_ts = time.time()
        self._realtime_stop_hours = float(life_cfg.get("realtime_stop_hours", 0.0))
        
        # --- VLM call limit ---
        vlm_call_limit = int(life_cfg.get("vlm_call_limit", 0))
        
        self._recorder = RunRecorder(
            agent_id=str(self.agent_id),
            lifecycle_s=float(life_s if life_s > 0 else 0.0),
            export_path=export_path,
            initial_balance=float(self.earnings_total),
            realtime_stop_hours=self._realtime_stop_hours,
            realtime_start_ts=self._realtime_start_ts,
            vlm_call_limit=vlm_call_limit,
        )
        self._recorder.start(self.clock.now_sim(), self._realtime_start_ts)


    def _save_png(self, data: bytes, path: str) -> bool:
        with open(path, "wb") as f:
            f.write(data)
        return True

    def _tol(self, key: str, fallback: float = 500.0) -> float:
        return float(self.cfg.get("tolerance_cm", {}).get(key, fallback))

    def _has_scooter(self) -> bool:
        return self.e_scooter is not None

    def _has_any_scooter(self) -> bool:
        return (self.e_scooter is not None) or (self.assist_scooter is not None)

    def _pace_scale(self) -> float:
        return float(self.pace_scales.get(self.pace_state, 1.0))

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
        self._recorder.model = str(self._vlm_client.model)

    # ============== VLM 异步最小封装（仅网络在线程池；取图在主线程） ==============
    def set_vlm_executor(self, executor: Executor):
        """主程序(main)里建好的线程池传进来。"""
        self._vlm_executor = executor

    def request_vlm_async(self, prompt: str) -> bool:
        """
        若没有在飞的请求，则：主线程先采集图片 -> 线程池里只做网络请求。
        返回 True 表示本次已成功发起请求。
        """
        if self._vlm_executor is None or self._vlm_client is None:
            self._log("[VLM] executor/client not set"); 
            return False
        if self._vlm_future and not self._vlm_future.done():
            # 仍有在飞请求，避免洪泛
            return False

        self.timers_pause()

        # 主线程采图（Qt/pyqtgraph 必须在主线程）
        images = self.vlm_collect_images()

        # 打标记，防止旧结果"晚归"
        self._vlm_token_ctr += 1
        token = self._vlm_token_ctr
        self._vlm_inflight_token = token
        self._waiting_vlm = True

        # 提交线程任务：仅做网络调用
        def _call():
            resp = self._vlm_client.generate(user_prompt=prompt, images=images)
            return {"token": token, "resp": resp}

        self._vlm_future = self._vlm_executor.submit(_call)

        # 线程回调里只入队，不碰 UI
        def _done(fut: Future):
            try:
                res = fut.result()
                self._vlm_results_q.append(res)
            except Exception as e:
                # 确保主线程能看到错误并复位等待态
                self._vlm_results_q.append({"token": token, "error": str(e)})

        self._vlm_future.add_done_callback(_done)
        return True

    def pump_vlm_results(self) -> bool:
        """
        主线程定时调用（main 里用 QTimer 30ms 一次）。
        把队列里的 VLM 结果拿出来应用到状态机。
        """
        processed = False
        while self._vlm_results_q:
            rec = self._vlm_results_q.popleft()
            if rec.get("token") != self._vlm_inflight_token:
                # 过期结果，丢弃
                continue

            # 清理 in-flight 状态
            self._waiting_vlm = False
            self._vlm_inflight_token = None
            self._vlm_future = None

            if "error" in rec:
                self._on_vlm_failed(rec["error"])
            else:
                self._handle_vlm_response(rec["resp"])
            processed = True

        if processed and not getattr(self, "_waiting_vlm", False) and self._timers_paused:
            self.timers_resume()

        return processed

    def _handle_vlm_response(self, resp: Any):
        """
        解析 VLM 输出 -> 动作；若为拒答/解析失败则自动重试（带格式提示），直到成功或达上限。
        """
        raw = str(resp)
        
        # 记录VLM调用
        if self._recorder:
            self._recorder.inc("vlm_calls")

        try:
            act = parse_vlm_action(raw, self)  # 解析失败会抛 ValueError（见补丁 #1）
            from Base.DeliveryMan import DMAction  # 避免循环导入
            if not isinstance(act, DMAction):
                raise ValueError(f"bad return type: {type(act)}")
        except Exception as e:
            # 解析失败 -> 记录并再次请求
            if self._recorder:
                self._recorder.inc("vlm_parse_failures")
            self._retry_vlm(str(e), sample=raw)
            return

        # 成功：清空重试计数 & 提示，正常入队
        self._vlm_retry_count = 0
        self.vlm_ephemeral.pop("format_hint", None)
        if self._recorder:
            self._recorder.inc("vlm_successes")
        # print(f"[VLM] parsed action: {act.kind} {act.data if act.data else ''}")
        self.logger.info(f"[VLM] parsed action: {act.kind} {act.data if act.data else ''}")
        self.enqueue_action(act)


    def _retry_vlm(self, reason: str, sample: Optional[str] = None):
        """记录错误并立刻再次请求 VLM（带格式提醒），到达上限后给个温和兜底，避免死循环。"""
        self._vlm_retry_count += 1
        if self._recorder:
            self._recorder.inc("vlm_retries")
        self._vlm_last_bad_output = str(sample)[:160] if sample is not None else None

        # 这条提示会进 build_vlm_input() 的 ### ephemeral_context，帮助模型"对齐格式"
        self.vlm_ephemeral["format_hint"] = (
            "Your previous output was invalid. Reply with exactly ONE action call from the Action API. "
            "No explanations or apologies."
        )

        # recent_error 也会进入 prompt，给更强的纠错信号
        self.vlm_add_error(f"VLM invalid output (attempt {self._vlm_retry_count}/{self._vlm_retry_max}): {reason}")

        if self._vlm_retry_count <= self._vlm_retry_max:
            # 直接用最新 prompt 再发一次（已自动带上 format_hint / recent_error）
            self.request_vlm_async(self.build_vlm_input())
        else:
            # 超过上限：重置计数，移除提示，给一个温和兜底动作避免挂死
            self._vlm_retry_count = 0
            self.vlm_ephemeral.pop("format_hint", None)
            try:
                from Base.DeliveryMan import DMAction, DMActionKind
                self.enqueue_action(DMAction(DMActionKind.VIEW_ORDERS, data={}))
            except Exception:
                # 兜底失败也别再卡住
                pass

    def _on_vlm_failed(self, msg: str):
        self._log(f"[VLM] error: {msg}")
        self._retry_vlm(msg)

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
        return self._vlm_client.generate(user_prompt=compiled_prompt, images=images)

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
        comms = get_comms()
        if comms: comms.pause_timers_for(str(self.agent_id))
        self._timers_paused = True

    def timers_resume(self):
        if not self._timers_paused: return
        if getattr(self, "_waiting_vlm", False):
            return
        now = self.clock.now_sim()
        self._ctx_mark_resume(self._charge_ctx, now)
        self._ctx_mark_resume(self._rest_ctx,  now)
        self._ctx_mark_resume(self._wait_ctx,  now)
        self._ctx_mark_resume(self._hospital_ctx, now)
        if self._rental_ctx is not None:
            self._rental_ctx["last_tick_sim"] = now
        comms = get_comms()
        if comms: comms.resume_timers_for(str(self.agent_id))
        self._timers_paused = False
        self._orders_last_tick_sim = now
        self._life_last_tick_sim = now
        self._advance_charge_to_now()

    # ===== viewer events =====
    def _on_view_event(self, agent_id: str, event: str, payload: Dict[str, Any]):
        fx = float(payload.get("x", self.x)) if payload.get("x", self.x) is not None else self.x
        fy = float(payload.get("y", self.y)) if payload.get("y", self.y) is not None else self.y
        self.x, self.y = fx, fy

        if event == "move":
            # 先尝试自动送达
            # self._auto_try_dropoff()

            self._refresh_poi_hints_nearby()

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
            self._viewer.log_action(prefix + text, also_print=False)
            self.logger.info(f"[Agent {self.agent_id}] {text}")
        else:
            print(f"[DeliveryMan {self.name}] {text}")

    def _fmt_inv_compact(self, inv: Dict[str, int]) -> str:
        if not inv:
            return "empty"
        parts = [f"{k} x{int(v)}" for k, v in inv.items() if int(v) > 0]
        return ", ".join(parts) if parts else "empty"

    def vlm_add_memory(self, text: str): self.vlm_past_memory.append(str(text))
    def vlm_clear_memory(self): self.vlm_past_memory.clear()
    def vlm_add_ephemeral(self, tag: str, text: str): self.vlm_ephemeral[str(tag)] = str(text)
    def vlm_clear_ephemeral(self): self.vlm_ephemeral.clear()
    def vlm_add_error(self, msg: str): 
        self.vlm_errors = f"You just tried {action_to_text(self._current)}, but it failed. Error message: '{msg}'."
    def vlm_clear_errors(self): self.vlm_errors = None
    def _register_success(self, note: str): self.vlm_last_actions.append(note); self.vlm_clear_errors()

    # ===== state/speed =====
    def is_busy(self) -> bool:
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "is_busy"):
            return bool(self._viewer.is_busy(self._viewer_agent_id))
        return self._current is not None

    def _recalc_towing(self):
        if self.mode == TransportMode.DRAG_SCOOTER:
            self.towing_scooter = True
        elif (
            self.e_scooter and
            getattr(self.e_scooter, "with_owner", True) and
            self.e_scooter.state == ScooterState.DEPLETED and
            not self.e_scooter.park_xy
        ):
            self.towing_scooter = True
        else:
            self.towing_scooter = False

    def get_current_speed_for_viewer(self) -> float:
        self._recalc_towing()
        ts = float(getattr(self.clock, "time_scale", 1.0) or 1.0)
        return float(self.speed_cm_s) * self._pace_scale() * ts

    def set_mode(self, mode: TransportMode, *, override_speed_cm_s: Optional[float] = None):
        mode = TransportMode(mode)
        if mode == TransportMode.SCOOTER and self.assist_scooter is not None:
            mode = TransportMode.DRAG_SCOOTER

        if mode == TransportMode.SCOOTER:
            if not self.e_scooter:
                mode = TransportMode.DRAG_SCOOTER
            else:
                owner_ok = (getattr(self.e_scooter, "owner_id", None) == str(self.agent_id))
                usable   = (self.e_scooter.state != ScooterState.DEPLETED)
                with_me  = bool(getattr(self.e_scooter, "with_owner", True))
                if not (owner_ok and usable and with_me):
                    mode = TransportMode.DRAG_SCOOTER

        self.mode = mode
        self.pace_state = "normal"

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
            base = float(self.avg_speed_by_mode.get(self.mode))
            if override_speed_cm_s is not None: base = float(override_speed_cm_s)
            self.speed_cm_s = base

        self._recalc_towing()

    def on_move_consumed(self, distance_cm: float):
        if distance_cm <= 0.0: return
        self._consume_by_distance(distance_cm)

    def _consume_personal_energy_by_distance(self, distance_m: float):
        """按当前交通方式，从 agent 的体力中扣除（单位：米）。"""
        # 拖车属于 DRAG_SCOOTER，会自动使用映射里的高消耗
        cost_per_m = float(self.energy_cost_by_mode.get(self.mode, 0.0)) * self._pace_scale()  # ← 乘节奏系数
        if cost_per_m > 0.0:
            self.energy_pct -= cost_per_m * max(0.0, float(distance_m))
            if self.energy_pct <= 0.0:
                self.energy_pct = 0.0
                self._trigger_hospital_if_needed()

    def _consume_vehicle_by_distance(self, distance_m: float):
        """车辆侧能量/电量消耗（目前仅电瓶车有电量）。"""
        if self.mode == TransportMode.SCOOTER and self.e_scooter and getattr(self.e_scooter, "with_owner", True):
            before = self.e_scooter.battery_pct
            delta_pct = max(0.0, float(distance_m)) * self.scooter_batt_decay_pct_per_m * self._pace_scale()  # ← 乘节奏系数
            self.e_scooter.consume_pct(delta_pct)

            # 电瓶耗尽 -> 触发中断并自动切换为拖车模式
            if self.e_scooter.battery_pct <= 0.0 and before > 0.0:
                self.e_scooter.state = ScooterState.DEPLETED
                self._interrupt_and_stop(
                    "escooter_depleted",
                    "Your e-scooter battery is depleted. You may SWITCH_TRANSPORT(to='walk') to leave the scooter, "
                    "or keep dragging it to a charging station and then CHARGE_ESCOOTER(target_pct=80)."
                )
                self.set_mode(TransportMode.DRAG_SCOOTER)
        # 其他载具（car/bus）目前不模拟载具能耗；后续要加"油/电"也可在这里扩展。

    def _consume_by_distance(self, distance_cm: float):
        distance_m = max(0.0, float(distance_cm) / 100.0)
        if distance_m <= 0.0:
            return

        self._recalc_towing()

        # 先统一从"人"的体力扣除（所有 mode 都会消耗，包括 WALK/SCOOTER/CAR/BUS/DRAG_SCOOTER）
        self._consume_personal_energy_by_distance(distance_m)

        # 再按需要扣"载具"的能量/电量（目前仅电瓶车）
        self._consume_vehicle_by_distance(distance_m)

    def _sync_help_lists(self) -> None:
        """与 Comms 同步我作为 helper 的请求对象，填充 accepted_help / completed_help。"""
        try:
            comms = get_comms()
            if not comms:
                return
            mine_active = {int(r.req_id): r for r in comms.list_my_helps(str(self.agent_id))}
            mine_done   = {int(r.req_id): r for r in comms.list_my_helps_completed(str(self.agent_id))}
            self.accepted_help = mine_active
            self.completed_help = mine_done
        except Exception:
            pass


    def rescue(self):
        self.energy_pct = float(self.cfg.get("energy_pct_max", 100.0))
        self.is_rescued = False

    # ===== map helpers =====
    def _xy_of_node(self, node: Any) -> Optional[Tuple[float, float]]:
        return float(node.position.x), float(node.position.y)

    def _is_at_xy(self, x: float, y: float, tol_cm: Optional[float] = None) -> bool:
        tol_cm = float(tol_cm) if tol_cm is not None else self._tol("nearby")
        return math.hypot(self.x - x, self.y - y) <= float(tol_cm)

    def _nearest_poi_xy(self, kind: str, tol_cm: Optional[float] = None) -> Optional[Tuple[float, float]]:
        tol_cm = float(tol_cm) if tol_cm is not None else self._tol("nearby")
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
        return float(self.e_scooter.battery_pct) / max(1e-9, self.scooter_batt_decay_pct_per_m)

    def _agent_state_text(self) -> str:
        active_ids = [getattr(o, "id", None) for o in self.active_orders if getattr(o, "id", None) is not None]
        help_ids   = list(getattr(self, "help_orders", {}).keys())
        carrying_ids = list(self.carrying)
        mode_str = "towing a scooter" if self.towing_scooter else self.mode.value
        speed_ms = self.speed_cm_s / 100
        lines = []
        lines.append(f"You are Agent {self.agent_id}. There are {self.cfg.get('agent_count', 0)} delivery agents in total in this city.")
        lines.append(f"Your current transport mode is {mode_str}, at {self._fmt_xy_m(self.x, self.y)}.")
        lines.append(f"Your speed is ~{speed_ms:.1f} m/s, energy is {self.energy_pct:.0f}%.")
        pace_map = {"accel":"accelerating", "normal":"normal", "decel":"decelerating"}
        lines.append(f"Your current pace is {pace_map.get(self.pace_state,'normal')} (×{self._pace_scale():.2f}).")
        lines.append(f"Earnings is ${self.earnings_total:.2f}.")
        if active_ids: lines.append(f"Active orders: {', '.join(map(str, active_ids))}.")
        if help_ids:   lines.append(f"Helping orders: {', '.join(map(str, help_ids))}.")
        if carrying_ids: lines.append(f"Carrying: {', '.join(map(str, carrying_ids))}.")
        lines.append(f"Rest energy recovery rate is +{self.rest_rate_pct_per_min:.1f}%/min.")

        if self.inventory:
            inv_str = ", ".join([f"{k} x{int(v)}" for k, v in self.inventory.items() if int(v) > 0]) or "empty"
        else:
            inv_str = "empty"
        lines.append(f"Inventory: {inv_str}.")

        if self.e_scooter:
            rng_m = self._remaining_range_m(); rng_km = (rng_m/1000.0) if rng_m is not None else None
            park_str = f"parked at {self._fmt_xy_m_opt(self.e_scooter.park_xy)}" if self.e_scooter.park_xy else "not parked"
            if rng_m is not None:
                lines.append(f"Scooter: {self.e_scooter.state.value}, batt {self.e_scooter.battery_pct:.0f}%, range {rng_m:.1f} m,")
            else:
                lines.append(f"Scooter: {self.e_scooter.state.value}, batt {self.e_scooter.battery_pct:.0f}%, range N/A,")
            lines.append(f"charge rate {self.e_scooter.charge_rate_pct_per_min:.1f}%/min, {park_str}.")
        
        if self._charge_ctx:
            ctx = self._charge_ctx
            sc = ctx.get("scooter_ref") or self.e_scooter
            if sc:
                cur  = float(ctx.get("paid_pct", getattr(sc, "battery_pct", 0.0)))
                pt   = float(ctx.get("target_pct", 100.0))
                spot = self._fmt_xy_m_opt(getattr(sc, "park_xy", None))
                which = ctx.get("which", "own")
                lines.append(
                    f"Charging in progress ({'assist' if which=='assist' else 'own'}): {cur:.0f}% → {pt:.0f}% at {spot}."
                )

        # --- assisting (foreign) scooter 展示 ---
        if self.assist_scooter:
            asc = self.assist_scooter
            owner = getattr(asc, "owner_id", "?")
            rng_m = float(asc.battery_pct) / max(1e-9, self.scooter_batt_decay_pct_per_m)
            rng_km = rng_m / 1000.0
            park_str = f"parked at {self._fmt_xy_m_opt(asc.park_xy)}" if asc.park_xy else "not parked"
            lines.append(f"Assisting scooter (owner agent {owner}), batt {asc.battery_pct:.0f}%, range {rng_m:.1f} m, {park_str}.")

        # --- 如果你的车已交给 TempBox，提示车在哪个 TempBox（publisher 视角） ---
        try:
            comms = get_comms()
            if comms:
                my_active = comms.list_my_posts_active(str(self.agent_id))
                for r in my_active:
                    if r.kind == HelpType.HELP_CHARGE:
                        info = comms.get_temp_box_info(int(r.req_id)) or {}
                        pub_box = info.get("publisher_box", {})
                        if pub_box.get("xy"):
                            lines.append(f"Your scooter is placed in TempBox at {self._fmt_xy_m_opt(pub_box['xy'])}. You can TAKE_FROM_TEMP_BOX(req_id={int(r.req_id)}) to reclaim it when ready.")
        except Exception:
            pass

        if self.car:
            lines.append(f"Car {self.car.state.value}, rate ${self.car.rate_per_min:.2f}/min, park_xy={self._fmt_xy_m_opt(self.car.park_xy) if self.car.park_xy else 'N/A'}, rental={'on' if self._rental_ctx else 'off'}.")

        return " ".join(lines)

    def _build_bag_place_hint(self) -> str:
        lines = [
            "You have UNPLACED FOOD items that must be arranged into the insulated bag.",
            "Output ONE combined bag_cmd per order, e.g.:",
            "  order <id>: 1,2 -> A; 3 -> B",
            ""
        ]
        # 展示待放物
        for oid, items in (self._pending_food_by_order or {}).items():
            lines.append(f"- Order #{int(oid)} items:")
            if items:
                for i, it in enumerate(items, start=1):
                    name = getattr(it, "name", None) or getattr(it, "title", None) or getattr(it, "label", None) or str(it)
                    lines.append(f"    {i}. {name}")
            else:
                lines.append("    (none)")
        # 展示当前袋布局
        lines += ["", "Current bag layout:"]
        if self.insulated_bag:
            lines.append(self.insulated_bag.list_items())
        else:
            lines.append("(no bag)")
        lines += ["", "Example (single order):", "  order 12: 1,2 -> A; 3 -> B",
          "", "Example (multiple orders):", "  order 2: 1,2 -> A",
          "  order 3: 1,2,3,4 -> B"]

        return "\n".join(lines)

    def _build_pickup_arrival_hint(self, ready_orders: List[Any], waiting_pairs: List[Tuple[Any, int]]) -> str:
        """
        waiting_pairs: [(order_obj, mins_remaining), ...]
        """
        lines: List[str] = []

        # Try to use the first order's pickup road name as the place label
        place = None
        if ready_orders:
            place = getattr(ready_orders[0], "pickup_road_name", None) or place
        if place is None and waiting_pairs:
            place = getattr(waiting_pairs[0][0], "pickup_road_name", None) or place

        if place:
            lines.append(f"You have arrived at the pickup location: {place}.")
        else:
            lines.append("You are near a pickup location.")

        if ready_orders:
            ready_ids = [getattr(o, "id", None) for o in ready_orders]
            ready_ids = [f"#{oid}" for oid in ready_ids if oid is not None]
            if ready_ids:
                lines.append(f"Orders ready for pickup: {', '.join(ready_ids)}")

        if waiting_pairs:
            waiting_texts = []
            for (o, mins) in waiting_pairs:
                oid = getattr(o, "id", None)
                if oid is not None:
                    waiting_texts.append(f"#{oid} ~{mins} min")
            if waiting_texts:
                lines.append(f"Still being prepared (ETA): {', '.join(waiting_texts)}")

        if ready_orders:
            ids_list = [str(getattr(o, "id")) for o in ready_orders if getattr(o, "id", None) is not None]
            if ids_list:
                lines.append("")
                lines.append(f"You can execute: PICKUP(orders=[{','.join(ids_list)}]) to collect the ready orders.")

        return "\n".join(lines)

    def _refresh_pickup_hint_nearby(self):
        """
        在取餐点附近时给出可 PICKUP 的提示。
        扫描自己的 active_orders 以及我作为 helper 的 help_orders。
        """
        tol = self._tol("nearby")

        here_orders_ready: List[Any] = []
        here_orders_waiting: List[Tuple[Any, int]] = []

        # 同时考虑自己的单和 helper 的单
        all_considered: List[Any] = list(self.active_orders or []) + list(getattr(self, "help_orders", {}).values())

        for o in all_considered:
            # 只看"尚未取餐"的订单
            if getattr(o, "has_picked_up", False):
                continue

            pu = self._xy_of_node(getattr(o, "pickup_node", None))
            if not pu or not self._is_at_xy(pu[0], pu[1], tol_cm=tol):
                continue

            # —— 使用"订单的活动时间"而不是全局 now_sim ——
            local_now = o.active_now()

            # 若有 is_ready_for_pickup 则用它；否则视为已备好
            if not hasattr(o, "is_ready_for_pickup"):
                is_ready = True
            else:
                is_ready = o.is_ready_for_pickup(local_now)

            if is_ready:
                here_orders_ready.append(o)
            else:
                remain_s = 0.0
                if hasattr(o, "remaining_prep_s"):
                    remain_s = float(o.remaining_prep_s(local_now))

                mins = max(0, int(math.ceil(remain_s / 60.0)))
                here_orders_waiting.append((o, mins))

        # 写入/清理临时提示
        if here_orders_ready or here_orders_waiting:
            self.vlm_ephemeral["pickup_hint"] = self._build_pickup_arrival_hint(here_orders_ready, here_orders_waiting)
        else:
            self.vlm_ephemeral.pop("pickup_hint", None)


    # === NEW: Unified POI arrival hints (EN) ===
    def _refresh_poi_hints_nearby(self):
        """
        Centralize all "arrived at POI" hints here. Only writes lightweight
        suggestions into `self.vlm_ephemeral` for the VLM to see.
        No side effects (no auto-buy/auto-charge/etc).
        """
        tol = self._tol("nearby")
        now_sim = self.clock.now_sim()

        # Clear this round's POI hints (keep bag_hint / pickup_hint / scooter_ready if any)
        for k in ("charging_hint", "rest_hint", "store_hint", "bus_hint",
                  "rental_hint", "hospital_hint", "tempbox_hint", "escooter_depleted"):
            self.vlm_ephemeral.pop(k, None)

        # Reuse your original pickup arrival logic
        self._refresh_pickup_hint_nearby()

        # (A) E-scooter depleted & currently towing -> show clear choices
        if self.e_scooter and self.e_scooter.state == ScooterState.DEPLETED and self.towing_scooter:
            self.vlm_ephemeral["escooter_depleted"] = (
                "Your e-scooter battery is depleted. You may SWITCH_TRANSPORT(to='walk') to leave the scooter, "
                "or keep dragging it to a charging station and then CHARGE_ESCOOTER(target_pct=80)."
            )

        # (B) At a charging station & scooter is with you (towing or parked here) -> hint charge + retrieve
        near_chg = self._nearest_poi_xy("charging_station", tol_cm=tol)
        if near_chg:
            # 仍按原规则选择"用于充电"的候选：优先 assist，其次 with_owner 的自车
            sc_charge = self.assist_scooter if self.assist_scooter is not None else (
                self.e_scooter if (self.e_scooter and getattr(self.e_scooter, "with_owner", True)) else None
            )

            # 额外：检测"任意一辆停在脚边的车"（不看 with_owner）
            parked_here_cmd = None
            for label, s in (("assist", self.assist_scooter), ("own", self.e_scooter)):
                if s is not None and getattr(s, "park_xy", None) and self._is_at_xy(s.park_xy[0], s.park_xy[1], tol_cm=tol):
                    # 根据是哪辆车给出对应的 Switch 提示
                    parked_here_cmd = 'Switch(to="assist_scooter")' if label == "assist" else 'Switch(to="escooter")'
                    break

            # 组装提示
            hints = []

            if sc_charge:
                scooter_here = (
                    self.towing_scooter or
                    (getattr(sc_charge, "park_xy", None) and self._is_at_xy(sc_charge.park_xy[0], sc_charge.park_xy[1], tol_cm=tol))
                )
                if scooter_here and float(getattr(sc_charge, "battery_pct", 0.0)) < 100.0:
                    hints.append(
                        "You are at a charging station. You can CHARGE_ESCOOTER(target_pct=60..100). "
                        "(If you are riding, it will park the scooter automatically before charging.)"
                    )

            if parked_here_cmd:
                hints.append(f"You have a parked scooter here. You MUST MUST MUST now {parked_here_cmd} to retrieve it.")

            if hints:
                self.vlm_ephemeral["charging_hint"] = " ".join(hints)


        # (C) At rest area -> hint REST
        if self._nearest_poi_xy("rest_area", tol_cm=tol) is not None and self.energy_pct < 100.0:
            self.vlm_ephemeral["rest_hint"] = "You are at a rest area. You can REST(target_pct=100)."

        # (D) At store -> hint BUY (common items)
        if self._nearest_poi_xy("store", tol_cm=tol) is not None:
            self.vlm_ephemeral["store_hint"] = (
                "You are inside a store. You can BUY(item_id='energy_drink', qty=XX) or "
                "BUY(items=[{'item_id':'energy_drink','qty':XX}, {'item_id':'escooter_battery_pack','qty':XX}, "
                "{'item_id':'ice_pack','qty':XX}, {'item_id':'heat_pack','qty':XX}])."
            )

        # (E) At bus stop -> hint (assume feature exists)
        if self._nearest_poi_xy("bus_station", tol_cm=tol) is not None:
            self.vlm_ephemeral["bus_hint"] = (
                "You are at a bus stop. When a bus arrives, you can board it."
            )

        # (F) At car rental -> hint rent/return
        if self._nearest_poi_xy("car_rental", tol_cm=tol) is not None:
            if self.car is None:
                self.vlm_ephemeral["rental_hint"] = (
                    "You are at a car rental. You can RENT_CAR()."
                )
            else:
                self.vlm_ephemeral["rental_hint"] = (
                    "You are at a car rental. You can RETURN_CAR() to stop billing."
                )

        # (G) Hospital rescue progress hint (your logic already teleports + counts down)
        if self._hospital_ctx:
            remain = max(0.0, float(self._hospital_ctx["end_sim"] - now_sim))
            mins = int(math.ceil(remain / 60.0))
            self.vlm_ephemeral["hospital_hint"] = (
                f"You are being treated in the hospital. About {mins} min to full energy."
            )

        # (H) TempBox hints (role- & help-type–aware; stacked)
        try:
            comms = get_comms()
            if comms:
                msgs: List[str] = []
                me = str(self.agent_id)

                def _as_kind(val):
                    """Robustly normalize to HelpType enum."""
                    if isinstance(val, HelpType):
                        return val
                    s = str(val)
                    for h in HelpType:
                        if s == h.name or s == h.value:
                            return h
                    return None

                def _tokens(kind_val):
                    """
                    Returns (place_noun, take_noun, content_example) for hints.
                    - place_noun: what the publisher should place
                    - take_noun : what the taker will retrieve
                    - content_example: example 'content=...' for PLACE_TEMP_BOX
                    """
                    k = _as_kind(kind_val)
                    if k == HelpType.HELP_CHARGE:
                        return ("an e-scooter", "the e-scooter", "{'escooter': ''}")
                    if k in (HelpType.HELP_DELIVERY, HelpType.HELP_PICKUP):
                        return ("food", "the food", "{'food': ''}")
                    if k == HelpType.HELP_BUY:
                        return ("inventory items", "the purchased items", "{'inventory': {...}}")
                    return ("items", "the items", "{'food': ''} or {'inventory': {...}}")

                # Helper view: near the publisher's TempBox -> TAKE
                for r in getattr(comms, "_active", {}).values():
                    if r.completed or str(getattr(r, "accepted_by", "")) != me:
                        continue
                    rid = int(getattr(r, "req_id"))
                    info = comms.get_temp_box_info(rid) or {}
                    pub_box = info.get("publisher_box", {})  # helper takes from here
                    if pub_box.get("xy"):
                        bx, by = pub_box["xy"]
                        if self._is_at_xy(float(bx), float(by), tol_cm=tol):
                            place_noun, take_noun, _ = _tokens(getattr(r, "kind", None))
                            if pub_box.get("has_content"):
                                msgs.append(f"[Help #{rid}] TempBox is here. Use TAKE_FROM_TEMP_BOX(req_id={rid}) to collect {take_noun}.")
                            else:
                                msgs.append(f"[Help #{rid}] TempBox is here but empty. Wait for the publisher to place {place_noun}.")

                # Publisher view A: at provide_xy (handover point) -> PLACE
                for r in getattr(comms, "_active", {}).values():
                    if r.completed or str(getattr(r, "publisher_id", "")) != me or not getattr(r, "accepted_by", None):
                        continue
                    rid = int(getattr(r, "req_id"))
                    det = comms.get_request_detail(rid) or {}
                    pxy = det.get("provide_xy")
                    if pxy and self._is_at_xy(float(pxy[0]), float(pxy[1]), tol_cm=tol):
                        place_noun, _take_noun, content_example = _tokens(det.get("kind"))
                        msgs.append(
                            f"[Help #{rid}] You are at the handover point. "
                            f"Use PLACE_TEMP_BOX(req_id={rid}, content={content_example}) to place {place_noun}."
                        )

                # Publisher view B: near the helper's TempBox -> TAKE (helper dropped off)
                for r in getattr(comms, "_active", {}).values():
                    if r.completed or str(getattr(r, "publisher_id", "")) != me:
                        continue
                    rid = int(getattr(r, "req_id"))
                    info = comms.get_temp_box_info(rid) or {}
                    helper_box = info.get("helper_box", {})
                    if helper_box.get("xy"):
                        bx, by = helper_box["xy"]
                        if self._is_at_xy(float(bx), float(by), tol_cm=tol):
                            _place_noun, take_noun, _ = _tokens(getattr(r, "kind", None))
                            if helper_box.get("has_content"):
                                msgs.append(f"[Help #{rid}] Helper's TempBox is here. Use TAKE_FROM_TEMP_BOX(req_id={rid}) to retrieve {take_noun}.")
                            else:
                                msgs.append(f"[Help #{rid}] Helper's TempBox is here but empty.")

                # Show stacked hints for all matched requests
                if msgs:
                    self.vlm_ephemeral["tempbox_hint"] = "\n".join(msgs)
        except Exception:
            pass

        # (I) Drop-off hints for manual DROP_OFF
        try:
            tol = self._tol("nearby")
            msgs = []

            # 自己的订单
            for o in (self.active_orders or []):
                if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                    dxy = self._xy_of_node(getattr(o, "dropoff_node", None))
                    if dxy and self._is_at_xy(dxy[0], dxy[1], tol_cm=tol):
                        oid = int(getattr(o, "id", -1))
                        msgs.append(f"You are at the drop-off for order #{oid}. Choose a delivery method and call DROP_OFF(oid={oid}, method='leave_at_door|knock|call|hand_to_customer').")

            # 我作为 helper 的订单
            for oid, o in (self.help_orders or {}).items():
                if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                    dxy = self._xy_of_node(getattr(o, "dropoff_node", None))
                    if dxy and self._is_at_xy(dxy[0], dxy[1], tol_cm=tol):
                        msgs.append(f"You are at the helper drop-off for order #{int(oid)}. Call DROP_OFF(oid={int(oid)}, method='leave_at_door|knock|call|hand_to_customer').")

            if msgs:
                self.vlm_ephemeral["dropoff_hint"] = "\n".join(msgs)
            else:
                self.vlm_ephemeral.pop("dropoff_hint", None)
        except Exception:
            pass


    def build_vlm_input(self) -> str:
        """分离system prompt和user prompt, 把不变的内容放在system prompt中, 把可变的内容放在user prompt中"""
        # parts: List[str] = ["### system_prompt\n "+self.vlm_prompt]
        parts: List[str] = []
        if self.vlm_past_memory:
            parts.append("### past_memory"); parts += [f"- {m}" for m in self.vlm_past_memory]
        parts.append("### agent_state"); parts.append(self._agent_state_text())

        if self._store_manager and hasattr(self._store_manager, "to_text"):
            try:
                parts.append("### store_catalog")
                parts.append(self._store_manager.to_text(title="Available items & effects"))
            except Exception:
                # 容错：就算 store 还没加载好，也不要影响主 prompt
                pass

        # === NEW: always show accepted-but-not-delivered orders ===
        active_blocks: List[str] = []
        for o in (self.active_orders or []):
            if getattr(o, "is_accepted", False) and not getattr(o, "has_delivered", False):
                active_blocks.append(o.to_text())
        # helper-side unfinished orders (I accepted as helper)
        for o in getattr(self, "help_orders", {}).values():
            if not getattr(o, "has_delivered", False):
                active_blocks.append(o.to_text())

        parts.append("### active_orders")
        if active_blocks:
            # keep it compact; separator line is helpful but lightweight
            parts.append("\n" + ("\n" + "-"*48 + "\n").join(active_blocks))
        else:
            # 没有接受任何订单时给出提示
            parts.append("You currently have no accepted orders.")

        # === accepted_help===
        try:
            self._sync_help_lists()
            comms = get_comms()
            parts.append("### accepted_help")
            if comms and self.accepted_help:
                now = self.clock.now_sim()
                for r in self.accepted_help.values():
                    parts.append(r.to_text_for(str(self.agent_id), comms, now=now, view_as="helper"))
            else:
                parts.append("(none)")
        except Exception:
            pass

        # === posted_help===
        try:
            comms = get_comms()
            parts.append("### posted_help")
            if comms:
                now = self.clock.now_sim()
                my_open   = comms.list_my_posts_open(str(self.agent_id))
                my_active = comms.list_my_posts_active(str(self.agent_id))
                if not (my_open or my_active):
                    parts.append("(none)")
                else:
                    # 先显示已被接的，再显示仍在板上的（你也可以反过来）
                    for r in (my_active + my_open):
                        parts.append(r.to_text_for(str(self.agent_id), comms, now=now, view_as="publisher"))
            else:
                parts.append("(none)")
        except Exception:
            pass

        parts.append("### pickables"); parts.append(comms.pickables_text_for(self.agent_id))
        
        parts.append("### map_snapshot"); parts.append(self._map_brief())
        # parts.append("### action_api"); parts.append(ACTION_API_SPEC.strip())
        if self.vlm_last_actions:
            parts.append("### recent_actions")
            actions = list(self.vlm_last_actions)
            for i, a in enumerate(actions):
                if i == len(actions) - 1:
                    parts.append(f"- [Your last successfully executed action] {a}")
                else:
                    parts.append(f"- {a}")

        if self.vlm_errors:
            parts.append("### recent_error"); parts.append(self.vlm_errors)

        # bag / ephemeral as-is
        if self._force_place_food_now:
            hint = self._build_bag_place_hint()
            parts.append("### ephemeral_context")
            parts.append(f"[bag_hint]\n{hint}")
        elif self.vlm_ephemeral:
            parts.append("### ephemeral_context")
            for k, v in self.vlm_ephemeral.items():
                parts.append(f"[{k}]\n{v}")

        txt = "\n".join(parts)
        self.vlm_last_compiled_input = txt
        return txt


    def _map_brief(self) -> str:
        limits = self.cfg.get("map_snapshot_limits", {})
        if hasattr(self.city_map, "agent_info_package_xy"):
            pkg = self.city_map.agent_info_package_xy(
                float(self.x), float(self.y),
                include_docks=False,
                limit_next=int(limits.get("next", 20)),
                limit_s=int(limits.get("s", 40)),
                limit_poi=int(limits.get("poi", 80))
            )
            if isinstance(pkg, dict) and pkg.get("text"): return str(pkg["text"])
        return "map_brief: N/A"

    # ===== VLM decider =====
    def _default_decider(self) -> Optional[DMAction]:
        # return
        # 不可决策状态：直接返回
        if self._lifecycle_done:
            return None
        if self.is_rescued or self._hospital_ctx is not None or self.energy_pct <= 0.0:
            return None

        # 已经在等一次 VLM 结果了，就不要重复发
        if getattr(self, "_waiting_vlm", False):
            return None

        # 组 prompt + 渲染图片（这部分必须在主线程；你接受这点）
        prompt = self.build_vlm_input()
        # print(f'User Prompt:\n{prompt}')
        self.logger.debug(f"[VLM] User Prompt:\n{prompt}")

        # 如果你希望渲染不计入虚拟时间，可临时 pause/resume（可选）
        # self.timers_pause()
        # images = self.vlm_collect_images()
        # self.timers_resume()

        # 让 request_vlm_async 自己去收集图片（或把上面 images 传进去，二选一）
        try:
            self.request_vlm_async(prompt)   # 非阻塞：把 generate 丢到线程池
            # 如果你选择上面先渲染 images，则用：self.request_vlm_async(prompt, images_override=images)
        except Exception as e:
            self.vlm_add_error(f"VLM dispatch error: {e}")

        # 立刻返回，不阻塞 UI
        return None

    # ===== loop/scheduling =====
    def kickstart(self):
        if self._current is None and not self._queue:
            self.timers_pause()
            act = self._default_decider()
            self.timers_resume()
            if act is not None: self.enqueue_action(act)

    def enqueue_action(self, action: DMAction, *, allow_interrupt: bool = False):
        if not isinstance(action, DMAction):
            self._log(f"ignore invalid action enqueued: {type(action)}")
            return
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

        if self._recorder:
            self._recorder.inc_nested(f"action_attempts.{act.kind.value}")

        handler(self, act, allow_interrupt)

    def _finish_action(self, *, success: bool):
        if self._current and callable(self._current.on_done):
            self._current.on_done(self)
        if success and self._current:
            self._register_success(action_to_text(self._current))

        if self._recorder and self._current and success:
            action_name = self._current.kind.value
            self._recorder.inc_nested(f"action_successes.{action_name}")
            
        self._current = None

        if self._lifecycle_done:
            self._current = None
            return
        
        self.timers_pause()
        next_act = self._default_decider()
        if next_act is not None:
            from PyQt5.QtCore import QTimer
            delay = int(self.cfg.get("vlm", {}).get("next_action_delay_ms", 300))
            QTimer.singleShot(delay, lambda: self.enqueue_action(next_act))
        self.timers_resume()

    def register_action(self, kind: DMActionKind, handler: Callable[['DeliveryMan', DMAction, bool], None]):
        self._action_handlers[kind] = handler

    # ===== Handlers =====
    def _handle_say(self, _self, act: DMAction, _allow_interrupt: bool):
        text = str(act.data.get("text", "") or "").strip()
        to   = act.data.get("to", None)

        if not text:
            self.vlm_add_error("say failed: empty text"); self._finish_action(success=False); return

        comms = get_comms()
        if not comms:
            self.vlm_add_error("say failed: comms not ready"); self._finish_action(success=False); return

        is_broadcast = (to is None) or (str(to).upper() in ("ALL", "*"))
        target_id = None if is_broadcast else str(to)

        ok, msg, _ = comms.send_chat(from_agent=str(self.agent_id), text=text, to_agent=target_id, broadcast=is_broadcast)
        if not ok:
            self.vlm_add_error(f"say failed: {msg}"); self._finish_action(success=False); return

        if is_broadcast:
            self._log(f"chat broadcast: {text}")
            self.vlm_ephemeral["chat_sent"] = f"(broadcast) {text}"
        else:
            self._log(f"chat to agent {target_id}: {text}")
            self.vlm_ephemeral["chat_sent"] = f"to {target_id}: {text}"

        if self._recorder:
            self._recorder.inc("say", 1)

        self._finish_action(success=True)

    def _handle_move_to(self, _self, act: DMAction, allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        self._interrupt_move_flag = False
        sx, sy = float(self.x), float(self.y)
        tx, ty = float(act.data.get("tx", self.x)), float(act.data.get("ty", self.y))
        tol = float(act.data.get("arrive_tolerance_cm", self._tol("nearby")))

        pace = str(act.data.get("pace", "normal")).strip().lower()
        if pace not in ("accel", "decel", "normal"):
            pace = "normal"
        self.pace_state = pace
        if pace == "accel" and self.insulated_bag:
            self.insulated_bag.bump_motion_damage(inc=1)

        self._recalc_towing()
        if "expected_dist_cm" not in act.data:
            act.data["expected_dist_cm"] = self._estimate_distance_cm(sx, sy, tx, ty, bool(act.data.get("use_route", True)), float(act.data.get("snap_cm", 120.0)))

        mode_str = 'towing' if self.towing_scooter else self.mode.value
        speed_to_use = float(self.speed_cm_s) * self._pace_scale()
        self._log(f"move from {self._fmt_xy_m(sx, sy)} to {self._fmt_xy_m(tx, ty)} [mode={mode_str}, speed={speed_to_use:.1f} cm/s, pace={self.pace_state}]")

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

        om = self._order_manager
        if om is None:
            self.vlm_add_error("accept_order failed: no order manager")
            self._finish_action(success=False)
            return

        # 1) 归一化为去重后的 id 列表
        if "oids" in act.data:
            oids = [int(x) for x in (act.data.get("oids") or [])]
        elif "oid" in act.data:
            oids = [int(act.data.get("oid"))]
        else:
            self.vlm_add_error("accept_order failed: need 'oid' or 'oids'")
            self._finish_action(success=False)
            return

        oids = list(dict.fromkeys([i for i in oids if isinstance(i, int)]))
        if not oids:
            self.vlm_add_error("accept_order failed: empty ids")
            self._finish_action(success=False)
            return

        # 2) 调 OM 批量接单（始终传列表，拿到 (accepted_ids, failed_ids)）
        accepted_ids, failed_ids = om.accept_order(oids)

        accepted_ids = list(accepted_ids or [])
        failed_ids   = list(failed_ids or [])

        # 3) 把成功的对象挂到 active_orders，并统一移出池子
        for oid in accepted_ids:
            o = om.get(int(oid))
            if o is not None and all(o is not x for x in self.active_orders):
                self.active_orders.append(o)

        if accepted_ids:
            om.remove_order(accepted_ids, self.city_map, self.world_nodes)

        # 4) 汇总日志 / 成功与否
        acc_txt = " ".join(f"#{i}" for i in accepted_ids) if accepted_ids else "none"
        if failed_ids:
            fail_txt = " ".join(f"#{i}" for i in failed_ids)
            msg = f"accept orders: accepted {acc_txt}; failed {fail_txt} (not found or already accepted by others)"
        else:
            msg = f"accept orders: accepted {acc_txt}"

        self._log(msg)

        if accepted_ids:
            self._finish_action(success=True)
        else:
            self.vlm_add_error(f"accept_order failed: {msg}")
            self._finish_action(success=False)


    def _handle_view_orders(self, _self, act: DMAction, _allow_interrupt: bool):
        om = act.data.get("order_manager") or self._order_manager
        if om and hasattr(om, "orders_text"):
            pool_text = om.orders_text()
            if pool_text: self.vlm_add_ephemeral("order_pool", pool_text); self._log("view orders")
        self._finish_action(success=True)

    
    def _handle_view_help_board(self, _self, act: DMAction, _allow_interrupt: bool):
        """拉取求助板文本，写入 ephemeral，并打日志。"""
        try:
            comms = get_comms()
            if comms is None:
                text = "(help board unavailable: comms not initialized)"
                # 这里视作成功，让上层继续流程；也可改为 self.vlm_add_error(...) + fail
                self.vlm_add_ephemeral("help_board", text)
                self._log("view help board (comms not ready)")
                self._finish_action(success=True)
                return

            # 可选：允许通过动作参数控制是否包含 active/completed
            include_active    = False
            include_completed = False
            max_items         = 50

            text = comms.board_to_text(
                include_active=include_active,
                include_completed=include_completed,
                max_items=max_items,
                exclude_publisher=str(self.agent_id),
            )

            # 塞进 ephemeral，供 VLM prompt 使用
            self.vlm_add_ephemeral("help_board", text)
            self._log("view help board")
            self._finish_action(success=True)

        except Exception as e:
            self.vlm_add_error(f"view_help_board failed: {e}")
            self._finish_action(success=False)

    def _handle_view_bag(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        显示保温袋当前排布和每个隔层温度。仅写入 ephemeral，交给下一轮 prompt 展示。
        """
        try:
            if not self.insulated_bag:
                self.vlm_add_ephemeral("bag_layout", "(no insulated bag)")
                self._log("view bag (no bag)")
                self._finish_action(success=True)
                return

            layout = self.insulated_bag.list_items()  # 你已有的文本化排布
            temps  = getattr(self.insulated_bag, "_comp_temp_c", None)
            if isinstance(temps, (list, tuple)):
                temps_txt = " | ".join([f"Comp {i}: {float(t):.1f}°C" for i, t in enumerate(temps)])
            else:
                temps_txt = "(no per-compartment temps)"
            text = f"{layout}\n\n[compartment temps] {temps_txt}"

            self.vlm_add_ephemeral("bag_layout", text)
            self._log("view bag")
            self._finish_action(success=True)
        except Exception as e:
            self.vlm_add_error(f"view_bag failed: {e}")
            self._finish_action(success=False)


    def _handle_pickup_food(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        两步制：
        第一步（本函数）：到取餐口执行 PICKUP，将订单的 items 记入 _pending_food_by_order，
        标记 has_picked_up / carrying，并强制下一步让模型输出 bag_cmd。
        第二步：模型输出 place_food_in_bag(bag_cmd=...)，由 _handle_place_food_in_bag 完成入袋。
        入参：
        data = {"orders": [order_obj,...], "tol_cm": 300.0}
        """
        self.vlm_clear_ephemeral()
        orders = list(act.data.get("orders") or [])
        tol = float(act.data.get("tol_cm", self._tol("nearby")))

        # 找到当前可取且在阈值内的订单
        here_orders = []
        for o in orders:
            if getattr(o, "has_picked_up", False):
                continue
            pu_xy = self._xy_of_node(getattr(o, "pickup_node", None))
            if pu_xy and self._is_at_xy(pu_xy[0], pu_xy[1], tol_cm=tol):
                # 基于活动时钟判断就绪
                is_ready = True if not hasattr(o, "is_ready_for_pickup") else o.is_ready_for_pickup(o.active_now())
                if is_ready:
                    here_orders.append(o)

        if not here_orders:
            self.vlm_add_error("Nothing to pick up here.")
            self._finish_action(success=False); return

        amb_pickup = self.ambient_temp_c
        k = self.k_food_per_s

        picked_ids: List[int] = []
        for o in here_orders:
            with o._lock:  # 可选：加锁更稳
                oid = int(getattr(o, "id", -1))
                items = list(getattr(o, "items", []) or [])

                now_active = o.active_now()
                ready_at   = o.ready_at()
                # 只计算活动时间差，并且夹成非负
                time_from_ready = max(0.0, float(getattr(o, "sim_elapsed_active_s", 0.0)) - float(getattr(o, "prep_longest_s", 0.0)))

                for it in items:
                    # 温度初始化：hasattr 不行，要判 NaN
                    tc = float(getattr(it, "temp_c", float("nan")))
                    if math.isnan(tc):
                        it.temp_c = float(getattr(it, "serving_temp_c", 25.0))

                    # 记录基于活动时间轴的时间戳
                    if hasattr(it, "prepared_at_sim"):
                        it.prepared_at_sim = float(ready_at)
                    if hasattr(it, "picked_at_sim"):
                        it.picked_at_sim = float(now_active)

                    # 衰减（注意 time_from_ready 已经 >=0）
                    it.temp_c = amb_pickup + (it.temp_c - amb_pickup) * math.exp(-k * time_from_ready)

                # 状态：已取到手
                o.has_picked_up = True
                if oid not in self.carrying:
                    self.carrying.append(oid)

                # 合并到待放队列
                cur = self._pending_food_by_order.get(oid, [])
                cur += items
                self._pending_food_by_order[oid] = cur
                picked_ids.append(oid)


        # 强制下一步让模型输出 bag_cmd
        self._force_place_food_now = True
        self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()

        self.vlm_ephemeral["XXX"] = "Picked up items (pending bag placement)."
        if picked_ids:
            self._log(f"picked (pending) order #{picked_ids[0]}" if len(picked_ids) == 1 else f"picked (pending) orders {picked_ids}")
        self._finish_action(success=True)

    def _handle_drop_off(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        手动投递一个订单；入参：
        - oid: int  订单号（必填）
        - method: str in {"leave_at_door","knock","call","hand_to_customer"}（必填）
        - tol_cm: float 位置容差（可选，默认 300）
        规则：
        - 必须在该订单 dropoff 点附近（tol 内）
        - 必须是"已取餐且未送达"的订单
        - 一次只处理一个订单
        - 对于自己的订单：物理卸载 -> 结算记录（保持你原先自动结算的效果）
        - 对于我作为 helper 的订单：仅物理卸载 + 向 Comms 推送 helper delivered
        """
        # 1) 参数
        try:
            oid = int(act.data.get("oid"))
        except Exception:
            self.vlm_add_error("drop_off failed: need integer 'oid'")
            self._finish_action(success=False); return

        method = str(act.data.get("method", "")).strip().lower()
        if method not in VALID_DELIVERY_METHODS:
            self.vlm_add_error("drop_off failed: invalid 'method' (use one of leave_at_door|knock|call|hand_to_customer)")
            self._finish_action(success=False); return

        tol = float(act.data.get("tol_cm", self._tol("nearby")))

        # 2) 判定是"自己的订单"还是"帮别人送的订单"
        order_obj = next((o for o in self.active_orders if int(getattr(o, "id", -1)) == oid), None)
        is_helper = False
        if order_obj is None:
            order_obj = self.help_orders.get(int(oid))
            is_helper = order_obj is not None

        if order_obj is None:
            self.vlm_add_error(f"drop_off failed: order #{oid} not found on this agent")
            self._finish_action(success=False); return

        # 3) 核对状态与位置
        if not getattr(order_obj, "has_picked_up", False):
            self.vlm_add_error("drop_off failed: order not picked up yet")
            self._finish_action(success=False); return

        if getattr(order_obj, "has_delivered", False):
            self.vlm_add_error("drop_off failed: order already delivered")
            self._finish_action(success=False); return

        # 检查是否在有效的投递位置
        allowed_methods = getattr(order_obj, "allowed_delivery_methods", [])
        is_handoff_allowed = 'hand_to_customer' in allowed_methods
        handoff_address = getattr(order_obj, "handoff_address", None)

        # 检查是否在 dropoff_node 位置
        dxy = self._xy_of_node(getattr(order_obj, "dropoff_node", None))
        at_dropoff = dxy and self._is_at_xy(dxy[0], dxy[1], tol_cm=tol)

        # 检查是否在 handoff_address 位置
        at_handoff = False
        if handoff_address:
            hx, hy = float(handoff_address.x), float(handoff_address.y)
            at_handoff = self._is_at_xy(hx, hy, tol_cm=tol)

        # 根据交付方式检查位置要求
        if method == 'hand_to_customer':
            # hand_to_customer 必须在 handoff_address 位置
            if not at_handoff:
                self.vlm_add_error("drop_off failed: hand_to_customer delivery must be near the customer. You are not close to the customer.")
                self._finish_action(success=False); return
        else:
            # 其他交付方式可以在 dropoff_node 位置
            if not at_dropoff:
                self.vlm_add_error("drop_off failed: not at the drop-off location")
                self._finish_action(success=False); return

        # 4) 记录交付方式（后续结算可用，但当前不改变 compute_settlement）
        try:
            setattr(order_obj, "delivery_method", method)
        except Exception:
            pass

        # 5) 物理卸载（移出保温袋/待放/携带列表）
        self._dropoff_physical_unload(order_obj)

        # 6) 分流：自己的单 -> 直接结算；帮别人 -> 推消息，等待对方结算
        if not is_helper:
            # 自己订单：保持你原本"自动 dropoff 时"的结算效果
            self._dropoff_settle_record(order_obj)
            self._finish_action(success=True)

            if is_handoff_allowed and handoff_address:
                self._ue.destroy_customer(order_obj.id)
            return

        # helper：推送"我已送达"，移出本地 help_orders，进入等待 ACK 集
        comms = get_comms()
        if not comms:
            self.vlm_add_error("drop_off failed: comms unavailable for helper delivery")
            self._finish_action(success=False); return

        req_id = int(self._help_delivery_req_by_oid.get(int(oid), 0))
        if req_id <= 0:
            self.vlm_add_error("drop_off failed: no req_id bound for this helper delivery")
            self._finish_action(success=False); return

        ok, msg = comms.push_helper_delivered(
            req_id=req_id,
            by_agent=str(self.agent_id),
            order_id=int(oid),
            at_xy=(self.x, self.y),
        )
        if not ok:
            self.vlm_add_error(f"drop_off failed: {msg}")
            self._finish_action(success=False); return

        # 本地清理 helper 态
        self.help_orders.pop(int(oid), None)
        self.help_orders_completed.add(int(oid))
        self._helping_wait_ack_oids.add(int(oid))
        self._log(f"helper delivered order #{oid} with method '{method}', pushed to Comm (req #{req_id})")

        self._finish_action(success=True)

        if is_handoff_allowed and handoff_address:
            self._ue.destroy_customer(order_obj.id)


    # ===== Charging (ESCOOTER) =====
    def _advance_charge_to_now(self):
        if self._charge_ctx and self._charge_ctx.get("scooter_ref"):
            sc = self._charge_ctx["scooter_ref"]
            cur = float(self._charge_ctx.get("paid_pct", getattr(sc, "battery_pct", 0.0)))
            sc.charge_to(cur)


    def _handle_charge_escooter(self, _self, act: DMAction, _allow_interrupt: bool):
        if self._charge_ctx is not None:
            self.vlm_add_error("charge failed: already charging"); self._finish_action(success=False); return

        station_xy = self._nearest_poi_xy("charging_station", tol_cm=self._tol("nearby"))
        if station_xy is None:
            self.vlm_add_error("charge failed: not near a charging station"); self._finish_action(success=False); return

        tol = float(act.data.get("tol_cm", self._tol("nearby")))

        def _with_me(s):
            if not s: return False
            if self.mode == TransportMode.SCOOTER:
                return s is self.e_scooter
            if self.mode == TransportMode.DRAG_SCOOTER:
                return (self.assist_scooter is not None and s is self.assist_scooter) or \
                    (self.assist_scooter is None and s is self.e_scooter)
            return False

        def _parked_nearby(s):
            return bool(s and s.park_xy and self._is_at_xy(s.park_xy[0], s.park_xy[1], tol_cm=tol))

        # 只在"在身边或停在附近"中选；优先他人车
        candidates = []
        if self.assist_scooter:
            candidates.append(("assist", self.assist_scooter))
        if self.e_scooter and getattr(self.e_scooter, "with_owner", True):
            candidates.append(("own", self.e_scooter))

        sc, which, with_me = None, None, False
        for kind, s in candidates:
            if _with_me(s) or _parked_nearby(s):
                sc, which, with_me = s, kind, _with_me(s)
                break

        if not sc:
            self.vlm_add_error("charge failed: scooter not with you and not parked nearby")
            self._finish_action(success=False); return

        # 只有真在身边（骑/拖）时才就地 park；停在附近则保持原 park_xy，不许"瞬移"
        if with_me:
            sc.park_here(self.x, self.y)
            if self.mode in (TransportMode.SCOOTER, TransportMode.DRAG_SCOOTER):
                self.set_mode(TransportMode.WALK)
        else:
            px, py = sc.park_xy
            if not self._is_at_xy(px, py, tol_cm=tol):
                self.vlm_add_error("charge failed: not at parked scooter location")
                self._finish_action(success=False); return

        comms = get_comms()
        station_key = None
        if comms and hasattr(comms, "reserve_charging_spot"):
            ok, msg, station_key = comms.reserve_charging_spot(station_xy, str(self.agent_id))
            if not ok:
                self.vlm_add_error(f"charge failed: {msg}")
                self._finish_action(success=False); return

        target_pct = float(act.data.get("target_pct", self.cfg.get("defaults", {}).get("charge_target_pct", 100.0))); target_pct = max(0.0, min(100.0, target_pct))
        before = float(sc.battery_pct)
        if target_pct <= before + 1e-6:
            if station_key is not None and comms:
                comms.release_charging_spot(station_key, agent_id=str(self.agent_id))
            self._finish_action(success=True); return
        rate_m = float(sc.charge_rate_pct_per_min)
        if rate_m <= 0.0:
            if station_key is not None and comms and hasattr(comms, "release_charging_spot"):
                comms.release_charging_spot(station_key, agent_id=str(self.agent_id))
            self.vlm_add_error("charge failed: invalid rate"); self._finish_action(success=False); return

        duration_sim_s = (target_pct - before) / rate_m * 60.0
        now_sim = self.clock.now_sim()
        self._charge_ctx = dict(
            start_sim=now_sim,
            end_sim=now_sim + duration_sim_s,
            start_pct=before,
            target_pct=target_pct,
            paid_pct=before,
            price_per_pct=float(self.charge_price_per_pct),
            scooter_ref=sc,
            which=("assist" if which == "assist" else "own"),
            park_xy_start = tuple(sc.park_xy) if sc.park_xy else None,
            station_key = station_key,
            station_xy  = tuple(station_xy),
        )
        self._log(f"start charging scooter ({'assist' if which=='assist' else 'own'}): {before:.0f}% -> {target_pct:.0f}% (~{duration_sim_s/60.0:.1f} min @virtual)")
        self._finish_action(success=True)



    # ===== WAIT / REST =====
    def _handle_wait(self, _self, act: DMAction, _allow_interrupt: bool):
        # 等到充电完成
        if act.data.get("until") == "charge_done" and self._charge_ctx:
            now_sim = self.clock.now_sim()
            end_sim = float(self._charge_ctx["end_sim"])
            if end_sim <= now_sim:
                self._log("wait skipped: charge already done")
                self._finish_action(success=True)
                return
            remain_s = max(0.0, end_sim - now_sim)
            self._log(f"start waiting until charge done: ~{remain_s/60.0:.1f} min @virtual")
            self._wait_ctx = dict(start_sim=now_sim, end_sim=end_sim)
            return

        # 固定时长等待
        duration_s = float(act.data.get("duration_s", 0.0))
        if duration_s <= 0.0:
            self._log("wait skipped: duration <= 0s")
            self._finish_action(success=True)
            return

        now_sim = self.clock.now_sim()
        self._wait_ctx = dict(start_sim=now_sim, end_sim=now_sim + float(duration_s))
        self._log(f"start waiting: {duration_s:.1f}s (~{duration_s/60.0:.1f} min) @virtual")

    def _handle_rest(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if self.is_rescued:
            self.vlm_add_error("rest failed: in hospital rescue"); self._finish_action(success=False); return
        if self._nearest_poi_xy("rest_area", tol_cm=self._tol("nearby")) is None:
            self.vlm_add_error("rest failed: not near a rest_area"); self._finish_action(success=False); return

        target = float(act.data.get("target_pct", self.cfg.get("defaults", {}).get("rest_target_pct", 100.0)))

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
        """
        支持两种用法：
        1) 单品：BUY(item_id="energy_drink", qty=2)  或  BUY(name="energy_drink", qty=2)
        2) 多种：BUY(items=[{"item_id":"energy_drink","qty":2}, {"name":"escooter_battery_pack","qty":1}])

        说明：
        - 如果同时传了单品参数和 items，会合并到同一批购买里；
        - 数量 <=0 的条目会被忽略；
        - 只要有一种买成，就算 success=True；失败项会写日志。
        """
        self.vlm_clear_ephemeral()

        # 位置与依赖检查
        if self._nearest_poi_xy("store", tol_cm=self._tol("nearby")) is None:
            self.vlm_add_error("buy failed: not in a store")
            self._finish_action(success=False)
            return
        if not self._store_manager:
            self.vlm_add_error("buy failed: no store manager")
            self._finish_action(success=False)
            return

        # 归并待购清单：item_id -> qty
        purchases: Dict[str, int] = {}

        def _merge(iid: Optional[str], qty: Any):
            if iid is None:
                return
            sid = str(iid).strip()
            try:
                q = int(qty)
            except Exception:
                q = 0
            if sid and q > 0:
                purchases[sid] = purchases.get(sid, 0) + q

        # 2) 多种：严格只接受 list/tuple 且项为 dict，键为 item_id/name + qty
        if "items" in act.data:
            raw_items = act.data.get("items")
            if not isinstance(raw_items, (list, tuple)):
                self.vlm_add_error("buy failed: 'items' must be a list of dicts like {'item_id':..., 'qty':...}")
                self._finish_action(success=False); return
            for entry in raw_items:
                if not isinstance(entry, dict):
                    self.vlm_add_error("buy failed: each element in 'items' must be a dict with item_id/name and qty")
                    self._finish_action(success=False); return
                _merge(entry.get("item_id") or entry.get("name"), entry.get("qty", 1))

        # 1) 单品：item_id/name + qty
        if "item_id" in act.data or "name" in act.data:
            _merge(act.data.get("item_id") or act.data.get("name"), act.data.get("qty", 1))

        if not purchases:
            self.vlm_add_error("buy failed: provide either item_id+qty or items=[{item_id/name, qty}, ...]")
            self._finish_action(success=False)
            return

        # 逐项购买（不改 StoreManager 接口）
        total_cost = 0.0
        bought_lines: List[str] = []
        failed_lines: List[str] = []
        for iid, q in purchases.items():
            ok, msg, cost = self._store_manager.purchase(self, item_id=iid, qty=int(q))
            if ok:
                total_cost += float(cost or 0.0)
                bought_lines.append(f"{q} x {iid}")
            else:
                failed_lines.append(f"{iid} ({msg})")

        # 结果处理
        if bought_lines:
            self._log(f"bought {', '.join(bought_lines)} for ${total_cost:.2f}")
            if failed_lines:
                self._log("buy partial fails: " + "; ".join(failed_lines))
            if self._recorder:
                self._recorder.on_purchase(self.clock.now_sim(), items=", ".join(bought_lines), cost=float(total_cost))
            self._finish_action(success=True)
        else:
            self.vlm_add_error("buy failed: " + ("; ".join(failed_lines) if failed_lines else "unknown reason"))
            self._finish_action(success=False)



    def _handle_use_battery_pack(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()

        # 0) 基础校验：有车、有电池包库存
        if not self.e_scooter:
            self.vlm_add_error("use_battery_pack failed: no e-scooter")
            self._finish_action(success=False)
            return

        if not getattr(self.e_scooter, "with_owner", True):
            self.vlm_add_error("use_battery_pack failed: your e-scooter is currently handed off (in a TempBox). Retrieve it first.")
            self._finish_action(success=False)
            return

        item_id = act.data.get("item_id", ITEM_ESC_BATTERY_PACK)
        cnt = int(self.inventory.get(item_id, 0))
        if cnt <= 0:
            self.vlm_add_error(f"use_battery_pack failed: inventory=0 ({item_id})")
            self._finish_action(success=False)
            return

        # 1) 位置/持有性校验：必须在身边或停在附近
        tol = float(act.data.get("tol_cm", self._tol("nearby")))  # 可选：允许外部传入容差
        own_scooter_in_hand = (
            getattr(self.e_scooter, "with_owner", True) and
            (self.mode == TransportMode.SCOOTER or (self.mode == TransportMode.DRAG_SCOOTER and self.assist_scooter is None))
        )
        scooter_is_with_me = own_scooter_in_hand
        scooter_is_parked_nearby = False

        if getattr(self.e_scooter, "with_owner", True) and self.e_scooter.park_xy:
            px, py = self.e_scooter.park_xy
            scooter_is_parked_nearby = self._is_at_xy(float(px), float(py), tol_cm=tol)

        if not (scooter_is_with_me or scooter_is_parked_nearby):
            if self.e_scooter.park_xy:
                px, py = self.e_scooter.park_xy
                self.vlm_add_error(
                    f"use_battery_pack failed: not near your e-scooter (parked at {self._fmt_xy_m(px, py)}). MOVE there first."
                )
            else:
                self.vlm_add_error("use_battery_pack failed: scooter location unknown")
            # 友情提示：引导去车边再用
            self.vlm_ephemeral["charging_hint"] = (
                "Go to your parked scooter (MOVE to its coordinates) before using a battery pack."
            )
            self._finish_action(success=False)
            return

        # 2) 消耗并充满
        self.inventory[item_id] = cnt - 1
        target = float(self.cfg.get("items", {}).get(ITEM_ESC_BATTERY_PACK, {}).get("target_charge_pct", 100))
        self.e_scooter.charge_to(target)

        # 3) 状态与模式处理：
        # - 正在拖车：立刻切换为骑行（已满电）
        # - 正在骑行：保持骑行
        # - 车停在旁边：保持"停放"状态，不自动上车
        if self.mode == TransportMode.DRAG_SCOOTER:
            self.e_scooter.state = ScooterState.USABLE
            self.set_mode(TransportMode.SCOOTER)
        elif self.mode == TransportMode.SCOOTER:
            self.e_scooter.state = ScooterState.USABLE
        else:
            # 这里表示车是"停在附近"的场景，充完仍保持 PARKED，更贴近现实
            self.e_scooter.state = ScooterState.PARKED

        self._log(f"used '{item_id}': scooter battery -> 100% (remaining {self.inventory.get(item_id, 0)})")
        self._finish_action(success=True)


    def _handle_use_energy_drink(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if self._hospital_ctx is not None:
            self.vlm_add_error("use_energy_drink failed: in hospital rescue"); self._finish_action(success=False); return
        if float(self.energy_pct) >= 100.0 - 1e-6:
            self.vlm_add_error("use_energy_drink failed: Your energy is full.")
            self._finish_action(success=False)
            return

        item_id = act.data.get("item_id", ITEM_ENERGY_DRINK)
        cnt = int(self.inventory.get(item_id, 0))
        if cnt <= 0:
            self.vlm_add_error(f"use_energy_drink failed: inventory=0 ({item_id})"); self._finish_action(success=False); return
        self.inventory[item_id] = cnt - 1
        gain = float(self.cfg.get("items", {}).get(ITEM_ENERGY_DRINK, {}).get("energy_gain_pct", 50))
        before = float(self.energy_pct)
        self.energy_pct = float(min(self.cfg.get("energy_pct_max", 100), before + gain))
        self._log(f"used '{item_id}': energy {before:.0f}% -> {self.energy_pct:.0f}% (remaining {self.inventory[item_id]})")
        self._finish_action(success=True)

    def _handle_use_ice_pack(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if not self.insulated_bag:
            self.vlm_add_error("use_ice_pack failed: no insulated bag")
            self._finish_action(success=False); return

        cnt = int(self.inventory.get(ITEM_ICE_PACK, 0))
        if cnt <= 0:
            self.vlm_add_error("use_ice_pack failed: inventory=0 (ice_pack)")
            self._finish_action(success=False); return

        # 仅支持 A/B/C... 的字母（你前面已经希望简化为这样）
        lab = str(act.data.get("comp") or "").strip().upper() or "A"
        try:
            # 让 InsulatedBag 校验并加入
            self.insulated_bag.add_misc_item(lab, IcePack())
        except Exception as e:
            self.vlm_add_error(f"use_ice_pack failed: {e}")
            self._finish_action(success=False); return

        self.inventory[ITEM_ICE_PACK] = cnt - 1
        self._log(f"inserted 'ice_pack' into compartment {lab} (remaining {self.inventory.get(ITEM_ICE_PACK,0)})")
        self._finish_action(success=True)

    def _handle_use_heat_pack(self, _self, act: DMAction, _allow_interrupt: bool):
        self.vlm_clear_ephemeral()
        if not self.insulated_bag:
            self.vlm_add_error("use_heat_pack failed: no insulated bag")
            self._finish_action(success=False); return

        cnt = int(self.inventory.get(ITEM_HEAT_PACK, 0))
        if cnt <= 0:
            self.vlm_add_error("use_heat_pack failed: inventory=0 (heat_pack)")
            self._finish_action(success=False); return

        lab = str(act.data.get("comp") or "").strip().upper() or "A"
        try:
            self.insulated_bag.add_misc_item(lab, HeatPack())
        except Exception as e:
            self.vlm_add_error(f"use_heat_pack failed: {e}")
            self._finish_action(success=False); return

        self.inventory[ITEM_HEAT_PACK] = cnt - 1
        self._log(f"inserted 'heat_pack' into compartment {lab} (remaining {self.inventory.get(ITEM_HEAT_PACK,0)})")
        self._finish_action(success=True)


    # ===== NEW: Comms =====
    def _handle_post_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        约束：
        - HELP_DELIVERY：必须有 order_id，且必须给出 provide_xy；不允许传 deliver_xy
        - HELP_PICKUP  ：必须有 order_id，且必须给出 deliver_xy；忽略/不传 provide_xy
        - HELP_BUY     ：必须有 buy_list，且必须给出 deliver_xy
        - HELP_CHARGE  ：必须给出 provide_xy 与 deliver_xy；target_pct 可选（默认 100）
        禁止任何坐标"缺省兜底"。缺字段直接报错并失败。
        """
        self.vlm_clear_ephemeral()
        comms = get_comms()
        if not comms:
            self.vlm_add_error("post_help_request failed: no comms")
            self._finish_action(success=False)
            return

        help_type = act.data.get("help_type")
        if isinstance(help_type, str):
            help_type = HelpType[help_type]
        bounty = float(act.data.get("bounty", 0.0))
        ttl_s  = float(act.data.get("ttl_s", 0.0))
        payload = dict(act.data.get("payload") or {})

        def _as_xy(xy):
            if not xy or len(xy) != 2: return None
            x, y = xy
            return (float(x), float(y))

        kwargs = dict(
            publisher_id=str(self.agent_id),
            kind=help_type,
            reward=bounty,
            time_limit_s=ttl_s,
        )


        if help_type == HelpType.HELP_DELIVERY:
            if "order_id" not in payload:
                self.vlm_add_error("post_help_request failed: HELP_DELIVERY needs payload.order_id")
                self._finish_action(success=False); return
            if "provide_xy" not in payload or _as_xy(payload.get("provide_xy")) is None:
                self.vlm_add_error("post_help_request failed: HELP_DELIVERY needs payload.provide_xy")
                self._finish_action(success=False); return

            oid = int(payload["order_id"])
            order_obj = next((o for o in self.active_orders
                            if int(getattr(o, "id", -1)) == oid), None)
            if order_obj is None:
                self.vlm_add_error("post_help_request failed: order_ref not found on publisher")
                self._finish_action(success=False); return

            kwargs["order_id"]   = oid
            kwargs["provide_xy"] = _as_xy(payload["provide_xy"])
            kwargs["order_ref"]  = order_obj  # ← 把实例传进 Comms

        
        elif help_type == HelpType.HELP_PICKUP:
            # 必填：order_id + deliver_xy；忽略 provide_xy（不传给 comms）
            if "order_id" not in payload:
                self.vlm_add_error("post_help_request failed: HELP_PICKUP needs payload.order_id")
                self._finish_action(success=False); return
            if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
                self.vlm_add_error("post_help_request failed: HELP_PICKUP needs payload.deliver_xy")
                self._finish_action(success=False); return

            oid = int(payload["order_id"])
            order_obj = next((o for o in self.active_orders
                              if int(getattr(o, "id", -1)) == oid), None)
            if order_obj is None:
                self.vlm_add_error("post_help_request failed: order_ref not found on publisher")
                self._finish_action(success=False); return
            if getattr(order_obj, "has_picked_up", False):
                self.vlm_add_error("post_help_request failed: order already picked up; use HELP_DELIVERY instead")
                self._finish_action(success=False); return

            kwargs["order_id"]   = oid
            kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])
            kwargs["order_ref"]  = order_obj  # 传进 Comms；不传 provide_xy


        elif help_type == HelpType.HELP_BUY:
            # 必填：buy_list + deliver_xy
            raw = list(payload.get("buy_list") or [])
            buy_items = {}
            for item_id, qty in raw:
                q = int(qty)
                if q > 0:
                    buy_items[str(item_id)] = q
            if not buy_items:
                self.vlm_add_error("post_help_request failed: HELP_BUY needs non-empty payload.buy_list")
                self._finish_action(success=False); return
            if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
                self.vlm_add_error("post_help_request failed: HELP_BUY needs payload.deliver_xy")
                self._finish_action(success=False); return
            kwargs["buy_items"]  = buy_items
            kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])

        elif help_type == HelpType.HELP_CHARGE:
            # 必填：provide_xy + deliver_xy；target_pct 可选
            if "provide_xy" not in payload or _as_xy(payload.get("provide_xy")) is None:
                self.vlm_add_error("post_help_request failed: HELP_CHARGE needs payload.provide_xy")
                self._finish_action(success=False); return
            if "deliver_xy" not in payload or _as_xy(payload.get("deliver_xy")) is None:
                self.vlm_add_error("post_help_request failed: HELP_CHARGE needs payload.deliver_xy")
                self._finish_action(success=False); return
            target = float(payload.get("want_charge_pct", payload.get("target_pct", 100.0)))
            kwargs["target_pct"] = max(0.0, min(100.0, target))
            kwargs["provide_xy"] = _as_xy(payload["provide_xy"])
            kwargs["deliver_xy"] = _as_xy(payload["deliver_xy"])

        else:
            self.vlm_add_error(f"post_help_request failed: unsupported help_type={help_type}")
            self._finish_action(success=False); return

        ok, msg, rid = comms.post_request(**kwargs)
        if not ok:
            self.vlm_add_error(f"post_help_request failed: {msg}")
            self._finish_action(success=False); return

        def _fmt(xy): 
            return f"({xy[0]/100.0:.2f}m,{xy[1]/100.0:.2f}m)" if xy else "N/A"
        self._log(
            f"posted help request #{rid} ({help_type.name}) "
            f"bounty=${bounty:.2f} provide={_fmt(kwargs.get('provide_xy'))}"
        )
        if self._recorder:
            self._recorder.inc("help_posted", 1)
        self._finish_action(success=True)


    def _attach_helper_order(self, order_obj: Any) -> None:
        """把别人的订单引用挂到 helper 专属容器，不触碰 is_accepted、不动订单池。"""
        oid = int(getattr(order_obj, "id", -1))
        if oid <= 0:
            return
        self.help_orders[oid] = order_obj
        self.helping_order_ids.add(oid)  # 兼容原有标记
        # 仅用于本地显示的起始时间（不参与计费、也不写回 pool）
        if getattr(order_obj, "sim_started_s", None) is None:
            order_obj.sim_started_s = float(self.clock.now_sim())
            order_obj.sim_elapsed_active_s = 0.0
        self._log(f"attached helper order #{oid} (kept outside active_orders)")

    
    def _handle_accept_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        if not comms:
            self.vlm_add_error("accept_help_request failed: no comms")
            self._finish_action(success=False)
            return

        req_id = int(act.data.get("req_id"))
        ok, msg = comms.accept_request(req_id=req_id, helper_id=self.agent_id)
        if not ok:
            self.vlm_add_error(f"accept_help_request failed: {msg}")
            self._finish_action(success=False)
            return

        self._log(f"accepted help request #{req_id}")

        # 记录 HELP_DELIVERY / HELP_PICKUP 的 order_id <-> req_id，并把订单挂到 help_orders
        det = get_comms().get_request_detail(req_id=req_id) or {}
        kind = det.get("kind")
        kind_str = str(kind)

        if det.get("order_id") is not None and kind_str in (
            HelpType.HELP_DELIVERY.value, HelpType.HELP_DELIVERY.name,
            HelpType.HELP_PICKUP.value,  HelpType.HELP_PICKUP.name
        ):
            oid = int(det["order_id"])
            self.helping_order_ids.add(oid)
            # 统一用这张映射，后续 DROP_OFF（helper）会从这里取到 req_id
            self._help_delivery_req_by_oid[oid] = int(det.get("id", req_id))

            order_obj = det.get("order_ref")
            if order_obj is not None:
                self._attach_helper_order(order_obj)

        # 记录到 accepted_help，便于展示
        req_obj = get_comms().get_request(req_id)
        if req_obj is not None:
            self.accepted_help[int(req_id)] = req_obj

        if self._recorder:
            self._recorder.inc("help_accepted", 1)
        self._finish_action(success=True)


    def _handle_edit_help_request(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        req_id = int(act.data.get("req_id"))
        new_bounty = act.data.get("new_bounty", None)
        new_ttl_s  = act.data.get("new_ttl_s", None)
        ok, msg = comms.modify_request(publisher_id=str(self.agent_id), req_id=req_id, reward=new_bounty, time_limit_s=new_ttl_s)
        if not ok:
            self.vlm_add_error(f"edit_help_request failed: {msg}"); self._finish_action(success=False); return
        self._log(f"edited help request #{req_id}")
        self._finish_action(success=True)

    # ===== NEW: Transport switching / car rental =====
    def _handle_switch_transport(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        支持以下切换：
        - to="walk"
        - to="e-scooter" / "scooter"：若存在 assist_scooter，则不报错，强制降级为 DRAG_SCOOTER
        - to="car"
        - to="drag_scooter" / "drag"：显式切到拖拽（优先拖 assist 车；否则拖自车）
        规则要点：
        - 切到"拖拽"时，不会把电瓶车先 park_here（避免状态抖动）；
        - 切到其它模式时：
            * 汽车：必须先 park_here（若当前在开车）；
            * 自己的电瓶车：只有在"非拖拽切换"时才自动 park_here。
        """
        to = str(act.data.get("to", "")).strip().lower()
        tol = self._tol("nearby")
        want_drag = to in ("drag_scooter", "drag")

        # --- 收尾当前载具 ---
        # 车：无论切到什么都先停好
        if self.mode == TransportMode.CAR and self.car:
            self.car.park_here(self.x, self.y)
        # 自己的电瓶车：仅当不是"切到拖拽"时，才自动在脚下 park
        if not want_drag and self.e_scooter and getattr(self.e_scooter, "with_owner", True):
            if self.mode == TransportMode.SCOOTER or (self.mode == TransportMode.DRAG_SCOOTER and self.assist_scooter is None):
                self.e_scooter.park_here(self.x, self.y)

        # --- 显式切到拖拽（优先拖 assist 车） ---
        if want_drag:
            # (A) 拖助援车
            if self.assist_scooter is not None:
                sc = self.assist_scooter
                if sc.park_xy:
                    px, py = sc.park_xy
                    if not self._is_at_xy(px, py, tol_cm=tol):
                        self.vlm_add_error("switch failed: not near the assisting scooter")
                        self._finish_action(success=False); return
                    try:
                        sc.unpark()
                    except Exception:
                        sc.park_xy = None
                self.set_mode(TransportMode.DRAG_SCOOTER)
                self._log("switch -> DRAG_SCOOTER (assisting)")
                self._finish_action(success=True); return

            # (B) 拖自车
            if not self.e_scooter:
                self.vlm_add_error("switch failed: no scooter to drag")
                self._finish_action(success=False); return
            if not getattr(self.e_scooter, "with_owner", True):
                self.vlm_add_error("switch failed: your e-scooter is currently handed off (TempBox). Retrieve it first.")
                self._finish_action(success=False); return
            if self.e_scooter.park_xy:
                px, py = self.e_scooter.park_xy
                if not self._is_at_xy(px, py, tol_cm=tol):
                    self.vlm_add_error("switch failed: not near your scooter")
                    self._finish_action(success=False); return
                try:
                    self.e_scooter.unpark()
                except Exception:
                    self.e_scooter.park_xy = None
            self.set_mode(TransportMode.DRAG_SCOOTER)
            self._log("switch -> DRAG_SCOOTER")
            self._finish_action(success=True); return

        # --- 走路 ---
        if to in ("walk", TransportMode.WALK.value):
            self.set_mode(TransportMode.WALK)
            self._log("switch -> WALK")
            self._finish_action(success=True); return

        # --- 电瓶车（骑行）。若有 assist 车，允许调用但强制降级为拖拽 ---
        if to in ("e-scooter", "scooter", TransportMode.SCOOTER.value):
            if self.assist_scooter is not None:
                sc = self.assist_scooter
                if sc.park_xy:
                    px, py = sc.park_xy
                    if not self._is_at_xy(px, py, tol_cm=tol):
                        self.vlm_add_error("switch failed: not near the assisting scooter")
                        self._finish_action(success=False); return
                    try:
                        sc.unpark()
                    except Exception:
                        sc.park_xy = None
                self.set_mode(TransportMode.DRAG_SCOOTER)
                self._log("switch -> DRAG_SCOOTER (assist scooter cannot be ridden)")
                self._finish_action(success=True); return

            if not self.e_scooter:
                self.vlm_add_error("switch failed: no scooter")
                self._finish_action(success=False); return
            if self.e_scooter.park_xy:
                px, py = self.e_scooter.park_xy
                if not self._is_at_xy(px, py, tol_cm=tol):
                    self.vlm_add_error("switch failed: not near your scooter")
                    self._finish_action(success=False); return
                try:
                    self.e_scooter.unpark()
                except Exception:
                    self.e_scooter.park_xy = None

            if self.e_scooter.state == ScooterState.DEPLETED:
                self.set_mode(TransportMode.DRAG_SCOOTER)
                self._log("switch -> DRAG_SCOOTER (battery depleted)")
            else:
                self.set_mode(TransportMode.SCOOTER)
                self._log("switch -> SCOOTER")
            self._finish_action(success=True); return

        # --- 汽车 ---
        if to in ("car", TransportMode.CAR.value):
            if not self.car:
                self.vlm_add_error("switch failed: no rented car")
                self._finish_action(success=False); return
            if self.car.park_xy:
                px, py = self.car.park_xy
                if not self._is_at_xy(px, py, tol_cm=tol):
                    self.vlm_add_error("switch failed: not near your car")
                    self._finish_action(success=False); return
                self.car.unpark()
            self.set_mode(TransportMode.CAR)
            self._log("switch -> CAR")
            self._finish_action(success=True); return

        # --- 其它 ---
        self.vlm_add_error("switch failed: invalid target")
        self._finish_action(success=False)


    def _handle_rent_car(self, _self, act: DMAction, _allow_interrupt: bool):
        tol_xy = self._nearest_poi_xy("car_rental", tol_cm=self._tol("nearby"))
        if tol_xy is None:
            self.vlm_add_error("rent_car failed: not at car_rental"); self._finish_action(success=False); return
        if self.car is not None:
            self.vlm_add_error("rent_car failed: already have a car"); self._finish_action(success=False); return

        if self.e_scooter and (
            self.mode == TransportMode.SCOOTER or
            (self.mode == TransportMode.DRAG_SCOOTER and self.assist_scooter is None)
        ):
            self.e_scooter.park_here(self.x, self.y)

        defs = self.cfg.get("rent_car_defaults", {})
        rate = float(act.data.get("rate_per_min", defs.get("rate_per_min", 1.0)))
        speed = float(act.data.get("avg_speed_cm_s", defs.get("avg_speed_cm_s", 1200)))
        self.car = Car(owner_id=str(self.agent_id), avg_speed_cm_s=speed, rate_per_min=rate, state=CarState.USABLE, park_xy=None)
        self.set_mode(TransportMode.CAR)
        self._rental_ctx = {"last_tick_sim": self.clock.now_sim(), "rate_per_min": float(self.car.rate_per_min)}
        self._log(f"rent car @ ${self.car.rate_per_min:.2f}/min")
        self._finish_action(success=True)

    def _handle_return_car(self, _self, act: DMAction, _allow_interrupt: bool):
        tol_xy = self._nearest_poi_xy("car_rental", tol_cm=self._tol("nearby"))
        if tol_xy is None:
            self.vlm_add_error("return_car failed: not at car_rental"); self._finish_action(success=False); return
        if not self.car:
            self.vlm_add_error("return_car failed: no car"); self._finish_action(success=False); return

        if self.mode == TransportMode.CAR: self.set_mode(TransportMode.WALK)
        self._log("return car: stop billing")
        self.car = None
        self._rental_ctx = None
        self._finish_action(success=True)

    def _handle_place_temp_box(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        将内容放到临时盒。输入格式（均可选）：
        data = {
            "req_id": 123,
            "location_xy": (x, y)  # 缺省用当前位置
            "content": {
                "inventory": {"item_id": qty, ...},  # 从本地库存扣减
                "food": "any value or empty key",     # 只要出现该键，就表示要放**全部食物**
                "escooter": ""                        # 只要出现该键，就表示交接整车
            }
        }
        成功后才扣本地；失败不改本地。
        """
        comms = get_comms()
        if not comms:
            self.vlm_add_error("place_temp_box failed: no comms"); self._finish_action(success=False); return

        req_id = int(act.data.get("req_id"))
        location_xy = tuple(act.data.get("location_xy") or (self.x, self.y))
        content_req = dict(act.data.get("content") or {})

        payload: Dict[str, Any] = {}

        # 1) 校验 & 组装 inventory
        inv_req = {str(k): int(v) for k, v in (content_req.get("inventory") or {}).items()}
        for k, q in inv_req.items():
            if int(self.inventory.get(k, 0)) < int(q):
                self.vlm_add_error(f"place_temp_box failed: lacking '{k}' x{int(q)}")
                self._finish_action(success=False); return
        if inv_req:
            payload["inventory"] = dict(inv_req)

        # 2) 组装食物：出现键 "food" 即放**全部**尚未送达且已取餐的订单里的全部条目
        want_food = ("food" in content_req)
        food_by_order: Dict[int, List[Any]] = {}
        if want_food:
            for o in list(self.active_orders or []):
                if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                    items = list(getattr(o, "items", []) or [])
                    if items:
                        food_by_order[int(getattr(o, "id", -1))] = items
            if not food_by_order:
                self.vlm_add_error("place_temp_box failed: no food to place")
                self._finish_action(success=False); return
            payload["food_by_order"] = {int(k): list(v) for k, v in food_by_order.items()}

        # 3) 交接 e-scooter：出现键 "escooter" 即表示放车
        give_scooter = ("escooter" in content_req)
        if give_scooter:
            scooter_to_place = self.assist_scooter if self.assist_scooter is not None else self.e_scooter
            if scooter_to_place is None:
                self.vlm_add_error("place_temp_box failed: no e-scooter to place")
                self._finish_action(success=False); return

            is_my_scooter = (getattr(scooter_to_place, "owner_id", None) == str(self.agent_id))

            # NEW: 二次放车拦截（我的车已交接过）
            if is_my_scooter and not getattr(self.e_scooter, "with_owner", True):
                self.vlm_add_error("place_temp_box failed: your e-scooter has already been handed off")
                self._finish_action(success=False); return

            if scooter_to_place.park_xy:
                px, py = scooter_to_place.park_xy
                if not self._is_at_xy(px, py, tol_cm=self._tol("nearby")):
                    self.vlm_add_error("place_temp_box failed: not near the e-scooter to place")
                    self._finish_action(success=False); return
                scooter_to_place.unpark()

            lx, ly = location_xy
            scooter_to_place.park_here(float(lx), float(ly))
            payload["escooter"] = scooter_to_place

            # NEW: 如果放的是"我的车"，标记交接态（不断引用）
            if is_my_scooter:
                setattr(self.e_scooter, "with_owner", False)

        if not (payload.get("inventory") or payload.get("food_by_order") or payload.get("escooter")):
            self.vlm_add_error("place_temp_box failed: empty content")
            self._finish_action(success=False); return

        # 4) 调 Comms
        _inv_before = dict(self.inventory)
        _had_scooter_before = self._has_scooter()
        ok, msg = comms.place_temp_box(req_id=req_id, by_agent=str(self.agent_id),
                                    location_xy=location_xy, content=payload)
        if not ok:
            self.vlm_add_error(f"place_temp_box failed: {msg}")
            self._finish_action(success=False); return

        # 5) 成功 -> 本地扣减
        for k, q in inv_req.items():
            self.inventory[k] = int(self.inventory.get(k, 0)) - int(q)
            if self.inventory[k] <= 0: self.inventory.pop(k, None)

        if want_food:
            # 从保温袋移除（若之前已经放袋）+ 从"待放队列"清理
            if self.insulated_bag:
                all_items = []
                for items in food_by_order.values():
                    all_items.extend(items)
                if all_items:
                    self.insulated_bag.remove_items(all_items)
            for oid in list(food_by_order.keys()):
                if oid in self._pending_food_by_order:
                    self._pending_food_by_order.pop(oid, None)
                if oid in self.carrying:
                    self.carrying.remove(oid)

        if give_scooter:
            was_using = (self.mode in (TransportMode.SCOOTER, TransportMode.DRAG_SCOOTER))
            # 放的是谁就清谁
            if self.assist_scooter is not None:
                self.assist_scooter = None

            if was_using:
                self.set_mode(TransportMode.WALK)


        self._log(f"placed TempBox for request #{req_id}")
        if inv_req:
            self._log(
                f"TempBox[#{req_id}] placed inventory: {self._fmt_inv_compact(_inv_before)} -> {self._fmt_inv_compact(self.inventory)}"
            )

        if want_food:
            for _oid in sorted(food_by_order.keys()):
                self._log(f"TempBox[#{req_id}] placed food for order #{int(_oid)}")

        if give_scooter:
            _had_scooter_after = self._has_scooter()
            self._log(
                f"TempBox[#{req_id}] placed e-scooter: {'present' if _had_scooter_before else 'absent'} -> {'present' if _had_scooter_after else 'absent'}"
            )
        self._finish_action(success=True)

    def _handle_take_from_temp_box(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        if not comms:
            self.vlm_add_error("take_from_temp_box failed: no comms")
            self._finish_action(success=False); return

        req_id = int(act.data.get("req_id"))
        tol = float(act.data.get("tol_cm", self._tol("nearby")))

        # 1) 判定我在这单里的身份 -> 应取哪个盒
        det = comms.get_request_detail(req_id) or {}
        me  = str(self.agent_id)
        role = None
        if str(det.get("accepted_by", "")) == me and str(det.get("publisher_id", "")) != me:
            role = "helper";  expect_key = "publisher_box"  # 帮手从发布者的盒子取
        elif str(det.get("publisher_id", "")) == me:
            role = "publisher"; expect_key = "helper_box"   # 发布者从帮手的盒子取
        else:
            self.vlm_add_error("take_from_temp_box failed: not a participant of this request")
            self._finish_action(success=False); return

        info = comms.get_temp_box_info(req_id) or {}
        box  = info.get(expect_key) or {}

        # 2) 位置与内容硬校验
        if not box.get("xy"):
            self.vlm_add_error("take_from_temp_box failed: temp box not available yet")
            self._finish_action(success=False); return

        bx, by = box["xy"]
        if not self._is_at_xy(float(bx), float(by), tol_cm=tol):
            self.vlm_add_error(f"take_from_temp_box failed: not near the TempBox (at {self._fmt_xy_m(bx, by)}). MOVE there first.")
            # 给 VLM 一个轻提示
            self.vlm_ephemeral["tempbox_hint"] = f"Go to the TempBox at {self._fmt_xy_m(bx, by)} for request #{req_id}."
            self._finish_action(success=False); return

        if not box.get("has_content", False):
            self.vlm_add_error("take_from_temp_box failed: the TempBox is empty")
            self.vlm_ephemeral["tempbox_hint"] = f"[Help #{req_id}] This TempBox is currently empty."
            self._finish_action(success=False); return

        # 3) 真正取物
        ok, msg, payload = comms.take_from_temp_box(req_id=req_id, by_agent=me)
        if not ok:
            self.vlm_add_error(f"take_from_temp_box failed: {msg}")
            self._finish_action(success=False); return

        # 4) 空 payload 也算失败，避免"成功但什么都没变"的卡死
        if not (payload.get("inventory") or payload.get("food_by_order") or (payload.get("escooter") is not None)):
            self.vlm_add_error("take_from_temp_box failed: TempBox is empty")
            self._finish_action(success=False); return

        _inv_before = dict(self.inventory)
        _had_scooter_before = self._has_any_scooter()

        # ===== 原有合并逻辑（不变） =====
        inv = dict(payload.get("inventory") or {})
        for k, q in inv.items():
            self.inventory[str(k)] = int(self.inventory.get(str(k), 0)) + int(q)

        if payload.get("escooter") is not None:
            sc = payload["escooter"]
            if getattr(sc, "owner_id", None) == str(self.agent_id):
                # ---- NEW: ensure we are holding the canonical instance
                com = get_comms()
                if com:
                    canon = com.get_scooter_by_owner(str(self.agent_id)) or sc
                    self.e_scooter = canon
                else:
                    self.e_scooter = sc

                # 取回自己的车：解除停放 + with_owner=True；不自动上车
                try:
                    self.e_scooter.unpark()
                except Exception:
                    self.e_scooter.park_xy = None
                setattr(self.e_scooter, "with_owner", True)

            else:
                # 别人的车：沿用你原逻辑作为 assist_scooter（这就是"同一台车"的共享引用）
                if self.assist_scooter is None:
                    self.assist_scooter = sc
                    setattr(self.assist_scooter, "proxy_helper_id", str(self.agent_id))
                    try:
                        self.assist_scooter.unpark()
                    except AttributeError:
                        self.assist_scooter.park_xy = None
                    self.set_mode(TransportMode.DRAG_SCOOTER)
                    if self.e_scooter and self.e_scooter.park_xy == None and getattr(self.e_scooter, "with_owner", True):
                        self.e_scooter.park_here(self.x, self.y)
                else:
                    self._log("take_from_temp_box: already assisting another scooter; ignoring extra")


        fbo = payload.get("food_by_order") or {}
        if fbo:
            now_sim = self.clock.now_sim()
            for oid, items in fbo.items():
                oid = int(oid)
                items_list = list(items or [])
                if not items_list: continue
                order_obj = self.help_orders.get(oid)
                for it in items_list:
                    if hasattr(it, "picked_at_sim"): it.picked_at_sim = float(now_sim)
                if order_obj is not None:
                    order_obj.has_picked_up = True
                    if oid not in self.carrying: self.carrying.append(oid)
                if oid not in self.carrying:
                    self.carrying.append(oid)
                cur = self._pending_food_by_order.get(oid, [])
                cur += items_list
                self._pending_food_by_order[oid] = cur

            self._force_place_food_now = True
            self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()

        # ===== 日志（沿用你原来的） =====
        if inv:
            self._log(f"TempBox[#{req_id}] took inventory: {self._fmt_inv_compact(_inv_before)} -> {self._fmt_inv_compact(self.inventory)}")
        if fbo:
            for _oid in sorted(fbo.keys()):
                self._log(f"TempBox[#{req_id}] took food for order #{int(_oid)}")
        if "escooter" in payload:
            _had_scooter_after = self._has_any_scooter()
            self._log(f"TempBox[#{req_id}] took e-scooter: {'present' if _had_scooter_before else 'absent'} -> {'present' if _had_scooter_after else 'absent'}")

        self._log(f"took items from TempBox for request #{req_id}")
        self._finish_action(success=True)



    def _handle_place_food_in_bag(self, _self, act: DMAction, _allow_interrupt: bool):
        """
        将 _pending_food_by_order 的食物按 bag_cmd 放入保温袋（支持多订单一次性放）。
        data = {"bag_cmd": "..."}  # 必填（可以是单行，包含多个 "order <id>:" 片段）
        
        事务语义：
        - 任意一步失败 -> 回滚到放置前状态，报错给 VLM，不抛异常，下一步仍提示放置。
        - 只处理 bag_cmd 中明确出现的订单，其它待放订单保持原样。
        """
        self.vlm_clear_ephemeral()
        spec_text = (act.data.get("bag_cmd") or "").strip()
        if not spec_text:
            self.vlm_add_error("place_food_in_bag failed: need bag_cmd")
            self._finish_action(success=False)
            return
        if not self._pending_food_by_order:
            self._finish_action(success=True)
            return

        if not self.insulated_bag:
            self.insulated_bag = InsulatedBag()

        # --- 解析：把单条 bag_cmd 中每个 "order <id>:" 独立切片 ---
        # 例： "order 2: 1,2 -> A; order 3: 1,2,3,4 -> B"
        # re.split 会得到 ["", "2", "1,2 -> A; ", "3", "1,2,3,4 -> B"]
        tokens = re.split(r'(?i)order\s+(\d+)\s*:\s*', spec_text)
        per_order_cmd: Dict[int, str] = {}
        if len(tokens) >= 3:
            # tokens: [prefix, oid1, tail1, oid2, tail2, ...]
            for i in range(1, len(tokens), 2):
                try:
                    oid = int(tokens[i])
                    tail = tokens[i + 1].strip()
                except Exception:
                    continue
                if tail:
                    per_order_cmd[oid] = tail
        else:
            # 没写 "order <id>:" 的简写：只在"仅有一个待放订单"时允许
            if len(self._pending_food_by_order) == 1:
                only_oid = next(iter(self._pending_food_by_order.keys()))
                per_order_cmd[int(only_oid)] = spec_text
            else:
                self.vlm_add_error(
                    "place_food_in_bag failed: multiple pending orders, please prefix each line with 'order <id>:'"
                )
                self._force_place_food_now = True
                self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()
                self._finish_action(success=False)
                return

        # 没有任何命中的订单 id
        hit_oids = [oid for oid in per_order_cmd.keys() if oid in self._pending_food_by_order]
        if not hit_oids:
            self.vlm_add_error("place_food_in_bag failed: no matching pending orders for provided bag_cmd")
            self._force_place_food_now = True
            self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()
            self._finish_action(success=False)
            return

        # --- 事务：在临时副本 tmp_bag 上尝试操作，全部成功后一次性提交 ---
        bag_before = copy.deepcopy(self.insulated_bag)
        pending_before = {k: list(v) for k, v in self._pending_food_by_order.items()}

        try:
            tmp_bag = copy.deepcopy(self.insulated_bag)
            for oid in hit_oids:
                items = self._pending_food_by_order.get(int(oid)) or []
                if not items:
                    continue
                order_cmd = per_order_cmd[int(oid)]
                # 针对该订单重新编号 1..N
                items_map = {i + 1: items[i] for i in range(len(items))}
                # 先按命令调整已有物品布局，再把"待放物"放入
                tmp_bag.move_items(order_cmd)
                tmp_bag.add_items(order_cmd, items_map)

            # 全部成功 -> 提交：替换保温袋对象，清理对应订单的 pending 队列
            self.insulated_bag = tmp_bag
            for oid in hit_oids:
                self._pending_food_by_order.pop(int(oid), None)

        except Exception as e:
            # 失败 -> 回滚到放置前状态；报错给 VLM；下一步继续提示放置
            self.insulated_bag = bag_before
            self._pending_food_by_order = pending_before
            self.vlm_add_error(f"place_food_in_bag failed: {e}")
            self._force_place_food_now = True
            self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()
            self._finish_action(success=False)
            return

        # --- 成功后的善后：如果还有待放项，则继续强制放置；否则收起提示 ---
        if self._pending_food_by_order:
            self._force_place_food_now = True
            self.vlm_ephemeral["bag_hint"] = self._build_bag_place_hint()
        else:
            self._force_place_food_now = False
            self.vlm_ephemeral.pop("bag_hint", None)

        self.vlm_ephemeral["ZZZ"] = "Bag placement succeeded."
        self._log(f"placed pending food into bag for orders {hit_oids}")
        self._finish_action(success=True)


    def _handle_report_help_finished(self, _self, act: DMAction, _allow_interrupt: bool):
        comms = get_comms()
        if not comms:
            self.vlm_add_error("report_help_finished failed: no comms")
            self._finish_action(success=False); return

        req_id = int(act.data.get("req_id"))
        ok, msg, _res = comms.report_help_finished(req_id=req_id, by_agent=str(self.agent_id), at_xy=(self.x, self.y))
        if not ok:
            self.vlm_add_error(f"report_help_finished failed: {msg}")
            self._finish_action(success=False); return

        self._log(f"reported help finished for request #{req_id}")
        self._finish_action(success=True)

    # ===== auto dropoff =====
    def _dropoff_physical_unload(self, order: Any) -> None:
        """只做'物理卸载'：从保温袋移除该单所有 item、caring 列表移除、清理待放队列。"""
        oid = int(getattr(order, "id", -1))
        items = list(getattr(order, "items", []) or [])
        if self.insulated_bag and hasattr(self.insulated_bag, "remove_items") and items:
            self.insulated_bag.remove_items(items)
        if oid in self.carrying:
            self.carrying.remove(oid)
        if self._pending_food_by_order and oid in self._pending_food_by_order:
            self._pending_food_by_order.pop(oid, None)

    def _dropoff_settle_record(self, order: Any) -> None:
        """只做'结算+记录+日志'，不动保温袋、不动 carrying。"""
        oid = getattr(order, "id", None)

        # 标记送达&时间戳
        order.has_delivered = True
        now_sim = float(self.clock.now_sim())
        order.sim_delivered_s = now_sim
        for it in getattr(order, "items", []) or []:
            it.delivered_at_sim = now_sim

        # 结算
        duration_s   = float(getattr(order, "sim_elapsed_active_s", 0.0) or 0.0)
        time_limit_s = float(getattr(order, "time_limit_s", 0.0) or 0.0)
        base_earn    = float(getattr(order, "earnings", 0.0) or 0.0)
        items        = list(getattr(order, "items", []) or [])

        settle_res = compute_settlement(
            order_base_earnings=base_earn,
            duration_s=duration_s,
            time_limit_s=time_limit_s,
            items=items,
            order_allowed_delivery_methods=getattr(order, "allowed_delivery_methods", []),
            actual_delivery_method=getattr(order, "delivery_method", None),
            config=self.cfg.get("settlement")
        )
        self.add_earnings(settle_res.total_pay)

        # 写入完成记录
        _bd = settle_res.breakdown or {}
        _time_star   = int((_bd.get("time")   or {}).get("time_star",   0))
        _food_star   = int((_bd.get("food")   or {}).get("food_star",   0))
        _method_star = int((_bd.get("method") or {}).get("method_star", 0))
        _flags = dict((_bd.get("flags") or {}))  # 兼容缺字段
        _on_time  = bool(_flags.get("on_time", True))
        _temp_ok  = bool(_flags.get("temp_ok_all", True))
        _odor_ok  = bool(_flags.get("odor_ok_all", True))
        _dmg_ok   = bool(_flags.get("damage_ok_all", True))

        # 用于日志的紧凑文本
        _flags_detail = (
            f" [on_time={'Y' if _on_time else 'N'}, "
            f"temp={'OK' if _temp_ok else 'BAD'}, "
            f"odor={'OK' if _odor_ok else 'BAD'}, "
            f"damage={'OK' if _dmg_ok else 'BAD'}]"
        )

        # 写入完成记录
        self.completed_orders.append(dict(
            id=oid,
            duration_s=duration_s,
            rating=float(settle_res.stars),
            earnings=base_earn,
            bonus_extra=float(settle_res.extra_pay),
            paid_total=float(settle_res.total_pay),
            breakdown=settle_res.breakdown,
            pickup=getattr(order, "pickup_road_name", ""),
            dropoff=getattr(order, "dropoff_road_name", ""),
            # 新增：记录允许的配送方式以及实际选择的方式
            allowed_delivery_methods=list(getattr(order, "allowed_delivery_methods", []) or []),
            delivery_method=getattr(order, "delivery_method", None),
            stars=dict(
                overall=int(settle_res.stars),
                time=_time_star,
                food=_food_star,
                method=_method_star,
            ),
            # === NEW: 汇总布尔指标（用于后续 failure 统计） ===
            flags=dict(
                on_time=_on_time,
                temp_ok_all=_temp_ok,
                odor_ok_all=_odor_ok,
                damage_ok_all=_dmg_ok,
            ),
        ))


        # 日志也打印三颗星
        extra_str   = f" (extra {settle_res.extra_pay:+.2f}, stars={settle_res.stars})"
        star_detail = f" [time={_time_star}, food={_food_star}, method={_method_star}]"
        self._log(
            f"dropped off order #{oid}{extra_str}{star_detail}{_flags_detail}"
            if oid is not None else
            f"dropped off order{extra_str}{star_detail}{_flags_detail}"
        )

        # 从 active_orders 移除本单
        self.active_orders = [o for o in self.active_orders if getattr(o, "id", None) != oid]

        # 若已无待放条目，关掉 bag hint
        if not self._pending_food_by_order:
            self._force_place_food_now = False
            self.vlm_ephemeral.pop("bag_hint", None)

        self.vlm_clear_errors()

    def _auto_try_dropoff(self):
        tol = self._tol("nearby")
        comms = get_comms()

        # ===== 吃 Comm 消息（我作为 publisher 的单被 helper 完成）
        if comms:
            msgs = comms.pop_msgs_for_publisher(str(self.agent_id))  # 消息即清空
            for m in msgs:
                if m.get("type") != "HELP_DELIVERY_DONE":
                    continue
                oid = int(m.get("order_id", -1))
                if oid <= 0 or oid in self.help_completed_order_ids:
                    continue
                # 找到我这边的订单对象（还未结算）
                order_obj = next((o for o in self.active_orders
                                  if int(getattr(o, "id", -1)) == oid
                                  and not getattr(o, "has_delivered", False)), None)
                if order_obj is None:
                    # 可能已被我手动结算或不在 active；忽略
                    continue

                # 只做'结算+记录'，不做物理卸载（包与 carrying 由对方实际处理）
                self._dropoff_settle_record(order_obj)
                self.help_completed_order_ids.add(oid)


        # ===== 清理 helper 侧已被对方结算的单 =====
        for oid, o in list(self.help_orders.items()):
            if getattr(o, "has_delivered", False):
                self._helping_wait_ack_oids.discard(int(oid))
                self.help_orders.pop(int(oid), None)


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
            f"  Pace     : {self.pace_state} (×{self._pace_scale():.2f})",
            f"  Energy   : {self.energy_pct:.0f}%",
            f"  Earnings : ${self.earnings_total:.2f}",
            f"  Active Orders : {active_ids}",
            f"  Helping Orders: {list(self.help_orders.keys())}",
            f"  Carrying : {self.carrying}",
            f"  Queue    : {len(self._queue)} action(s), Busy: {self.is_busy()}",
        ]
        if self.e_scooter:
            rng_m = self._remaining_range_m(); rng_km = (rng_m/1000.0) if rng_m is not None else None
            lines += [f"  Scooter  : state={self.e_scooter.state.value}, battery={self.e_scooter.battery_pct:.0f}% "
                      f"({self.e_scooter.charge_rate_pct_per_min:.1f}%/min), avg_speed={self.e_scooter.avg_speed_cm_s:.0f} cm/s, "
                      f"park_xy={self._fmt_xy_m_opt(self.e_scooter.park_xy)}, remaining={'{:.1f} km'.format(rng_km) if rng_km is not None else 'N/A'}"]
        if self.assist_scooter:
            s = self.assist_scooter
            lines += [f"  AssistScooter : owner={getattr(s, 'owner_id','?')}, battery={s.battery_pct:.0f}% "
                      f"({s.charge_rate_pct_per_min:.1f}%/min), park_xy={self._fmt_xy_m_opt(s.park_xy)}"]

        if self.car:
            lines += [f"  Car      : state={self.car.state.value}, rate=${self.car.rate_per_min:.2f}/min, park_xy={self._fmt_xy_m_opt(self.car.park_xy)}, rental={'on' if self._rental_ctx else 'off'}"]
        return "\n".join(lines)

    # ===== progress for UI =====
    def charging_progress(self) -> Optional[Dict[str, Any]]:
        if self._charge_ctx and self._charge_ctx.get("scooter_ref"):
            ctx = self._charge_ctx; now = self.clock.now_sim()
            sc = ctx["scooter_ref"]
            t0, t1 = ctx["start_sim"], ctx["end_sim"]; p0, pt = ctx["start_pct"], ctx["target_pct"]
            if t1 <= t0: cur = pt; prog = 1.0
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0))); cur = p0 + (pt - p0) * r; prog = 0.0 if pt <= p0 else (cur - p0) / max(1e-9, pt - p0)
            xy = sc.park_xy if sc.park_xy else (self.x, self.y)
            return dict(progress=float(max(0.0, min(1.0, prog))), current_pct=float(cur), target_pct=float(pt), xy=xy, which=ctx.get("which","own"))
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

        fee = float(getattr(self, "hospital_rescue_fee", 0.0))
        if fee > 1e-9:
            deduct = min(fee, max(0.0, float(self.earnings_total)))
            if deduct > 0.0:
                self.earnings_total -= deduct
                self._log(f"hospital rescue fee charged: ${deduct:.2f} (balance ${self.earnings_total:.2f})")
            if self._recorder:
                self._recorder.on_hospital_fee(self.clock.now_sim(), fee=deduct)
                self._recorder.inc("hospital_rescue", 1)

        hxy = self._closest_poi_xy("hospital") or (self.x, self.y)
        if self._ue and hasattr(self._ue, "teleport_xy"): self._ue.teleport_xy(str(self.agent_id), float(hxy[0]), float(hxy[1]))
        self.x, self.y = float(hxy[0]), float(hxy[1])
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "set_agent_xy"): self._viewer.set_agent_xy(self._viewer_agent_id, self.x, self.y)
        self._queue.clear(); self._current = None
        now_sim = self.clock.now_sim()
        self._hospital_ctx = dict(start_sim=now_sim, end_sim=now_sim + float(self.hospital_duration_s))

    # ===== tick =====
    def _interrupt_and_stop(self, reason: str, hint: Optional[str] = None):
        """统一的中断行为：标记中断原因、打断当前移动、并在 UE 停车。可选给 VLM 提示。"""
        self._interrupt_reason = str(reason)
        self._interrupt_move_flag = True
        if self._ue and hasattr(self._ue, "delivery_man_stop"):
            try:
                self._ue.delivery_man_stop(str(self.agent_id))
            except Exception:
                pass
        if hint:
            # 用"原因"作为 ephemeral key，便于在 prompt 里按需展示
            self.vlm_ephemeral[str(reason)] = str(hint)
        self._log(f"interrupt: {reason} -> stop moving & wait for decision")
        if reason == "escooter_depleted" and self._recorder:
            self._recorder.inc("scooter_depleted", 1)
        if reason == "car_rental_ended":
            self._recorder.inc("rent_insufficient", 1)
    
    def poll_time_events(self):
        now = self.clock.now_sim()

        self._auto_try_dropoff()

        # === active orders elapsed time ===
        if self._orders_last_tick_sim is None:
            self._orders_last_tick_sim = now
        if not self._timers_paused:
            delta = max(0.0, now - self._orders_last_tick_sim)
            if delta > 0:
                for o in self.active_orders:
                    if getattr(o, "is_accepted", False) and not getattr(o, "has_delivered", False):
                        cur = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0)
                        o.sim_elapsed_active_s = cur + delta
            self._orders_last_tick_sim = now

        rec = getattr(self, "_recorder", None)
        if self._life_last_tick_sim is None:
            self._life_last_tick_sim = now
        if rec:
            if rec.started_sim_s is None:
                rec.start(now_sim=now)
            if not self._timers_paused:
                delta = max(0.0, now - self._life_last_tick_sim)
                rec.tick_active(delta)
                # 按当前交通方式累计活跃时间
                try:
                    rec.tick_transport(getattr(self.mode, "value", str(self.mode)), delta)
                except Exception:
                    pass
                self._life_last_tick_sim = now

            # 到点立即停止（只触发一次），并导出报告
            if rec.should_end():
                # 确定停止原因
                stop_reason = "unknown"
                stop_message = "Lifecycle reached. Stopping this run."
                
                # 检查虚拟时间停止
                sim_time_end = (rec.lifecycle_s > 0) and (rec.active_elapsed_s >= rec.lifecycle_s)
                # 检查现实时间停止
                realtime_end = False
                if rec.realtime_stop_hours > 0 and rec.realtime_start_ts is not None:
                    current_realtime = time.time()
                    elapsed_realtime_hours = (current_realtime - rec.realtime_start_ts) / 3600.0
                    realtime_end = elapsed_realtime_hours >= rec.realtime_stop_hours
                # 检查 VLM call 次数限制
                vlm_call_end = (rec.vlm_call_limit > 0) and (rec.counters.vlm_calls >= rec.vlm_call_limit)
                
                if sim_time_end and realtime_end and vlm_call_end:
                    stop_reason = "all_limits_reached"
                    stop_message = f"All limits reached: simulation time ({rec.active_elapsed_s/3600:.2f}h), real time ({elapsed_realtime_hours:.2f}h), and VLM calls ({rec.counters.vlm_calls}). Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif sim_time_end and realtime_end:
                    stop_reason = "both_times_reached"
                    stop_message = f"Both simulation time ({rec.active_elapsed_s/3600:.2f}h) and real time ({elapsed_realtime_hours:.2f}h) reached. Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif sim_time_end and vlm_call_end:
                    stop_reason = "sim_time_and_vlm_reached"
                    stop_message = f"Simulation time ({rec.active_elapsed_s/3600:.2f}h) and VLM calls ({rec.counters.vlm_calls}) reached. Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif realtime_end and vlm_call_end:
                    stop_reason = "realtime_and_vlm_reached"
                    stop_message = f"Real time ({elapsed_realtime_hours:.2f}h) and VLM calls ({rec.counters.vlm_calls}) reached. Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif sim_time_end:
                    stop_reason = "sim_time_reached"
                    stop_message = f"Simulation time reached ({rec.active_elapsed_s/3600:.2f}h). Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif realtime_end:
                    stop_reason = "realtime_reached"
                    stop_message = f"Real time reached ({elapsed_realtime_hours:.2f}h). Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                elif vlm_call_end:
                    stop_reason = "vlm_call_limit_reached"
                    stop_message = f"VLM call limit reached ({rec.counters.vlm_calls}). Stopping this run."
                    self.logger.info("Agent " + self.agent_id + ": " + stop_message)
                
                rec.mark_end(now_sim=now)
                # 先收口持续计时会话，避免漏账/爆量
                rec.finish_charging(end_ts=now, reason="lifecycle_end")
                rec.finish_rental(end_ts=now)

                if self._charge_ctx:
                    comms = get_comms()
                    if comms and hasattr(comms, "release_charging_spot"):
                        key = self._charge_ctx.get("station_key") or self._charge_ctx.get("station_xy")
                        comms.release_charging_spot(key, agent_id=str(self.agent_id))
                    self._charge_ctx = None

                self._interrupt_and_stop("lifecycle_ended", stop_message)
                try:
                    path = rec.export(self)
                    self._log(f"run report exported to {path}")
                except Exception as e:
                    self._log(f"run report export failed: {e}")
                self._lifecycle_done = True
                return  # 当次 tick 不再继续推进

        # === MOVE arrival / blocked ===
        if self._move_ctx is not None:
            if self._interrupt_move_flag:
                self._move_ctx["blocked"] = 1.0
                self._interrupt_move_flag = False

            tx = float(self._move_ctx["tx"])
            ty = float(self._move_ctx["ty"])
            tol = float(self._move_ctx["tol"])

            if self._move_ctx.get("blocked", 0.0) == 1.0:
                self._move_ctx = None
                if self._current and self._current.kind == DMActionKind.MOVE_TO:
                    self._finish_action(success=False)

            elif self._is_at_xy(tx, ty, tol_cm=tol):
                self._move_ctx = None
                # 到点先尝试自动投递/结算
                # self._auto_try_dropoff()
                if self._current and self._current.kind == DMActionKind.MOVE_TO:
                    self._finish_action(success=True)

        # === WAIT ===
        if self._current and self._current.kind == DMActionKind.WAIT and self._wait_ctx and not self._timers_paused:
            if now >= self._wait_ctx["end_sim"]:
                self._wait_ctx = None
                self._finish_action(success=True)

        # === CHARGE (MODIFIED: use ctx['scooter_ref'] instead of self.e_scooter) ===
        if self._charge_ctx and self._charge_ctx.get("scooter_ref") and not self._timers_paused:
            ctx = self._charge_ctx
            sc = ctx["scooter_ref"]  # 充电中的那台车：assist 优先，否则 own

            t0, t1 = ctx["start_sim"], ctx["end_sim"]
            p0, pt = float(ctx["start_pct"]), float(ctx["target_pct"])
            paid_pct = float(ctx.get("paid_pct", p0))
            price_per_pct = float(ctx.get("price_per_pct", self.charge_price_per_pct))

            # 理论应达到的电量（按时间线）
            if t1 <= t0:
                cur_should = pt
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0)))
                cur_should = p0 + (pt - p0) * r

            # 需要推进的百分点 & 钱包可承受的百分点
            add_pct_need = max(0.0, cur_should - paid_pct)
            max_afford_pct = float("inf") if price_per_pct <= 0.0 else max(0.0, self.earnings_total) / price_per_pct
            add_pct_can = min(add_pct_need, max_afford_pct)

            # 扣费并推进
            if add_pct_can > 0.0:
                cost = add_pct_can * price_per_pct
                self.earnings_total = max(0.0, self.earnings_total - cost)
                paid_pct = paid_pct + add_pct_can
                ctx["paid_pct"] = paid_pct
                sc.charge_to(paid_pct)  # 写回这台车
                rec = getattr(self, "_recorder", None)
                if rec:
                    rec.accrue_charging(
                        ts_sim=now,
                        which=str(ctx.get("which", "own")),
                        delta_pct=float(add_pct_can),
                        cost=float(cost),
                        req_id=(int(ctx.get("req_id")) if ctx.get("req_id") is not None else None),
                        start_ts=float(ctx.get("start_sim", now))
                    )

            # 结束条件
            finished_by_target = (paid_pct + 1e-6) >= pt
            out_of_money = add_pct_need > 0.0 and add_pct_can + 1e-9 < add_pct_need

            if finished_by_target:
                sc.charge_to(pt)
                if rec:
                    rec.finish_charging(end_ts=now, reason="finished", target_pct=pt)
                self._charge_ctx = None
                if sc.park_xy:
                    px, py = sc.park_xy
                else:
                    px, py = self.x, self.y
                loc = self._fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                self._log(f"charging finished ({which}): {p0:.0f}% -> {pt:.0f}% at {loc}")
                self.vlm_ephemeral["scooter_ready"] = (
                    f"{'Assisting scooter' if which=='assist' else 'Your scooter'} charged to {pt:.0f}%. "
                    f"It's parked at {loc}. You can come here to retrieve it."
                )
                # 如果当前在 WAIT("charge_done")，立即完成等待
                if self._current and self._current.kind == DMActionKind.WAIT and self._wait_ctx:
                    self._wait_ctx = None
                    self._finish_action(success=True)
                comms = get_comms()
                if comms and hasattr(comms, "release_charging_spot"):
                    key = ctx.get("station_key") or ctx.get("station_xy")
                    comms.release_charging_spot(key, agent_id=str(self.agent_id))

            elif out_of_money:
                # 钱不够：中断
                self._charge_ctx = None
                if rec:
                    rec.finish_charging(end_ts=now, reason="no_money", target_pct=pt)
                    self._recorder.inc("charge_insufficient", 1)
                if sc.park_xy:
                    px, py = sc.park_xy
                else:
                    px, py = self.x, self.y
                loc = self._fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                self._log(f"charging interrupted ({which}) for insufficient funds at {paid_pct:.0f}% (target {pt:.0f}%)")
                self.vlm_ephemeral["charging_interrupted"] = (
                    f"Charging was interrupted due to insufficient funds at {paid_pct:.0f}%. "
                    f"The scooter is parked at {loc}. Earn more money, then CHARGE_ESCOOTER again."
                )
                # 如果当前在 WAIT("charge_done")，也结束等待
                if self._current and self._current.kind == DMActionKind.WAIT and self._wait_ctx:
                    self._wait_ctx = None
                    self._finish_action(success=True)
                comms = get_comms()
                if comms and hasattr(comms, "release_charging_spot"):
                    key = ctx.get("station_key") or ctx.get("station_xy")
                    comms.release_charging_spot(key, agent_id=str(self.agent_id))

            elif (
                getattr(sc, "park_xy", None) is None
                or (ctx.get("park_xy_start") and tuple(sc.park_xy or ()) != tuple(ctx["park_xy_start"]))
            ):
                self._charge_ctx = None
                if rec:
                    rec.finish_charging(end_ts=now, reason="moved", target_pct=pt)
                if sc.park_xy:
                    px, py = sc.park_xy
                else:
                    px, py = self.x, self.y
                loc = self._fmt_xy_m(px, py)
                which = ctx.get("which", "own")
                self._log(f"charging interrupted ({which}) at {paid_pct:.0f}% (scooter moved)")
                self.vlm_ephemeral["charging_interrupted"] = (
                    f"Charging was interrupted at {paid_pct:.0f}% because the scooter was moved."
                )
                # 若当前在 WAIT("charge_done")，也结束等待
                if self._current and self._current.kind == DMActionKind.WAIT and self._wait_ctx:
                    self._wait_ctx = None
                    self._finish_action(success=True)
                comms = get_comms()
                if comms and hasattr(comms, "release_charging_spot"):
                    key = ctx.get("station_key") or ctx.get("station_xy")
                    comms.release_charging_spot(key, agent_id=str(self.agent_id))


        # === REST ===
        if self._rest_ctx and not self._timers_paused:
            t0, t1 = self._rest_ctx["start_sim"], self._rest_ctx["end_sim"]
            e0, et = self._rest_ctx["start_pct"], self._rest_ctx["target_pct"]
            if t1 <= t0:
                cur = et
            else:
                r = max(0.0, min(1.0, (now - t0) / (t1 - t0)))
                cur = e0 + (et - e0) * r
            self.energy_pct = float(cur)
            if now >= t1:
                self.energy_pct = float(et)
                self._log(f"rest finished: {e0:.0f}% -> {et:.0f}%")
                self._rest_ctx = None
                if self._current and self._current.kind == DMActionKind.REST:
                    self._finish_action(success=True)

        # === HOSPITAL ===
        if self._hospital_ctx and now >= self._hospital_ctx["end_sim"] and not self._timers_paused:
            self.rescue()
            self._hospital_ctx = None
            self._log("rescue finished: full energy at Hospital")
            self.kickstart()

        # === rental billing ===
        if self._rental_ctx and not self._timers_paused:
            dt = max(0.0, now - float(self._rental_ctx["last_tick_sim"]))
            if dt > 0:
                rate = float(self._rental_ctx["rate_per_min"])
                cost = rate * (dt / 60.0)
                old_balance = float(self.earnings_total)
                if self.earnings_total - cost <= 0.0:
                    in_car = (self.mode == TransportMode.CAR)
                    self.car = None
                    self._rental_ctx = None
                    self.earnings_total = max(0.0, self.earnings_total - cost)
                    self._interrupt_and_stop(
                        "car_rental_ended",
                        "Your car rental has ended (insufficient funds). You may SWITCH_TRANSPORT(to='walk'), "
                        "RENT_CAR(...) again, or choose another mode."
                    )
                    self._log("rental ended (no money) -> interrupt; waiting for decision")
                    charge_amount = min(cost, max(0.0, old_balance))
                    rec = getattr(self, "_recorder", None)
                    if rec and charge_amount > 1e-12:
                        rec.accrue_rental(dt_s=float(dt), cost=float(charge_amount),
                                        start_ts=float(self._rental_ctx.get("start_sim", now)))
                    if rec:
                        rec.finish_rental(end_ts=now)
                else:
                    self.earnings_total -= cost
                    self._rental_ctx["last_tick_sim"] = now
                    rec = getattr(self, "_recorder", None)
                    if rec and cost > 1e-12:
                        rec.accrue_rental(dt_s=float(dt), cost=float(cost),
                                        start_ts=float(self._rental_ctx.get("start_sim", now)))

        # === insulated bag temperatures ===
        if self.insulated_bag and not self._timers_paused:
            if self._last_bag_tick_sim is None:
                self._last_bag_tick_sim = now
            else:
                dt = max(0.0, now - self._last_bag_tick_sim)
                if dt > 0:
                    self.insulated_bag.tick_temperatures(dt)
                    self.insulated_bag.tick_odor(dt)
                    self._last_bag_tick_sim = now

        comms = get_comms()
        if comms:
            inbox = comms.pop_chat(str(self.agent_id), max_items=20)
            if inbox:
                # 简单渲染：最新在下
                lines = []
                for m in inbox:
                    ts = float(m.get("ts_sim", 0.0))
                    src = str(m.get("from", ""))
                    kind = m.get("kind", "direct")
                    txt = str(m.get("text", ""))
                    if kind == "broadcast":
                        lines.append(f"[broadcast] from {src}: {txt}")
                    else:
                        lines.append(f"from {src}: {txt}")
                # 放到临时上下文，供 prompt 消费
                self.vlm_ephemeral["chat_inbox"] = "\n".join(lines[-20:])

        # 刷新附近 POI 提示（MODIFIED: 会针对 assist/own 选择提示充电）
        self._refresh_poi_hints_nearby()

        # 公交状态推进
        if self._bus_ctx and self.mode == TransportMode.BUS:
            self._update_bus_riding(now)

    # ===== Bus handlers =====
    def _handle_board_bus(self, _self, act: DMAction, _allow_interrupt: bool):
        """上车动作"""
        self.vlm_clear_ephemeral()
        if not self._bus_manager:
            self.vlm_add_error("board_bus failed: no bus manager")
            self._finish_action(success=False)
            return

        bus_id = act.data.get("bus_id")
        target_stop_id = act.data.get("target_stop_id")

        if not bus_id:
            self.vlm_add_error("board_bus failed: need bus_id")
            self._finish_action(success=False)
            # print("board_bus failed: need bus_id")
            return

        if not target_stop_id:
            self.vlm_add_error("board_bus failed: need target_stop")
            self._finish_action(success=False)
            # print("board_bus failed: need target_stop")
            return

        bus = self._bus_manager.get_bus(bus_id)
        if not bus:
            self.vlm_add_error(f"board_bus failed: bus {bus_id} not found")
            self._finish_action(success=False)
            # print(f"board_bus failed: bus {bus_id} not found")
            return

        # 检查bus是否在当前车站且为stopped状态
        if not bus.is_at_stop() or math.hypot(bus.x - self.x, bus.y - self.y) > 1000.0:
            self.vlm_add_error(f"board_bus failed: bus {bus_id} not at stop")
            self._finish_action(success=False)
            # print(f"board_bus failed: bus {bus_id} not at stop")
            return

        # 验证目标站点是否在bus的路线上
        target_stop = None
        for stop in bus.route.stops:
            if stop.id == target_stop_id:
                target_stop = stop
                break

        if not target_stop:
            self.vlm_add_error(f"board_bus failed: target stop {target_stop_id} not on bus route")
            self._finish_action(success=False)
            # print(f"board_bus failed: target stop {target_stop_id} not on bus route")
            return

        # 检查当前站点是否就是目标站点（避免无意义的上车）
        current_stop = bus.get_current_stop()
        if current_stop and current_stop.id == target_stop_id:
            self.vlm_add_error(f"board_bus failed: already at target stop {target_stop_id}")
            self._finish_action(success=False)
            # print(f"board_bus failed: already at target stop {target_stop_id}")
            return

        # 上车
        # 先检查余额是否足够（$1）
        if float(self.earnings_total) + 1e-9 < 1.0:
            self.vlm_add_error("board_bus failed: insufficient funds ($1 required)")
            self._finish_action(success=False)
            return

        if bus.board_passenger(str(self.agent_id)):
            # 上车成功，扣费 $1
            old_balance = float(self.earnings_total)
            self.earnings_total = max(0.0, old_balance - 1.0)
            if self._recorder:
                self._recorder.inc("bus_board", 1)
            self._bus_ctx = {
                "bus_id": bus_id,
                "boarding_stop": current_stop.id if current_stop else "",
                "target_stop": target_stop_id,
                "transport_mode": self.mode,
                "boarded_time": self.clock.now_sim()
            }
            self.set_mode(TransportMode.BUS)
            self._log(f"boarded bus {bus_id} at {current_stop.id if current_stop else 'unknown'} heading to {target_stop_id}")
            self._register_success(f"boarded bus {bus_id}")
        else:
            self.vlm_add_error("board_bus failed: could not board")
            self._finish_action(success=False)
            # print("board_bus failed: could not board")
            
    def _update_bus_riding(self, now: float):
        """更新乘坐公交状态"""
        if not self._bus_ctx or not self._bus_manager:
            return

        bus_id = self._bus_ctx.get("bus_id")
        bus = self._bus_manager.get_bus(bus_id)

        # 跟随公交位置
        self.x = bus.x
        self.y = bus.y

        # 检查是否到达目标站点
        target_stop_id = self._bus_ctx.get("target_stop")
        if target_stop_id and bus.is_at_stop():
            current_stop = bus.get_current_stop()
            if current_stop and current_stop.id == target_stop_id:
                # 到达目标站点，自动下车
                bus.alight_passenger(str(self.agent_id))
                self._log(f"arrived at target stop {target_stop_id}, auto alighting")
                self.set_mode(self._bus_ctx.get("transport_mode"))
                if self._ue and hasattr(self._ue, "teleport_xy"): 
                    self._ue.teleport_xy(str(self.agent_id), float(self.x), float(self.y))
                self._bus_ctx = None
                self._finish_action(success=True)

    def _handle_view_bus_schedule(self, _self, act: DMAction, _allow_interrupt: bool):
        """查看公交时刻表"""
        try:
            if not self._bus_manager:
                self.vlm_add_ephemeral("bus_schedule", "(no bus schedule)")
                self._log("view bus schedule (no bus manager)")
                self._finish_action(success=True)
                return

            # 获取所有路线信息
            routes_info = self._bus_manager.get_all_routes_info()
            
            # 获取所有公交车状态
            buses_status = self._bus_manager.get_all_buses_status()
            
            # 构建时刻表文本
            schedule_text = ""
            
            # 添加路线信息
            if routes_info:
                schedule_text += "Routes:\n"
                for route_id, route_info in routes_info.items():
                    schedule_text += f"Route{route_info['name']}:\n"
                    schedule_text += f"  Stops ({len(route_info['stops'])}):\n"
                    
                    for i, stop in enumerate(route_info['stops']):
                        schedule_text += f"  {stop['name']} - Wait: {stop['wait_time_s']:.1f}s\n"
            else:
                schedule_text += "No routes available.\n"
            
            # 添加当前公交车状态
            if buses_status:
                schedule_text += "\nCurrent bus status:\n"
                for status in buses_status:
                    schedule_text += f"  {status}\n"
            else:
                schedule_text += "\nNo buses currently running.\n"
            
            print(schedule_text)
            # 塞进 ephemeral，供 VLM prompt 使用
            self.vlm_add_ephemeral("bus_schedule", schedule_text)
            self._log("view bus schedule")
            self._finish_action(success=True)
        except Exception as e:
            self.vlm_add_error(f"view_bus_schedule failed: {e}")
            self._finish_action(success=False)


    # ======low-level actions======
    def _handle_turn_around(self, _self, act: DMAction, _allow_interrupt: bool):
        """转身"""
        try:
            angle = act.data.get("angle")
            direction = act.data.get("direction")
            self._ue.delivery_man_turn_around(self.agent_id, angle, direction)
            self._finish_action(success=True)
        except Exception as e:
            self.vlm_add_error(f"turn_around failed: {e}")
            self._finish_action(success=False)
        
    def _handle_step_forward(self, _self, act: DMAction, _allow_interrupt: bool):
        """前进一步"""
        try:
            self._ue.delivery_man_step_forward(self.agent_id, 100, 1)
            self._finish_action(success=True)
        except Exception as e:
            self.vlm_add_error(f"step_forward failed: {e}")
            self._finish_action(success=False)
        