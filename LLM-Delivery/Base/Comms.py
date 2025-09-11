# -*- coding: utf-8 -*-
# Base/Comms.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Dict, Any, List, Optional, Tuple

from Base.Timer import VirtualClock

# ===== Help Types =====
class HelpType(str, Enum):
    HELP_PICKUP   = "HELP_PICKUP"    # Helper picks up at restaurant -> drop at deliver_xy (TempBox)
    HELP_DELIVERY = "HELP_DELIVERY"  # Helper delivers an existing order (publisher prepares/places food)
    HELP_BUY      = "HELP_BUY"       # Helper buys items -> drop at deliver_xy (TempBox)
    HELP_CHARGE   = "HELP_CHARGE"    # Helper charges e-scooter to target_pct -> drop at deliver_xy (TempBox)

# ===== TempBox =====
@dataclass
class TempBox:
    box_id: int
    req_id: int
    owner_id: str                  # who placed (for logs)
    role: str                      # 'publisher' or 'helper'
    xy: Tuple[float, float]
    created_sim: float

    # unified content
    inventory: Dict[str, int] = field(default_factory=dict)
    food_by_order: Dict[int, List[Any]] = field(default_factory=dict)  # {order_id: [FoodItem,...]}
    escooter: Optional[Any] = None

    def merge_payload(self, payload: Dict[str, Any]) -> None:
        """Merge incoming content into the box (caller is responsible for deducting from agent inventory)."""
        if not payload:
            return
        # inventory
        inv = payload.get("inventory") or {}
        for k, v in inv.items():
            self.inventory[k] = int(self.inventory.get(k, 0)) + int(v)
        # food items (support short form {order_id, food_items})
        fbo = payload.get("food_by_order") or {}
        if "order_id" in payload and "food_items" in payload:
            fbo = {int(payload["order_id"]): list(payload["food_items"])}
        for oid, items in fbo.items():
            oid = int(oid)
            cur = self.food_by_order.get(oid, [])
            cur += list(items or [])
            self.food_by_order[oid] = cur
        # e-scooter handover
        if "escooter" in payload:
            self.escooter = payload["escooter"]

# ===== Help Request =====
@dataclass
class HelpRequest:
    req_id: int
    publisher_id: str
    kind: HelpType
    reward: float
    time_limit_s: float
    created_sim: float

    # task params
    order_id: Optional[int] = None
    buy_items: Optional[Dict[str, int]] = None
    target_pct: Optional[float] = None

    # handover/finish locations
    provide_xy: Optional[Tuple[float, float]] = None
    deliver_xy: Optional[Tuple[float, float]] = None

    brief: str = ""

    # runtime state
    accepted_by: Optional[str] = None
    completed: bool = False
    order_ref: Optional[Any] = None

    # ==== Timer (starts when the task becomes "ready") ====
    timer_started_sim: Optional[float] = None      # None = not started
    timer_paused_accum_s: float = 0.0
    timer_paused_at: Optional[float] = None
    start_note: Optional[str] = None

    # ==== Funds (escrow and upfront) ====
    escrow_reward: float = 0.0
    upfront_paid_to_helper: float = 0.0

    # ==== Result ====
    result: Optional[str] = None                   # 'success' | 'partial' | 'fail'

    # ---- Text rendering (English only) ----
    def to_text_for(
        self,
        agent_id: str,
        comms: "CommsSystem",
        now: Optional[float] = None,
        view_as: Optional[str] = None,  # "helper" / "publisher" / None(auto)
    ) -> str:
        """
        Human-readable instruction block for a given agent:
        - Clear next steps (where to go / what to do / when to REPORT).
        - Shows TempBox state (placed? what/where?).
        - Explicitly states timer rules and both agent IDs.
        """
        now = comms.clock.now_sim() if now is None else float(now)

        def _fmt_xy(xy):
            return f"({xy[0]/100.0:.2f}m, {xy[1]/100.0:.2f}m)" if xy else "N/A"

        def _fmt_buy_list(buy_items: Optional[Dict[str, int]]) -> str:
            if not buy_items:
                return "the required items"
            parts = [f"{k} x{int(v)}" for k, v in buy_items.items()]
            return ", ".join(parts)

        # time text (shows overdue explicitly)
        if self.timer_started_sim is None:
            head_time = "time left — not started"
        else:
            elapsed = comms._effective_elapsed_s(self, now)
            remaining = float(self.time_limit_s) - float(elapsed)
            if remaining >= 0:
                mins = int(math.ceil(remaining / 60.0))
                head_time = f"time left {mins} min"
            else:
                over = int(math.ceil((-remaining) / 60.0))
                head_time = f"time left 0 min (overdue by {over} min)"


        reward = f"${self.reward:.2f}"
        head = f"[Help #{self.req_id}] {self.kind.value.replace('HELP_', '').title()} — reward {reward}, {head_time}."

        # roles
        am_helper = (self.accepted_by == str(agent_id))
        am_publisher = (self.publisher_id == str(agent_id))
        if view_as == "helper":
            am_helper, am_publisher = True, False
        elif view_as == "publisher":
            am_helper, am_publisher = False, True

        helper_label = self.accepted_by if self.accepted_by else "TBD"
        role_line = f"Participants: help-seeker (agent {self.publisher_id}), helper (agent {helper_label})."

        # timer rule line
        when_line = ""
        if self.kind in (HelpType.HELP_DELIVERY, HelpType.HELP_CHARGE):
            if self.timer_started_sim is None:
                thing = "order food" if self.kind == HelpType.HELP_DELIVERY else "e-scooter"
                when_line = (
                    f"Note: the timer starts when the help-seeker (agent {self.publisher_id}) "
                    f"places the {thing} into the TempBox (handover ready). Not started yet."
                )
            else:
                when_line = "Note: timer has started (counting from the handover-ready moment)."
        else:
            # BUY / PICKUP
            if self.accepted_by:
                when_line = "Note: this task requires no handover from the help-seeker; timer runs from acceptance."
            else:
                when_line = "Note: once accepted, the timer runs immediately."

        # TempBox overview
        info = comms.get_temp_box_info(int(self.req_id)) or {}
        pub_box = info.get("publisher_box", {}) or {}
        hel_box = info.get("helper_box", {}) or {}

        # destination
        dst_txt = "N/A"
        if self.kind == HelpType.HELP_DELIVERY and self.order_ref is not None:
            try:
                name = getattr(self.order_ref, "dropoff_road_name", None)
                if name:
                    dst_txt = name
                else:
                    dn = getattr(self.order_ref, "dropoff_node", None)
                    if dn is not None:
                        x = float(getattr(dn.position, "x", 0.0)); y = float(getattr(dn.position, "y", 0.0))
                        dst_txt = _fmt_xy((x, y))
            except Exception:
                pass
        elif self.deliver_xy:
            dst_txt = _fmt_xy(self.deliver_xy)

        provide_txt = _fmt_xy(self.provide_xy)

        plan = ""
        plan_intro = f"{role_line} {when_line} "

        if self.kind == HelpType.HELP_BUY:
            buy_txt = _fmt_buy_list(self.buy_items)
            if am_helper:
                if hel_box.get("has_content") and hel_box.get("ready"):
                    plan = (
                        f"Your TempBox at {_fmt_xy(hel_box.get('xy'))} already has items. "
                        f"If the list is complete ({buy_txt}), REPORT_HELP_FINISHED. "
                        f"Otherwise buy missing items and update the box."
                    )
                else:
                    plan = (
                        f"Go to a store and buy {buy_txt}. "
                        f"Then deliver to {dst_txt}. "
                        f"On arrival, PLACE_TEMP_BOX with the purchased items and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                if hel_box.get("has_content"):
                    plan = (
                        f"Helper has dropped the purchased items at {_fmt_xy(hel_box.get('xy'))}. "
                        f"You can TAKE_FROM_TEMP_BOX when convenient."
                    )
                else:
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (waiting for a helper)"
                    plan = (
                        f"Helper will buy {buy_txt} and drop at {dst_txt}{who}. "
                        f"No action required from you until they place the TempBox."
                    )
            else:
                plan = f"Buy {buy_txt} and deliver to {dst_txt}. PLACE_TEMP_BOX on arrival, then REPORT_HELP_FINISHED."

        elif self.kind == HelpType.HELP_CHARGE:
            tgt = int(self.target_pct if self.target_pct is not None else 100)
            if am_helper:
                if pub_box.get("has_content"):
                    plan = (
                        f"Pick the e-scooter from the help-seeker's TempBox at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Charge to {tgt}% at a charging station, then deliver to {dst_txt}. "
                        f"PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) and REPORT_HELP_FINISHED."
                    )
                elif pub_box.get("xy"):
                    # NEW: if the timer already started but the box is empty, the scooter was placed and has been taken.
                    if self.timer_started_sim is not None:
                        sc = comms.get_scooter_by_owner(self.publisher_id)
                        cur = int(round(float(getattr(sc, "battery_pct", 0.0)))) if sc is not None else None
                        if sc and cur is not None and cur >= tgt:
                            loc_txt = _fmt_xy(getattr(sc, "park_xy", None)) if getattr(sc, "park_xy", None) else "your current location"
                            plan = (
                                f"You have taken the e-scooter. It's charged to {cur}% and parked at {loc_txt}. "
                                f"Deliver to {dst_txt}, PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) "
                                f"and REPORT_HELP_FINISHED."
                            )
                        else:
                            cur_txt = f" (current ~{cur}%)" if cur is not None else ""
                            plan = (
                                f"You have taken the e-scooter from the handover point {_fmt_xy(pub_box.get('xy'))}. "
                                f"Go charge to {tgt}%{cur_txt}, then deliver to {dst_txt}. "
                                f"PLACE_TEMP_BOX(req_id={self.req_id}, content={{'escooter':''}}) and REPORT_HELP_FINISHED."
                            )
                    else:
                        plan = (
                            f"Handover TempBox set at {_fmt_xy(pub_box.get('xy'))} but empty. "
                            f"Wait for the help-seeker to place the scooter, then charge to {tgt}%, "
                            f"deliver to {dst_txt}, PLACE_TEMP_BOX and REPORT_HELP_FINISHED."
                        )
                else:
                    plan = (
                        f"Help-seeker hasn't placed the scooter yet. Handover point: {provide_txt}. "
                        f"Once picked, charge to {tgt}% at a charging station, deliver to {dst_txt}, "
                        f"PLACE_TEMP_BOX and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                if pub_box.get("has_content"):
                    plan = (
                        f"You placed the scooter at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Waiting for the helper to charge it to {tgt}% and drop at {dst_txt}."
                    )
                else:
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                    plan = (
                        f"Please place your e-scooter at the handover point {provide_txt} (TempBox){who} for the helper. "
                        f"Target charge {tgt}%, drop at {dst_txt}."
                    )
            else:
                plan = f"Charge the e-scooter to {tgt}%, deliver to {dst_txt}. PLACE_TEMP_BOX, then REPORT_HELP_FINISHED."

        elif self.kind == HelpType.HELP_DELIVERY:
            if am_helper:
                if pub_box.get("has_content"):
                    plan = (
                        f"Pick up the food from the help-seeker's TempBox at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Deliver to {dst_txt}. After auto-delivery, REPORT_HELP_FINISHED."
                    )
                elif pub_box.get("xy"):
                    if self.timer_started_sim is not None:
                        plan = (
                            f"You have taken the food from {_fmt_xy(pub_box.get('xy'))}. "
                            f"Deliver to {dst_txt}. After auto-delivery, REPORT_HELP_FINISHED."
                        )
                    else:
                        plan = (
                            f"TempBox set at {_fmt_xy(pub_box.get('xy'))} but empty. "
                            f"Wait for the help-seeker to place the food, then deliver to {dst_txt} and REPORT_HELP_FINISHED."
                        )
                else:
                    plan = (
                        f"Help-seeker hasn't placed the food yet. Handover point: {provide_txt}. "
                        f"Once picked, deliver to {dst_txt} and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                need_pickup_first = False
                try:
                    if self.order_ref is not None:
                        has_picked = bool(getattr(self.order_ref, "has_picked_up", False))
                        has_delivered = bool(getattr(self.order_ref, "has_delivered", False))
                        need_pickup_first = (not has_picked) and (not has_delivered)
                except Exception:
                    pass

                if pub_box.get("has_content"):
                    plan = (
                        f"You placed the food at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Waiting for the helper to deliver to {dst_txt}."
                    )
                else:
                    prefix = ""
                    if need_pickup_first and self.order_id is not None:
                        prefix = f"First PICK UP order #{int(self.order_id)} at the restaurant, then "
                    who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                    plan = (
                        f"{prefix}place the food at the handover point {provide_txt} (TempBox){who}. "
                        f"The helper will take it and deliver to {dst_txt}."
                    )
            else:
                plan = f"Handover at {provide_txt}. Deliver to {dst_txt}. REPORT_HELP_FINISHED after dropoff."

        elif self.kind == HelpType.HELP_PICKUP:
            if am_helper:
                if pub_box.get("has_content"):
                    plan = (
                        f"Items available at {_fmt_xy(pub_box.get('xy'))}. "
                        f"Deliver to {dst_txt}. PLACE_TEMP_BOX on arrival and REPORT_HELP_FINISHED."
                    )
                else:
                    oid_txt = f"order #{int(self.order_id)}" if self.order_id is not None else "the order"
                    plan = (
                        f"Go to the restaurant and pick up {oid_txt}, then deliver to {dst_txt}. "
                        f"On arrival, PLACE_TEMP_BOX with the food and REPORT_HELP_FINISHED."
                    )
            elif am_publisher:
                who = f" (accepted by agent {self.accepted_by})" if self.accepted_by else " (on board)"
                plan = (
                    f"Helper will pick up the order and drop at {dst_txt}{who}. "
                    f"You can TAKE_FROM_TEMP_BOX when it's ready."
                )
            else:
                plan = f"Pick up the order and deliver to {dst_txt}. PLACE_TEMP_BOX at destination, then REPORT_HELP_FINISHED."
        else:
            plan = f"Destination: {dst_txt}."

        return f"{head} {plan_intro}{plan}"

# ===== Comms System =====
class CommsSystem:
    """
    - Help board: post / accept / modify
    - TempBox: one per role (publisher/helper) per request; used for handoff
    - Settlement: escrow bounty, upfront costs, 5-minute grace (half pay), refunds
    - Timing: DELIVERY/CHARGE start when publisher places items into TempBox; BUY/PICKUP start on accept
    - Thinking pause: pause_timers_for / resume_timers_for (per helper)
    """
    def __init__(self, clock: Optional[VirtualClock] = None):
        self.clock = clock if clock is not None else VirtualClock()
        self._lock = RLock()  # re-entrant to avoid deadlocks

        # help board pools
        self._next_req_id = 1
        self._board: Dict[int, HelpRequest] = {}     # open
        self._active: Dict[int, HelpRequest] = {}    # accepted but not finished
        self._completed: Dict[int, HelpRequest] = {} # finished

        # temp boxes
        self._next_box_id = 1
        self._boxes: Dict[int, TempBox] = {}
        self._role_index: Dict[Tuple[int, str], int] = {}  # (req_id, role)->box_id

        # agents registry
        self._agents: Dict[str, Any] = {}

        # publisher notifications
        self._msgs_for_publisher: Dict[str, List[Dict[str, Any]]] = {}

        self._scooters_by_owner: Dict[str, Any] = {}

        # distance tolerance (cm)
        self._tol_cm_default = 300.0

        # price fallbacks
        self._DEFAULT_PRICE_TABLE = {
            "energy_drink": 2.0,
            "escooter_battery_pack": 10.0,
        }
        self._DEFAULT_CHARGE_PRICE_PER_PERCENT = 0.5

    # ----- Agent registry -----
    def register_agent(self, dm: Any):
        with self._lock:
            self._agents[str(dm.agent_id)] = dm

    # ----- Helpers -----
    def _log_agent(self, agent_id: str, text: str):
        dm = self._agents.get(str(agent_id))
        if dm is not None and hasattr(dm, "_log"):
            dm._log(text)
        else:
            print(f"[Comms] agent {agent_id}: {text}")

    def _within(self, xy: Optional[Tuple[float, float]],
                target_xy: Optional[Tuple[float, float]],
                tol_cm: float) -> bool:
        if xy is None or target_xy is None:
            return False
        dx = float(xy[0]) - float(target_xy[0])
        dy = float(xy[1]) - float(target_xy[1])
        return (math.hypot(dx, dy) <= float(tol_cm))

    # ----- Timer helpers -----
    def _start_timer_if_needed(self, req: HelpRequest, reason: str):
        if req.timer_started_sim is None:
            req.timer_started_sim = float(self.clock.now_sim())
            req.start_note = str(reason)

    def _effective_elapsed_s(self, req: HelpRequest, now: Optional[float] = None) -> float:
        if req.timer_started_sim is None:
            return 0.0
        now = self.clock.now_sim() if now is None else float(now)
        paused = float(req.timer_paused_accum_s or 0.0)
        if req.timer_paused_at is not None:
            paused += max(0.0, now - float(req.timer_paused_at))
        return max(0.0, now - float(req.timer_started_sim) - paused)

    def _time_left_s(self, req: HelpRequest, now: Optional[float] = None) -> Optional[float]:
        if req.timer_started_sim is None:
            return None
        now = self.clock.now_sim() if now is None else float(now)
        used = self._effective_elapsed_s(req, now)
        return max(0.0, float(req.time_limit_s) - used)

    def pause_timers_for(self, helper_id: str):
        with self._lock:
            now = self.clock.now_sim()
            for r in self._active.values():
                if r.accepted_by == str(helper_id) and r.timer_started_sim is not None and r.timer_paused_at is None:
                    r.timer_paused_at = float(now)

    def resume_timers_for(self, helper_id: str):
        with self._lock:
            now = self.clock.now_sim()
            for r in self._active.values():
                if r.accepted_by == str(helper_id) and r.timer_paused_at is not None:
                    r.timer_paused_accum_s += max(0.0, float(now) - float(r.timer_paused_at))
                    r.timer_paused_at = None

    # ----- Pricing helpers (upfront estimation) -----
    def _any_store_manager(self):
        for dm in self._agents.values():
            sm = getattr(dm, "_store_manager", None)
            if sm is not None:
                return sm
        return None

    def _estimate_buy_cost(self, buy_items: Optional[Dict[str, int]]) -> float:
        if not buy_items:
            return 0.0
        sm = self._any_store_manager()
        total = 0.0
        for item_id, qty in buy_items.items():
            q = int(qty or 0)
            if q <= 0:
                continue
            unit = 0.0
            if sm:
                if hasattr(sm, "get_price"):
                    try:
                        unit = float(sm.get_price(item_id))
                    except Exception:
                        unit = 0.0
                elif hasattr(sm, "get_unit_price"):
                    try:
                        unit = float(sm.get_unit_price(item_id))
                    except Exception:
                        unit = 0.0
            if unit <= 0.0:
                unit = float(self._DEFAULT_PRICE_TABLE.get(str(item_id), 0.0))
            total += unit * q
        return float(total)

    def _estimate_upfront_cost(self, req: HelpRequest) -> float:
        if req.kind == HelpType.HELP_BUY:
            return self._estimate_buy_cost(req.buy_items)
        if req.kind == HelpType.HELP_CHARGE:
            tgt = int(req.target_pct or 100)
            return float(tgt) * float(self._DEFAULT_CHARGE_PRICE_PER_PERCENT)
        return 0.0

    # ----- Listing -----
    def list_open_requests(self) -> List[HelpRequest]:
        with self._lock:
            return list(self._board.values())

    def list_my_posts_open(self, publisher_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._board.values() if r.publisher_id == str(publisher_id)]

    def list_my_posts_active(self, publisher_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._active.values() if r.publisher_id == str(publisher_id)]

    def list_my_helps(self, helper_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._active.values() if r.accepted_by == str(helper_id)]

    def list_my_helps_completed(self, helper_id: str) -> List[HelpRequest]:
        with self._lock:
            return [r for r in self._completed.values() if r.accepted_by == str(helper_id)]

    # ----- Post / Accept / Modify -----
    def _auto_brief(self, kind: HelpType, order_id, buy_items, target_pct) -> str:
        if kind == HelpType.HELP_DELIVERY:
            return f"Deliver order #{order_id}."
        if kind == HelpType.HELP_PICKUP:
            return f"Pick up order #{order_id}."
        if kind == HelpType.HELP_BUY:
            return "Buy items."
        if kind == HelpType.HELP_CHARGE:
            tgt = int(target_pct or 100)
            return f"Charge e-scooter to {tgt}%."
        return "Help request."

    def post_request(
        self,
        publisher_id: str,
        kind: HelpType,
        *,
        reward: float,
        time_limit_s: float,
        order_id: Optional[int] = None,
        buy_items: Optional[Dict[str, int]] = None,
        target_pct: Optional[float] = None,
        brief: Optional[str] = None,
        location_xy: Optional[Tuple[float, float]] = None,
        provide_xy: Optional[Tuple[float, float]] = None,
        deliver_xy: Optional[Tuple[float, float]] = None,
        order_ref: Optional[Any] = None,
    ) -> Tuple[bool, str, Optional[int]]:
        now = self.clock.now_sim()
        with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1

            if deliver_xy is None:
                deliver_xy = location_xy  # backward compatibility

            req = HelpRequest(
                req_id=rid,
                publisher_id=str(publisher_id),
                kind=HelpType(kind),
                reward=float(reward),
                time_limit_s=float(time_limit_s),
                created_sim=float(now),
                order_id=int(order_id) if order_id is not None else None,
                buy_items=dict(buy_items) if buy_items else None,
                target_pct=float(target_pct) if target_pct is not None else None,
                brief=str(brief) if brief else self._auto_brief(kind, order_id, buy_items, target_pct),
                provide_xy=(tuple(provide_xy) if provide_xy else None),
                deliver_xy=(tuple(deliver_xy) if deliver_xy else None),
                order_ref=order_ref,
            )
            self._board[rid] = req
            return True, "posted", rid

    def accept_request(self, req_id: int, helper_id: str) -> Tuple[bool, str]:
        with self._lock:
            req = self._board.get(int(req_id))
            if not req:
                return False, "not_on_board"
            pub = self._agents.get(str(req.publisher_id))
            hel = self._agents.get(str(helper_id))
            if pub is None or hel is None:
                return False, "agent_not_registered"
            if pub.agent_id == hel.agent_id:
                return False, "cannot_accept_own_request"

            # upfront estimation (BUY/CHARGE) and total needed: escrow bounty + upfront
            upfront = self._estimate_upfront_cost(req)
            need_total = float(req.reward) + float(upfront)

            # funds check (publisher)
            pub_balance = float(getattr(pub, "earnings_total", 0.0))
            if pub_balance < need_total - 1e-9:
                shortfall = need_total - pub_balance
                # notify both sides; do not accept
                self._log_agent(str(helper_id),
                    f"[Help #{req.req_id}] Accept failed: publisher (agent {req.publisher_id}) insufficient funds. "
                    f"need=${need_total:.2f} (escrow=${req.reward:.2f} + upfront=${upfront:.2f}), "
                    f"has=${pub_balance:.2f}, shortfall=${shortfall:.2f}."
                )
                self._log_agent(str(req.publisher_id),
                    (f"[Help #{req.req_id}] Insufficient funds; the request cannot be accepted. "
                    f"Required pre-hold=${need_total:.2f} (escrow=${req.reward:.2f} + upfront=${upfront:.2f}), "
                    f"current balance=${pub_balance:.2f}, shortfall=${shortfall:.2f}.")
                )

                return False, "publisher_insufficient_funds"

            # deduct/transfer: publisher pays escrow + upfront; helper receives upfront immediately
            pub_before = float(pub.earnings_total)
            hel_before = float(hel.earnings_total)

            pub.earnings_total -= need_total
            hel.earnings_total += float(upfront)

            pub_after = float(pub.earnings_total)
            hel_after = float(hel.earnings_total)

            # bookkeeping
            req.escrow_reward = float(req.reward)
            req.upfront_paid_to_helper = float(upfront)
            req.accepted_by = str(helper_id)

            # logs: money only
            self._log_agent(str(req.publisher_id),
                (f"[Help #{req.req_id}] Funds held: escrow=${req.reward:.2f} + upfront=${upfront:.2f} "
                f"= total ${need_total:.2f}. Balance: ${pub_before:.2f} -> ${pub_after:.2f}.")
            )
            self._log_agent(str(helper_id),
                (f"[Help #{req.req_id}] Upfront received=${upfront:.2f}. "
                f"Balance: ${hel_before:.2f} -> ${hel_after:.2f}.")
            )


            # BUY / PICKUP: timer starts on accept (仅记录逻辑，不做资金日志)
            if req.kind in (HelpType.HELP_BUY, HelpType.HELP_PICKUP):
                self._start_timer_if_needed(req, reason="start_on_accept")

            # move to active
            self._board.pop(int(req_id), None)
            self._active[int(req_id)] = req
            return True, "accepted"

    def modify_request(
        self,
        publisher_id: str,
        req_id: int,
        *,
        reward: Optional[float] = None,
        time_limit_s: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Modify reward/ttl if open; if active, only ttl can be modified."""
        with self._lock:
            req = self._board.get(int(req_id))
            if req and req.publisher_id == str(publisher_id):
                if reward is not None:
                    req.reward = float(reward)
                if time_limit_s is not None:
                    req.time_limit_s = float(time_limit_s)
                return True, "modified"
            req = self._active.get(int(req_id))
            if req and req.publisher_id == str(publisher_id):
                if reward is not None:
                    return False, "cannot_modify_reward_after_accept"
                if time_limit_s is not None:
                    req.time_limit_s = float(time_limit_s)
                    return True, "modified_ttl"
                return False, "no_change"
            return False, "not_found_or_not_owner"

    # ----- Scooter -----
    def register_scooter(self, owner_id: str, scooter: Any) -> Any:
        """Register/lock-in the canonical scooter instance for this owner and return it."""
        with self._lock:
            key = str(owner_id)
            canon = self._scooters_by_owner.get(key)
            if canon is None:
                self._scooters_by_owner[key] = scooter
                return scooter
            return canon  # already have a canonical instance

    def get_scooter_by_owner(self, owner_id: str) -> Optional[Any]:
        with self._lock:
            return self._scooters_by_owner.get(str(owner_id))

    def _canonicalize_escooter(self, sc: Optional[Any]) -> Optional[Any]:
        """Make sure we always store/return the canonical instance for the scooter's owner."""
        if sc is None:
            return None
        owner = getattr(sc, "owner_id", None)
        if not owner:
            return sc
        with self._lock:
            key = str(owner)
            canon = self._scooters_by_owner.get(key)
            if canon is None:
                # first time seeing this scooter -> remember this instance as canonical
                self._scooters_by_owner[key] = sc
                return sc
            return canon

    # ----- TempBox -----
    def _get_or_create_box(self, req_id: int, role: str, xy: Tuple[float, float]) -> TempBox:
        key = (int(req_id), str(role))
        box_id = self._role_index.get(key)
        if box_id is None:
            box_id = self._next_box_id
            self._next_box_id += 1
            box = TempBox(
                box_id=box_id,
                req_id=int(req_id),
                owner_id="",
                role=str(role),
                xy=tuple(xy),
                created_sim=float(self.clock.now_sim()),
            )
            self._boxes[box_id] = box
            self._role_index[key] = box_id
        else:
            box = self._boxes[box_id]
            box.xy = tuple(xy)  # update to latest location
        return box

    def place_temp_box(self, req_id: int, by_agent: str,
                       location_xy: Tuple[float, float],
                       content: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lock:
            req = self._active.get(int(req_id)) or self._board.get(int(req_id))
            if not req:
                return False, "request_not_found"

            role = "publisher" if str(by_agent) == str(req.publisher_id) else "helper"
            
            if "escooter" in (content or {}):
                content = dict(content)  # shallow copy
                content["escooter"] = self._canonicalize_escooter(content["escooter"])
            
            box = self._get_or_create_box(int(req_id), role, tuple(location_xy))
            box.owner_id = str(by_agent)
            box.merge_payload(content or {})

            # trigger timer:
            # DELIVERY -> start when publisher places food (has food content or compatible short form)
            # CHARGE   -> start when publisher places scooter
            by_publisher = (str(by_agent) == str(req.publisher_id))
            if by_publisher:
                if req.kind == HelpType.HELP_DELIVERY and (
                    content.get("food_by_order")
                    or ("order_id" in content and "food_items" in content)
                    or ("food" in content)  # compatibility with your short form
                ):
                    self._start_timer_if_needed(req, reason="publisher_placed_food")
                elif req.kind == HelpType.HELP_CHARGE and ("escooter" in content):
                    self._start_timer_if_needed(req, reason="publisher_placed_escooter")

            return True, "placed"

    def take_from_temp_box(self, req_id: int, by_agent: str):
        """Take from the opposite role's box: helper takes from publisher_box; publisher takes from helper_box."""
        with self._lock:
            req = self._active.get(int(req_id)) or self._board.get(int(req_id))
            if not req:
                return False, "request_not_found", None

            target_role = "publisher" if str(by_agent) != str(req.publisher_id) else "helper"
            key = (int(req_id), target_role)
            box_id = self._role_index.get(key)
            if box_id is None or box_id not in self._boxes:
                return False, "box_not_found", None

            box = self._boxes[box_id]
            sc = self._canonicalize_escooter(box.escooter)
            payload = dict(
                inventory=dict(box.inventory),
                food_by_order={int(k): list(v) for k, v in box.food_by_order.items()},
                escooter=sc,
                xy=box.xy,
            )
            # clear content (box stays)
            box.inventory.clear()
            box.food_by_order.clear()
            box.escooter = None
            return True, "ok", payload

    def get_temp_box_info(self, req_id: int) -> Dict[str, Any]:
        """Return lightweight TempBox status for UI/copy."""
        with self._lock:
            out = {"publisher_box": {}, "helper_box": {}}
            for role in ("publisher", "helper"):
                key = (int(req_id), role)
                bid = self._role_index.get(key)
                if bid and bid in self._boxes:
                    b = self._boxes[bid]
                    has_content = bool(b.escooter or b.inventory or any(b.food_by_order.values()))
                    out[f"{role}_box"] = {
                        "xy": b.xy,
                        "has_content": has_content,
                        "ready": has_content,  # simple heuristic: any content => ready
                        "box_id": bid,
                    }
            return out

    # ----- Finish / Settlement -----
    def push_helper_delivered(self, req_id: int, by_agent: str, order_id: int, at_xy: Tuple[float, float]) -> Tuple[bool, str]:
        """Helper reached destination and dropped off. Publisher will auto-settle in their loop; finalization still requires REPORT_HELP_FINISHED."""
        with self._lock:
            req = self._active.get(int(req_id))
            if not req:
                return False, "not_active"
            if str(req.accepted_by) != str(by_agent):
                return False, "not_helper"
            self._msgs_for_publisher.setdefault(str(req.publisher_id), []).append({
                "type": "HELP_DELIVERY_DONE",
                "req_id": int(req.req_id),
                "order_id": int(order_id),
                "at_xy": tuple(at_xy),
            })
            return True, "ok"

    def pop_msgs_for_publisher(self, publisher_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            arr = self._msgs_for_publisher.pop(str(publisher_id), [])
            return list(arr or [])

    def report_help_finished(self, req_id: int, by_agent: str, at_xy: Tuple[float, float]):
        with self._lock:
            req = self._active.get(int(req_id))
            if not req:
                return False, "not_active", None
            if str(req.accepted_by) != str(by_agent):
                return False, "not_helper", None

            now = self.clock.now_sim()

            # effective elapsed time
            if req.timer_started_sim is None:
                elapsed = 0.0
            else:
                elapsed = self._effective_elapsed_s(req, now)

            # status: on time / within 5-min grace / fail
            limit = float(req.time_limit_s or 0.0)
            grace = 300.0  # 5 minutes
            if req.timer_started_sim is None:
                status = "success"
            elif elapsed <= limit + 1e-9:
                status = "success"
            elif elapsed <= limit + grace + 1e-9:
                status = "partial"
            else:
                status = "fail"

            pub = self._agents.get(str(req.publisher_id))
            hel = self._agents.get(str(req.accepted_by or ""))
            if pub is None or hel is None:
                return False, "agent_not_found", None

            # snapshot before settlement for logging
            pub_before = float(getattr(pub, "earnings_total", 0.0))
            hel_before = float(getattr(hel, "earnings_total", 0.0))

            # escrow settlement
            escrow = float(req.escrow_reward or 0.0)
            if status == "success":
                hel.earnings_total += escrow
            elif status == "partial":
                hel.earnings_total += escrow * 0.5
                pub.earnings_total += escrow * 0.5
            else:  # fail
                pub.earnings_total += escrow

            # upfront: success/partial -> helper keeps; fail -> claw back to publisher
            upfront = float(req.upfront_paid_to_helper or 0.0)
            if status == "fail" and upfront > 0.0:
                hel.earnings_total -= upfront
                pub.earnings_total += upfront

            # snapshot after settlement for logging
            pub_after = float(getattr(pub, "earnings_total", 0.0))
            hel_after = float(getattr(hel, "earnings_total", 0.0))

            # logs: money only,分别告知双方
            if status == "success":
                self._log_agent(str(req.publisher_id),
                    (f"[Help #{req.req_id}] Settlement: success. Escrow ${escrow:.2f} paid to helper (agent {req.accepted_by}). "
                    f"Your balance: ${pub_before:.2f} -> ${pub_after:.2f}.")
                )
                self._log_agent(str(req.accepted_by or ""),
                    (f"[Help #{req.req_id}] Settlement: success. Received escrow=${escrow:.2f}; "
                    f"upfront=${upfront:.2f} kept. Your balance: ${hel_before:.2f} -> ${hel_after:.2f}.")
                )
            elif status == "partial":
                self._log_agent(str(req.publisher_id),
                    (f"[Help #{req.req_id}] Settlement: partial. 50% of escrow (${escrow*0.5:.2f}) returned to you. "
                    f"Your balance: ${pub_before:.2f} -> ${pub_after:.2f}.")
                )
                self._log_agent(str(req.accepted_by or ""),
                    (f"[Help #{req.req_id}] Settlement: partial. Received 50% of escrow (${escrow*0.5:.2f}); "
                    f"upfront=${upfront:.2f} kept. Your balance: ${hel_before:.2f} -> ${hel_after:.2f}.")
                )
            else:  # fail
                self._log_agent(str(req.publisher_id),
                    (f"[Help #{req.req_id}] Settlement: fail. Full escrow (${escrow:.2f}) returned; "
                    f"upfront (${upfront:.2f}) clawed back from helper. Your balance: ${pub_before:.2f} -> ${pub_after:.2f}.")
                )
                self._log_agent(str(req.accepted_by or ""),
                    (f"[Help #{req.req_id}] Settlement: fail. Escrow ${escrow:.2f} returned to publisher; "
                    f"upfront ${upfront:.2f} clawed back. Your balance: ${hel_before:.2f} -> ${hel_after:.2f}.")
                )

            # finalize
            req.completed = True
            req.result = status
            self._active.pop(int(req_id), None)
            self._completed[int(req_id)] = req

            # notify publisher
            self._msgs_for_publisher.setdefault(str(req.publisher_id), []).append({
                "type": "HELP_FINISHED",
                "req_id": int(req.req_id),
                "status": status,
                "at_xy": tuple(at_xy),
            })
            return True, "ok", {"status": status}

    # ----- Query -----
    def get_request(self, req_id: int) -> Optional[HelpRequest]:
        with self._lock:
            return self._board.get(int(req_id)) or self._active.get(int(req_id)) or self._completed.get(int(req_id))

    def get_request_detail(self, req_id: int) -> Dict[str, Any]:
        with self._lock:
            r = self._board.get(int(req_id)) or self._active.get(int(req_id)) or self._completed.get(int(req_id))
            if not r:
                return {}
            return {
                "id": int(r.req_id),
                "req_id": int(r.req_id),
                "publisher_id": str(r.publisher_id),
                "accepted_by": r.accepted_by,
                "kind": r.kind.value,
                "reward": float(r.reward),
                "time_limit_s": float(r.time_limit_s),
                "order_id": r.order_id,
                "buy_items": dict(r.buy_items) if r.buy_items else None,
                "target_pct": r.target_pct,
                "provide_xy": tuple(r.provide_xy) if r.provide_xy else None,
                "deliver_xy": tuple(r.deliver_xy) if r.deliver_xy else None,
                "order_ref": r.order_ref,
                "completed": bool(r.completed),
                "result": r.result,
                "timer_started_sim": r.timer_started_sim,
                "escrow_reward": r.escrow_reward,
                "upfront_paid_to_helper": r.upfront_paid_to_helper,
            }

    # ----- Board text -----
    def board_to_text(
        self,
        *,
        include_active: bool = False,
        include_completed: bool = False,
        max_items: int = 50,
        exclude_publisher: Optional[str] = None,
        viewer_id: Optional[str] = None,
    ) -> str:
        """
        Build help-board text in English for agents.
        - By default lists open requests only.
        - exclude_publisher: hide requests posted by this agent.
        NOTE: snapshot inside lock, render outside to avoid re-entrant locking.
        """
        def _one(req: HelpRequest) -> str:
            # use neutral perspective; to_text_for already embeds agent IDs and timer rules
            return req.to_text_for(viewer_id or "", self, view_as=None)

        # snapshots under lock
        with self._lock:
            opens = list(self._board.values())
            act = list(self._active.values()) if include_active else []
            fin = list(self._completed.values()) if include_completed else []

        # filter & render outside lock
        if exclude_publisher:
            opens = [r for r in opens if r.publisher_id != str(exclude_publisher)]
            act   = [r for r in act   if r.publisher_id != str(exclude_publisher)]
            fin   = [r for r in fin   if r.publisher_id != str(exclude_publisher)]

        lines: List[str] = []
        if opens:
            lines.append("=== OPEN REQUESTS ===")
            for r in opens[:max_items]:
                lines.append(_one(r))
        if include_active and act:
            lines.append("=== ACTIVE REQUESTS ===")
            for r in act[:max_items]:
                lines.append(_one(r))
        if include_completed and fin:
            lines.append("=== COMPLETED REQUESTS ===")
            for r in fin[:max_items]:
                status = r.result or "unknown"
                lines.append(_one(r) + f" [completed: {status}]")

        return "\n".join(lines) if lines else "(empty help board)"

# ==== Module-level singleton helpers ====
_comms_singleton_lock = Lock()
_comms_singleton: Optional[CommsSystem] = None

def init_comms(clock: Optional[VirtualClock] = None) -> CommsSystem:
    """Create the global CommsSystem once (or return existing)."""
    global _comms_singleton
    with _comms_singleton_lock:
        if _comms_singleton is None:
            _comms_singleton = CommsSystem(clock=clock)
    return _comms_singleton

def set_comms(instance: Optional[CommsSystem]) -> None:
    """Explicitly set/replace the global CommsSystem (mainly for tests)."""
    global _comms_singleton
    with _comms_singleton_lock:
        _comms_singleton = instance

def get_comms() -> Optional[CommsSystem]:
    """Get the global CommsSystem (may be None if not initialized)."""
    return _comms_singleton

def is_comms_ready() -> bool:
    return _comms_singleton is not None

__all__ = [
    "HelpType", "TempBox", "HelpRequest", "CommsSystem",
    "init_comms", "get_comms", "set_comms", "is_comms_ready"
]