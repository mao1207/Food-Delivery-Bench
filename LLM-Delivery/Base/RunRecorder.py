# -*- coding: utf-8 -*-
# Base/RunRecorder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json, time

def _now_wall_ts() -> float:
    return time.time()

def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else float(a) / float(b)

@dataclass
class MoneyFlow:
    purchases: List[Dict[str, Any]] = field(default_factory=list)     # {ts_sim, items:str, cost}
    charging:  List[Dict[str, Any]] = field(default_factory=list)     # {ts_sim, which, req_id, delta_pct, cost}
    rental:    List[Dict[str, Any]] = field(default_factory=list)     # {ts_sim, dt_s, cost}
    hospital:  List[Dict[str, Any]] = field(default_factory=list)     # {ts_sim, fee}
    help_income: List[Dict[str, Any]] = field(default_factory=list)   # {ts_sim, req_id, kind, amount}

    help_expense: List[Dict[str, Any]] = field(default_factory=list)  # {ts_sim, req_id, kind, amount}
    # “垫付”两条流水（先记出，再记回收；净额 = out - returned）
    advance_out:     List[Dict[str, Any]] = field(default_factory=list)   # {ts_sim, req_id, category, amount}
    advance_return:  List[Dict[str, Any]] = field(default_factory=list)   # {ts_sim, req_id, amount}

    def sum_purchases(self) -> float:  return float(sum(x.get("cost", 0.0) for x in self.purchases))
    def sum_charging(self)  -> float:  return float(sum(x.get("cost", 0.0) for x in self.charging))
    def sum_rental(self)    -> float:  return float(sum(x.get("cost", 0.0) for x in self.rental))
    def sum_hospital(self)  -> float:  return float(sum(x.get("fee",  0.0) for x in self.hospital))
    def sum_help_income(self)-> float:  return float(sum(x.get("amount",0.0) for x in self.help_income))
    def sum_help_expense(self)-> float: return float(sum(x.get("amount",0.0) for x in self.help_expense))
    def sum_advance_out(self)-> float:  return float(sum(x.get("amount",0.0) for x in self.advance_out))
    def sum_advance_ret(self)-> float:  return float(sum(x.get("amount",0.0) for x in self.advance_return))

@dataclass
class Counters:
    # 中断/事件计数
    scooter_depleted: int = 0
    hospital_rescue:  int = 0
    rent_insufficient: int = 0
    charge_insufficient: int = 0

    preventive_actions: int = 0                           # 预防行为总次数
    preventive_by_kind: Dict[str, int] = field(default_factory=dict)  # 各类预防细分

    # 社会化
    help_posted:   int = 0
    help_accepted: int = 0
    help_given:    int = 0   # 我作为 helper 完成（push 给对方）
    help_received: int = 0   # 我作为 publisher 收到 helper 完成
    say          : int = 0   # 我发出的 chat 数量
    
    # 新增：动作统计
    action_attempts: Dict[str, int] = field(default_factory=dict)  # 动作尝试次数
    action_successes: Dict[str, int] = field(default_factory=dict)  # 动作成功次数
    
    # 新增：VLM统计
    vlm_calls: int = 0
    vlm_successes: int = 0
    vlm_parse_failures: int = 0
    vlm_retries: int = 0

@dataclass
class RunRecorder:
    agent_id: str
    lifecycle_s: float
    export_path: str
    initial_balance: float = 0.0
    model: str = ""

    # 内部时钟与累积
    started_sim_s: Optional[float] = None
    ended_sim_s:   Optional[float] = None
    active_elapsed_s: float = 0.0  # 仅在"未暂停"时累加（由外部驱动）
    done: bool = False
    
    # 现实时间停止支持
    realtime_stop_hours: float = 0.0
    realtime_start_ts: Optional[float] = None
    
    # VLM call 次数限制
    vlm_call_limit: int = 0

    # 订单/活动统计（导出时也会从 dm.completed_orders 再扫一遍补全）
    # 这里保持轻量，避免和主逻辑重叠
    counters: Counters = field(default_factory=Counters)
    money: MoneyFlow   = field(default_factory=MoneyFlow)
    # 交通方式累计时长（仅在未暂停时累加，单位：秒）
    transport_time_s: Dict[str, float] = field(default_factory=dict)

    # 新增：无效时间累计（等待/排队/阻塞等，单位：秒）
    inactive_time_s: Dict[str, float] = field(default_factory=dict)

    # —— 能量消耗累计（百分比，按“扣减后的实际消耗”统计）——
    energy_personal_consumed_pct: float = 0.0
    scooter_batt_consumed_pct: float = 0.0

    # ===== 内部会话累加器（仅内存，不直接写 details）=====
    _charging_acc: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _rental_acc:   Optional[Dict[str, Any]] = field(default=None, repr=False)

    # —— 事件接口（在 DM 各位置调用）——
    def start(self, now_sim: float, realtime_start_ts: Optional[float] = None):
        if self.started_sim_s is None:
            self.started_sim_s = float(now_sim)
        if realtime_start_ts is not None:
            self.realtime_start_ts = float(realtime_start_ts)

    def tick_active(self, delta_s: float):
        if delta_s > 0:
            self.active_elapsed_s += float(delta_s)

    def tick_transport(self, mode: str, delta_s: float):
        """按交通方式累计活跃时间（秒）。"""
        if delta_s <= 0:
            return
        key = str(mode)
        self.transport_time_s[key] = float(self.transport_time_s.get(key, 0.0) + float(delta_s))

    def tick_inactive(self, kind: str, delta_s: float):
        """
        记录无效时间：如等待(prepare/wait)、排队(queue)、阻塞(blocked)、暂停(paused)等。
        使用方式：在对应逻辑触发点调用，如 self.recorder.tick_inactive("wait", dt)
        """
        if delta_s <= 0:
            return
        key = str(kind)
        self.inactive_time_s[key] = float(self.inactive_time_s.get(key, 0.0) + float(delta_s))

    def should_end(self) -> bool:
        if self.done:
            return False
        
        # 检查虚拟时间停止
        sim_time_end = (self.lifecycle_s > 0) and (self.active_elapsed_s >= self.lifecycle_s)
        
        # 检查现实时间停止
        realtime_end = False
        if self.realtime_stop_hours > 0 and self.realtime_start_ts is not None:
            current_realtime = time.time()
            elapsed_realtime_hours = (current_realtime - self.realtime_start_ts) / 3600.0
            realtime_end = elapsed_realtime_hours >= self.realtime_stop_hours
        
        # 检查 VLM call 次数限制
        vlm_call_end = (self.vlm_call_limit > 0) and (self.counters.vlm_calls >= self.vlm_call_limit)
        
        return sim_time_end or realtime_end or vlm_call_end

    def mark_end(self, now_sim: float):
        self.ended_sim_s = float(now_sim)
        self.done = True

    # —— 记账接口（被动调用，不改主逻辑的扣款/加钱，只做记录）——
    def on_purchase(self, ts_sim: float, items: str, cost: float):
        self.money.purchases.append(dict(ts_sim=float(ts_sim), items=items, cost=float(cost)))

    def on_charge_payment(self, ts_sim: float, which: str, delta_pct: float, cost: float, req_id: Optional[int] = None):
        """老接口：直接 append 一条明细（不建议在 tick 内频繁使用）。"""
        self.money.charging.append(dict(ts_sim=float(ts_sim), which=str(which), req_id=(int(req_id) if req_id else None),
                                        delta_pct=float(delta_pct), cost=float(cost)))

    def on_rental_billing(self, ts_sim: float, dt_s: float, cost: float):
        """老接口：直接 append 一条明细（不建议在 tick 内频繁使用）。"""
        self.money.rental.append(dict(ts_sim=float(ts_sim), dt_s=float(dt_s), cost=float(cost)))

    def on_hospital_fee(self, ts_sim: float, fee: float):
        self.money.hospital.append(dict(ts_sim=float(ts_sim), fee=float(fee)))

    def on_help_income(self, ts_sim: float, req_id: int, kind: str, amount: float):
        self.money.help_income.append(dict(ts_sim=float(ts_sim), req_id=int(req_id), kind=str(kind), amount=float(amount)))
    
    def on_help_expense(self, ts_sim: float, req_id: int, kind: str, amount: float):
        """Publisher 支付赏金（成功 100%，宽限 50%，失败 0），在结算时调用。"""
        if amount > 1e-9:
            self.money.help_expense.append(dict(
                ts_sim=float(ts_sim),
                req_id=int(req_id),
                kind=str(kind),
                amount=float(amount),
            ))

    # —— 垫付（Comms 接好钩子后使用）——
    def note_advance_out(self, ts_sim: float, req_id: int, category: str, amount: float):
        """category in {'charge','buy'}"""
        self.money.advance_out.append(dict(ts_sim=float(ts_sim), req_id=int(req_id), category=str(category), amount=float(amount)))

    def note_advance_return(self, ts_sim: float, req_id: int, amount: float):
        self.money.advance_return.append(dict(ts_sim=float(ts_sim), req_id=int(req_id), amount=float(amount)))

    # —— 计数器 —— 
    def inc(self, name: str, k: int = 1):
        if hasattr(self.counters, name):
            setattr(self.counters, name, int(getattr(self.counters, name)) + int(k))

    def inc_nested(self, key: str, k: int = 1):
        """增加嵌套字典计数，如 action_attempts.VIEW_ORDERS"""
        if "." not in key:
            self.inc(key, k)
            return
        
        prefix, suffix = key.split(".", 1)
        if not hasattr(self.counters, prefix):
            setattr(self.counters, prefix, {})
        
        nested_dict = getattr(self.counters, prefix)
        if not isinstance(nested_dict, dict):
            setattr(self.counters, prefix, {})
            nested_dict = getattr(self.counters, prefix)
        
        nested_dict[suffix] = nested_dict.get(suffix, 0) + k

    def inc_preventive(self, kind: str = "generic", k: int = 1):
        """记录一次预防行为；kind 如 'early_charge'/'stock_up' 等。"""
        self.counters.preventive_actions += int(k)
        kind = str(kind)
        self.counters.preventive_by_kind[kind] = int(self.counters.preventive_by_kind.get(kind, 0)) + int(k)


    # ===== 新接口：持续计时型（充电/租车）聚合 =====
    # 充电：tick 内仅累计；结束时一次 append
    def accrue_charging(self, ts_sim: float, which: str, delta_pct: float, cost: float,
                        req_id: Optional[int] = None, start_ts: Optional[float] = None):
        """充电会话累计：只累加金额/百分点，不写明细。"""
        if delta_pct <= 0.0 and cost <= 0.0:
            return
        acc = self._charging_acc
        if acc is None:
            self._charging_acc = acc = dict(
                start_ts=float(start_ts if start_ts is not None else ts_sim),
                which=str(which),
                req_id=(int(req_id) if req_id is not None else None),
                delta_pct=0.0,
                cost=0.0,
            )
        acc["delta_pct"] = float(acc["delta_pct"]) + float(max(0.0, delta_pct))
        acc["cost"]      = float(acc["cost"])      + float(max(0.0, cost))

    def finish_charging(self, end_ts: float, reason: str = "finished", target_pct: Optional[float] = None):
        """充电会话结束：一次性 append 聚合明细，并清空累加器。"""
        acc = self._charging_acc
        if acc is None:
            return
        self.money.charging.append(dict(
            ts_sim=float(end_ts),
            which=acc["which"],
            req_id=acc["req_id"],
            delta_pct=float(acc["delta_pct"]),
            cost=float(acc["cost"]),
            start_ts=float(acc["start_ts"]),
            end_ts=float(end_ts),
            reason=str(reason),
            aggregated=True,
            target_pct=(float(target_pct) if target_pct is not None else None),
        ))
        self._charging_acc = None

    # 租车：tick 内仅累计；结束时一次 append
    def accrue_rental(self, dt_s: float, cost: float, start_ts: Optional[float] = None):
        """租车会话累计：只累加时长/金额，不写明细。"""
        if dt_s <= 0.0 and cost <= 0.0:
            return
        acc = self._rental_acc
        if acc is None:
            self._rental_acc = acc = dict(
                start_ts=float(start_ts if start_ts is not None else (self.started_sim_s or 0.0)),
                dt_s=0.0,
                cost=0.0,
            )
        if dt_s > 0.0: acc["dt_s"] = float(acc["dt_s"]) + float(dt_s)
        if cost > 0.0: acc["cost"] = float(acc["cost"]) + float(cost)

    def finish_rental(self, end_ts: float):
        """租车会话结束：一次性 append 聚合明细，并清空累加器。"""
        acc = self._rental_acc
        if acc is None:
            return
        self.money.rental.append(dict(
            ts_sim=float(end_ts),
            dt_s=float(acc["dt_s"]),
            cost=float(acc["cost"]),
            start_ts=float(acc["start_ts"]),
            end_ts=float(end_ts),
            aggregated=True,
        ))
        self._rental_acc = None

    # —— 导出 —— 
    def build_report(self, dm: Any) -> Dict[str, Any]:
        hrs = _safe_div(self.active_elapsed_s, 3600.0)

        # 从 completed_orders 汇总“基础工资 & 奖金惩罚 & 订单统计”
        completed = list(getattr(dm, "completed_orders", []) or [])
        active_list = list(getattr(dm, "active_orders", []) or [])

        open_overdue_cnt = 0        # 已超时的活跃单（失败）
        open_not_due_cnt = 0        # 未到时限的活跃单（不进分母）
        open_order_details: List[Dict[str, Any]] = []

        for o in active_list:
            try:
                elapsed = float(getattr(o, "sim_elapsed_active_s", 0.0) or 0.0)
                limit   = float(getattr(o, "time_limit_s", 0.0) or 0.0)
                slack   = float(limit - elapsed) if limit > 0.0 else float("inf")  # 无时限视为 +inf

                is_overdue = (limit > 0.0) and (slack < 0.0)
                if is_overdue:
                    open_overdue_cnt += 1
                else:
                    open_not_due_cnt += 1  # 包含“无时限”与“未到时限”

                open_order_details.append(dict(
                    id=int(getattr(o, "id", -1)),
                    elapsed_s=elapsed,
                    time_limit_s=limit,
                    deadline_slack_s=slack,
                    is_overdue=bool(is_overdue),
                    pick_score=float(getattr(o, "pick_score", 0.0) or 0.0),
                    pickup=getattr(o, "pickup_road_name", ""),
                    dropoff=getattr(o, "dropoff_road_name", ""),
                    earnings=float(getattr(o, "earnings", 0.0) or 0.0),
                    allowed_delivery_methods=list(getattr(o, "allowed_delivery_methods", []) or []),
                ))
            except Exception:
                pass

        base_sum   = float(sum(float(r.get("earnings", 0.0))     for r in completed))
        bonus_sum  = float(sum(float(r.get("bonus_extra", 0.0))  for r in completed))
        stars_avg  = _safe_div(sum(float(r.get("rating", 0.0))   for r in completed), max(1, len(completed)))

        # flags 统计
        n_done = len(completed)
        late_cnt = 0
        temp_ok_cnt = odor_ok_cnt = dmg_ok_cnt = 0
        method_success_cnt = 0
        total_order_time_s = 0.0
        food_star_sum = 0.0

        completed_order_details: List[Dict[str, Any]] = []
        for r in completed:
            flags = dict(r.get("flags") or {})
            # bd = dict(r.get("breakdown") or {})  # 如需可再用
            total_order_time_s += float(r.get("duration_s", 0.0))
            if flags.get("temp_ok_all", True):  temp_ok_cnt += 1
            if flags.get("odor_ok_all", True):  odor_ok_cnt += 1
            if flags.get("damage_ok_all", True):dmg_ok_cnt  += 1
            stars = dict(r.get("stars") or {})
            if int(stars.get("method", 0)) >= 5: method_success_cnt += 1
            food_star_sum += float(stars.get("food", 0) or 0)

            # 尝试从完成记录里拿 time_limit_s（如果 DM 已保存）
            r_time_limit = r.get("time_limit_s", None)
            r_duration   = float(r.get("duration_s", 0.0))
            if isinstance(r_time_limit, (int, float)):
                r_slack = float(r_time_limit) - r_duration
                r_is_overdue = (float(r_time_limit) > 0.0) and (r_slack < 0.0)
            else:
                # 回退：没有 time_limit_s 时，用 flags.on_time 估；仅用于布尔，slack 置 None
                r_slack = None
                r_is_overdue = (not bool(flags.get("on_time", True)))

            if r_is_overdue:
                late_cnt += 1

            completed_order_details.append(dict(
                id=r.get("id"),
                duration_s=r_duration,
                time_limit_s=(float(r_time_limit) if isinstance(r_time_limit, (int, float)) else None),
                deadline_slack_s=r_slack,
                is_overdue=bool(r_is_overdue),
                temp_ok_all=bool(flags.get("temp_ok_all", True)),
                odor_ok_all=bool(flags.get("odor_ok_all", True)),
                damage_ok_all=bool(flags.get("damage_ok_all", True)),
                pick_score=float(r.get("pick_score", 0.0) or 0.0),
                stars=dict(
                    overall=int(stars.get("overall", r.get("rating", 0))),
                    time=int(stars.get("time", 0)),
                    food=int(stars.get("food", 0)),
                    method=int(stars.get("method", 0))
                ),
                earnings=float(r.get("earnings", 0.0)),
                bonus_extra=float(r.get("bonus_extra", 0.0)),
                paid_total=float(r.get("paid_total", 0.0)),
                pickup=r.get("pickup",""),
                dropoff=r.get("dropoff",""),
                allowed_delivery_methods=list(r.get("allowed_delivery_methods", []) or []),
                delivery_method=r.get("delivery_method", None),
            ))


        # 费用合计
        exp_purchases = self.money.sum_purchases()
        exp_charging  = self.money.sum_charging()
        exp_rental    = self.money.sum_rental()
        exp_hospital  = self.money.sum_hospital()
        exp_help      = self.money.sum_help_expense()
        expense_total = exp_purchases + exp_charging + exp_rental + exp_hospital + exp_help

        # 帮助收入与垫付
        help_income_sum = self.money.sum_help_income()
        adv_out  = self.money.sum_advance_out()
        adv_ret  = self.money.sum_advance_ret()
        adv_net  = adv_out - adv_ret

        # 观察到的余额（由主逻辑实时增减）
        end_balance = float(getattr(dm, "earnings_total", 0.0))

        # 初始钱优先取 dm.cfg["initial_earnings"]，否则回退 RunRecorder.initial_balance
        cfg_initial = None
        try:
            cfg = getattr(dm, "cfg", None) or {}
            if "initial_earnings" in cfg:
                cfg_initial = float(cfg.get("initial_earnings"))
        except Exception:
            cfg_initial = None
        start_balance = float(cfg_initial if cfg_initial is not None else self.initial_balance)

        # 衍生项
        orders_income = float(base_sum + bonus_sum)
        net_growth    = float(end_balance - start_balance)

        # 三大能力分桶（供 capability_buckets 使用）
        active_time_ratio = _safe_div(total_order_time_s, max(self.active_elapsed_s, 1e-9))
        orders_per_hour   = _safe_div(n_done, max(hrs, 1e-9))
        avg_order_time_s  = _safe_div(total_order_time_s, max(n_done, 1))
        high_level_planning = dict(
            active_time_ratio=active_time_ratio,
            orders_per_hour=orders_per_hour,
            avg_order_time_s=avg_order_time_s,
            scooter_depleted_events=int(self.counters.scooter_depleted),
            hospital_rescue_times=int(self.counters.hospital_rescue),
            rent_insufficient_times=int(self.counters.rent_insufficient),
            charge_insufficient_times=int(self.counters.charge_insufficient),
        )
        common_sense_reasoning = dict(
            temp_ok_rate=_safe_div(temp_ok_cnt, max(n_done,1)),
            odor_ok_rate=_safe_div(odor_ok_cnt, max(n_done,1)),
            damage_ok_rate=_safe_div(dmg_ok_cnt, max(n_done,1)),
            method_success_rate=_safe_div(method_success_cnt, max(n_done,1)),
        )
        social_reasoning = dict(
            help_posted=int(self.counters.help_posted),
            help_accepted=int(self.counters.help_accepted),
            help_given=int(self.counters.help_given),
            help_received=int(self.counters.help_received),
            say=int(self.counters.say),
        )

        # 计算动作成功率
        action_stats = {}
        for action_name in self.counters.action_attempts:
            attempts = self.counters.action_attempts.get(action_name, 0)
            successes = self.counters.action_successes.get(action_name, 0)
            success_rate = _safe_div(successes, attempts)
            action_stats[action_name] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": success_rate
            }
        
        # VLM统计
        vlm_stats = {
            "total_calls": int(self.counters.vlm_calls),
            "successes": int(self.counters.vlm_successes),
            "parse_failures": int(self.counters.vlm_parse_failures),
            "retries": int(self.counters.vlm_retries),
            "success_rate": _safe_div(self.counters.vlm_successes, self.counters.vlm_calls),
            "parse_success_rate": _safe_div(self.counters.vlm_calls - self.counters.vlm_parse_failures, self.counters.vlm_calls)
        }

        # ===== 汇总（六/八项在最前，其余保留在后）=====
        money_totals = dict(
            initial_earnings=start_balance,   # 1
            current_balance=end_balance,      # 2
            net_growth=net_growth,            # 3（可为负）
            orders_income=orders_income,      # 4（base_wage + bonus）
            help_income=help_income_sum,      # 5
            expense_total=expense_total,      # 6（含 help_expense）
            base_wage=base_sum,               # 7
            bonus=bonus_sum,                  # 8

            # 其余原有信息照旧保留
            help_expense=exp_help,
            expense_breakdown=dict(
                purchases=exp_purchases,
                charging=exp_charging,
                rental=exp_rental,
                hospital=exp_hospital,
                help_expense=exp_help,
            ),
            advances=dict(out=adv_out, returned=adv_ret, net=adv_net),
            observed_agent_balance_end=end_balance,  # 兼容旧字段
        )

        # ===== 每小时版本 =====
        denom = max(hrs, 1e-9)
        all_ph = {k: _safe_div(v, denom) for k, v in money_totals.items() if isinstance(v, (int, float))}
        money_per_hour: Dict[str, float] = {}
        # 先放你关心的六项（顺序固定）
        for k in ("net_growth", "orders_income", "help_income", "expense_total", "base_wage", "bonus"):
            if k in all_ph:
                money_per_hour[k] = all_ph[k]
        # 再把剩余数值项按原顺序追加
        for k, v in all_ph.items():
            if k not in money_per_hour:
                money_per_hour[k] = v

        # 把“活跃超时”并入 timeout_count
        timeout_count_all = int(late_cnt) + int(open_overdue_cnt)

        # 分母：成功 + 失败；未到时限的活跃单不计入
        success_done = int(n_done - late_cnt)          # 已完成且未超时
        failed_all   = int(late_cnt + open_overdue_cnt)  # 已完成超时 + 活跃超时
        denom_for_ontime = max(1, success_done + failed_all)
        ontime_rate = float(success_done) / float(denom_for_ontime)

        # ===== 有效/无效时间统计 =====
        inactive_total_time_s = float(sum(self.inactive_time_s.values())) if self.inactive_time_s else 0.0
        effective_active_time_s = max(0.0, float(self.active_elapsed_s) - inactive_total_time_s)
        effective_time_ratio = _safe_div(effective_active_time_s, max(self.active_elapsed_s, 1e-9))

        interruptions_total = int(
            self.counters.scooter_depleted
            + self.counters.hospital_rescue
            + self.counters.rent_insufficient
            + self.counters.charge_insufficient
        )
        preventive_total = int(self.counters.preventive_actions)
        prevention_rate = _safe_div(preventive_total, preventive_total + interruptions_total)

        report = dict(
            meta=dict(
                agent_id=str(self.agent_id),
                model=str(self.model),
                lifecycle_hours=_safe_div(self.lifecycle_s, 3600.0),
                started_sim_s=float(self.started_sim_s or 0.0),
                ended_sim_s=float(self.ended_sim_s or 0.0),
                active_elapsed_s=float(self.active_elapsed_s),
                active_hours=hrs,
                wall_finished_ts=_now_wall_ts(),
            ),
            money=dict(
                totals=money_totals,
                per_hour=money_per_hour,
                details=dict(
                    purchases=self.money.purchases,
                    charging=self.money.charging,
                    rental=self.money.rental,
                    hospital=self.money.hospital,
                    help_income=self.money.help_income,
                    help_expense=self.money.help_expense,
                    advance_out=self.money.advance_out,
                    advance_return=self.money.advance_return,
                )
            ),
            energy=dict(
                personal_consumed_pct=float(self.energy_personal_consumed_pct),
                scooter_batt_consumed_pct=float(self.scooter_batt_consumed_pct),
                personal_remaining_pct=float(getattr(dm, "energy_pct", 0.0)),
                scooter_remaining_pct=(float(getattr(getattr(dm, "e_scooter", None), "battery_pct", 0.0))
                                       if getattr(dm, "e_scooter", None) is not None else None),
            ),
            orders=dict(
                # 已完成
                completed_count=n_done,
                timeout_count=timeout_count_all,         # ④ ✅ 统一口径：含活跃超时
                ontime_rate=ontime_rate,                 # ④ ✅ 统一口径：成功 / (成功+失败)

                # 质量统计（已完成）
                avg_stars=stars_avg,
                avg_food_stars=_safe_div(food_star_sum, max(n_done,1)),
                temp_ok_rate=_safe_div(temp_ok_cnt, max(n_done,1)),
                odor_ok_rate=_safe_div(odor_ok_cnt, max(n_done,1)),
                damage_ok_rate=_safe_div(dmg_ok_cnt, max(n_done,1)),
                method_success_rate=_safe_div(method_success_cnt, max(n_done,1)),

                # 持续时长（已完成）
                total_active_time_s=total_order_time_s,
                avg_time_per_order_s=avg_order_time_s,

                # 活跃单
                open_count=len(active_list),
                open_overdue_count=open_overdue_cnt,
                open_not_due_count=open_not_due_cnt,

                # 详情（两类都统一字段：deadline_slack_s / is_overdue）
                completed_order_details=completed_order_details,            # 已完成明细
                open_order_details=open_order_details,  # 活跃明细
            ),
            activity=dict(
                effective_time_ratio=effective_time_ratio,
                inactive_time_s_breakdown=dict(self.inactive_time_s or {}), # NEW
                inactive_total_time_s=inactive_total_time_s,              # NEW
                orders_per_hour=orders_per_hour,
                avg_order_time_s=avg_order_time_s,
                mode_time_s=dict(self.transport_time_s or {}),
                mode_time_ratio={k: _safe_div(v, max(self.active_elapsed_s, 1e-9)) for k, v in (self.transport_time_s or {}).items()}
            ),
            interruptions=dict(
                scooter_depleted=int(self.counters.scooter_depleted),
                hospital_rescue=int(self.counters.hospital_rescue),
                rent_insufficient=int(self.counters.rent_insufficient),
                charge_insufficient=int(self.counters.charge_insufficient),
                total=interruptions_total,                     # NEW
            ),
            prevention=dict(                                   # NEW block
                total=preventive_total,
                by_kind=dict(self.counters.preventive_by_kind or {}),
                prevention_rate=prevention_rate,               # 预防 / (预防 + 中断)
            ),
            social=dict(
                help_posted=int(self.counters.help_posted),
                help_accepted=int(self.counters.help_accepted),
                help_given=int(self.counters.help_given),
                help_received=int(self.counters.help_received),
            ),
            capability_buckets=dict(
                high_level_planning=high_level_planning,
                common_sense_reasoning=common_sense_reasoning,
                social_reasoning=social_reasoning
            ),
            actions=action_stats,
            vlm=vlm_stats
        )
        return report

    def export(self, dm: Any) -> str:
        # 导出前兜底：如果还有未结束的充电/租车会话，先各自收口为一条聚合明细
        end_ts = float(self.ended_sim_s or self.started_sim_s or _now_wall_ts())
        if self._charging_acc is not None:
            self.finish_charging(end_ts=end_ts, reason="export")
        if self._rental_acc is not None:
            self.finish_rental(end_ts=end_ts)

        report = self.build_report(dm)
        path = str(self.export_path or "run_report.json")  # 如果你在外面那一行已把目录+文件名拼好，这里不需要再动
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path