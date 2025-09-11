# -*- coding: utf-8 -*-
"""
Insulated_bag.py
简化热学模型（弱空气结点）：
- 每个隔层有一个“空气/袋体结点”温度 Ta，初始化为 ambient_temp_c（室温），热容 Cab 很小（air_heat_capacity），
  这样它对食物的影响较弱，但仍保持能量守恒。
- 动力学（离散欧拉，同步更新）：
    设食物 i 的热容 Ci、温度 Ti，空气温度 Ta、热容 Cab，时间步 dt：
      S = sum_i Ci * (Ti - Ta)
      Ta_new = Ta + alpha * (S / Cab)
      Ti_new = Ti + alpha * (Ta - Ti)
    其中 alpha = dt / (exchange_tau_min * 60)；上限夹取（<=0.5）保证稳定性。
- 不与外界交换（保温理想化），因此总能量守恒。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any

def _letters(n: int) -> List[str]:
    n = max(1, min(26, int(n)))
    return [chr(ord('A') + i) for i in range(n)]

@dataclass
class InsulatedBag:
    num_compartments: int = 4

    # —— 环境与交换参数 —— 
    ambient_temp_c: float = 23.0        # 室温；隔层初始温度
    exchange_tau_min: float = 10.0      # 趋同时间常数（分钟）；越大收敛越慢
    air_heat_capacity: float = 0.4      # 空气/袋体结点等效热容（相对量，远小于食物 => 影响弱）

    # 运行态
    _comps: List[List[Any]] = field(default_factory=list, repr=False)   # 每层食物对象列表
    _comp_temp_c: List[float] = field(default_factory=list, repr=False) # 每层空气温度

    def __post_init__(self):
        n = int(self.num_compartments)
        self._comps = [[] for _ in range(n)]
        self._comp_temp_c = [float(self.ambient_temp_c) for _ in range(n)]

    # ---------------- 基础 ----------------
    @property
    def labels(self) -> List[str]:
        return _letters(self.num_compartments)

    def _idx(self, label: str) -> int:
        lab = str(label).strip().upper()
        if lab not in self.labels:
            raise ValueError(f"invalid compartment: {lab}")
        return self.labels.index(lab)

    # ---------------- 展示 ----------------
    def list_items(self) -> str:
        out: List[str] = []
        for i, lab in enumerate(self.labels):
            tc = self._comp_temp_c[i]
            out.append(f"{lab}: ({tc:.1f}°C)")
            for j, obj in enumerate(self._comps[i], start=1):
                name = getattr(obj, "name", None) or getattr(obj, "title", None) \
                       or getattr(obj, "label", None) or str(obj)
                if hasattr(obj, "temp_c") and isinstance(getattr(obj, "temp_c"), (int, float)):
                    out.append(f"  {lab}.{j} {name} ({obj.temp_c:.1f}°C)")
                else:
                    out.append(f"  {lab}.{j} {name}")
        return "\n".join(out) if out else "(empty)"

    # ---------------- 内部：放入/取出 ----------------
    def _append_to_comp(self, i: int, item: Any) -> None:
        # 隔层空气温度保持当前值（初始化为室温）；食物应在“备好餐/取餐”时已经有正确 temp_c
        self._comps[i].append(item)

    def _pop_from_comp(self, i: int, k: int) -> Any:
        item = self._comps[i].pop(k - 1)
        if not self._comps[i]:
            # 清空后把空气温度重置回室温
            self._comp_temp_c[i] = float(self.ambient_temp_c)
        return item

    # ---------------- 指令：移动/加入 ----------------
    def move_items(self, spec: str) -> None:
        """
        形如： "A.1, A.3 -> B; C.2 -> A"
        只移动袋中已有的条目（不新增）。
        """
        if not spec: return
        for clause in spec.split(";"):
            clause = clause.strip()
            if not clause or "->" not in clause: 
                continue
            left, right = clause.split("->", 1)
            dst = right.strip().upper()
            ptrs = []
            for tok in left.split(","):
                tok = tok.strip()
                if "." in tok:
                    lab, k = tok.split(".", 1)
                    ptrs.append((lab.strip().upper(), int(k)))
            if not ptrs:
                continue
            # 按源隔层分组，并在各自隔层内按索引降序 pop，避免漂移
            by_lab: Dict[str, List[int]] = {}
            for lab, k in ptrs:
                by_lab.setdefault(lab, []).append(k)
            for lab, ks in by_lab.items():
                ks.sort(reverse=True)
                src_i = self._idx(lab)
                dst_i = self._idx(dst)
                for k in ks:
                    if not (1 <= k <= len(self._comps[src_i])):
                        raise IndexError(f"{lab}.{k} not found")
                    item = self._pop_from_comp(src_i, k)
                    self._append_to_comp(dst_i, item)

    def add_items(self, spec: str, items_by_number: Dict[int, Any]) -> None:
        """
        形如： "1,2 -> A; 3 -> B"
        把“编号食物”加入到某隔层；典型用于‘取餐’时将新食物放入袋。
        假定食物在“备好餐/取餐”时已把 item.temp_c 设为 item.serving_temp_c。
        """
        if not spec: return
        for clause in spec.split(";"):
            clause = clause.strip()
            if not clause or "->" not in clause:
                continue
            left, right = clause.split("->", 1)
            dst = right.strip().upper()
            dst_i = self._idx(dst)

            nums: List[int] = []
            for tok in left.split(","):
                tok = tok.strip()
                if not tok or "." in tok:
                    continue
                try:
                    nums.append(int(tok))
                except Exception:
                    pass
            for n in nums:
                if n not in items_by_number:
                    raise KeyError(f"item #{n} not found")
                self._append_to_comp(dst_i, items_by_number[n])

    # ---------------- 虚拟时间推进 ----------------
    def tick_temperatures(self, delta_s: float) -> None:
        """
        仅“空气/袋体结点 ↔ 食物”交换；不与外界交换。能量守恒。
        delta_s: 虚拟时间步长（秒）
        """
        if delta_s <= 0:
            return
        alpha = delta_s / max(1e-6, self.exchange_tau_min * 60.0)
        if alpha <= 0:
            return
        if alpha > 0.5:
            alpha = 0.5  # 避免大步长振荡

        Cab = float(self.air_heat_capacity)

        for i, comp in enumerate(self._comps):
            if not comp:
                continue

            Ta0 = float(self._comp_temp_c[i])

            # 采样一次“旧温度”，做同步更新
            Ci_list: List[float] = []
            Ti0_list: List[float] = []
            for it in comp:
                Ci = float(getattr(it, "heat_capacity", 1.0) or 1.0)
                Ti0 = float(getattr(it, "temp_c", Ta0))
                Ci_list.append(Ci)
                Ti0_list.append(Ti0)
            if not Ti0_list:
                continue

            # 能量守恒：先更新空气温度
            S = 0.0
            for Ci, Ti0 in zip(Ci_list, Ti0_list):
                S += Ci * (Ti0 - Ta0)
            Ta_new = Ta0 + alpha * (S / max(1e-9, Cab))
            self._comp_temp_c[i] = Ta_new

            # 再更新所有食物（同步使用 Ta0）
            for it, Ti0 in zip(comp, Ti0_list):
                Ti_new = Ti0 + alpha * (Ta0 - Ti0)
                it.temp_c = Ti_new

    def remove_items(self, items: List[Any]) -> None:
        # 用对象 id 做身份匹配，避免不可哈希/等价关系的坑
        target_ids = {id(obj) for obj in (items or [])}
        if not target_ids:
            return
        for i, comp in enumerate(list(self._comps)):
            if not comp:
                continue
            new_comp = [obj for obj in comp if id(obj) not in target_ids]
            if len(new_comp) != len(comp):
                self._comps[i] = new_comp
                if not new_comp:
                    # 清空后重置为室温
                    self._comp_temp_c[i] = float(self.ambient_temp_c)