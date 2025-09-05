# -*- coding: utf-8 -*-
# StoreManager: catalog + payment + inventory only.
import json, os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class StoreItem:
    id: str
    name: str
    price: float

class StoreManager:
    def __init__(self, json_path: Optional[str] = None):
        self.items: Dict[str, StoreItem] = {}
        self.last_cost: float = 0.0
        self.json_path: Optional[str] = None
        if json_path:
            self.load_items(json_path)

    # 统一的加载入口：支持 {"items":[...]} 或顶层列表 [...]
    def load_items(self, json_path: str) -> None:
        self.json_path = os.path.abspath(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 兼容两种结构
        data = raw.get("items", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, list):
            raise ValueError("store json must be a list or a dict with 'items' list")

        self.items.clear()
        for rec in data:
            if not isinstance(rec, dict):  # 跳过非法项
                continue
            _id = str(rec["id"])
            _name = str(rec.get("name", _id))
            _price = float(rec["price"])
            self.items[_id] = StoreItem(id=_id, name=_name, price=_price)

    # 旧接口别名，兼容你现在写的 _load() 风格（可选）
    def _load(self):
        if not self.json_path:
            raise RuntimeError("no json_path set; call load_items(path) first or pass it to __init__")
        self.load_items(self.json_path)

    def list_items(self) -> List[StoreItem]:
        return list(self.items.values())

    def get_item(self, item_id: str) -> Optional[StoreItem]:
        return self.items.get(str(item_id))

    def can_afford(self, dm, item_or_id) -> bool:
        itm = item_or_id if isinstance(item_or_id, StoreItem) else self.get_item(str(item_or_id))
        return bool(itm and float(getattr(dm, "earnings_total", 0.0)) >= float(itm.price))

    def purchase(self, dm, item_id: str, *, qty: int = 1) -> Tuple[bool, str, float]:
        """
        扣费并把物品数量加入 dm.inventory
        返回: (ok, msg, total_cost)
        """
        itm = self.get_item(item_id)
        if not itm:
            return False, f"unknown item '{item_id}'", 0.0
        qty = max(1, int(qty))
        unit = float(itm.price)
        cost = unit * qty
        funds = float(getattr(dm, "earnings_total", 0.0))
        if funds < cost:
            return False, f"insufficient funds (${funds:.2f} < ${cost:.2f})", 0.0

        # pay
        dm.earnings_total = funds - cost
        self.last_cost = cost

        # inventory
        inv = getattr(dm, "inventory", None)
        if inv is None:
            dm.inventory = {}
            inv = dm.inventory
        inv[item_id] = int(inv.get(item_id, 0)) + qty

        return True, "ok", cost
