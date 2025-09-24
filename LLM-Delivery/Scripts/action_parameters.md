# DeliveryMan 动作参数完整文档
注意：标注“必需”的字段缺失会导致动作失败；未标注的为可选并有默认值。

## 1) 移动/视角

### move_to — 移动到指定坐标
```json
{
  "tx": 100.0,                    // 必需：目标 X 坐标
  "ty": 200.0,                    // 必需：目标 Y 坐标
  "pace": "normal",             // 可选："accel" | "normal" | "decel"
  "arrive_tolerance_cm": 120.0,  // 可选：到达容差（厘米）
  "use_route": true,              // 可选：是否走规划路径（解析层会用到）
  "snap_cm": 120.0               // 可选：路径吸附容差（解析层会用到）
}
```

### turn_around — 转身
```json
{
  "angle": 180.0,                 // 可选：转身角度
  "direction": "left"           // 可选：方向（字符串，由 UE 处理）
}
```

### step_forward — 前进一步
```json
{}
```

## 2) 订单流转

### accept_order — 接受订单
方式一：单个订单
```json
{ "oid": 12 }
```
方式二：多个订单
```json
{ "oids": [12, 18, 25] }
```

### view_orders — 查看可接订单
```json
{}
```

### pickup — 取餐（到店取餐口附近）
```json
{
  "orders": [],                   // 可选：订单id数组
  "tol_cm": 300.0                 // 可选：位置容差
}
```

### place_food_in_bag — 放入保温袋
```json
{
  "bag_cmd": "order 12: A cold; order 18: B hot"  // 必需：放置规范文本
}
```

### drop_off — 投递订单（到收货点附近）
```json
{
  "oid": 12,                      // 必需：订单 ID
  "method": "leave_at_door",    // 必需："leave_at_door" | "knock" | "call" | "hand_to_customer"
  "tol_cm": 300.0                 // 可选：位置容差
}
```

### view_bag — 查看保温袋
```json
{}
```

## 3) 等待与体力

### wait — 等待
固定时长：
```json
{ "duration_s": 30.0 }
```
或等待充电完成：
```json
{ "until": "charge_done" }
```

### rest — 休息（休息区附近）
```json
{ "target_pct": 100.0 }
```

## 4) 交通工具 / 充电 / 租车

### charge_escooter — 给电瓶车充电（充电站附近）
```json
{
  "target_pct": 100.0,            // 可选：目标电量
  "tol_cm": 300.0                 // 可选：位置容差
}
```

### switch_transport — 切换交通方式
```json
{ "to": "walk" }                 // 必需："walk" | "e-scooter" | "scooter" | "car" | "drag_scooter" | "drag"
```

### rent_car — 租车（租车点附近）
```json
{ "rate_per_min": 1.0 }
```

### return_car — 还车（租车点附近）
```json
{}
```

## 5) 购买与消耗品

### buy — 购买
单品：
```json
{ "item_id": "energy_drink", "qty": 2 }
```
或批量：
```json
{
  "items": [
    { "item_id": "energy_drink", "qty": 2 },
    { "name": "escooter_battery_pack", "qty": 1 }
  ]
}
```

### use_battery_pack — 使用电池包（需自有电瓶车在手边）
```json
{ "item_id": "escooter_battery_pack" }
```

### use_energy_drink — 使用能量饮料
```json
{ "item_id": "energy_drink" }
```

### use_ice_pack — 使用冰袋（需有保温袋）
```json
{ "comp": "A" }                  // 必需：隔层标签 "A" | "B" | ...
```

### use_heat_pack — 使用热袋（需有保温袋）
```json
{ "comp": "A" }                  // 必需：隔层标签
```

## 6) 社交 / 通信

### say — 发送消息
```json
{
  "text": "Hello!",              // 必需：消息文本（非空）
  "to": "7"                      // 可选：目标 Agent ID；省略或 "ALL"/"*" 则广播
}
```

## 7) 求助系统（Comms）

### view_help_board — 查看求助板
```json
{}
```

### post_help_request — 发布求助请求
不同求助类型需要在 payload 中提供不同字段：
```json
{
  "help_type": "HELP_DELIVERY",  // 必需：HELP_DELIVERY | HELP_PICKUP | HELP_BUY | HELP_CHARGE
  "bounty": 10.0,                 // 可选：悬赏
  "ttl_s": 3600.0,                // 可选：有效期（秒）
  "payload": {                    // 必需：按类型提供
    "order_id": 12,               // HELP_DELIVERY/HELP_PICKUP 需要
    "provide_xy": [100.0, 200.0], // HELP_DELIVERY 需要
    "deliver_xy": [300.0, 400.0], // HELP_PICKUP/HELP_BUY/HELP_CHARGE 需要
    "buy_list": [                 // HELP_BUY 需要
      { "item_id": "energy_drink", "qty": 1 }
    ],
    "target_pct": 100.0           // HELP_CHARGE 可选
  }
}
```

### accept_help_request — 接受求助
```json
{ "req_id": 123 }
```

### edit_help_request — 编辑求助
```json
{
  "req_id": 123,                  // 必需
  "new_bounty": 15.0,             // 可选
  "new_ttl_s": 7200.0             // 可选
}
```

### report_help_finished — 报告求助完成
```json
{ "req_id": 123 }
```

### place_temp_box — 放临时盒（交接）
```json
{
  "req_id": 123,                  // 必需
  "location_xy": [100.0, 200.0],  // 可选：默认当前位置
  "content": {                    // 可选：以下任一或多项
    "inventory": { "energy_drink": 2 }, // 从库存转移
    "food": "",                 // 出现该键即代表“全部食物”
    "escooter": ""              // 出现该键即代表交接整车
  }
}
```

### take_from_temp_box — 从临时盒取物
```json
{
  "req_id": 123,                  // 必需
  "tol_cm": 300.0                 // 可选
}
```

## 8) 公交系统

### board_bus — 上车
```json
{
  "bus_id": "bus_001",           // 必需
  "target_stop_id": "stop_5"     // 必需
}
```

### view_bus_schedule — 查看公交时刻表
```json
{}
```

---

提示：
- 坐标字段通常是浮点数；容差以厘米计（如 `tol_cm`）。
- 百分比字段范围 0–100（如 `target_pct`）。
- 时间以秒计（如 `duration_s`, `ttl_s`）。
- 物品 ID 须与配置一致（如 `energy_drink`, `escooter_battery_pack`）。


