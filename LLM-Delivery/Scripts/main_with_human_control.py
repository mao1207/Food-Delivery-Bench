# main_with_human_control.py
# -*- coding: utf-8 -*-
import copy
import sys
import os
import json
import time
import random
import argparse

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from concurrent.futures import ThreadPoolExecutor

# === 路径按你的工程实际调整 ===
SIMWORLD_DIR      = r"D:\Projects\Food-Delivery-Bench\SimWorld"
LLM_DELIVERY_DIR  = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery"
sys.path.insert(0, SIMWORLD_DIR); sys.path.insert(0, LLM_DELIVERY_DIR)

from Base.Map import Map
from Base.Order import OrderManager
from Base.DeliveryMan import DeliveryMan, TransportMode
from Base.Store import StoreManager
from utils.map_observer import MapObserver
from Base.Timer import VirtualClock
from Base.Comms import init_comms
from Base.Bus import Bus, BusRoute, BusStop
from Base.BusManager import BusManager

from Communicator import Communicator

# ✅ 新增：引入 MapExportor
from utils.map_exportor import MapExportor

# ✅ 极简 VLM 客户端
from llm.base_model import BaseModel

ROADS_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\maps\medium-city-20roads\roads.json"
WORLD_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\maps\medium-city-20roads\progen_world_enriched.json"
STORE_ITEMS_JSON  = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\store_items.json"
FOOD_JSON         = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\food.json"
CONFIG_JSON       = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\config.json"
SPECIAL_NOTES_JSON = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\special_notes.json"
MODELS_JSON       = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\models.json"

# result路径
OUTPUT_PATH = r"D:\Projects\Food-Delivery-Bench\results"

# random seed
random.seed(42)


def _load_world_nodes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("nodes", [])
    
def _load_cfg(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Config JSON parse error in {path}: {e}")

def _load_models(path: str) -> dict:
    """Load agent-specific model configurations from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data
    except FileNotFoundError:
        raise RuntimeError(f"Models file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Models JSON parse error in {path}: {e}")

def _get_agent_model_config(agent_id: str, models_config: dict) -> dict:
    """Get model configuration for a specific agent, falling back to default if not found."""
    agents = models_config.get("agents", {})
    default = models_config.get("default", {})
    
    agent_config = agents.get(agent_id, {})
    
    # Merge with default values, agent-specific values take precedence
    config = default.copy()
    config.update(agent_config)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="DeliveryMan with Human Control")
    parser.add_argument("--human-control", choices=["gui", "none"], default="gui",
                       help="启用人类控制 (gui=图形界面, none=禁用)")
    parser.add_argument("--agent-id", type=str, default="1",
                       help="要控制的agent ID")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)

    # --- 地图/订单/商店 ---
    m = Map(); m.import_roads(ROADS_JSON); m.import_pois(WORLD_JSON)
    nodes = _load_world_nodes(WORLD_JSON)

    # 读取 food.json，直接把 data["items"]（字典列表）交给 OrderManager
    with open(FOOD_JSON, "r", encoding="utf-8") as f:
        food_data = json.load(f) or {}
    menu_items = food_data.get("items", [])

    with open(SPECIAL_NOTES_JSON, "r", encoding="utf-8") as f:
        special_notes_data = json.load(f) or {}

    cfg = _load_cfg(CONFIG_JSON)
    models_config = _load_models(MODELS_JSON)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg['lifecycle']['export_path'] = os.path.join(OUTPUT_PATH, timestamp)
    os.makedirs(cfg['lifecycle']['export_path'], exist_ok=True)

    # Clock
    clock = VirtualClock(time_scale=3.0)

    # Comms
    comms = init_comms(clock=clock, ambient_temp_c=cfg.get("ambient_temp_c", 22.0), k_food_per_s=cfg.get("k_food_per_s", 1.0 / 1200.0))
    agent_count = 1

    om = OrderManager(capacity=10, menu=menu_items, clock=clock, special_notes_map=special_notes_data, note_prob=0.4)
    om.fill_pool(m, nodes)

    sm = StoreManager(); sm.load_items(STORE_ITEMS_JSON)

    bus_manager = BusManager(clock=clock, waiting_time_s=cfg.get("bus", {}).get("waiting_time_s", 180.0), speed_cm_s=cfg.get("bus", {}).get("speed_cm_s", 1200.0))
    with open(WORLD_JSON, "r", encoding="utf-8") as f:
        world_data = json.load(f)
    bus_manager.init_bus_system(world_data)

    # --- 单实例 UE 通信（9000口） ---
    communicator = Communicator(port=9000, ip='127.0.0.1', resolution=(640, 480))

    # --- Viewer（不做位移动画，只用于显示/高亮/回调） + 虚拟时间 ---
    v = MapObserver(title="Map Observer — UE moves; viewer displays", clock=clock)
    v.draw_map(m, WORLD_JSON, show_bus=True, show_docks=False,
               show_building_links=True, show_road_names=True, plain_mode="pudo")
    v.resize(1200, 900); v.show()
    v.attach_order_manager(om)
    v.attach_comms(comms)
    v.attach_bus_manager(bus_manager)

    # --- 工具：随机道路坐标 ---
    def rand_xy():
        xy = v.random_xy_on_roads()
        return xy if xy else (0.0, 0.0)

    # ✅ VLM 客户端配置
    OPENAI_KEY = os.getenv("OPENAI_KEY", "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")
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "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")
    if not OPENAI_KEY and not OPENROUTER_KEY:
        raise RuntimeError("Please set OPENROUTER_API_KEY or OPENAI_API_KEY in environment.")

    # ✅ 初始化 MapExportor
    map_exportor = MapExportor(
        map_obj=m,
        world_json_path=WORLD_JSON,
        show_road_names=True,
    )
    map_exportor.prepare_base()
    print("[exportor] base ready")

    # === VLM 线程池 ===
    executor = ThreadPoolExecutor(max_workers=6)

    # --- 初始化多个 agent ---
    dms = []
    target_dm = None  # 用于人类控制的agent
    
    for i in range(int(agent_count)):
        aid = str(i + 1)
        mode = TransportMode.SCOOTER
        ax, ay = rand_xy()
        dm = DeliveryMan(aid, m, nodes, ax, ay, mode=mode, clock=clock, cfg=copy.deepcopy(cfg))

        dm.bind_viewer(v)
        dm.set_order_manager(om)
        dm.set_store_manager(sm)
        dm.set_bus_manager(bus_manager)
        dm.set_ue(communicator)
        dm.bind_simworld()
        dm.register_to_comms()

        # ✅ 为每个 agent 创建独立的 VLM 客户端
        agent_model_config = _get_agent_model_config(aid, models_config)
        llm = BaseModel(
            url=agent_model_config.get("url"),
            api_key=OPENAI_KEY if agent_model_config.get("provider", "openai") == "openai" else OPENROUTER_KEY,
            model=agent_model_config.get("model")
        )
        
        dm.set_vlm_client(llm)
        dm.set_vlm_executor(executor)
        dm.map_exportor = map_exportor

        dms.append(dm)
        
        # 选择要控制的agent
        if aid == args.agent_id:
            target_dm = dm
            
        print(f"[Agent {aid}] Using model: {agent_model_config.get('model', 'gpt-4o-mini')} with config: {agent_model_config}")

    # ✅ 设置人类控制
    if args.human_control != "none" and target_dm:
        print(f"🎮 为Agent {args.agent_id} 启用人类控制")
        
        # 设置人类控制模式
        target_dm.set_human_control_mode(True)
        
        # 设置人类动作回调
        def human_callback(dm):
            print(f"\n🤖 Agent {dm.agent_id} 需要人类输入动作...")
            print(f"当前位置: ({dm.x:.1f}, {dm.y:.1f})")
            print(f"可用动作: {', '.join(dm.get_available_actions()[:10])}...")
        
        target_dm.set_human_action_callback(human_callback)
    else:
        print("🤖 使用VLM控制模式")

    # --- 同步屏障：轮询 UE，等所有 agent 真正出现后统一开跑 ---
    ready = set()

    def check_all_ready():
        for dm in dms:
            if dm.agent_id in ready:
                continue
            rec = communicator.get_position_and_direction(str(dm.agent_id))
            tup = rec.get(str(dm.agent_id)) if rec else None
            if tup:
                ready.add(dm.agent_id)
                dm._log(f"Agent {dm.agent_id} initialized successfully at ({dm.x/100.0:.2f}m, {dm.y/100.0:.2f}m)")

        if len(ready) == len(dms):
            ready_timer.stop()
            # ✅ 小错位启动
            STEP_MS   = 120
            JITTER_MS = 60
            base = random.randint(0, 80)
            for i, dm in enumerate(dms):
                delay = base + i * STEP_MS + random.randint(0, JITTER_MS)
                QTimer.singleShot(delay, dm.kickstart)

    ready_timer = QTimer(v)
    ready_timer.setInterval(100)
    ready_timer.timeout.connect(check_all_ready)
    ready_timer.start()

    # === 主线程定时泵出 VLM 结果 ===
    def pump_all_vlm():
        for dm in dms:
            dm.pump_vlm_results()

    vlm_timer = QTimer(v)
    vlm_timer.setInterval(1000)
    vlm_timer.timeout.connect(pump_all_vlm)
    vlm_timer.start()

    # === 推进仿真 ===
    def tick_sim():
        for dm in dms:
            dm.poll_time_events()

    sim_timer = QTimer(v)
    sim_timer.setInterval(100)
    sim_timer.timeout.connect(tick_sim)
    sim_timer.start()

    # 退出时关闭线程池
    app.aboutToQuit.connect(lambda: executor.shutdown(wait=False, cancel_futures=True))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
