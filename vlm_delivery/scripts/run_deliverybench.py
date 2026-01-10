# run_gym_qt_route_a.py
import sys
sys.path.insert(0, r"D:\DeliveryBench-gym")
sys.path.insert(0, r"D:\DeliveryBench-gym\vlm_delivery")

import os
import threading
import traceback
from PyQt5.QtCore import QTimer
from gym_like_interface.gym_like_interface import DeliveryBenchGymEnvQtRouteA

def main():
    base = os.environ.get("DELIVERYBENCH_BASE_DIR", r"D:\DeliveryBench-gym")
    env = DeliveryBenchGymEnvQtRouteA(
        base_dir=base,
        ue_ip="127.0.0.1",
        ue_port=9000,
        sim_tick_ms=100,
        vlm_pump_ms=100,
        enable_viewer=True,
        map_name="medium-city-22roads",
        max_steps=20,
    )

    # 必须主线程
    env.bootstrap_qt()

    def rl_loop():
        try:
            obs, info = env.reset(seed=0)
            print("reset info:", info)
            for i in range(20):
                obs, r, term, trunc, info2 = env.step(None)
                print(f"[RL] step={i+1} info:", info2)
                if term or trunc or "error" in info2:
                    break
        except Exception:
            traceback.print_exc()
        finally:
            try: env.close()
            except: pass
            try:
                if env._app is not None:
                    env._app.quit()
            except: pass

    def start_rl_thread():
        t = threading.Thread(target=rl_loop, name="RLWorker", daemon=True)
        t.start()

    QTimer.singleShot(0, start_rl_thread)
    env.run_qt_loop()

if __name__ == "__main__":
    main()