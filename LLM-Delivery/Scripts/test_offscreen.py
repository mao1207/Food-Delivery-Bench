#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单验证 MapExportor 在离屏模式下的 PNG 导出功能
包括测试 show_road_names 功能
"""
import os
import sys
import json
from PyQt5.QtWidgets import QApplication
from PyQt5.Qt import Qt

# 设置离屏模式

# 添加路径
SIMWORLD_DIR = r"D:\Projects\Food-Delivery-Bench\SimWorld"
LLM_DELIVERY_DIR = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery"
sys.path.insert(0, SIMWORLD_DIR)
sys.path.insert(0, LLM_DELIVERY_DIR)

from Base.Map import Map
from utils.map_exportor import MapExportor
from utils.map_debug_viewer import MapDebugViewer
from utils.map_canvas_base import _node_xy  # 导入模块级别的函数

def debug_road_names(map_obj):
    """调试路名显示问题"""
    print("\n 调试路名显示问题:")
    
    # 检查道路元数据
    road_count = 0
    for a, nbs in map_obj.adjacency_list.items():
        for b in nbs:
            meta = map_obj._get_edge_meta(a, b)
            if meta and meta.get("kind") == "road":
                road_count += 1
                print(f"  道路 {road_count}: {meta}")
                
                # 测试 _split_name_side 方法
                from utils.map_canvas_base import MapCanvasBase
                viewer = MapCanvasBase()
                base, side = viewer._split_name_side(meta)
                print(f"    原始名称: '{meta.get('name')}'")
                print(f"    解析结果: base='{base}', side='{side}'")
                
                # 测试 _lr_label 方法
                label = viewer._lr_label(base, side)
                print(f"    最终标签: '{label}'")
                print()
    
    print(f"总共找到 {road_count} 条道路")

def debug_road_names_detailed(map_obj):
    """详细调试路名显示问题"""
    print("\n 详细调试路名显示问题:")
    
    from utils.map_canvas_base import MapCanvasBase
    viewer = MapCanvasBase()
    viewer.set_context(map_obj, None)
    
    # 模拟 _draw_road_names 的完整逻辑
    print("📋 模拟路名绘制逻辑:")
    
    agg = {}
    seen_pairs = set()
    road_count = 0
    
    for a, nbs in map_obj.adjacency_list.items():
        for b in nbs:
            keyp = tuple(sorted((id(a), id(b))))
            if keyp in seen_pairs: continue
            seen_pairs.add(keyp)
            
            meta = map_obj._get_edge_meta(a, b)
            if not isinstance(meta, dict) or (meta.get("kind") != "road"): continue
            
            road_count += 1
            base, side = viewer._split_name_side(meta)
            
            if not base or side not in ("left", "right"):
                continue
            
            ax, ay = _node_xy(a); bx, by = _node_xy(b)  # 使用导入的函数
            dx, dy = bx - ax, by - ay
            L = (dx*dx + dy*dy)**0.5
            if L < 1e-6: continue
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            
            rec = agg.setdefault((base, side), dict(sumL=0.0, cx=0.0, cy=0.0, vx=0.0, vy=0.0))
            rec["sumL"] += L
            rec["cx"]   += mx * L
            rec["cy"]   += my * L
            rec["vx"]   += dx
            rec["vy"]   += dy
    
    print(f"总共处理了 {road_count} 条道路")
    print(f"聚合后有 {len(agg)} 个唯一的路名")
    
    # 检查聚合结果
    print("\n📊 聚合结果:")
    for (base, side), rec in agg.items():
        sumL = rec["sumL"]
        if sumL <= 0: continue
        mx = rec["cx"] / sumL
        my = rec["cy"] / sumL
        vx, vy = rec["vx"], rec["vy"]
        vlen = (vx*vx + vy*vy)**0.5
        tx, ty = ((vx / vlen, vy / vlen) if vlen > 1e-6 else (1.0, 0.0))
        nx, ny = -ty, tx
        
        label = viewer._lr_label(base, side)
        print(f"  路名: {base} ({side}) -> 标签: {label}")
        print(f"    位置: ({mx:.1f}, {my:.1f})")
        print(f"    方向: ({tx:.3f}, {ty:.3f})")
        print(f"    法向: ({nx:.3f}, {ny:.3f})")
        print(f"    长度: {sumL:.1f}")
        
        # 计算路名位置
        sgn = -1 if side == "left" else 1
        offset = 140.0  # ROAD_NAME_OFFSET_CM
        px_ = mx + sgn * nx * offset
        py_ = my + sgn * ny * offset
        print(f"    路名位置: ({px_:.1f}, {py_:.1f})")
        print()
    
    # 测试实际绘制
    print("🎨 测试实际绘制...")
    viewer.draw_map(map_obj, show_road_names=True)
    
    # 检查 plot 中的 items
    items = viewer.plot.listDataItems()
    text_items = [item for item in items if hasattr(item, 'text')]
    print(f"Plot 中有 {len(text_items)} 个文本项")
    
    for i, item in enumerate(text_items):
        if hasattr(item, 'text') and item.text():
            print(f"  文本项 {i}: '{item.text()}' 位置: {item.pos()}")
    
    # 检查字体设置
    print(f"\n🔤 字体设置:")
    print(f"  字体大小: {viewer._label_font.pixelSize()}")
    print(f"  字体族: {viewer._label_font.family()}")
    print(f"  粗体: {viewer._label_font.bold()}")

def test_map_exportor():
    """测试 MapExportor 的 PNG 导出功能"""
    print("🚀 开始测试 MapExportor 离屏导出...")
    
    # 1. 初始化 QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    print("✅ QApplication 初始化成功")
    
    # 2. 创建地图对象
    try:
        map_obj = Map()
        # 导入道路数据
        roads_json = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\roads.json"
        world_json = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\progen_world_enriched.json"
        
        if os.path.exists(roads_json):
            map_obj.import_roads(roads_json)
            print(f"✅ 导入道路数据: {roads_json}")
        else:
            print(f"⚠️  道路文件不存在: {roads_json}")
            return False
            
        if os.path.exists(world_json):
            map_obj.import_pois(world_json)
            print(f"✅ 导入 POI 数据: {world_json}")
        else:
            print(f"⚠️  World 文件不存在: {world_json}")
            
    except Exception as e:
        print(f"❌ 地图初始化失败: {e}")
        return False
    
    # 调试路名问题
    debug_road_names_detailed(map_obj)
    
    # 3. 创建 MapDebugViewer 并设置离屏模式
    try:
        viewer = MapDebugViewer(title="Test Viewer (headless)")
        # 设置为离屏模式
        viewer.setAttribute(Qt.WA_DontShowOnScreen, True)
        viewer.hide()
        
        # 绘制地图到 viewer
        viewer.draw_map(
            map_obj=map_obj,
            world_json_path=world_json if os.path.exists(world_json) else None,
            show_bus=True,
            show_docks=False,
            show_building_links=True,
            show_road_names=True,  # 启用路名显示
            plain_mode="pudo"
        )
        print("✅ MapDebugViewer 创建并配置成功")
    except Exception as e:
        print(f"❌ MapDebugViewer 创建失败: {e}")
        return False
    
    # 4. 创建 MapExportor 并传入 viewer
    try:
        exportor = MapExportor(
            map_obj=map_obj,
            world_json_path=world_json if os.path.exists(world_json) else None,
            show_road_names=True,  # 启用路名显示
            viewer=viewer  # 传入配置好的 viewer
        )
        print("✅ MapExportor 创建成功")
    except Exception as e:
        print(f"❌ MapExportor 创建失败: {e}")
        return False
    
    # 5. 准备底图
    try:
        exportor.prepare_base()
        print("✅ 底图准备完成")
    except Exception as e:
        print(f"❌ 底图准备失败: {e}")
        return False
    
    # 6. 测试导出（带路名和不带路名）
    try:
        # 模拟一些订单数据
        mock_orders = [
            {
                "id": "order_001",
                "pickup_xy": [1000.0, 2000.0],  # 坐标单位：cm
                "dropoff_xy": [3000.0, 4000.0]
            },
            {
                "id": "order_002", 
                "pickup_xy": [1500.0, 2500.0],
                "dropoff_xy": [3500.0, 4500.0]
            }
        ]
        
        # 模拟 agent 位置
        agent_xy = (2000.0, 3000.0)
        
        # 测试1: 导出带路名的图片
        print(" 测试导出带路名的图片...")
        global_bytes_with_names, local_bytes_with_names = exportor.export(
            agent_xy=agent_xy,
            orders=mock_orders
        )
        
        print(f"✅ 带路名导出成功!")
        print(f"   - 全局图大小: {len(global_bytes_with_names)} bytes")
        print(f"   - 局部图大小: {len(local_bytes_with_names)} bytes")
        
        # 保存带路名的图片
        with open("test_global_with_road_names.png", "wb") as f:
            f.write(global_bytes_with_names)
        with open("test_local_with_road_names.png", "wb") as f:
            f.write(local_bytes_with_names)
        
        # 测试2: 创建不带路名的导出器
        print("📸 测试导出不带路名的图片...")
        exportor_no_names = MapExportor(
            map_obj=map_obj,
            world_json_path=world_json if os.path.exists(world_json) else None,
            show_road_names=False,  # 不显示路名
            viewer=viewer
        )
        exportor_no_names.prepare_base()
        
        global_bytes_no_names, local_bytes_no_names = exportor_no_names.export(
            agent_xy=agent_xy,
            orders=mock_orders
        )
        
        print(f"✅ 不带路名导出成功!")
        print(f"   - 全局图大小: {len(global_bytes_no_names)} bytes")
        print(f"   - 局部图大小: {len(local_bytes_no_names)} bytes")
        
        # 保存不带路名的图片
        with open("test_global_no_road_names.png", "wb") as f:
            f.write(global_bytes_no_names)
        with open("test_local_no_road_names.png", "wb") as f:
            f.write(local_bytes_no_names)
            
        print("✅ PNG 文件已保存:")
        print(f"   - test_global_with_road_names.png")
        print(f"   - test_local_with_road_names.png")
        print(f"   - test_global_no_road_names.png")
        print(f"   - test_local_no_road_names.png")
        
        # 比较文件大小（带路名的应该稍大一些）
        size_diff_global = len(global_bytes_with_names) - len(global_bytes_no_names)
        size_diff_local = len(local_bytes_with_names) - len(local_bytes_no_names)
        print(f"📊 文件大小差异:")
        print(f"   - 全局图差异: {size_diff_global} bytes")
        print(f"   - 局部图差异: {size_diff_local} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MapExportor 离屏模式验证测试 (包含路名测试)")
    print("=" * 60)
    
    success = test_map_exportor()
    
    print("=" * 60)
    if success:
        print(" 测试通过! MapExportor 在离屏模式下工作正常")
        print("📁 请检查生成的 PNG 文件来验证路名显示效果")
    else:
        print("❌ 测试失败! 请检查错误信息")
    print("=" * 60)