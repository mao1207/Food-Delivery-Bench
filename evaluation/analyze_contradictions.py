#!/usr/bin/env python3
"""
深入分析性能指标中的相悖现象
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_contradictions():
    """分析性能指标中的相悖现象"""
    
    # 读取四个数据集
    gemini_base_path = "evaluation/ablation/gemini_medium20_base/results.csv"
    gemini_single_path = "evaluation/ablation/gemini_medium20_single/model_performance_summary.csv"
    qwen_base_path = "evaluation/ablation/qwen72_medium20_base/model_performance_summary.csv"
    qwen_single_path = "evaluation/ablation/qwen72_medium20_single/model_performance_summary.csv"
    
    gemini_base_df = pd.read_csv(gemini_base_path)
    gemini_single_df = pd.read_csv(gemini_single_path)
    qwen_base_df = pd.read_csv(qwen_base_path)
    qwen_single_df = pd.read_csv(qwen_single_path)
    
    print("="*80)
    print("多模型性能对比分析")
    print("="*80)
    
    # 创建对比表格
    print("\n关键指标对比表")
    print("-" * 80)
    print(f"{'指标':<20} {'Gemini 8A':<12} {'Gemini 1A':<12} {'Qwen 8A':<12} {'Qwen 1A':<12}")
    print("-" * 80)
    
    # 净利润对比
    print(f"{'Net Profit':<20} {gemini_base_df['Avg_Per_Hour_Net_Growth'].iloc[0]:<12.2f} {gemini_single_df['Avg_Per_Hour_Net_Growth'].iloc[0]:<12.2f} {qwen_base_df['Avg_Per_Hour_Net_Growth'].iloc[0]:<12.2f} {qwen_single_df['Avg_Per_Hour_Net_Growth'].iloc[0]:<12.2f}")
    
    # 完成订单数对比
    print(f"{'Complete Orders':<20} {gemini_base_df['Avg_Completed_Orders_Per_Agent'].iloc[0]:<12.2f} {gemini_single_df['Avg_Completed_Orders_Per_Agent'].iloc[0]:<12.2f} {qwen_base_df['Avg_Completed_Orders_Per_Agent'].iloc[0]:<12.2f} {qwen_single_df['Avg_Completed_Orders_Per_Agent'].iloc[0]:<12.2f}")
    
    # 星级评分对比
    print(f"{'Stars':<20} {gemini_base_df['Avg_Stars'].iloc[0]:<12.2f} {gemini_single_df['Avg_Stars'].iloc[0]:<12.2f} {qwen_base_df['Avg_Stars'].iloc[0]:<12.2f} {qwen_single_df['Avg_Stars'].iloc[0]:<12.2f}")
    
    # 准时率对比
    print(f"{'On Time Rate':<20} {gemini_base_df['Avg_On_Time_Order_Rate'].iloc[0]:<12.2f} {gemini_single_df['Avg_On_Time_Order_Rate'].iloc[0]:<12.2f} {qwen_base_df['Avg_On_Time_Order_Rate'].iloc[0]:<12.2f} {qwen_single_df['Avg_On_Time_Order_Rate'].iloc[0]:<12.2f}")
    
    print("-" * 80)
    
    # 分析单代理vs多代理的改进
    print("\n单代理vs多代理性能改进分析")
    print("-" * 50)
    
    # Gemini模型改进
    gemini_net_improvement = ((gemini_single_df['Avg_Per_Hour_Net_Growth'].iloc[0] - gemini_base_df['Avg_Per_Hour_Net_Growth'].iloc[0]) / gemini_base_df['Avg_Per_Hour_Net_Growth'].iloc[0] * 100) if gemini_base_df['Avg_Per_Hour_Net_Growth'].iloc[0] != 0 else 0
    gemini_stars_improvement = ((gemini_single_df['Avg_Stars'].iloc[0] - gemini_base_df['Avg_Stars'].iloc[0]) / gemini_base_df['Avg_Stars'].iloc[0] * 100) if gemini_base_df['Avg_Stars'].iloc[0] != 0 else 0
    gemini_ontime_improvement = ((gemini_single_df['Avg_On_Time_Order_Rate'].iloc[0] - gemini_base_df['Avg_On_Time_Order_Rate'].iloc[0]) / gemini_base_df['Avg_On_Time_Order_Rate'].iloc[0] * 100) if gemini_base_df['Avg_On_Time_Order_Rate'].iloc[0] != 0 else 0
    # Qwen模型改进
    qwen_net_improvement = ((qwen_single_df['Avg_Per_Hour_Net_Growth'].iloc[0] - qwen_base_df['Avg_Per_Hour_Net_Growth'].iloc[0]) / abs(qwen_base_df['Avg_Per_Hour_Net_Growth'].iloc[0]) * 100) if qwen_base_df['Avg_Per_Hour_Net_Growth'].iloc[0] != 0 else 0
    qwen_stars_improvement = ((qwen_single_df['Avg_Stars'].iloc[0] - qwen_base_df['Avg_Stars'].iloc[0]) / qwen_base_df['Avg_Stars'].iloc[0] * 100) if qwen_base_df['Avg_Stars'].iloc[0] != 0 else 0
    qwen_ontime_improvement = ((qwen_single_df['Avg_On_Time_Order_Rate'].iloc[0] - qwen_base_df['Avg_On_Time_Order_Rate'].iloc[0]) / qwen_base_df['Avg_On_Time_Order_Rate'].iloc[0] * 100) if qwen_base_df['Avg_On_Time_Order_Rate'].iloc[0] != 0 else 0
    
    # 5. 创建可视化
    create_contradiction_visualization(gemini_base_df, gemini_single_df, qwen_base_df, qwen_single_df)
    

def create_contradiction_visualization(gemini_base_df, gemini_single_df, qwen_base_df, qwen_single_df):
    """创建关键指标对比的柱状图，区分不同模型"""
    
    # 创建单个柱状图
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    
    # 选择关键指标
    metrics = ['Net Profit', 'Complete Orders', 'Stars', 'On Time Rate']
    
    # 获取Gemini数据
    gemini_base_values = [
        gemini_base_df['Avg_Per_Hour_Net_Growth'].iloc[0],
        gemini_base_df['Avg_Completed_Orders_Per_Agent'].iloc[0],
        gemini_base_df['Avg_Stars'].iloc[0],
        gemini_base_df['Avg_On_Time_Order_Rate'].iloc[0]
    ]
    
    gemini_single_values = [
        gemini_single_df['Avg_Per_Hour_Net_Growth'].iloc[0],
        gemini_single_df['Avg_Completed_Orders_Per_Agent'].iloc[0],
        gemini_single_df['Avg_Stars'].iloc[0],
        gemini_single_df['Avg_On_Time_Order_Rate'].iloc[0]
    ]
    
    # 获取Qwen数据
    qwen_base_values = [
        qwen_base_df['Avg_Per_Hour_Net_Growth'].iloc[0],
        qwen_base_df['Avg_Completed_Orders_Per_Agent'].iloc[0],
        qwen_base_df['Avg_Stars'].iloc[0],
        qwen_base_df['Avg_On_Time_Order_Rate'].iloc[0]
    ]
    
    qwen_single_values = [
        qwen_single_df['Avg_Per_Hour_Net_Growth'].iloc[0],
        qwen_single_df['Avg_Completed_Orders_Per_Agent'].iloc[0],
        qwen_single_df['Avg_Stars'].iloc[0],
        qwen_single_df['Avg_On_Time_Order_Rate'].iloc[0]
    ]
    
    # 设置柱状图位置 - 为四个系列设置位置
    x = np.arange(len(metrics))
    width = 0.15  # 减小柱体宽度
    series_gap = 0.2  # 系列之间的间隔
    inner_gap = 0.05  # 系列内部的间隔
    
    # 绘制柱状图 - 系列内部有适当间隔，系列之间有更大间隔
    bars1 = ax.bar(x - series_gap, gemini_base_values, width, label='Gemini 8 Agents', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x - series_gap + width + inner_gap, gemini_single_values, width, label='Gemini 1 Agent', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + series_gap, qwen_base_values, width, label='Qwen 8 Agents', alpha=0.8, color='lightgreen')
    bars4 = ax.bar(x + series_gap + width + inner_gap, qwen_single_values, width, label='Qwen 1 Agent', alpha=0.8, color='orange')
    
    # 设置标签和标题
    # ax.set_xlabel('Performance Metrics', fontsize=16)
    # ax.set_ylabel('Values', fontsize=16)
    ax.set_title('Performance Comparison: Different Models and Agent Configurations', fontsize=44)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=40)
    ax.legend(fontsize=40, loc='upper right')
    
    # 设置y轴刻度字体大小
    ax.tick_params(axis='y', labelsize=40)
    
    # 调整y轴范围，确保负值也能显示，并为最高柱体顶部留出空间
    max_value = max(max(gemini_base_values), max(gemini_single_values), 
                   max(qwen_base_values), max(qwen_single_values))
    min_value = min(min(gemini_base_values), min(gemini_single_values), 
                   min(qwen_base_values), min(qwen_single_values))
    
    # 如果有负值，确保y轴下限包含负值
    if min_value < 0:
        ax.set_ylim(min_value * 1.1, max_value * 1.15)  # 负值向下扩展10%，正值向上扩展15%
    else:
        ax.set_ylim(0, max_value * 1.15)  # 只有正值时，从0开始
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=40)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('evaluation/contradiction_analysis.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_contradictions()
