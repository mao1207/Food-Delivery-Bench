#!/usr/bin/env python3
"""
Script to visualize and compare performance differences between two ablation study settings.
Compares gemini_medium20_base vs gemini_medium20_single settings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the performance data from both settings."""
    base_path = Path("evaluation/ablation/gemini_medium20_base/model_performance_summary.csv")
    single_path = Path("evaluation/ablation/gemini_medium20_single/model_performance_summary.csv")
    
    base_df = pd.read_csv(base_path)
    single_df = pd.read_csv(single_path)
    
    return base_df, single_df

def create_comparison_visualization(base_df, single_df):
    """Create comprehensive comparison visualizations."""
    
    # Extract key metrics for comparison
    key_metrics = {
        'Financial Performance': [
            'Avg_Per_Hour_Net_Growth',
            'Avg_Per_Hour_Orders_Income', 
            'Avg_Per_Hour_Expense_Total',
            'Avg_Net_Growth'
        ],
        'Order Performance': [
            'Avg_On_Time_Order_Rate',
            'Avg_Completed_Orders_Per_Agent',
            'Avg_Timeout_Orders_Per_Agent',
            'Avg_Orders_Per_Hour'
        ],
        'Quality Metrics': [
            'Avg_Stars',
            'Avg_Food_Stars',
            'Avg_Temp_Ok_Rate',
            'Avg_Odor_Ok_Rate',
            'Avg_Damage_Ok_Rate'
        ],
        'Efficiency Metrics': [
            'Avg_Interruptions_Per_Hour',
            'Avg_Violation_Rate',
            'Avg_VLM_Success_Rate',
            'Avg_Active_Hours_Per_Agent'
        ]
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Financial Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    financial_data = []
    for metric in key_metrics['Financial Performance']:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        financial_data.append([metric.replace('Avg_', '').replace('_', ' '), base_val, single_val])
    
    df_financial = pd.DataFrame(financial_data, columns=['Metric', 'Base', 'Single'])
    df_financial.set_index('Metric').plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Financial Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.legend(['Base Setting', 'Single Setting'])
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Order Performance Comparison
    ax2 = plt.subplot(2, 3, 2)
    order_data = []
    for metric in key_metrics['Order Performance']:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        order_data.append([metric.replace('Avg_', '').replace('_', ' '), base_val, single_val])
    
    df_order = pd.DataFrame(order_data, columns=['Metric', 'Base', 'Single'])
    df_order.set_index('Metric').plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Order Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.legend(['Base Setting', 'Single Setting'])
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Quality Metrics Comparison
    ax3 = plt.subplot(2, 3, 3)
    quality_data = []
    for metric in key_metrics['Quality Metrics']:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        quality_data.append([metric.replace('Avg_', '').replace('_', ' '), base_val, single_val])
    
    df_quality = pd.DataFrame(quality_data, columns=['Metric', 'Base', 'Single'])
    df_quality.set_index('Metric').plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Quality Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.legend(['Base Setting', 'Single Setting'])
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Efficiency Metrics Comparison
    ax4 = plt.subplot(2, 3, 4)
    efficiency_data = []
    for metric in key_metrics['Efficiency Metrics']:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        efficiency_data.append([metric.replace('Avg_', '').replace('_', ' '), base_val, single_val])
    
    df_efficiency = pd.DataFrame(efficiency_data, columns=['Metric', 'Base', 'Single'])
    df_efficiency.set_index('Metric').plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Efficiency Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.legend(['Base Setting', 'Single Setting'])
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Performance Improvement Heatmap
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate percentage improvements
    improvement_data = []
    all_metrics = []
    for category, metrics in key_metrics.items():
        for metric in metrics:
            base_val = base_df[metric].iloc[0]
            single_val = single_df[metric].iloc[0]
            if base_val != 0:
                improvement = ((single_val - base_val) / base_val) * 100
            else:
                improvement = 0 if single_val == 0 else float('inf')
            improvement_data.append(improvement)
            all_metrics.append(metric.replace('Avg_', '').replace('_', ' '))
    
    # Create heatmap data
    improvement_matrix = np.array(improvement_data).reshape(-1, 1)
    sns.heatmap(improvement_matrix, 
                xticklabels=['Improvement %'],
                yticklabels=all_metrics,
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax5)
    ax5.set_title('Performance Improvement (%)', fontsize=14, fontweight='bold')
    
    # 6. Key Metrics Radar Chart
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Select key metrics for radar chart
    radar_metrics = [
        'Avg_Per_Hour_Net_Growth',
        'Avg_On_Time_Order_Rate', 
        'Avg_Stars',
        'Avg_VLM_Success_Rate',
        'Avg_Completed_Orders_Per_Agent',
        'Avg_Active_Hours_Per_Agent'
    ]
    
    # Normalize values for radar chart (0-1 scale)
    base_values = []
    single_values = []
    labels = []
    
    for metric in radar_metrics:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        
        # Normalize based on the maximum value between both settings
        max_val = max(base_val, single_val)
        if max_val > 0:
            base_values.append(base_val / max_val)
            single_values.append(single_val / max_val)
        else:
            base_values.append(0)
            single_values.append(0)
        
        labels.append(metric.replace('Avg_', '').replace('_', ' '))
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    base_values += base_values[:1]  # Complete the circle
    single_values += single_values[:1]
    angles += angles[:1]
    
    ax6.plot(angles, base_values, 'o-', linewidth=2, label='Base Setting', color='blue')
    ax6.fill(angles, base_values, alpha=0.25, color='blue')
    ax6.plot(angles, single_values, 'o-', linewidth=2, label='Single Setting', color='red')
    ax6.fill(angles, single_values, alpha=0.25, color='red')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels)
    ax6.set_title('Key Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation/ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(base_df, single_df):
    """Create a summary comparison table."""
    
    key_metrics = [
        'Avg_Per_Hour_Net_Growth',
        'Avg_Per_Hour_Orders_Income',
        'Avg_Per_Hour_Expense_Total',
        'Avg_On_Time_Order_Rate',
        'Avg_Completed_Orders_Per_Agent',
        'Avg_Timeout_Orders_Per_Agent',
        'Avg_Stars',
        'Avg_Food_Stars',
        'Avg_Violation_Rate',
        'Avg_VLM_Success_Rate',
        'Avg_Active_Hours_Per_Agent'
    ]
    
    comparison_data = []
    for metric in key_metrics:
        base_val = base_df[metric].iloc[0]
        single_val = single_df[metric].iloc[0]
        
        if base_val != 0:
            improvement = ((single_val - base_val) / base_val) * 100
        else:
            improvement = 0 if single_val == 0 else float('inf')
        
        comparison_data.append({
            'Metric': metric.replace('Avg_', '').replace('_', ' '),
            'Base_Setting': f"{base_val:.3f}",
            'Single_Setting': f"{single_val:.3f}",
            'Improvement_%': f"{improvement:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return comparison_df

def main():
    """Main function to run the comparison analysis."""
    print("Loading performance data...")
    base_df, single_df = load_data()
    
    print("Creating comparison visualizations...")
    fig = create_comparison_visualization(base_df, single_df)
    
    print("Generating summary table...")
    summary_df = create_summary_table(base_df, single_df)
    
    print("\nAnalysis complete! Check 'evaluation/ablation_comparison.png' for visualizations.")
    
    # Save summary table
    summary_df.to_csv('evaluation/ablation_comparison_summary.csv', index=False)
    print("Summary table saved to 'evaluation/ablation_comparison_summary.csv'")

if __name__ == "__main__":
    main()
