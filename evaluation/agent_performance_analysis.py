#!/usr/bin/env python3
"""
Agent Performance Analysis
=========================

This script analyzes the performance of different AI agents in the food delivery simulation.
It compares key metrics across all agents and provides insights into their capabilities.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class AgentPerformanceAnalyzer:
    def __init__(self, results_dir="results/20250915_184732"):
        self.results_dir = Path(results_dir)
        self.agents_data = {}
        self.load_agent_data()
    
    def load_agent_data(self):
        """Load all agent run reports from the results directory."""
        for agent_file in self.results_dir.glob("run_report_agent*.json"):
            with open(agent_file, 'r') as f:
                data = json.load(f)
                agent_id = data['meta']['agent_id']
                self.agents_data[agent_id] = data
        print(f"Loaded data for {len(self.agents_data)} agents")
    
    def extract_key_metrics(self):
        """Extract key performance metrics from all agents."""
        metrics = []
        
        for agent_id, data in self.agents_data.items():
            meta = data['meta']
            money = data['money']['totals']
            orders = data['orders']
            vlm = data['vlm']
            
            agent_metrics = {
                'agent_id': agent_id,
                'model': meta['model'],
                'net_growth': money['net_growth'],
                'completed_orders': orders['completed_count'],
                'timeout_orders': orders['timeout_count'],
                'avg_stars': orders['avg_stars'],
                'vlm_total_calls': vlm['total_calls'],
                'vlm_success_calls': vlm['successes'],
                'vlm_success_rate': vlm['success_rate'],
                'active_hours': meta['active_hours'],
                'orders_per_hour': data['activity']['orders_per_hour'],
                'temp_ok_rate': orders['temp_ok_rate'],
                'odor_ok_rate': orders['odor_ok_rate'],
                'damage_ok_rate': orders['damage_ok_rate'],
                'method_success_rate': orders['method_success_rate']
            }
            metrics.append(agent_metrics)
        
        return pd.DataFrame(metrics)
    
    def create_comparison_table(self):
        """Create a comprehensive comparison table."""
        df = self.extract_key_metrics()
        
        # Sort by net_growth descending
        df_sorted = df.sort_values('net_growth', ascending=False)
        
        print("=" * 80)
        print("AGENT PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Main comparison table
        comparison_cols = [
            'agent_id', 'model', 'net_growth', 'completed_orders', 
            'timeout_orders', 'avg_stars', 'vlm_total_calls', 'vlm_success_calls'
        ]
        
        print("\nKey Performance Metrics:")
        print("-" * 80)
        print(df_sorted[comparison_cols].to_string(index=False, float_format='%.2f'))
        
        # Additional metrics table
        print("\n\nAdditional Performance Metrics:")
        print("-" * 80)
        additional_cols = [
            'agent_id', 'model', 'vlm_success_rate', 'active_hours', 
            'orders_per_hour', 'temp_ok_rate', 'odor_ok_rate', 'damage_ok_rate'
        ]
        print(df_sorted[additional_cols].to_string(index=False, float_format='%.3f'))
        
        return df_sorted
    
    def analyze_performance_patterns(self, df):
        """Analyze performance patterns and provide insights."""
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS & INSIGHTS")
        print("=" * 80)
        
        # Top performers
        print("\n🏆 TOP PERFORMERS:")
        print("-" * 40)
        top_3 = df.head(3)
        for _, agent in top_3.iterrows():
            print(f"Agent {agent['agent_id']} ({agent['model']}):")
            print(f"  • Net Growth: ${agent['net_growth']:.2f}")
            print(f"  • Completed Orders: {agent['completed_orders']}")
            print(f"  • Average Stars: {agent['avg_stars']:.1f}")
            print(f"  • VLM Success Rate: {agent['vlm_success_rate']:.1%}")
            print()
        
        # Performance correlations
        print("\n📊 PERFORMANCE CORRELATIONS:")
        print("-" * 40)
        
        # VLM success rate vs performance
        vlm_correlation = df[['vlm_success_rate', 'net_growth', 'completed_orders', 'avg_stars']].corr()
        print("VLM Success Rate correlations:")
        print(f"  • With Net Growth: {vlm_correlation.loc['vlm_success_rate', 'net_growth']:.3f}")
        print(f"  • With Completed Orders: {vlm_correlation.loc['vlm_success_rate', 'completed_orders']:.3f}")
        print(f"  • With Average Stars: {vlm_correlation.loc['vlm_success_rate', 'avg_stars']:.3f}")
        
        # Model performance analysis
        print("\n🤖 MODEL PERFORMANCE ANALYSIS:")
        print("-" * 40)
        model_stats = df.groupby('model').agg({
            'net_growth': ['mean', 'std'],
            'completed_orders': ['mean', 'std'],
            'avg_stars': ['mean', 'std'],
            'vlm_success_rate': ['mean', 'std']
        }).round(3)
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            print(f"\n{model}:")
            print(f"  • Agents: {len(model_data)}")
            print(f"  • Avg Net Growth: ${model_data['net_growth'].mean():.2f} ± {model_data['net_growth'].std():.2f}")
            print(f"  • Avg Completed Orders: {model_data['completed_orders'].mean():.1f} ± {model_data['completed_orders'].std():.1f}")
            print(f"  • Avg Stars: {model_data['avg_stars'].mean():.2f} ± {model_data['avg_stars'].std():.2f}")
            print(f"  • Avg VLM Success Rate: {model_data['vlm_success_rate'].mean():.1%} ± {model_data['vlm_success_rate'].std():.1%}")
        
        # Key insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 40)
        
        # Best overall performer
        best_agent = df.iloc[0]
        print(f"• Best Overall Performer: Agent {best_agent['agent_id']} ({best_agent['model']})")
        print(f"  - Highest net growth: ${best_agent['net_growth']:.2f}")
        print(f"  - Most completed orders: {best_agent['completed_orders']}")
        
        # VLM performance insights
        high_vlm_success = df[df['vlm_success_rate'] > 0.9]
        print(f"• {len(high_vlm_success)} agents achieved >90% VLM success rate")
        
        # Zero performance agents
        zero_performance = df[df['completed_orders'] == 0]
        if len(zero_performance) > 0:
            print(f"• {len(zero_performance)} agents completed no orders:")
            for _, agent in zero_performance.iterrows():
                print(f"  - Agent {agent['agent_id']} ({agent['model']})")
        
        # Temperature control issues
        temp_issues = df[df['temp_ok_rate'] < 0.5]
        print(f"• {len(temp_issues)} agents had temperature control issues (<50% temp_ok_rate)")
        
        return df
    
    def create_visualizations(self, df):
        """Create performance visualization charts."""
        plt.style.use('seaborn-v0_8')
        
        # Sort by agent_id to maintain order 1-8
        df_sorted = df.sort_values('agent_id')
        
        # Create model name labels for x-axis
        model_labels = []
        for _, row in df_sorted.iterrows():
            model_name = row['model'].split('/')[-1]  # Get the last part after '/'
            model_labels.append(f"{model_name}")
        
        # Chart 1: Financial Performance
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('Financial Performance Analysis', fontsize=16, fontweight='bold')
        
        # Net Growth
        bars1 = axes1[0].bar(range(len(df_sorted)), df_sorted['net_growth'], 
                             color=['green' if x >= 0 else 'red' for x in df_sorted['net_growth']], 
                             alpha=0.7)
        axes1[0].set_title('Net Growth by Agent', fontsize=14)
        axes1[0].set_xlabel('Agent & Model', fontsize=12)
        axes1[0].set_ylabel('Net Growth ($)', fontsize=12)
        axes1[0].set_xticks(range(len(df_sorted)))
        axes1[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes1[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, df_sorted['net_growth'])):
            height = bar.get_height()
            axes1[0].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                         f'${value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Orders Performance
        x = np.arange(len(df_sorted))
        width = 0.35
        bars2 = axes1[1].bar(x - width/2, df_sorted['completed_orders'], width, 
                             label='Completed', color='green', alpha=0.7)
        bars3 = axes1[1].bar(x + width/2, df_sorted['timeout_orders'], width, 
                             label='Timeout', color='red', alpha=0.7)
        axes1[1].set_title('Completed vs Timeout Orders', fontsize=14)
        axes1[1].set_xlabel('Agent & Model', fontsize=12)
        axes1[1].set_ylabel('Number of Orders', fontsize=12)
        axes1[1].set_xticks(x)
        axes1[1].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes1[1].legend()
        
        # Add value labels
        for bar, value in zip(bars2, df_sorted['completed_orders']):
            if value > 0:
                height = bar.get_height()
                axes1[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars3, df_sorted['timeout_orders']):
            if value > 0:
                height = bar.get_height()
                axes1[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('evaluation/financial_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 2: Quality and VLM Performance
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('Quality and VLM Performance Analysis', fontsize=16, fontweight='bold')
        
        # Average Stars
        bars4 = axes2[0].bar(range(len(df_sorted)), df_sorted['avg_stars'], 
                             color='gold', alpha=0.7)
        axes2[0].set_title('Average Customer Rating (Stars)', fontsize=14)
        axes2[0].set_xlabel('Agent & Model', fontsize=12)
        axes2[0].set_ylabel('Average Stars', fontsize=12)
        axes2[0].set_xticks(range(len(df_sorted)))
        axes2[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes2[0].set_ylim(0, 5)
        
        # Add value labels
        for bar, value in zip(bars4, df_sorted['avg_stars']):
            if value > 0:
                height = bar.get_height()
                axes2[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # VLM Performance - Grouped Bar Chart
        x_pos = np.arange(len(df_sorted))
        width = 0.35
        
        # Create bars for total and successful calls
        bars1 = axes2[1].bar(x_pos - width/2, df_sorted['vlm_total_calls'], width, 
                            label='Total VLM Calls', color='lightcoral', alpha=0.8)
        bars2 = axes2[1].bar(x_pos + width/2, df_sorted['vlm_success_calls'], width, 
                            label='Successful VLM Calls', color='lightgreen', alpha=0.8)
        
        axes2[1].set_title('VLM Performance: Total vs Successful Calls', fontsize=14)
        axes2[1].set_xlabel('Model', fontsize=12)
        axes2[1].set_ylabel('Number of Calls', fontsize=12)
        axes2[1].set_xticks(x_pos)
        axes2[1].set_xticklabels(model_labels, rotation=45, ha='right')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        
        plt.tight_layout()
        plt.savefig('evaluation/quality_vlm_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 3: Food Quality Metrics
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Food Quality Metrics Analysis', fontsize=16, fontweight='bold')
        
        quality_metrics = ['temp_ok_rate', 'odor_ok_rate', 'damage_ok_rate', 'method_success_rate']
        quality_titles = ['Temperature Control', 'Odor Preservation', 'Damage Prevention', 'Delivery Method']
        colors = ['red', 'orange', 'yellow', 'green']
        
        for i, (metric, title, color) in enumerate(zip(quality_metrics, quality_titles, colors)):
            row = i // 2
            col = i % 2
            
            bars = axes3[row, col].bar(range(len(df_sorted)), df_sorted[metric], 
                                      color=color, alpha=0.7)
            axes3[row, col].set_title(f'{title} Success Rate', fontsize=14)
            axes3[row, col].set_xlabel('Agent & Model', fontsize=12)
            axes3[row, col].set_ylabel('Success Rate', fontsize=12)
            axes3[row, col].set_xticks(range(len(df_sorted)))
            axes3[row, col].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
            axes3[row, col].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, df_sorted[metric]):
                height = bar.get_height()
                axes3[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{value:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('evaluation/food_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 4: Activity and Efficiency Metrics
        fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
        fig4.suptitle('Activity and Efficiency Metrics', fontsize=16, fontweight='bold')
        
        # Active Hours
        bars5 = axes4[0].bar(range(len(df_sorted)), df_sorted['active_hours'], 
                             color='lightblue', alpha=0.7)
        axes4[0].set_title('Active Hours by Agent', fontsize=14)
        axes4[0].set_xlabel('Agent & Model', fontsize=12)
        axes4[0].set_ylabel('Active Hours', fontsize=12)
        axes4[0].set_xticks(range(len(df_sorted)))
        axes4[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars5, df_sorted['active_hours']):
            height = bar.get_height()
            axes4[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{value:.2f}h', ha='center', va='bottom', fontsize=9)
        
        # Orders per Hour
        bars6 = axes4[1].bar(range(len(df_sorted)), df_sorted['orders_per_hour'], 
                             color='lightgreen', alpha=0.7)
        axes4[1].set_title('Orders per Hour by Agent', fontsize=14)
        axes4[1].set_xlabel('Agent & Model', fontsize=12)
        axes4[1].set_ylabel('Orders per Hour', fontsize=12)
        axes4[1].set_xticks(range(len(df_sorted)))
        axes4[1].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars6, df_sorted['orders_per_hour']):
            height = bar.get_height()
            axes4[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('evaluation/activity_efficiency_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report."""
        report = f"""
# Agent Performance Analysis Report

## Executive Summary
This report analyzes the performance of {len(df)} AI agents in a food delivery simulation environment.

## Key Findings

### Top Performers
1. **Agent {df.iloc[0]['agent_id']}** ({df.iloc[0]['model']})
   - Net Growth: ${df.iloc[0]['net_growth']:.2f}
   - Completed Orders: {df.iloc[0]['completed_orders']}
   - Average Stars: {df.iloc[0]['avg_stars']:.1f}

2. **Agent {df.iloc[1]['agent_id']}** ({df.iloc[1]['model']})
   - Net Growth: ${df.iloc[1]['net_growth']:.2f}
   - Completed Orders: {df.iloc[1]['completed_orders']}
   - Average Stars: {df.iloc[1]['avg_stars']:.1f}

3. **Agent {df.iloc[2]['agent_id']}** ({df.iloc[2]['model']})
   - Net Growth: ${df.iloc[2]['net_growth']:.2f}
   - Completed Orders: {df.iloc[2]['completed_orders']}
   - Average Stars: {df.iloc[2]['avg_stars']:.1f}

### Performance Statistics
- **Total Agents**: {len(df)}
- **Average Net Growth**: ${df['net_growth'].mean():.2f} ± ${df['net_growth'].std():.2f}
- **Total Completed Orders**: {df['completed_orders'].sum()}
- **Average Stars**: {df['avg_stars'].mean():.2f} ± {df['avg_stars'].std():.2f}
- **Average VLM Success Rate**: {df['vlm_success_rate'].mean():.1%} ± {df['vlm_success_rate'].std():.1%}

### Model Performance Ranking
"""
        
        # Add model ranking
        model_ranking = df.groupby('model')['net_growth'].mean().sort_values(ascending=False)
        for i, (model, avg_growth) in enumerate(model_ranking.items(), 1):
            report += f"{i}. {model}: ${avg_growth:.2f}\n"
        
        report += f"""
### Key Insights
- **Best Performing Model**: {model_ranking.index[0]}
- **Most Reliable VLM**: {df.loc[df['vlm_success_rate'].idxmax(), 'model']} (Agent {df.loc[df['vlm_success_rate'].idxmax(), 'agent_id']})
- **Most Productive Agent**: Agent {df.loc[df['completed_orders'].idxmax(), 'agent_id']} ({df.loc[df['completed_orders'].idxmax(), 'model']})

### Recommendations
1. Focus on agents with high VLM success rates for better task execution
2. Investigate temperature control issues affecting food quality
3. Analyze successful strategies from top-performing agents
4. Consider model-specific optimizations based on performance patterns
"""
        
        return report
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Agent Performance Analysis...")
        
        # Extract and display metrics
        df = self.create_comparison_table()
        
        # Analyze patterns
        df = self.analyze_performance_patterns(df)
        
        # Create visualizations
        print("\n📊 Creating performance visualizations...")
        self.create_visualizations(df)
        
        # Generate summary report
        # print("\n📝 Generating summary report...")
        # report = self.generate_summary_report(df)
        
        # Save report
        # with open('evaluation/performance_analysis_report.md', 'w') as f:
        #     f.write(report)
        
        print("\n✅ Analysis complete!")
        print("Files generated:")
        print("  - evaluation/financial_performance.png")
        print("  - evaluation/quality_vlm_performance.png")
        print("  - evaluation/food_quality_metrics.png")
        print("  - evaluation/activity_efficiency_metrics.png")
        print("  - evaluation/performance_analysis_report.md")
        
        return df

def main():
    """Main function to run the analysis."""
    analyzer = AgentPerformanceAnalyzer()
    results_df = analyzer.run_analysis()
    return results_df

if __name__ == "__main__":
    results = main()
