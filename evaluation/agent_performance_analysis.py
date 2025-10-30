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
import os
from typing import List, Union

class AgentPerformanceAnalyzer:
    def __init__(self, results_dirs: Union[str, List[str]] = "results/20250915_184732", output_dir=None):
        # Handle both single directory and multiple directories
        if isinstance(results_dirs, str):
            self.results_dirs = [Path(results_dirs)]
        else:
            self.results_dirs = [Path(d) for d in results_dirs]
        
        self.agents_data = {}
        self.model_grouped_data = {}  # Store data grouped by model for averaging
        
        # Generate output directory name based on input paths if not specified
        if output_dir is None:
            # Create a combined name from all input directories
            if len(self.results_dirs) == 1:
                self.output_dir = Path("evaluation") / self.results_dirs[0].name
            else:
                # Use a combined name for multiple directories
                dir_names = [d.name for d in self.results_dirs]
                combined_name = "_".join(sorted(dir_names))
                self.output_dir = Path("evaluation") / f"combined_{combined_name}"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.load_agent_data()
    
    def load_agent_data(self):
        """Load all agent run reports from the results directories."""
        total_agents = 0
        for results_dir in self.results_dirs:
            print(f"Loading data from: {results_dir}")
            for agent_file in results_dir.glob("run_report_agent*.json"):
                with open(agent_file, 'r') as f:
                    data = json.load(f)
                    agent_id = data['meta']['agent_id']
                    model = data['meta']['model']
                    
                    # Create unique key combining directory and agent_id
                    unique_key = f"{results_dir.name}_{agent_id}"
                    self.agents_data[unique_key] = data
                    
                    # Group by model for averaging
                    if model not in self.model_grouped_data:
                        self.model_grouped_data[model] = []
                    self.model_grouped_data[model].append(data)
                    
                    total_agents += 1
        
        print(f"Loaded data for {total_agents} agents from {len(self.results_dirs)} directories")
        print(f"Models found: {list(self.model_grouped_data.keys())}")
        print(f"Output directory: {self.output_dir}")
    
    def extract_key_metrics(self):
        """Extract key performance metrics from all agents."""
        metrics = []
        
        for unique_key, data in self.agents_data.items():
            meta = data['meta']
            money = data['money']['totals']
            orders = data['orders']
            vlm = data['vlm']
            
            agent_metrics = {
                'unique_key': unique_key,
                'agent_id': meta['agent_id'],
                'model': meta['model'],
                'source_dir': unique_key.split('_')[0],  # Extract directory name
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
    
    def extract_detailed_metrics_for_csv(self):
        """Extract detailed metrics for CSV output grouped by model."""
        detailed_metrics = []
        
        for unique_key, data in self.agents_data.items():
            meta = data['meta']
            money = data['money']
            orders = data['orders']
            interruptions = data['interruptions']
            social = data['social']
            actions = data['actions']
            
            # Calculate on_time_order_rate
            total_orders = orders['completed_count'] + orders['timeout_count']
            on_time_orders = 0
            if 'details' in orders:
                for order in orders['details']:
                    if order.get('on_time', False):
                        on_time_orders += 1
            on_time_order_rate = on_time_orders / total_orders if total_orders > 0 else 0
            
            # Calculate total interruptions
            interruptions_num = (interruptions.get('scooter_depleted', 0) + 
                               interruptions.get('hospital_rescue', 0) + 
                               interruptions.get('rent_insufficient', 0) + 
                               interruptions.get('charge_insufficient', 0))
            
            # Extract social metrics
            num_help_posted = social.get('help_posted', 0)
            num_help_accepted = social.get('help_accepted', 0)
            
            # Extract say count from actions if available
            num_say = actions.get('say', {}).get('attempts', 0) if 'say' in actions else 0
            
            agent_metrics = {
                'unique_key': unique_key,
                'agent_id': meta['agent_id'],
                'model': meta['model'],
                'source_dir': unique_key.split('_')[0],
                'per_hour_net_growth': money['per_hour']['net_growth'],
                'per_hour_orders_income': money['per_hour']['orders_income'],
                'per_hour_expense_total': money['per_hour']['expense_total'],
                'on_time_order_rate': on_time_order_rate,
                'avg_stars': orders['avg_stars'],
                'avg_food_stars': orders.get('avg_food_stars', 0),
                'interruptions_num': interruptions_num,
                'num_help_posted': num_help_posted,
                'num_help_accepted': num_help_accepted,
                'num_say': num_say,
                'completed_orders': orders['completed_count'],
                'timeout_orders': orders['timeout_count'],
                'active_hours': meta['active_hours']
            }
            detailed_metrics.append(agent_metrics)
        
        return pd.DataFrame(detailed_metrics)
    
    def create_model_averaged_data(self):
        """Create averaged data for each model across all directories."""
        model_averaged = {}
        
        for model, data_list in self.model_grouped_data.items():
            if not data_list:
                continue
                
            # Initialize aggregated data structure
            aggregated = {
                'model': model,
                'agent_count': len(data_list),
                'source_directories': list(set([d['meta'].get('source_dir', 'unknown') for d in data_list])),
            }
            
            # Aggregate financial data
            money_totals = [d['money']['totals'] for d in data_list]
            money_per_hour = [d['money']['per_hour'] for d in data_list]
            
            aggregated.update({
                'net_growth_mean': np.mean([m['net_growth'] for m in money_totals]),
                'net_growth_std': np.std([m['net_growth'] for m in money_totals]),
                'per_hour_net_growth_mean': np.mean([m['net_growth'] for m in money_per_hour]),
                'per_hour_net_growth_std': np.std([m['net_growth'] for m in money_per_hour]),
                'per_hour_orders_income_mean': np.mean([m['orders_income'] for m in money_per_hour]),
                'per_hour_orders_income_std': np.std([m['orders_income'] for m in money_per_hour]),
                'per_hour_expense_total_mean': np.mean([m['expense_total'] for m in money_per_hour]),
                'per_hour_expense_total_std': np.std([m['expense_total'] for m in money_per_hour]),
            })
            
            # Aggregate order data
            orders_data = [d['orders'] for d in data_list]
            aggregated.update({
                'completed_orders_mean': np.mean([o['completed_count'] for o in orders_data]),
                'completed_orders_std': np.std([o['completed_count'] for o in orders_data]),
                'completed_orders_sum': np.sum([o['completed_count'] for o in orders_data]),
                'timeout_orders_mean': np.mean([o['timeout_count'] for o in orders_data]),
                'timeout_orders_std': np.std([o['timeout_count'] for o in orders_data]),
                'timeout_orders_sum': np.sum([o['timeout_count'] for o in orders_data]),
                'avg_stars_mean': np.mean([o['avg_stars'] for o in orders_data]),
                'avg_stars_std': np.std([o['avg_stars'] for o in orders_data]),
                'avg_food_stars_mean': np.mean([o.get('avg_food_stars', 0) for o in orders_data]),
                'avg_food_stars_std': np.std([o.get('avg_food_stars', 0) for o in orders_data]),
                'temp_ok_rate_mean': np.mean([o['temp_ok_rate'] for o in orders_data]),
                'temp_ok_rate_std': np.std([o['temp_ok_rate'] for o in orders_data]),
                'odor_ok_rate_mean': np.mean([o['odor_ok_rate'] for o in orders_data]),
                'odor_ok_rate_std': np.std([o['odor_ok_rate'] for o in orders_data]),
                'damage_ok_rate_mean': np.mean([o['damage_ok_rate'] for o in orders_data]),
                'damage_ok_rate_std': np.std([o['damage_ok_rate'] for o in orders_data]),
                'method_success_rate_mean': np.mean([o['method_success_rate'] for o in orders_data]),
                'method_success_rate_std': np.std([o['method_success_rate'] for o in orders_data]),
            })
            
            # Calculate on_time_order_rate
            on_time_rates = []
            for orders in orders_data:
                total_orders = orders['completed_count'] + orders['timeout_count']
                on_time_orders = 0
                if 'details' in orders:
                    for order in orders['details']:
                        if order.get('on_time', False):
                            on_time_orders += 1
                on_time_rate = on_time_orders / total_orders if total_orders > 0 else 0
                on_time_rates.append(on_time_rate)
            
            aggregated.update({
                'on_time_order_rate_mean': np.mean(on_time_rates),
                'on_time_order_rate_std': np.std(on_time_rates),
            })
            
            # Aggregate VLM data
            vlm_data = [d['vlm'] for d in data_list]
            aggregated.update({
                'vlm_total_calls_mean': np.mean([v['total_calls'] for v in vlm_data]),
                'vlm_total_calls_std': np.std([v['total_calls'] for v in vlm_data]),
                'vlm_success_calls_mean': np.mean([v['successes'] for v in vlm_data]),
                'vlm_success_calls_std': np.std([v['successes'] for v in vlm_data]),
                'vlm_success_rate_mean': np.mean([v['success_rate'] for v in vlm_data]),
                'vlm_success_rate_std': np.std([v['success_rate'] for v in vlm_data]),
            })
            
            # Aggregate activity data
            activity_data = [d['activity'] for d in data_list]
            aggregated.update({
                'orders_per_hour_mean': np.mean([a['orders_per_hour'] for a in activity_data]),
                'orders_per_hour_std': np.std([a['orders_per_hour'] for a in activity_data]),
            })
            
            # Aggregate interruptions and social data
            interruptions_data = [d['interruptions'] for d in data_list]
            social_data = [d['social'] for d in data_list]
            
            interruptions_totals = []
            help_posted_totals = []
            help_accepted_totals = []
            say_totals = []
            
            for i, (interruptions, social) in enumerate(zip(interruptions_data, social_data)):
                interruptions_total = (interruptions.get('scooter_depleted', 0) + 
                                     interruptions.get('hospital_rescue', 0) + 
                                     interruptions.get('rent_insufficient', 0) + 
                                     interruptions.get('charge_insufficient', 0))
                interruptions_totals.append(interruptions_total)
                
                help_posted_totals.append(social.get('help_posted', 0))
                help_accepted_totals.append(social.get('help_accepted', 0))
                
                # Extract say count from actions
                actions = data_list[i]['actions']
                say_count = actions.get('say', {}).get('attempts', 0) if 'say' in actions else 0
                say_totals.append(say_count)
            
            aggregated.update({
                'interruptions_num_mean': np.mean(interruptions_totals),
                'interruptions_num_std': np.std(interruptions_totals),
                'interruptions_num_sum': np.sum(interruptions_totals),
                'num_help_posted_mean': np.mean(help_posted_totals),
                'num_help_posted_std': np.std(help_posted_totals),
                'num_help_posted_sum': np.sum(help_posted_totals),
                'num_help_accepted_mean': np.mean(help_accepted_totals),
                'num_help_accepted_std': np.std(help_accepted_totals),
                'num_help_accepted_sum': np.sum(help_accepted_totals),
                'num_say_mean': np.mean(say_totals),
                'num_say_std': np.std(say_totals),
                'num_say_sum': np.sum(say_totals),
            })
            
            # Aggregate active hours
            active_hours = [d['meta']['active_hours'] for d in data_list]
            aggregated.update({
                'active_hours_mean': np.mean(active_hours),
                'active_hours_std': np.std(active_hours),
                'active_hours_sum': np.sum(active_hours),
            })
            
            model_averaged[model] = aggregated
        
        return model_averaged
    
    def create_model_summary_csv(self):
        """Create a CSV summary grouped by model with averaged data."""
        model_averaged = self.create_model_averaged_data()
        
        # Convert to DataFrame
        df_list = []
        for model, data in model_averaged.items():
            # Remove unwanted fields for CSV output
            data_clean = {k: v for k, v in data.items() if k not in ['agent_count', 'source_directories']}
            df_list.append(data_clean)
        
        model_summary = pd.DataFrame(df_list)
        
        # Rename columns for better readability
        column_mapping = {
            'model': 'Model',
            'net_growth_mean': 'Avg_Net_Growth',
            'net_growth_std': 'Std_Net_Growth',
            'per_hour_net_growth_mean': 'Avg_Per_Hour_Net_Growth',
            'per_hour_net_growth_std': 'Std_Per_Hour_Net_Growth',
            'per_hour_orders_income_mean': 'Avg_Per_Hour_Orders_Income',
            'per_hour_orders_income_std': 'Std_Per_Hour_Orders_Income',
            'per_hour_expense_total_mean': 'Avg_Per_Hour_Expense_Total',
            'per_hour_expense_total_std': 'Std_Per_Hour_Expense_Total',
            'completed_orders_mean': 'Avg_Completed_Orders_Per_Agent',
            'completed_orders_std': 'Std_Completed_Orders_Per_Agent',
            'completed_orders_sum': 'Total_Completed_Orders',
            'timeout_orders_mean': 'Avg_Timeout_Orders_Per_Agent',
            'timeout_orders_std': 'Std_Timeout_Orders_Per_Agent',
            'timeout_orders_sum': 'Total_Timeout_Orders',
            'avg_stars_mean': 'Avg_Stars',
            'avg_stars_std': 'Std_Avg_Stars',
            'avg_food_stars_mean': 'Avg_Food_Stars',
            'avg_food_stars_std': 'Std_Avg_Food_Stars',
            'on_time_order_rate_mean': 'Avg_On_Time_Order_Rate',
            'on_time_order_rate_std': 'Std_On_Time_Order_Rate',
            'temp_ok_rate_mean': 'Avg_Temp_Ok_Rate',
            'temp_ok_rate_std': 'Std_Temp_Ok_Rate',
            'odor_ok_rate_mean': 'Avg_Odor_Ok_Rate',
            'odor_ok_rate_std': 'Std_Odor_Ok_Rate',
            'damage_ok_rate_mean': 'Avg_Damage_Ok_Rate',
            'damage_ok_rate_std': 'Std_Damage_Ok_Rate',
            'method_success_rate_mean': 'Avg_Method_Success_Rate',
            'method_success_rate_std': 'Std_Method_Success_Rate',
            'vlm_total_calls_mean': 'Avg_VLM_Total_Calls',
            'vlm_total_calls_std': 'Std_VLM_Total_Calls',
            'vlm_success_calls_mean': 'Avg_VLM_Success_Calls',
            'vlm_success_calls_std': 'Std_VLM_Success_Calls',
            'vlm_success_rate_mean': 'Avg_VLM_Success_Rate',
            'vlm_success_rate_std': 'Std_VLM_Success_Rate',
            'orders_per_hour_mean': 'Avg_Orders_Per_Hour',
            'orders_per_hour_std': 'Std_Orders_Per_Hour',
            'interruptions_num_mean': 'Avg_Interruptions_Per_Agent',
            'interruptions_num_std': 'Std_Interruptions_Per_Agent',
            'interruptions_num_sum': 'Total_Interruptions',
            'num_help_posted_mean': 'Avg_Help_Posted_Per_Agent',
            'num_help_posted_std': 'Std_Help_Posted_Per_Agent',
            'num_help_posted_sum': 'Total_Help_Posted',
            'num_help_accepted_mean': 'Avg_Help_Accepted_Per_Agent',
            'num_help_accepted_std': 'Std_Help_Accepted_Per_Agent',
            'num_help_accepted_sum': 'Total_Help_Accepted',
            'num_say_mean': 'Avg_Say_Per_Agent',
            'num_say_std': 'Std_Say_Per_Agent',
            'num_say_sum': 'Total_Say',
            'active_hours_mean': 'Avg_Active_Hours_Per_Agent',
            'active_hours_std': 'Std_Active_Hours_Per_Agent',
            'active_hours_sum': 'Total_Active_Hours'
        }
        
        model_summary = model_summary.rename(columns=column_mapping)
        
        # Save to CSV
        csv_path = self.output_dir / 'model_performance_summary.csv'
        model_summary.to_csv(csv_path, index=False)
        
        print(f"\n✅ Model Performance Summary CSV saved to: {csv_path}")
        print("\nModel Performance Summary (Averaged across directories):")
        print("=" * 120)
        
        # Display key metrics
        key_metrics = [
            'Model', 'Avg_Net_Growth', 'Avg_On_Time_Order_Rate',
            'Avg_Stars', 'Avg_Food_Stars', 'Total_Interruptions', 'Total_Help_Posted',
            'Total_Help_Accepted', 'Total_Say'
        ]
        
        print(model_summary[key_metrics].to_string(index=False, float_format='%.4f'))
        
        return model_summary
    
    def create_comparison_table(self):
        """Create a comprehensive comparison table."""
        df = self.extract_key_metrics()
        
        # Sort by net_growth descending
        df_sorted = df.sort_values('net_growth', ascending=False)
        
        print("=" * 100)
        print("AGENT PERFORMANCE COMPARISON (All Agents)")
        print("=" * 100)
        
        # Main comparison table
        comparison_cols = [
            'agent_id', 'model', 'source_dir', 'net_growth', 'completed_orders', 
            'timeout_orders', 'avg_stars', 'vlm_total_calls', 'vlm_success_calls'
        ]
        
        print("\nKey Performance Metrics:")
        print("-" * 100)
        print(df_sorted[comparison_cols].to_string(index=False, float_format='%.2f'))
        
        # Additional metrics table
        print("\n\nAdditional Performance Metrics:")
        print("-" * 100)
        additional_cols = [
            'agent_id', 'model', 'source_dir', 'vlm_success_rate', 'active_hours', 
            'orders_per_hour', 'temp_ok_rate', 'odor_ok_rate', 'damage_ok_rate'
        ]
        print(df_sorted[additional_cols].to_string(index=False, float_format='%.3f'))
        
        return df_sorted
    
    def analyze_performance_patterns(self, df):
        """Analyze performance patterns and provide insights."""
        print("\n" + "=" * 100)
        print("PERFORMANCE ANALYSIS & INSIGHTS")
        print("=" * 100)
        
        # Top performers
        print("\n🏆 TOP PERFORMERS:")
        print("-" * 50)
        top_3 = df.head(3)
        for _, agent in top_3.iterrows():
            print(f"Agent {agent['agent_id']} ({agent['model']}) from {agent['source_dir']}:")
            print(f"  • Net Growth: ${agent['net_growth']:.2f}")
            print(f"  • Completed Orders: {agent['completed_orders']}")
            print(f"  • Average Stars: {agent['avg_stars']:.1f}")
            print(f"  • VLM Success Rate: {agent['vlm_success_rate']:.1%}")
            print()
        
        # Model performance analysis
        print("\n🤖 MODEL PERFORMANCE ANALYSIS (Averaged):")
        print("-" * 50)
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            print(f"\n{model}:")
            print(f"  • Total Agents: {len(model_data)}")
            print(f"  • Source Directories: {model_data['source_dir'].unique()}")
            print(f"  • Avg Net Growth: ${model_data['net_growth'].mean():.2f} ± {model_data['net_growth'].std():.2f}")
            print(f"  • Avg Completed Orders: {model_data['completed_orders'].mean():.1f} ± {model_data['completed_orders'].std():.1f}")
            print(f"  • Avg Stars: {model_data['avg_stars'].mean():.2f} ± {model_data['avg_stars'].std():.2f}")
            print(f"  • Avg VLM Success Rate: {model_data['vlm_success_rate'].mean():.1%} ± {model_data['vlm_success_rate'].std():.1%}")
        
        # Key insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 50)
        
        # Best overall performer
        best_agent = df.iloc[0]
        print(f"• Best Overall Performer: Agent {best_agent['agent_id']} ({best_agent['model']}) from {best_agent['source_dir']}")
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
                print(f"  - Agent {agent['agent_id']} ({agent['model']}) from {agent['source_dir']}")
        
        # Temperature control issues
        temp_issues = df[df['temp_ok_rate'] < 0.5]
        print(f"• {len(temp_issues)} agents had temperature control issues (<50% temp_ok_rate)")
        
        return df
    
    def create_visualizations(self, df):
        """Create performance visualization charts using model-averaged data."""
        plt.style.use('seaborn-v0_8')
        
        # Get model-averaged data
        model_averaged = self.create_model_averaged_data()
        
        # Convert to DataFrame for easier plotting
        model_df = pd.DataFrame(list(model_averaged.values()))
        model_df = model_df.sort_values('net_growth_mean', ascending=False)
        
        # Create model name labels for x-axis
        model_labels = []
        for _, row in model_df.iterrows():
            model_name = row['model'].split('/')[-1]  # Get the last part after '/'
            agent_count = int(row['agent_count'])
            model_labels.append(f"{model_name}\n({agent_count} agents)")
        
        # Chart 1: Financial Performance
        fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
        fig1.suptitle('Financial Performance Analysis (Model Averaged)', fontsize=16, fontweight='bold')
        
        # Net Growth with error bars
        x_pos = range(len(model_df))
        bars1 = axes1[0].bar(x_pos, model_df['net_growth_mean'], 
                             yerr=model_df['net_growth_std'],
                             color=['green' if x >= 0 else 'red' for x in model_df['net_growth_mean']], 
                             alpha=0.7, capsize=5)
        axes1[0].set_title('Net Growth by Model (Mean ± Std)', fontsize=14)
        axes1[0].set_xlabel('', fontsize=12)
        axes1[0].set_ylabel('Net Growth ($)', fontsize=12)
        axes1[0].set_xticks(x_pos)
        axes1[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes1[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars1, model_df['net_growth_mean'], model_df['net_growth_std'])):
            height = bar.get_height()
            axes1[0].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                         f'${mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # Orders Performance with error bars
        x = np.arange(len(model_df))
        width = 0.35
        bars2 = axes1[1].bar(x - width/2, model_df['completed_orders_mean'], width, 
                             yerr=model_df['completed_orders_std'],
                             label='Completed', color='green', alpha=0.7, capsize=5)
        bars3 = axes1[1].bar(x + width/2, model_df['timeout_orders_mean'], width, 
                             yerr=model_df['timeout_orders_std'],
                             label='Timeout', color='red', alpha=0.7, capsize=5)
        axes1[1].set_title('Completed vs Timeout Orders (Mean ± Std)', fontsize=14)
        axes1[1].set_xlabel('', fontsize=12)
        axes1[1].set_ylabel('Number of Orders', fontsize=12)
        axes1[1].set_xticks(x)
        axes1[1].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes1[1].legend()
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars2, model_df['completed_orders_mean'], model_df['completed_orders_std']):
            if mean_val > 0:
                height = bar.get_height()
                axes1[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar, mean_val, std_val in zip(bars3, model_df['timeout_orders_mean'], model_df['timeout_orders_std']):
            if mean_val > 0:
                height = bar.get_height()
                axes1[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'financial_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 2: Quality and VLM Performance
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
        fig2.suptitle('Quality and VLM Performance Analysis (Model Averaged)', fontsize=16, fontweight='bold')
        
        # Average Stars with error bars
        bars4 = axes2[0].bar(x_pos, model_df['avg_stars_mean'], 
                             yerr=model_df['avg_stars_std'],
                             color='gold', alpha=0.7, capsize=5)
        axes2[0].set_title('Average Customer Rating (Mean ± Std)', fontsize=14)
        axes2[0].set_xlabel('', fontsize=12)
        axes2[0].set_ylabel('Average Stars', fontsize=12)
        axes2[0].set_xticks(x_pos)
        axes2[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes2[0].set_ylim(0, 5)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars4, model_df['avg_stars_mean'], model_df['avg_stars_std']):
            if mean_val > 0:
                height = bar.get_height()
                axes2[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # VLM Performance - Grouped Bar Chart with error bars
        x_pos = np.arange(len(model_df))
        width = 0.35
        
        # Create bars for total and successful calls
        bars1 = axes2[1].bar(x_pos - width/2, model_df['vlm_total_calls_mean'], width, 
                            yerr=model_df['vlm_total_calls_std'],
                            label='Total VLM Calls', color='lightcoral', alpha=0.8, capsize=5)
        bars2 = axes2[1].bar(x_pos + width/2, model_df['vlm_success_calls_mean'], width, 
                            yerr=model_df['vlm_success_calls_std'],
                            label='Successful VLM Calls', color='lightgreen', alpha=0.8, capsize=5)
        
        axes2[1].set_title('VLM Performance: Total vs Successful Calls (Mean ± Std)', fontsize=14)
        axes2[1].set_xlabel('', fontsize=12)
        axes2[1].set_ylabel('Number of Calls', fontsize=12)
        axes2[1].set_xticks(x_pos)
        axes2[1].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars1, model_df['vlm_total_calls_mean'], model_df['vlm_total_calls_std']):
            height = bar.get_height()
            axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=7)
        
        for bar, mean_val, std_val in zip(bars2, model_df['vlm_success_calls_mean'], model_df['vlm_success_calls_std']):
            height = bar.get_height()
            axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_vlm_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 3: Food Quality Metrics
        fig3, axes3 = plt.subplots(2, 2, figsize=(20, 12))
        fig3.suptitle('Food Quality Metrics Analysis (Model Averaged)', fontsize=16, fontweight='bold')
        
        quality_metrics = ['temp_ok_rate_mean', 'odor_ok_rate_mean', 'damage_ok_rate_mean', 'method_success_rate_mean']
        quality_stds = ['temp_ok_rate_std', 'odor_ok_rate_std', 'damage_ok_rate_std', 'method_success_rate_std']
        quality_titles = ['Temperature Control', 'Odor Preservation', 'Damage Prevention', 'Delivery Method']
        colors = ['red', 'orange', 'yellow', 'green']
        
        for i, (metric, std_metric, title, color) in enumerate(zip(quality_metrics, quality_stds, quality_titles, colors)):
            row = i // 2
            col = i % 2
            
            bars = axes3[row, col].bar(x_pos, model_df[metric], 
                                      yerr=model_df[std_metric],
                                      color=color, alpha=0.7, capsize=5)
            axes3[row, col].set_title(f'{title} Success Rate (Mean ± Std)', fontsize=14, pad=20)
            axes3[row, col].set_xlabel('', fontsize=12)
            axes3[row, col].set_ylabel('Success Rate', fontsize=12)
            axes3[row, col].set_xticks(x_pos)
            axes3[row, col].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
            axes3[row, col].set_ylim(0, 1)
            
            # Add value labels
            for bar, mean_val, std_val in zip(bars, model_df[metric], model_df[std_metric]):
                height = bar.get_height()
                axes3[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{mean_val:.1%}±{std_val:.1%}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'food_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 4: Activity and Efficiency Metrics
        fig4, axes4 = plt.subplots(1, 2, figsize=(20, 8))
        fig4.suptitle('Activity and Efficiency Metrics (Model Averaged)', fontsize=16, fontweight='bold')
        
        # Active Hours with error bars
        bars5 = axes4[0].bar(x_pos, model_df['active_hours_mean'], 
                             yerr=model_df['active_hours_std'],
                             color='lightblue', alpha=0.7, capsize=5)
        axes4[0].set_title('Active Hours by Model (Mean ± Std)', fontsize=14)
        axes4[0].set_xlabel('', fontsize=12)
        axes4[0].set_ylabel('Active Hours', fontsize=12)
        axes4[0].set_xticks(x_pos)
        axes4[0].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars5, model_df['active_hours_mean'], model_df['active_hours_std']):
            height = bar.get_height()
            axes4[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{mean_val:.2f}±{std_val:.2f}h', ha='center', va='bottom', fontsize=8)
        
        # Orders per Hour with error bars
        bars6 = axes4[1].bar(x_pos, model_df['orders_per_hour_mean'], 
                             yerr=model_df['orders_per_hour_std'],
                             color='lightgreen', alpha=0.7, capsize=5)
        axes4[1].set_title('Orders per Hour by Model (Mean ± Std)', fontsize=14)
        axes4[1].set_xlabel('', fontsize=12)
        axes4[1].set_ylabel('Orders per Hour', fontsize=12)
        axes4[1].set_xticks(x_pos)
        axes4[1].set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars6, model_df['orders_per_hour_mean'], model_df['orders_per_hour_std']):
            height = bar.get_height()
            axes4[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activity_efficiency_metrics.png', dpi=300, bbox_inches='tight')
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
        print(f"Analyzing {len(self.results_dirs)} directories:")
        for i, dir_path in enumerate(self.results_dirs, 1):
            print(f"  {i}. {dir_path}")
        
        # Extract and display metrics
        df = self.create_comparison_table()
        
        # Analyze patterns
        df = self.analyze_performance_patterns(df)
        
        # Create model summary CSV
        print("\n📊 Creating model performance summary CSV...")
        model_summary = self.create_model_summary_csv()
        
        # Create visualizations
        print("\n📊 Creating performance visualizations...")
        self.create_visualizations(df)
        
        print("\n✅ Analysis complete!")
        print("Files generated:")
        print(f"  - {self.output_dir}/model_performance_summary.csv")
        print(f"  - {self.output_dir}/financial_performance.png")
        print(f"  - {self.output_dir}/quality_vlm_performance.png")
        print(f"  - {self.output_dir}/food_quality_metrics.png")
        print(f"  - {self.output_dir}/activity_efficiency_metrics.png")
        
        return df

def main(paths, output_dir=None):
    """Main function to run the analysis.
    
    Args:
        paths: Single directory path (str) or list of directory paths
        output_dir: Optional output directory path
    """
    analyzer = AgentPerformanceAnalyzer(results_dirs=paths, output_dir=output_dir)
    results_df = analyzer.run_analysis()
    return results_df

if __name__ == "__main__":
    # Example usage with multiple directories
    results = main(paths=[
        r"D:\Projects\Food-Delivery-Bench\results\20250917_101705",
        r"D:\Projects\Food-Delivery-Bench\results\20250917_111220",
        r"D:\Projects\Food-Delivery-Bench\results\20250917_104816",
    ], output_dir='initial_results')
