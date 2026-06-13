import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import uuid

# Set professional style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.facecolor'] = '#F8FAFC'
plt.rcParams['grid.color'] = '#E2E8F0'

class VeritasCharts:
    """
    Generates professional-grade visualizations for the VERITAS AI PDF report.
    """
    
    def __init__(self, output_dir="veritas/temp"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_trend_chart(self, history_data=None):
        """
        Generates a professional 24H environmental trend chart.
        """
        try:
            # Fallback data if none provided
            if history_data is None:
                times = [datetime.now() - timedelta(hours=i) for i in range(24)]
                times.reverse()
                history_data = {
                    'timestamp': times,
                    'co2': np.random.normal(800, 100, 24),
                    'temperature': np.random.normal(24, 2, 24),
                    'humidity': np.random.normal(45, 5, 24)
                }

            df = pd.DataFrame(history_data)
            
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            # Primary axis: CO2
            color_co2 = '#065F46' # Emerald
            ax1.plot(df['timestamp'], df['co2'], color=color_co2, linewidth=2, label='CO2 (ppm)')
            ax1.set_ylabel('CO2 (ppm)', color=color_co2, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color_co2)
            
            # Secondary axis: Temp & Humidity
            ax2 = ax1.twinx()
            color_temp = '#EF4444' # Red
            color_hum = '#3B82F6'  # Blue
            
            ax2.plot(df['timestamp'], df['temperature'], color=color_temp, linewidth=1.5, linestyle='--', label='Temp (°C)')
            ax2.plot(df['timestamp'], df['humidity'], color=color_hum, linewidth=1.5, linestyle=':', label='Humidity (%)')
            ax2.set_ylabel('Temp / Humidity', color='#475569', fontweight='bold')
            
            # Formatting
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.title('24H Environmental Trend Analysis', color='#0F172A', pad=20, fontweight='bold')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
            
            plt.tight_layout()
            
            filename = f"trend_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            return filepath
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None

    def generate_contribution_bars(self, feature_importance):
        """
        Generates horizontal bars for XAI feature contribution.
        """
        try:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            features = [features[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            colors = ['#10B981' if v < 0.3 else '#F59E0B' if v < 0.6 else '#EF4444' for v in values]
            
            bars = ax.barh(features, values, color=colors, height=0.6)
            
            # Clean up UI
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                        va='center', fontweight='bold', color='#475569')

            plt.title('Risk Factor Contribution (XAI)', color='#0F172A', fontweight='bold', pad=15)
            plt.tight_layout()
            
            filename = f"xai_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            return filepath
        except Exception as e:
            print(f"XAI chart error: {e}")
            return None
