#!/usr/bin/env python3
"""
Real-Time GPU & CPU Monitor for Apple Silicon
Displays live bar plots of GPU and CPU core usage
"""
import os
import sys

# Set matplotlib backend BEFORE importing matplotlib
os.environ['MPLBACKEND'] = 'TkAgg'

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import subprocess
import re
import psutil
import numpy as np
from datetime import datetime

class SystemMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU & CPU Monitor - Apple M3 Ultra")
        self.root.geometry("1200x800")
        
        # Get system info
        self.get_system_info()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # System info label
        info_text = f"System: {self.chip_model} | GPU Cores: {self.gpu_cores} | CPU Cores: {self.cpu_cores}"
        ttk.Label(main_frame, text=info_text, font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=10)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 8), facecolor='#f0f0f0')
        
        # GPU usage subplot
        self.ax_gpu = self.fig.add_subplot(211)
        self.ax_gpu.set_title('GPU Usage (%)', fontsize=14, fontweight='bold')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.set_ylabel('Usage %', fontsize=10)
        self.ax_gpu.grid(True, alpha=0.3)
        
        # CPU usage subplot
        self.ax_cpu = self.fig.add_subplot(212)
        self.ax_cpu.set_title('CPU Core Usage (%)', fontsize=14, fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_ylabel('Usage %', fontsize=10)
        self.ax_cpu.set_xlabel('Core #', fontsize=10)
        self.ax_cpu.grid(True, alpha=0.3)
        
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Monitoring started...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Pause", command=self.toggle_pause).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_app).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Animation control
        self.paused = False
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                          interval=1000, blit=False, cache_frame_data=False)
        
        # Initialize data storage
        self.gpu_usage = 0.0
        self.cpu_usage = [0] * self.cpu_cores
        
    def get_system_info(self):
        """Get system information"""
        try:
            # Get GPU info
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=5)
            
            chip_match = re.search(r'Chipset Model:\s*(.+)', result.stdout)
            cores_match = re.search(r'Total Number of Cores:\s*(\d+)', result.stdout)
            
            self.chip_model = chip_match.group(1) if chip_match else "Apple Silicon"
            self.gpu_cores = int(cores_match.group(1)) if cores_match else 0
            
        except Exception as e:
            self.chip_model = "Apple Silicon"
            self.gpu_cores = 0
        
        # Get CPU info
        self.cpu_cores = psutil.cpu_count(logical=False)
    
    def get_gpu_usage(self):
        """Get GPU usage percentage"""
        try:
            result = subprocess.run(['sudo', '-n', 'powermetrics', '--samplers', 'gpu_power', 
                                   '-i', '500', '-n', '1'],
                                  capture_output=True, text=True, timeout=3, 
                                  stderr=subprocess.DEVNULL)
            
            # Parse GPU active residency
            match = re.search(r'GPU active residency:\s*([\d.]+)%', result.stdout)
            if match:
                return float(match.group(1))
            
            # Alternative: parse GPU frequency ratio
            freq_match = re.search(r'GPU active frequency:\s*([\d.]+)\s*MHz', result.stdout)
            if freq_match:
                # Estimate usage from frequency (rough approximation)
                freq = float(freq_match.group(1))
                max_freq = 1500  # Approximate max for M3 Ultra
                return min(100, (freq / max_freq) * 100)
                
        except Exception:
            pass
        
        return 0.0
    
    def get_cpu_usage(self):
        """Get per-core CPU usage"""
        try:
            return psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception:
            return [0] * self.cpu_cores
    
    def update_plots(self, frame):
        """Update plots with new data"""
        if self.paused:
            return
        
        # Get current usage
        self.gpu_usage = self.get_gpu_usage()
        self.cpu_usage = self.get_cpu_usage()
        
        # Update GPU plot
        self.ax_gpu.clear()
        self.ax_gpu.set_title('GPU Usage (%)', fontsize=14, fontweight='bold')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.set_ylabel('Usage %', fontsize=10)
        self.ax_gpu.grid(True, alpha=0.3)
        
        color_gpu = 'green' if self.gpu_usage < 50 else 'orange' if self.gpu_usage < 80 else 'red'
        self.ax_gpu.bar(['GPU'], [self.gpu_usage], color=color_gpu, alpha=0.7, width=0.5)
        self.ax_gpu.text(0, self.gpu_usage + 3, f'{self.gpu_usage:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Update CPU plot
        self.ax_cpu.clear()
        self.ax_cpu.set_title('CPU Core Usage (%)', fontsize=14, fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_ylabel('Usage %', fontsize=10)
        self.ax_cpu.set_xlabel('Core #', fontsize=10)
        self.ax_cpu.grid(True, alpha=0.3)
        
        cores = range(len(self.cpu_usage))
        colors_cpu = ['green' if u < 50 else 'orange' if u < 80 else 'red' 
                     for u in self.cpu_usage]
        
        bars = self.ax_cpu.bar(cores, self.cpu_usage, color=colors_cpu, alpha=0.7)
        
        # Add value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, self.cpu_usage)):
            if val > 5:  # Only show label if bar is visible
                self.ax_cpu.text(i, val + 2, f'{val:.0f}', 
                               ha='center', va='bottom', fontsize=8)
        
        self.ax_cpu.set_xticks(cores)
        self.ax_cpu.set_xticklabels([str(i) for i in cores], fontsize=8)
        
        # Update status
        avg_cpu = np.mean(self.cpu_usage)
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_var.set(
            f"[{timestamp}] GPU: {self.gpu_usage:.1f}% | "
            f"CPU Avg: {avg_cpu:.1f}% | Max: {max(self.cpu_usage):.1f}%"
        )
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def toggle_pause(self):
        """Pause/resume monitoring"""
        self.paused = not self.paused
        status = "Paused" if self.paused else "Monitoring..."
        self.status_var.set(status)
    
    def reset_data(self):
        """Reset the plots"""
        self.gpu_usage = 0.0
        self.cpu_usage = [0] * self.cpu_cores
        self.update_plots(0)
    
    def exit_app(self):
        """Exit the application"""
        self.root.quit()
        self.root.destroy()

def main():
    # Check if running with sudo for GPU metrics
    if os.geteuid() != 0:
        print("\n" + "="*70)
        print("NOTE: Running without sudo - GPU usage will show as 0%")
        print("For GPU monitoring, you can:")
        print("  1. Run: sudo python3 gpu_cpu_monitor.py")
        print("  2. Or configure passwordless sudo for powermetrics")
        print("\nContinuing with CPU monitoring only...")
        print("="*70 + "\n")
    
    try:
        root = tk.Tk()
        app = SystemMonitor(root)
        root.mainloop()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure you're running in a GUI environment (not SSH)")
        print("  - Try: pip install --upgrade matplotlib")
        sys.exit(1)

if __name__ == "__main__":
    main()
