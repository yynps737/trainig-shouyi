#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import subprocess
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, interval=1.0, window_size=100, save_log=True):
        """
        初始化GPU监控器
        
        参数:
            interval (float): 采样间隔(秒)
            window_size (int): 显示窗口大小
            save_log (bool): 是否保存日志
        """
        self.interval = interval
        self.window_size = window_size
        self.save_log = save_log
        
        # 存储数据
        self.timestamps = []
        self.gpu_utils = []
        self.mem_utils = []
        self.temperatures = []
        self.power_usages = []
        
        # 停止标志
        self.stop_flag = threading.Event()
        
        # 记录线程
        self.thread = None
        
        # 日志文件
        if save_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("logs", exist_ok=True)
            self.log_file = f"logs/gpu_monitor_{timestamp}.csv"
            with open(self.log_file, "w") as f:
                f.write("timestamp,gpu_util,mem_util,temperature,power\n")
    
    def get_gpu_stats(self):
        """获取GPU统计信息"""
        try:
            # 使用nvidia-smi命令
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                universal_newlines=True
            )
            
            # 解析输出
            values = output.strip().split(",")
            gpu_util = float(values[0].strip())
            mem_util = float(values[1].strip())
            temperature = float(values[2].strip())
            power = float(values[3].strip())
            
            return gpu_util, mem_util, temperature, power
        except Exception as e:
            print(f"获取GPU统计信息时出错: {e}")
            return 0, 0, 0, 0
    
    def record_stats(self):
        """记录GPU统计信息"""
        while not self.stop_flag.is_set():
            # 获取当前时间
            now = time.time()
            
            # 获取GPU统计信息
            gpu_util, mem_util, temperature, power = self.get_gpu_stats()
            
            # 添加到列表
            self.timestamps.append(now)
            self.gpu_utils.append(gpu_util)
            self.mem_utils.append(mem_util)
            self.temperatures.append(temperature)
            self.power_usages.append(power)
            
            # 保持窗口大小
            if len(self.timestamps) > self.window_size:
                self.timestamps = self.timestamps[-self.window_size:]
                self.gpu_utils = self.gpu_utils[-self.window_size:]
                self.mem_utils = self.mem_utils[-self.window_size:]
                self.temperatures = self.temperatures[-self.window_size:]
                self.power_usages = self.power_usages[-self.window_size:]
            
            # 保存日志
            if self.save_log:
                with open(self.log_file, "a") as f:
                    f.write(f"{now},{gpu_util},{mem_util},{temperature},{power}\n")
            
            # 等待下一个间隔
            time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if self.thread is None or not self.thread.is_alive():
            print("开始监控GPU...")
            self.stop_flag.clear()
            self.thread = threading.Thread(target=self.record_stats)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """停止监控"""
        if self.thread and self.thread.is_alive():
            print("停止监控GPU...")
            self.stop_flag.set()
            self.thread.join()
    
    def plot_live(self):
        """实时绘图"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        
        # 设置标题
        fig.suptitle("NVIDIA GeForce RTX 4070 监控", fontsize=16)
        
        # 初始化线条
        line1, = ax1.plot([], [], 'b-', label='GPU利用率 (%)')
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('利用率 (%)')
        ax1.grid(True)
        ax1.legend()
        
        line2, = ax2.plot([], [], 'g-', label='显存利用率 (%)')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('利用率 (%)')
        ax2.grid(True)
        ax2.legend()
        
        line3, = ax3.plot([], [], 'r-', label='温度 (°C)')
        ax3.set_ylim(30, 90)
        ax3.set_ylabel('温度 (°C)')
        ax3.grid(True)
        ax3.legend()
        
        line4, = ax4.plot([], [], 'purple', label='功耗 (W)')
        ax4.set_ylim(0, 200)
        ax4.set_ylabel('功耗 (W)')
        ax4.set_xlabel('时间 (秒)')
        ax4.grid(True)
        ax4.legend()
        
        # 开始监控
        self.start()
        
        def update(frame):
            if len(self.timestamps) > 1:
                # 计算相对时间
                rel_times = [t - self.timestamps[0] for t in self.timestamps]
                
                # 更新数据
                line1.set_data(rel_times, self.gpu_utils)
                line2.set_data(rel_times, self.mem_utils)
                line3.set_data(rel_times, self.temperatures)
                line4.set_data(rel_times, self.power_usages)
                
                # 调整X轴范围
                for ax in (ax1, ax2, ax3, ax4):
                    ax.relim()
                    ax.autoscale_view(scalex=True, scaley=False)
            
            return line1, line2, line3, line4
        
        # 创建动画
        ani = FuncAnimation(fig, update, interval=self.interval * 1000, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        # 停止监控
        self.stop()

def parse_args():
    parser = argparse.ArgumentParser(description="GPU监控工具")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔(秒)")
    parser.add_argument("--window", type=int, default=60, help="显示窗口大小")
    parser.add_argument("--no-log", action="store_true", help="不保存日志")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    monitor = GPUMonitor(
        interval=args.interval,
        window_size=args.window,
        save_log=not args.no_log
    )
    
    try:
        monitor.plot_live()
    except KeyboardInterrupt:
        print("监控被用户中断")
    finally:
        monitor.stop()
