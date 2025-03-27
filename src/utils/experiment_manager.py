import torch


class ExperimentTracker:
    """
    实验跟踪器
    跟踪当前实验的进度，并在控制台显示一个好看的进度条
    """

    def __init__(self, total_epochs, total_steps_per_epoch=None,
                 desc="Training",
                 metrics_to_display=None):
        """
        初始化进度跟踪器

        参数:
            total_epochs: 总轮次
            total_steps_per_epoch: 每轮次的总步骤数
            desc: 描述
            metrics_to_display: 要显示的指标列表
        """
        self.total_epochs = total_epochs
        self.total_steps_per_epoch = total_steps_per_epoch
        self.desc = desc
        self.metrics_to_display = metrics_to_display or ["loss"]

        self.current_epoch = 0
        self.current_step = 0
        self.epoch_start_time = None
        self.total_start_time = None

        # 指标存储
        self.metrics = {}
        self.current_metrics = {}

        # 尝试获取终端宽度
        try:
            import os
            self.term_width = os.get_terminal_size().columns
        except:
            self.term_width = 80

    def start(self):
        """开始跟踪"""
        # 记录开始时间
        import time
        self.total_start_time = time.time()
        self.epoch_start_time = time.time()

        # 打印初始消息
        self._print_header()

    def update(self, step=None, metrics=None):
        """
        更新跟踪器

        参数:
            step: 当前步骤
            metrics: 指标字典
        """
        # 更新步骤
        if step is not None:
            self.current_step = step

        # 更新指标
        if metrics:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()

                self.current_metrics[name] = value
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(value)

        # 打印进度
        self._print_progress()

    def next_epoch(self):
        """进入下一轮次"""
        # 完成当前轮次
        import time
        elapsed = time.time() - self.epoch_start_time

        # 打印轮次摘要
        self._print_epoch_summary(elapsed)

        # 增加轮次计数
        self.current_epoch += 1

        # 重置步骤和指标
        self.current_step = 0
        self.current_metrics = {}

        # 如果已完成所有轮次，打印总结
        if self.current_epoch >= self.total_epochs:
            self._print_final_summary()
        else:
            # 重置轮次开始时间
            self.epoch_start_time = time.time()

            # 打印新轮次头部
            self._print_header()

    def _print_header(self):
        """打印头部"""
        print("\n" + "=" * self.term_width)
        print(f"{self.desc} - Epoch {self.current_epoch + 1}/{self.total_epochs}")
        print("-" * self.term_width)

        # 打印指标头部
        header = f"{'Step':>6} {'Progress':>10} {'Time':>10}"

        for metric in self.metrics_to_display:
            header += f" {metric:>10}"

        print(header)
        print("-" * self.term_width)

    def _print_progress(self):
        """打印进度"""
        # 清除当前行
        print("\r", end="")

        # 计算进度
        import time
        if self.total_steps_per_epoch:
            progress = min(1.0, self.current_step / self.total_steps_per_epoch)
            progress_bar = self._get_progress_bar(progress, width=10)
        else:
            progress_bar = "N/A       "

        # 计算经过的时间
        elapsed = time.time() - self.epoch_start_time
        time_str = self._format_time(elapsed)

        # 构建行
        progress_line = f"{self.current_step:6d} {progress_bar} {time_str:>10}"

        # 添加指标
        for metric in self.metrics_to_display:
            if metric in self.current_metrics:
                value = self.current_metrics[metric]
                progress_line += f" {value:10.4f}"
            else:
                progress_line += f" {'N/A':>10}"

        # 打印
        print(progress_line, end="")

    def _print_epoch_summary(self, elapsed):
        """
        打印轮次摘要

        参数:
            elapsed: 经过的时间
        """
        # 清除当前行
        print("\r", end="")

        # 计算平均指标
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = sum(values) / len(values)

        # 打印摘要
        print("\n" + "-" * self.term_width)
        print(f"Epoch {self.current_epoch + 1}/{self.total_epochs} 完成")
        print(f"时间: {self._format_time(elapsed)}")

        # 打印指标
        for name, value in summary.items():
            print(f"{name}: {value:.6f}")

        print("-" * self.term_width)

    def _print_final_summary(self):
        """打印最终摘要"""
        # 计算总时间
        import time
        elapsed = time.time() - self.total_start_time

        print("\n" + "=" * self.term_width)
        print(f"{self.desc} 完成")
        print(f"总时间: {self._format_time(elapsed)}")
        print("=" * self.term_width)

    def _get_progress_bar(self, progress, width=10, fill='█', empty='░'):
        """
        获取进度条字符串

        参数:
            progress: 进度 (0-1)
            width: 进度条宽度
            fill: 填充字符
            empty: 空字符

        返回:
            进度条字符串
        """
        filled_width = int(width * progress)
        bar = fill * filled_width + empty * (width - filled_width)
        return bar

    def _format_time(self, seconds):
        """
        格式化时间

        参数:
            seconds: 秒数

        返回:
            格式化时间字符串
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes, seconds = divmod(seconds, 60)
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m"


class ExperimentManager:
    pass