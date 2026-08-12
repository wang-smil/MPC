# 第4课：数字 PID 与串级伺服

运行：`conda run -n robot-control python .\lesson04_digital_pid_cascade\src\run_experiments.py`

本课比较 PD/PID 的恒定负载恢复、积分限幅 anti-windup，以及单环 PID 与位置 P—速度 PI 串级。D 使用测量速度反馈，避免设定值阶跃带来的 derivative kick；真实电机转矩还受底层电流环与驱动约束影响。
