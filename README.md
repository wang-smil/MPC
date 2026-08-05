# MPC

控制算法学习与仿真实验。

## 课程目录

- `lesson01_state_space/`：质量—弹簧—阻尼系统与状态空间模型。
- `lesson02_servo_control/`：单轴机器人关节、PD闭环、约束、扰动、传感器噪声、日志与性能指标。

## 环境

```powershell
conda activate robot-control
python -m pip install -r requirements.txt
python -m pip install -r .\lesson02_servo_control\requirements.txt
```

## 运行第二课

```powershell
python .\lesson02_servo_control\src\run_servo_test.py
```

## 工程时序验收

运行三次轨迹、采样周期、命令延迟和轻度抖动对比：

```powershell
python .\lesson02_servo_control\src\run_timing_acceptance.py
```

验收结果写入：

- `lesson02_servo_control/reports/engineering_acceptance.md`
- `lesson02_servo_control/figures/step_vs_cubic.png`
- `lesson02_servo_control/figures/timing_comparison.png`

本项目目前完成离线数值仿真验证，不代表 Python 或普通 Windows 已满足硬实时控制要求。

## 自动测试

```powershell
python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```
