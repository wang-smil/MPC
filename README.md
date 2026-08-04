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

## 自动测试

```powershell
python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```
