"""单环位置 PD/PID 控制器入口。"""
from run_experiments import simulate

def run_single_loop(controller_kind, load_start_s=1.0, load_torque_nm=1.5):
    del load_start_s
    return simulate(controller_kind, load_torque_nm)
