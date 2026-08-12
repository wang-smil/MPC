"""位置 P -> 速度 PI -> 转矩的串级控制器入口。"""
from run_experiments import simulate

def run_cascade(load_torque_nm=1.5):
    return simulate("cascade", load_torque_nm)
