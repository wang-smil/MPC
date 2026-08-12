from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from pid import PIDController

ROOT = Path(__file__).resolve().parents[1]
DT=0.001; DURATION=4.; TARGET=np.deg2rad(30); LIMIT=3.

def simulate(kind, load=0., anti=True, target=TARGET):
    n=int(DURATION/DT)+1; t=np.linspace(0,DURATION,n); q=np.zeros(n); w=np.zeros(n); integ=np.zeros(n); raw=np.zeros(n); cmd=np.zeros(n); ref=np.full(n,target); rng=np.random.default_rng(42)
    if kind == 'cascade': outer=8.; c=PIDController(1.8,5.,0,LIMIT,1. if anti else None)
    else: c=PIDController(18.,5. if kind=='pid' else 0.,1.2,LIMIT,1. if anti else None)
    for k in range(1,n):
        measured=q[k-1]+rng.normal(0,np.deg2rad(.05)); rate=(q[k-1]-q[k-2])/DT if k>1 else 0
        if kind=='cascade': error=outer*(target-measured)-rate
        else: error=target-measured
        raw[k],cmd[k],_=c.update(error,rate,DT); integ[k]=c.integral
        torque_load=load if t[k]>=1 else 0; w[k]=w[k-1]+DT*(cmd[k]-torque_load-.08*w[k-1])/.02; q[k]=q[k-1]+DT*w[k]
    return dict(time=t,q=q,w=w,ref=ref,error=ref-q,integral=integ,torque_unsat=raw,torque_cmd=cmd,load=np.where(t>=1,load,0.))

def _plot(filename, series, ylabel):
    fig,ax=plt.subplots(figsize=(9,5));
    for label,data in series.items(): ax.plot(data['time'], data[ylabel], label=label)
    ax.grid(); ax.legend(); ax.set_xlabel('Time / s'); ax.set_ylabel(ylabel); fig.tight_layout(); p=ROOT/'figures'/filename; p.parent.mkdir(exist_ok=True); fig.savefig(p,dpi=180); plt.close(fig); return p

def _log(name,data):
    p=ROOT/'logs'/f'{name}.csv'; p.parent.mkdir(exist_ok=True)
    with p.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f); keys=list(data); writer.writerow(keys); writer.writerows(zip(*(data[k] for k in keys)))
    return p

def run_all_experiments():
    pd=simulate('pd',1.5); pid=simulate('pid',1.5); _log('pd_load',pd); _log('pid_load',pid)
    p1=_plot('pd_vs_pid_load.png',{'PD':pd,'PID':pid},'error')
    no_aw=simulate('pid',0.,False,np.deg2rad(90)); aw=simulate('pid',0.,True,np.deg2rad(90)); _log('no_anti_windup',no_aw); _log('anti_windup',aw)
    p2=_plot('anti_windup.png',{'no anti-windup':no_aw,'integral clamp':aw},'integral')
    single=simulate('pid',1.5); cascade=simulate('cascade',1.5); _log('single',single); _log('cascade',cascade)
    p3=_plot('single_vs_cascade.png',{'single PID':single,'cascade':cascade},'error')
    rp=ROOT/'reports'/'digital_pid_cascade.md'; rp.parent.mkdir(exist_ok=True); rp.write_text('# 数字 PID 与串级伺服\n\n积分会累积恒定误差直到提供补偿转矩。执行器饱和时积分仍累积会形成 windup，积分限幅可限制它。D 对测量速度反馈以减少 setpoint kick，但差分会放大噪声。串级结构把位置 P 与速度 PI 分层，内速度环应更快。理想 torque command 不等于真实转矩：实际驱动还受电流环、PWM、电压、延迟和温度影响。\n',encoding='utf8')
    return {'pd_vs_pid_load':p1,'anti_windup':p2,'single_vs_cascade':p3,'report':rp}

if __name__=='__main__':
    for name,path in run_all_experiments().items(): print(f'{name}: {path}')
