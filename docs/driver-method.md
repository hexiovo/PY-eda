# Driver 分解与指标（v0.3.0）

选择 `single_driver` 预设，或在“分解与 SCR”页选择 `bateman_sparse`，即可启用 9 项 Driver 指标。参数编辑器应用前会执行校验。默认 `single` 仍使用独立滤波；该方法没有 Driver 输出。

## 模型与单位

本地方法 `local-convex-bateman-v1` 联合估计非负脉冲质量 a 和分段线性 tonic 系数 c：

```
y = H a + B c + residual, a >= 0
min 0.5*||H a + B c - y||² + alpha*sum(a)
    + 0.5*ridge*||a||² + 0.5*gamma*||D2 c||²
```

y 为预处理后的 clean_uS；H 是因果 Bateman 双指数卷积，B 为等距节点线性插值基，D2 为节点系数二阶差分。核 h(t)=(exp(-t/tau_decay)-exp(-t/tau_rise))/peak，t>=0；peak 为连续时间函数的解析峰高，使峰值归一为 1。核截断至配置的尾部上界。a 的单位是 µS，输出 `driver_uS_s=a/dt` 为 µS/s。tonic、phasic=Ha、residual 均为 µS，三者严格按上述定义重构 clean。

SciPy L-BFGS-B 使用解析梯度及 FFT 前向/伴随算子。只有优化器成功且投影梯度通过配置的 KKT 容差时才接受连续段。失败保留诊断，该段信号与特征缺失；不会换成滤波结果。该实现是独立研究模型，不是 Ledalab/CDA、cvxEDA 的复现或兼容实现。

## 默认参数与支持

上升/衰减时间常数为 0.7/2 秒，tonic 节点间隔 10 秒，L1 稀疏惩罚 0.01，ridge=1e-6，tonic 平滑惩罚 10。核尾部上界 0.001，最多 2000 次迭代，每段最多 100000 个样本；完整参数冻结到配置、特征定义与分解诊断。惩罚系数并未按采样率归一，改变采样率后应重新评估参数。

要求等间隔网格，采样间隔不得大于上升时间常数。分别拟合连续有效段，排除伪迹、插值点和预处理边缘，不跨缺口传播。每段两端各排除一个完整核支持（默认 4 Hz 下约 16 秒），短段不可用。因此短记录或靠近边缘的事件可能没有足够有效支持。窗口使用配置的最小覆盖率；真实时钟缺口使跨缺口窗口不可用。

## 九项指标

| ID | 含义与单位 |
|---|---|
| driver_integral_uS | 窗口 Driver 积分，µS |
| driver_time_mean_uS_s | 积分/有效相邻支持时长，µS/s |
| driver_max_uS_s | 半开窗口内有效采样最大值，µS/s |
| driver_active_time_proportion | 高于或等于阈值的时长/有效支持时长 |
| driver_burst_count | 峰时间位于窗口内的完整阈上脉冲数 |
| driver_burst_rate_per_min | 完整脉冲数/有效支持分钟数 |
| event_driver_integral_uS | 事件 response 窗口积分，µS |
| event_driver_max_uS_s | 事件 response 窗口最大值，µS/s |
| event_driver_peak_latency_s | 首个最大阈上采样峰时间减事件 onset，秒 |

默认阈值 0.01 µS/s。积分与阈上时长使用相邻有效点的分段线性插值，精确裁剪窗口边界，不跨缺失、不外推；因此裁剪窗口积分不等同于简单求和 a。脉冲是连续阈上区间，在支持边界未完整观察的脉冲标记 censored，排除计数。最大值/潜伏期按采样点定义，平峰取最早样本。有效无响应窗口积分/最大值/计数可以为零，没有阈上峰时潜伏期缺失。

事件指标不减基线；重叠事件窗口可能共享 Driver 活动，不能解释为唯一归属于某事件的反应。Driver 代表模型估计驱动，不能直接等同于生理神经放电；参数和模型适用性仍需真实设备/任务验证。

## 导出与目录

`signals` 包含 driver_uS_s 和 driver_eligible；`driver_bursts` 保存阈上事件、边界删失和积分；`driver_diagnostics` 保存各段状态、收敛信息、投影梯度、迭代数、误差及参数。缓存恢复保留 Driver 数组和诊断。

默认频带/滞后配置下，完整目录共 144 项；滤波 single/group_sync 适用 124/135 项，Driver single/group 适用 133/144 项。事件专用指标只在 event_response 窗口产生；目录项目数不等于每个窗口的结果行数。

## 方法参考与验证

非负分解背景：[Benedek & Kaernbach (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2904901/)。优化接口：[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)。此处的 tonic 基、惩罚、核归一化及边缘策略以本文件和源代码为准。

`tests/test_driver.py` 包含独立稠密卷积矩阵、有限差分梯度、增广有界最小二乘 oracle、已知三脉冲恢复、偏移不变性、手算三角波积分、分段隔离、失败与资源边界、缓存/导出/GUI 集成检查。合成测试证明这些计算路径符合指定模型，不构成真实设备科学资格。
