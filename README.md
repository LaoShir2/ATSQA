# 自适应时序质量评估原型Adaptive Time Series Quality Assessment

> 模型与数据集来源于 [https://github\.com/thuml/Time\-Series\-Library](https://github.com/thuml/Time-Series-Library)
> 
> 

本目录包含项目文档中七步工作流可直接运行、轻依赖的原型代码。

## 运行方式

使用 numpy \+ PatchTST 版：
```bash
python run_seven_step_demo.py
```

使用 PyTorch \+ PatchTST 版：

```powershell
python run_torch_patchtst_demo.py
```

基于 Time\-Series\-Library 格式的 CSV 数据集运行：

```powershell
run_torch_patchtst_demo.py 
  --data-path path\to\ETTh1.csv 
  --target OT 
  --seq-len 96 
  --pred-len 24
```

若省略 `--target` 参数，则基于全部数值列执行多变量预测。

原始脚本 `run_seven_step_demo.py` 仅依赖 numpy，可作为兜底基线方案。
`run_torch_patchtst_demo.py` 为主版本：脚本构建合成多变量时序预测任务，计算论文图 2 中的四项质量指标；通过单步可微元验证更新学习质量维度权重；随后利用更新后的梯度截断样本权重训练轻量化 PatchTST 预测模型。

## 文件说明

- `atsqa_quality.py`：时序质量指标计算模块

    - 可预测性：`1 - spectral_entropy / log(N_f)`

    - 季节性强度：基于 FFT 周期估计，计算公式 `1 - Var(R) / Var(S + R)`

    - 趋势强度：数据最小 \- 最大归一化后，`min(1, abs(beta_hat) * T)`

    - 稀疏度：`1 - N_unique(X) / T`

- `run_seven_step_demo.py`：完整七步流程可执行实验脚本

- `models/patchtst.py`：轻量化 PatchTST 预测器，输入张量形状 `[batch, seq_len, channels]`，输出张量形状 `[batch, pred_len, channels]`

- `run_torch_patchtst_demo.py`：基于 PyTorch 实现的七步工作流，内置 PatchTST 模型与等权重基线

## 注意事项

PyTorch 版本可对质量指标生成的样本权重进行求导，并借助 `torch.autograd.grad(create_graph=True)` 完成一次虚拟 PatchTST 参数更新。
在真实模型参数更新阶段，样本权重执行梯度截断操作，满足项目工作流中的梯度停止（Stop\-Gradient）约束。

内置 PatchTST 模块独立封装，张量输入输出规范与 THUML 时序库（Time\-Series\-Library）保持一致。后续若将完整时序库引入本项目，可直接替换 `run_torch_patchtst_demo.py` 中的 `build_model()` 函数。

---
