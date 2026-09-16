# causality-audit

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

检查序列模型中“修改最后一个输入位置，是否影响前面的输出”。这个基于 NumPy 的前缀不变性审计工具运行两次前向计算，逐层对比输出，并找出差异首次超过指定容差的位置。

项目附带正常版和故意注入缺陷的分块递推模型，方便验证审计流程。它们是教学及回归测试用的合成模型，**不是真实 Hugging Face checkpoint，也不复现论文中的基准数值**。

## 安装与运行

需要 Python 3.9+、NumPy 1.24+，不依赖 PyTorch。

```bash
git clone https://github.com/zhuhroscar-tech/causality-audit.git
cd causality-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
causality-audit --demo clean
causality-audit --demo buggy --json --epsilon-sweep
```

添加 `--check-clean` 后，检测到泄漏会以退出码 `1` 结束；不加时，demo 只报告结果，不用退出状态表示是否泄漏。

## Python API

```python
import numpy as np
from causality_audit.core import audit_prefix_invariance

x = np.random.default_rng(0).standard_normal((12, 3))
y = x.copy()
y[-1] += 1.0
layers = [("causal_sum", lambda a: np.cumsum(a, axis=0))]
result = audit_prefix_invariance(layers, x, y)
print(result.verdict, result.first_leak_index)
```

按顺序传入 `(名称, callable)`。每层接收上一层形状为 `(seq_len, dim)` 的输出，并返回同形状数组。两个输入只能在最后一个位置不同。结果包含逐层差异，以及 `clean`、`leak_detected` 或 `inconclusive` 判定。

## 结果解读与限制

- `clean` 只说明当前探针未发现问题，不覆盖所有输入。请先用已知会泄漏的正对照验证流程，并确认模型确实会响应输入变化。
- 测试序列应长于相关 chunk/window；单个 chunk 可能掩盖跨块泄漏。
- 容差应与数值精度匹配。`epsilon_sweep` 用于辅助区分随扰动变化的泄漏和数值噪声，但不是数学证明。
- 接入真实模型所需的 adapter、forward hook、状态重置和确定性推理需自行实现；本项目未验证下载的 checkpoint。

背景资料：[前缀不变性论文](https://arxiv.org/abs/2608.22876)、[Transformers #46741](https://github.com/huggingface/transformers/pull/46741)。

## 开发

```bash
pytest -v --cov=causality_audit
```

[测试目录](tests)覆盖输入前提、泄漏定位、合成对照及 CLI 行为。[MIT 许可证](LICENSE)。
