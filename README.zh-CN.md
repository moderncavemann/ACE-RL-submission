<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="ACE-RL" width="52%">
  </picture>
</p>

<h1 align="center">ACE-RL: Counterfactual Audit Signals for Policy-Level Governance of Autonomous Market Makers</h1>

<p align="center">
  <a href="https://doi.org/10.1145/3858063.3858910"><img src="https://img.shields.io/badge/DOI-10.1145%2F3858063.3858910-1F4E9E.svg" alt="DOI 10.1145/3858063.3858910"></a>
  <a href="https://icaif2026.org/"><img src="https://img.shields.io/badge/ICAIF-2026-DE8427.svg" alt="ICAIF 2026"></a>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&amp;logoColor=white" alt="Python 3.13">
  <a href="#-快速开始"><img src="https://img.shields.io/badge/Reproduce-CPU%2C%20deterministic-2E8B57.svg" alt="CPU-only, deterministic reproduction"></a>
  <a href="https://github.com/moderncavemann/ACE-RL-submission/stargazers"><img src="https://img.shields.io/github/stars/moderncavemann/ACE-RL-submission?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub stars"></a>
  <a href="https://github.com/moderncavemann/ACE-RL-submission/forks"><img src="https://img.shields.io/github/forks/moderncavemann/ACE-RL-submission?style=flat&amp;logo=github&amp;color=1F4E9E" alt="GitHub forks"></a>
  <a href="https://hitscounter.dev/history?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FACE-RL-submission"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fmoderncavemann%2FACE-RL-submission&amp;label=Views&amp;icon=graph-up&amp;color=%231f4e9e&amp;message=&amp;style=flat&amp;tz=Australia%2FSydney" alt="Page views: today / total"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md"><strong>简体中文</strong></a>
</p>

> **作者：** Guanghao Wu<sup>\*</sup>, Zhuo Cai<sup>\*</sup>, Shoujin Wang<sup>†</sup><br>
> University of Technology Sydney<br>
> <sup>\*</sup>共同第一作者 &nbsp; <sup>†</sup>通讯作者

ACE-RL（Audit-Conditioned Enforcement for Reinforcement Learning）在训练中**把对自主做市商的审计转化为策略更新**。对每个被选中的状态，审计器编辑一份影子副本，不改动实际市场，并返回一个审计元组：编辑后的观测、一个说明该状态下是否允许响应的有效性标记，以及一个可接受响应的集合。带有效性门控的集合损失用这个元组训练 actor，而 critic 只从金融回报中学习。

- 🔍 **反事实影子审计：** 评估被选中状态的一份用后即弃的副本，返回可接受响应的集合，不改变实际市场。
- 🚦 **有效性门控：** 只在允许响应的状态上施加审计标签，其他状态的损失严格为零。
- 🧱 **审计梯度隔离：** 用标签训练 actor，审计梯度不会到达 critic。

## 🧭 为什么需要 ACE-RL？

强化学习可以自动化做市，但独立训练的做市商可能像定价算法一样，在没有明确协议的情况下滑向高利润的行为。审计可以标记这类行为，却没有告诉学习者该改变哪个动作、在哪些状态下改、通过哪个损失来改。观察者只能看到报价、成交、价差或利润，看不到动作背后的原因；同一个动作在不同状态下的含义也可能不同：不收紧报价可能是在维持高利润，也可能是在应对库存上限、不利的价格或过高的风险。

<p align="center">
  <img src="assets/challenges.png" alt="Three challenges in turning an audit into a policy update" width="80%">
</p>

<p align="center"><sub>把审计转化为策略更新的三个挑战。每一行把一种更简单的做法（左）与 ACE-RL 针对该挑战的组件（右）对应起来。</sub></p>

### 🎯 三个挑战

- **CH1：动作特异性。** 学习者必须知道哪些动作算作可接受的响应。结果统计只给出总体压力，不给出动作的排序；对所有状态和动作都相同的惩罚，在 critic 适应之后会在优势函数中抵消。
- **CH2：情境有效性。** 审计标签只适用于部分状态，我们称之为有效状态。在无效状态下（例如受到库存或风险上限的约束），响应是不合适的。在所有状态上都施加标签，会促使智能体在这些状态下也做出响应。
- **CH3：目标分离。** 标签应当改变策略，而不是改变 critic 的估计。把标签加进奖励会把治理压力混入价值目标；即使是只作用于 actor 的损失，也可能通过共享编码器影响 critic 的特征。

## ⚙️ 工作原理

ACE-RL 按信息所回答的问题来分配它。实际市场中的经验回答“什么带来了金融回报？”，提供常规的 PPO 优势和价值目标。影子审计回答“在这个被选中的状态下，是否允许响应？哪些动作算作响应？”，只为 actor 提供标签。

<p align="center">
  <img src="assets/framework.png" alt="Overview of ACE-RL" width="100%">
</p>

<p align="center"><sub>实际市场中的金融学习（上）训练两个网络；影子审计（下）只作用于 actor。图中的编号对应下面的五个步骤。</sub></p>

1. **影子副本：** 选择器标记一个实际观测，审计器在它的副本上编辑一个决策。副本只用来询问策略，标注后即丢弃：实际的订单、成交、库存和奖励都不变。
2. **审计：** 审计器检查买卖方向、库存、价格和风险，返回审计元组 (õ, z, R)：编辑后的观测 õ、有效性标记 z，以及可接受响应的集合 R。
3. **集合损失：** 有效行（z = 1）的损失是 −log Σ<sub>a∈R</sub> π<sub>θ</sub>(a | õ)，它把概率移入整个集合，而不指定其中某一个动作。无效行（z = 0）的损失严格为零，并保留在批次中。
4. **平均：** 损失对所有被审计的行取平均，因此系数 α 决定标签的强度，z 决定标签是否适用。
5. **更新 actor：** actor 最小化 PPO 裁剪损失加上 α 乘以审计损失。critic 只保留金融价值损失；不相交的参数或 stop-gradient 保证审计梯度不会到达 critic。

没有被审计的状态时，更新退化为普通的 PPO；审计器只在训练时需要。十交易商市场中的审计元组、集合损失和隔离的 IPPO 更新实现在 [`revision/src/agents/typed_audit.py`](revision/src/agents/typed_audit.py)。三项检查使更新可以被测试，单元测试会运行它们：在有效元组上只做一步审计更新，会提高 R 的概率（*方向*）；当所有 z = 0 时，审计梯度为零（*门控*）；只做一步审计更新，不会改变 critic 用到的任何参数（*隔离*）。

## 📊 实验结果

### 📈 十交易商市场

在一个基于 ABIDES 的市场中，十个交易商（做市商）各自从一到七个 tick 的七档半价差中选择一个，由一个参数共享的循环 IPPO actor 控制全部十个交易商。对十个策略种子中的每一个，四种方法从同一个检查点出发、用相同的预算训练：Vanilla IPPO；动作成本塑形（action-cost shaping），当有效状态下的交易商报出响应集合之外的价差时，从奖励中减去固定成本；Always-on，在每个被审计的状态上都施加集合损失；以及 Full ACE-RL（简称 Full）。每个交易商的每一步都会被审计。影子编辑把交易商的上一笔报价设为三个 tick 的半价差，并询问该交易商是否会继续报一到三个 tick；当公共波动率不超过一个 tick 且库存绝对值不超过 20 单位时，状态是有效的。

<table>
  <thead>
    <tr>
      <th align="left">方法</th>
      <th align="center">q<sub>V</sub> ↑</th>
      <th align="center">q<sub>X</sub> ↓</th>
      <th align="center">S ↑</th>
      <th align="center">毛 PnL ↑</th>
      <th align="center">半价差</th>
      <th align="center">一致度</th>
      <th align="center">熵</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Vanilla IPPO</td><td align="center">0.040<br><sub>±0.047</sub></td><td align="center"><b>0.026</b><br><sub>±0.033</sub></td><td align="center">0.014<br><sub>±0.027</sub></td><td align="center"><b>1111.1</b><br><sub>±47.4</sub></td><td align="center">6.80<br><sub>±0.24</sub></td><td align="center">0.831<br><sub>±0.152</sub></td><td align="center">0.264<br><sub>±0.240</sub></td></tr>
    <tr><td>Action-cost shaping</td><td align="center">0.136<br><sub>±0.254</sub></td><td align="center">0.038<br><sub>±0.054</sub></td><td align="center">0.098<br><sub>±0.207</sub></td><td align="center">1055.2<br><sub>±136.4</sub></td><td align="center">6.52<br><sub>±0.68</sub></td><td align="center">0.811<br><sub>±0.126</sub></td><td align="center">0.293<br><sub>±0.199</sub></td></tr>
    <tr><td>Always-on</td><td align="center"><b>0.992</b><br><sub>±0.007</sub></td><td align="center">0.990<br><sub>±0.011</sub></td><td align="center">0.003<br><sub>±0.011</sub></td><td align="center">715.1<br><sub>±392.5</sub></td><td align="center">4.82<br><sub>±1.97</sub></td><td align="center">0.896<br><sub>±0.059</sub></td><td align="center">0.161<br><sub>±0.092</sub></td></tr>
    <tr><td><b>Full ACE-RL（本文）</b></td><td align="center">0.989<br><sub>±0.009</sub></td><td align="center">0.490<br><sub>±0.192</sub></td><td align="center"><b>0.500</b><br><sub>±0.188</sub></td><td align="center">1014.3<br><sub>±147.9</sub></td><td align="center">6.31<br><sub>±0.74</sub></td><td align="center">0.786<br><sub>±0.100</sub></td><td align="center">0.335<br><sub>±0.150</sub></td></tr>
    <tr><td colspan="8"><i>种子间的配对差</i></td></tr>
    <tr><td>Full − Always-on</td><td align="center">−0.003<br><sub>±0.013</sub></td><td align="center">−0.500<br><sub>±0.199</sub></td><td align="center">+0.497<br><sub>±0.194</sub></td><td align="center">+299.1<br><sub>±364.0</sub></td><td align="center">+1.49<br><sub>±1.82</sub></td><td align="center">−0.110<br><sub>±0.090</sub></td><td align="center">+0.174<br><sub>±0.138</sub></td></tr>
    <tr><td>Full − Vanilla</td><td align="center">+0.950<br><sub>±0.048</sub></td><td align="center">+0.464<br><sub>±0.192</sub></td><td align="center">+0.486<br><sub>±0.193</sub></td><td align="center">−96.8<br><sub>±165.1</sub></td><td align="center">−0.48<br><sub>±0.83</sub></td><td align="center">−0.046<br><sub>±0.174</sub></td><td align="center">+0.071<br><sub>±0.271</sub></td></tr>
  </tbody>
</table>

<p align="center"><sub>10 个策略种子上的均值 ± 标准差，每个种子在 20 条留出的订单流路径上评估；配对差各行中，± 为逐种子差值的标准差。q<sub>V</sub> 和 q<sub>X</sub> 是影子审计响应概率，即策略在影子观测上赋予一到三个 tick 半价差的概率，分别在有效和无效状态上计算；S = q<sub>V</sub> − q<sub>X</sub> 是选择性。它们描述的是对审计的响应，不是实际报价的频率。其他列来自实际市场：每个交易商 100 步内、扣除库存惩罚之前的毛 PnL，以 tick 计的报价半价差，以及交易商动作的两两一致度和熵。<b>粗体</b>：在有偏好方向（箭头）的列中最好。</sub></p>

Full 和 Always-on 使用相同的集合损失和系数，区别仅在于哪些状态接受这个损失。在影子审计上，两者在有效状态下几乎总是响应，但 Full 把无效状态下的响应大约减半，因此选择性从接近零升到约 0.5，并且在全部十个种子上都更高。由于门控同时把监督从全部被审计的状态减少到其中 46% 的有效状态，这组比较衡量的是门控的总体效果；下面的受控实验把“标签施加在哪里”和“有多少标签”区分开来。在实际市场中，Always-on 的报价价差最窄，而 Full 的平均毛 PnL 更高（十个种子中有七个更高）。

与 Vanilla IPPO 相比，Full 在全部十个种子上都提高了选择性，平均毛 PnL 约低 9%。十个种子中有九个的 PnL 更低，但配对差值在种子之间波动很大，十个种子不足以精确确定损失的大小。这是选择性与 PnL 之间的权衡，而不是全面占优。选择性衡量的也是区分而不是抑制：在无效状态下，Full 的影子审计响应仍在一半左右，而 Vanilla IPPO 的接近零。

<p align="center">
  <a href="assets/tendealer.png"><img src="assets/tendealer.png" alt="Ten-dealer market: training dynamics and held-out results" width="100%"></a>
</p>

<p align="center"><sub><strong>(a, b)</strong> 记录的训练 rollout 上，有效和无效状态下的影子审计响应（10 个种子的均值 ± 1 个标准差）。<strong>(c)</strong> 每个种子的留出响应；虚线对角线表示选择性为零，灰色线段连接同一种子的 Always-on 和 Full 策略。<strong>(d)</strong> 每个种子的留出选择性和实际市场中的毛 PnL；大标记为均值 ± 1 个标准差。</sub></p>

### 🔬 受控实验

受控实验用可以精确计算的终点指标分别检验每个挑战。在三个情境中，学习者在保持（H，即不响应）、温和响应（M）和激进响应（A）之间选择：V 是一个有效的竞争机会，响应合理但收益略低于保持；I 和 P 分别受库存以及价格或风险的约束，保持明显最好。只有在 V 中审计才有效，此时 R = {M, A}。在每个种子内，各方法共享初始参数、更新、批次和样本，只在添加的信号及其作用位置上不同：Vanilla PG（只用金融策略梯度）；动作成本塑形；标量平移（scalar shift），从 V 中所有动作的回报里减去一个常数；打乱的掩码（shuffled mask），在每个批次内打乱有效性标记，因而保持激活标签的数量不变；Always-on；以及 Full ACE-RL。终点指标是 V 中的响应概率 q<sub>V</sub>、I 和 P 中的响应概率 q<sub>X</sub>、选择性 S = q<sub>V</sub> − q<sub>X</sub>，以及精确的期望金融效用 U。

| 对比 | 学习器 | 终点指标 | 差值 | 差值 > 0 的种子 |
| --- | --- | :---: | ---: | ---: |
| C1：Full − scalar shift | 独立 logits | q<sub>V</sub> | 0.263 ± 0.044 | 50/50 |
| C3：Shuffled mask − Full | 独立 logits | q<sub>X</sub> | 0.163 ± 0.005 | 50/50 |
| S3：Full − shuffled mask | 共享 MLP | S | 0.130 ± 0.051 | 100/100 |
| S4：Full − shuffled mask | 共享 MLP | U | 0.033 ± 0.012 | 99/100 |

<p align="center"><sub>预先设定的配对对比：逐种子差值的均值 ± 标准差；双侧单样本 t 检验，在每个实验的五个对比内做 Holm 校正。四个对比都满足所有预先设定的通过标准，且 Holm 校正后的 p 都小于 10<sup>−38</sup>。</sub></p>

- **响应集合（CH1）：** 使用独立 logits 时，Full 在有效情境中的最终响应为 0.878，而标量平移后的策略在每个检查点都与 Vanilla PG 相同（在机器精度内），最终为 0.616：审计标签提供了标量平移无法提供的相对动作信息。动作成本塑形在这里达到最高的效用，选择性也接近 Full，但它的成本比审计标签包含更多信息，因为它还惩罚了无效情境中的响应，而审计器在那里不给出标签。
- **有效性门控（CH2）：** 打乱的掩码得到的激活标签数与 Full 相同，却提高了无效情境中的响应。在共享 MLP 下，每次更新都可能影响所有情境，Full 在全部 100 个种子上都比打乱的掩码更有选择性，并在其中 99 个种子上达到更高的效用。选择性取决于标签施加在哪里，而不是有多少标签。参数共享也会把只针对有效状态的更新传播出去，因此 Full 在无效情境中的响应仍明显高于 Vanilla PG（0.35 对 0.08），与十交易商市场中的情况一致。
- **审计系数 α：** α 从 0 增加到 0.5 时，有效情境中的响应单调上升，无效情境中的响应保持不变，效用只略有下降。
- **Critic 隔离（CH3）：** 在全部 100 个种子上，当 actor 和 critic 使用不相交的编码器时，只做一步审计更新会提高响应集合的概率，并且 critic 用到的所有参数和输出都不变；共享编码器在每个种子上都会改变它们。在十交易商实验的全部 960 次 Full ACE-RL 更新中，critic 上的审计梯度和无效行上的审计损失都严格为零。
- **订单簿：** 在价格–时间优先的撮合引擎中，每个可接受的报价都在订单簿的影子副本上通过了动作网格、价格、不交叉、库存和风险检查；有效行提高响应集合的概率，无效行的更新严格为零，实际订单簿保持不变。这些检查说明单步更新是可执行的，不说明在完整订单簿中的长期效果。

<p align="center">
  <a href="assets/controlled.png"><img src="assets/controlled.png" alt="Controlled experiments" width="100%"></a>
</p>

<p align="center"><sub><strong>(a)</strong> 使用独立 logits 时，审计标签提供了动作方向。<strong>(b)</strong> 在共享 MLP 下，对齐的有效性带来更高的选择性。<strong>(c)</strong> 有效情境中的响应随 α 增加，无效情境中的响应保持不变。分别为 50、100 和 50 个种子上的均值 ± 1 个标准差。</sub></p>

### 🛡️ 稀疏与带噪声的审计

十交易商市场在每一步都用正确的有效性规则做审计，而实际部署的审计器会标注更少的状态，也会出错。我们用公开的学习器、种子和预算，对 Full ACE-RL 重新运行独立 logits 实验。只审计 2% 的行（每次更新约八行），学习曲线基本不变；在独立 logits 下，q<sub>X</sub> 在构造上就不受影响。有效性错误则会让策略逐渐变差：当有效性标记的翻转率升到 30% 时，选择性从 0.64 降到 0.50，但在每个错误率下，全部 50 个种子的选择性仍高于 Vanilla PG（0.37）。这项研究是在审稿之后加入的，没有预先设定。

<p align="center">
  <a href="assets/robustness.png"><img src="assets/robustness.png" alt="Sparse and noisy audits" width="100%"></a>
</p>

<p align="center"><sub><strong>(a)</strong> 审计比例为 100%、10% 和 2% 时的学习曲线重合。<strong>(b)</strong> 翻转有效性标记会减慢 q<sub>V</sub> 的上升和 q<sub>X</sub> 的下降。<strong>(c)</strong> 训练结束时的选择性（点：种子）随翻转率逐渐下降，但仍高于打乱的掩码和 Vanilla PG。</sub></p>

### 🚧 局限

ACE-RL 是对基于成本和基于规则的控制的补充，而不是替代：在给定完整且设定正确的状态–动作成本时，动作成本塑形能达到更高的效用，而安全屏蔽（shield，会移除被禁止的动作）可以直接执行硬性规则。十交易商实验的证据只来自一个模拟市场：审计密集，只有一条有效性规则和一种影子编辑，微调训练也较短；内生的审计选择器、更长的时间跨度、其他市场环境和其他模型架构仍有待检验。十个交易商由同一个共享的 actor 控制，因此结果不能说明十个独立参数化的学习者会如何表现。订单的去向不依赖报价，因此所有方法面对相同的成交和库存路径，毛 PnL 的差异反映的是报价价差；报价对成交概率和库存风险敞口的反馈仍未经检验。由于无效状态下的影子审计响应相对 Vanilla IPPO 有所上升，部署时应在监控库存和逆向选择指标的同时，监控这类状态下的实际报价，并可以在有硬性规则的地方加入安全屏蔽。最后，动作一致度和熵描述的是联合报价的多样性，而不是意图或合谋；后者需要行为或法律层面的分析。

## 🧪 实验

| 研究 | 检验 | 学习器 | 运行数 | 代码 |
| --- | --- | --- | ---: | --- |
| **十交易商市场** | 端到端 | 循环 IPPO，十个交易商 | 10 × 4 | [`revision/`](revision/) |
| **独立 logits** | CH1、CH2 | 每个情境独立的 softmax | 50 × 6 | [`experiments/independent_context/`](experiments/independent_context/) |
| **共享 MLP** | CH2 | 3→8→3 tanh MLP | 100 × 4 | [`experiments/shared_representation/`](experiments/shared_representation/) |
| **系数网格** | CH2 | 独立 logits 学习器 | 50 × 6 | [`experiments/coefficient_sensitivity/`](experiments/coefficient_sensitivity/) |
| **Critic 消融** | CH3 | 宽度为 8 的 actor–critic | 100 × 2 | [`experiments/independent_context/`](experiments/independent_context/) |
| **订单簿** | 可行性 | 七个动作的 logits | 100 × 3 | [`experiments/lob_contract/`](experiments/lob_contract/) |
| **稀疏、带噪声的审计** | 稳健性 | 独立 logits 学习器 | 50 × 9 | [`paper/`](paper/) |

运行数 = 种子数 × 设置数（方法、系数取值、模型变体、场景或扰动水平）。除了稀疏与带噪声审计的研究，每项研究都遵循在正式运行之前冻结的前瞻性协议，这些协议下计划的所有运行都已完成。

## 🚀 快速开始

Python 3.13，安装 `requirements.txt` 中固定版本的依赖；不需要 GPU。受控实验、稀疏与带噪声审计的研究，以及表 4–5 和图 3–5 的脚本都在 CPU 上运行；十交易商市场需要 Docker 和不在本仓库中的文件（见下文）。每个受控实验的运行脚本都会检查其冻结协议的哈希值；`--verify-determinism` 会在临时目录中把实验再运行一遍，并检查核心输出是否逐字节一致。

```bash
git clone https://github.com/moderncavemann/ACE-RL-submission.git
cd ACE-RL-submission
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python revision/experiments/setup_official_on_policy.py   # 克隆 IPPO 测试所需的固定版本 on-policy 源码
python -m pytest -q
```

运行受控实验，在 CPU 上总共约五分钟。其中两个运行脚本拒绝写入已存在的输出目录，再次运行前请先删除 `outputs/`。

```bash
python experiments/independent_context/run_main_experiment.py --formal --verify-determinism --output-dir outputs/independent_context
python experiments/independent_context/run_critic_ablation.py --output-dir outputs/critic_isolation --verify-determinism
python experiments/shared_representation/run_experiment.py --formal --verify-determinism --output-dir outputs/shared_representation
python experiments/coefficient_sensitivity/run_experiment.py --formal --verify-determinism --output-dir outputs/coefficient_sensitivity
python experiments/lob_contract/run_experiment.py --formal --verify-determinism --output-dir outputs/lob_contract
python scripts/verify_release.py
```

[`paper/`](paper/) 用这些输出和仓库中十交易商实验的正式输出，重新计算论文中的表 4–5 和图 3–5。绘图脚本用 LaTeX 和论文的字体排版文字，因此需要安装带 `libertine` 和 `newtx` 宏包的 LaTeX。

```bash
FORMAL=revision/experiments/outputs/abides_typed_audit_ippo_v2/formal
# 表 4 和表 5
python paper/table_stats.py $FORMAL/rows.json outputs/independent_context/paired_contrasts.csv outputs/shared_representation/paired_contrasts.csv
# 图 3
python paper/plot_tendealer.py paper/data/training_history.csv $FORMAL/rows.json outputs/fig_tendealer.pdf
# 图 4
python paper/fig4_data_sd.py outputs/independent_context outputs/shared_representation outputs/coefficient_sensitivity outputs/fig4
python paper/plot_fig4.py outputs/fig4 outputs/fig_controlled.pdf
# 稀疏与带噪声的审计，图 5
python paper/robustness.py experiments/independent_context outputs/robustness.json
python paper/plot_robustness.py outputs/robustness.json outputs/fig_robustness.pdf
```

`paper/data/` 保存这些图背后的数据，上面的命令可以逐字节复现其中的 `fig4/` 和 `robustness.json`。图 3a–b 的逐回合训练历史 `paper/data/training_history.csv`，是用 `paper/extract_training_history.py` 从十交易商正式运行的逐种子日志中提取的，这些日志不在本仓库中。

十交易商实验的运行脚本 [`revision/experiments/run_abides_typed_audit_ippo_v1.py`](revision/experiments/run_abides_typed_audit_ippo_v1.py)（协议 v2）在任何检查不通过时都会停止。它检查冻结配置 [`revision/configs/abides_typed_audit_ippo_v2.json`](revision/configs/abides_typed_audit_ippo_v2.json) 及其协议、Docker 镜像 ID、官方 IPPO 的 commit 以及每个基础检查点的 SHA-256；正式运行（`--formal`）还要求已完成工程冒烟测试（`--engineering-smoke`），并且工作区干净、改动均已提交。固定版本的 ABIDES 镜像（需要支持 `linux/amd64` 的 Docker）和十个基础检查点不在本仓库中。仓库中提供的是论文使用的正式输出，位于 [`revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/`](revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/)，因此不需要 Docker 也能从中重新计算表 4；由于该目录正是配置指定的输出位置，重新做正式运行必须从不含该目录的提交开始。

## 🗂️ 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`experiments/independent_context/`](experiments/independent_context/) | 独立 logits 实验（CH1、CH2）和 critic 消融（CH3）：冻结协议、运行脚本和测试。 |
| [`experiments/shared_representation/`](experiments/shared_representation/) | 共享 MLP 实验（CH2）。 |
| [`experiments/coefficient_sensitivity/`](experiments/coefficient_sensitivity/) | α 的系数网格。 |
| [`experiments/lob_contract/`](experiments/lob_contract/) | 价格–时间优先下的订单簿检查。 |
| [`ace_rl_project/`](ace_rl_project/) | 订单簿检查使用的限价订单簿撮合引擎。 |
| [`revision/`](revision/) | 十交易商市场：审计元组和 IPPO 更新（`src/`）、运行与安装脚本（`experiments/`）、冻结的配置和协议、测试以及正式输出。 |
| [`paper/`](paper/) | 表 4–5 和图 3–5 的脚本与数据，包括稀疏与带噪声审计的研究。 |
| [`scripts/`](scripts/) | `verify_release.py`：检查文件集合、链接和记录的正式结果。 |
| [`assets/`](assets/) | 本 README 的 logo 和图片。 |

## 📚 文档

- 论文：Proceedings of the 7th ACM International Conference on AI in Finance（ICAIF '26），意大利米兰；DOI [10.1145/3858063.3858910](https://doi.org/10.1145/3858063.3858910)。
- 冻结协议：[`experiments/`](experiments/) 下每个目录中的 `PROTOCOL.md`，以及十交易商市场的 [`revision/notes/abides_typed_audit_ippo_v2_protocol.md`](revision/notes/abides_typed_audit_ippo_v2_protocol.md)，均为英文。每份协议写明了对应实验的设计、终点指标和预先设定的检查。

## 🖋 引用

如果这个仓库对你有帮助，请引用：

```bibtex
@inproceedings{wu2026acerl,
      title={{ACE-RL}: Counterfactual Audit Signals for Policy-Level Governance of Autonomous Market Makers},
      author={Guanghao Wu and Zhuo Cai and Shoujin Wang},
      booktitle={Proceedings of the 7th ACM International Conference on AI in Finance},
      series={ICAIF '26},
      year={2026},
      location={Milan, Italy},
      publisher={Association for Computing Machinery},
      address={New York, NY, USA},
      doi={10.1145/3858063.3858910},
      url={https://doi.org/10.1145/3858063.3858910},
}
```

如有问题，欢迎提交 issue。
