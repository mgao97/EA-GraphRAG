# EA-GraphRAG 实验实施指南

## 0. 实验目标

本项目验证一个新的 GraphRAG retrieval formulation：

> 将 GraphRAG 从传统的 relevance-driven graph retrieval 重新定义为
> sufficiency-aware evidence acquisition problem。

提出：

**EA-GraphRAG: Evidence-Aware GraphRAG**

核心模块：

**EAC: Evidence Acquisition Controller**

EAC 根据当前 evidence state：

1. Semantic Relevance
2. Structural Information Gain
3. Reasoning Completeness
4. Evidence Consistency

动态选择：

1. RETRIEVE
2. EXPAND
3. BRIDGE
4. VERIFY
5. STOP

最终目标：

minimize Evidence Acquisition Cost

subject to:

Evidence Sufficiency >= threshold

即：

    G* = argmin_G Cost(G)

    s.t.
    Sufficiency(Q, G) >= tau


============================================================
1. 实验总体原则
============================================================

第一阶段不要直接实现一个非常复杂的 LLM Agent。

实验必须采用：

    modular + reproducible + oracle-friendly

的方式。

所有实验都必须记录完整 trajectory：

    query
    initial evidence
    action
    retrieved evidence
    evidence state
    sufficiency score
    next action
    stopping decision
    final evidence
    final answer
    cost

每个 query 保存：

    trajectory.json

例如：

{
    "question": "...",
    "gold_answer": "...",

    "steps": [
        {
            "step": 0,
            "evidence_nodes": [...],
            "evidence_edges": [...],

            "semantic_score": ...,
            "structural_gain": ...,
            "reasoning_coverage": ...,
            "consistency_score": ...,

            "sufficiency_score": ...,

            "action": "RETRIEVE",

            "new_evidence": [...]
        },

        {
            "step": 1,
            ...
        }
    ],

    "final_answer": "...",

    "final_evidence": [...],

    "total_cost": ...
}


============================================================
2. 推荐第一阶段数据集
============================================================

优先使用：

1. HotpotQA
2. 2WikiMultiHopQA
3. MuSiQue

第一阶段实验优先级：

    HotpotQA > 2WikiMultiHopQA > MuSiQue

如果工程时间有限：

第一阶段只需要 HotpotQA。

完成核心实验后再扩展到：

    2WikiMultiHopQA
    MuSiQue


------------------------------------------------------------
2.1 HotpotQA
------------------------------------------------------------

主要用于：

- multi-hop reasoning
- bridge evidence
- evidence sufficiency
- evidence acquisition

需要保留：

    question
    answer
    supporting_facts
    context
    type
    level

重点使用：

    supporting_facts

作为 evidence ground truth。

不要只保存最终 answer。


------------------------------------------------------------
2.2 2WikiMultiHopQA
------------------------------------------------------------

用于验证：

- multi-hop graph reasoning
- bridge search
- structural expansion


------------------------------------------------------------
2.3 MuSiQue
------------------------------------------------------------

用于验证：

- longer reasoning chains
- harder multi-hop reasoning
- adaptive stopping


============================================================
3. Graph 构建
============================================================

第一阶段必须固定 graph construction pipeline。

不要在不同 baseline 中使用不同 KG。

所有 graph-based methods 使用相同的：

    entity
    relation
    node
    edge

定义。

推荐：

    Wikipedia passages / dataset context
          ↓
    entity extraction
          ↓
    relation extraction
          ↓
    directed graph

每个 node 至少保存：

    node_id
    entity_name
    text
    source_document

每条 edge 保存：

    source
    relation
    target
    source_document


============================================================
4. Evidence Representation
============================================================

Evidence 统一定义为：

    E = (V_E, E_E)

其中：

    V_E = evidence nodes
    E_E = evidence edges

同时保存：

    supporting documents
    supporting facts
    text spans

最终 Evidence State：

    S_t = {
        semantic,
        structural,
        reasoning,
        consistency
    }


============================================================
5. 四个核心 Evidence Signals
============================================================


------------------------------------------------------------
5.1 Semantic Relevance
------------------------------------------------------------

目标：

衡量当前 evidence 与 query 的语义相关程度。

第一版实现：

    embedding(query)
    embedding(evidence)

使用 cosine similarity。

推荐：

    BGE-M3
    或 sentence-transformers

不要一开始使用 LLM judge。

定义：

    SemanticScore(E, Q)

可以使用：

1. mean similarity
2. max similarity
3. top-k coverage

第一阶段推荐：

    SemanticScore =
        mean(top-k evidence similarities)


------------------------------------------------------------
5.2 Structural Information Gain
------------------------------------------------------------

目标：

判断新增 graph evidence 是否提供新的结构信息。

不要只计算：

    structural entropy

重点计算：

    ΔH_struct

定义：

    ΔH_t =
        H(G_t) - H(G_{t-1})

可以同时实现以下 structural statistics：

    node count
    edge count
    degree entropy
    path diversity
    connectivity
    structural entropy

第一阶段：

    structural_gain

优先采用：

    entropy difference
    + newly covered nodes
    + newly covered edges

不要把结构熵作为唯一指标。


------------------------------------------------------------
5.3 Reasoning Completeness
------------------------------------------------------------

目标：

判断当前 evidence 是否覆盖完成问题所需的 reasoning chain。

HotpotQA 中：

    supporting_facts

作为 gold evidence。

定义：

    ReasoningCoverage =
        #covered supporting facts
        /
        #gold supporting facts

第一阶段可以采用 oracle/evaluation version。

例如：

Gold:

    A -> B
    B -> C

Current:

    A -> B

则：

    Coverage = 0.5

这个指标用于机制验证。

注意：

不能把 gold supporting facts 直接用于真实 inference。

gold 只能用于 evaluation / oracle experiment。

真实 EA-GraphRAG 必须使用预测的 reasoning completeness。


------------------------------------------------------------
5.4 Evidence Consistency
------------------------------------------------------------

目标：

判断 evidence 是否存在：

    contradiction
    duplicate conflict
    unsupported answer
    incompatible relations

第一阶段实现：

    relation consistency

例如：

    Author X -> born_in -> City A
    Author X -> born_in -> City B

则存在 conflict。

定义：

    ConsistencyScore =
        1 - conflict_ratio


============================================================
6. Evidence Sufficiency
============================================================

第一阶段不要使用一个简单加权平均：

    alpha * semantic
    + beta * structural
    + gamma * reasoning
    + delta * consistency

推荐采用 constraint-based formulation：

    Sufficient(E, Q) =

        semantic >= tau_sem
        AND
        reasoning >= tau_reason
        AND
        consistency >= tau_cons


Structural Gain 主要用于：

    stopping
    expansion decision

而不是简单作为 sufficiency 的第四个 additive term。


============================================================
7. EAC: Evidence Acquisition Controller
============================================================

Action Space：

    RETRIEVE
    EXPAND
    BRIDGE
    VERIFY
    STOP


------------------------------------------------------------
7.1 RETRIEVE
------------------------------------------------------------

适用于：

当前 evidence 与 query 语义相关性不足。

执行：

    query -> semantic retrieval

返回：

    top-k entities / passages


------------------------------------------------------------
7.2 EXPAND
------------------------------------------------------------

适用于：

当前 evidence 已经相关，但 graph structure 不完整。

执行：

    selected node
        ↓
    1-hop / 2-hop neighbors

必须支持：

    top-k neighbor expansion


------------------------------------------------------------
7.3 BRIDGE
------------------------------------------------------------

适用于：

两个 evidence components 之间存在 reasoning gap。

例如：

    A -> B

    C -> D

寻找：

    B -> C

或者：

    shortest path(A, D)


------------------------------------------------------------
7.4 VERIFY
------------------------------------------------------------

适用于：

evidence 已基本完整，但存在：

    contradiction
    uncertainty
    unsupported relation

执行：

    evidence verification


------------------------------------------------------------
7.5 STOP
------------------------------------------------------------

当：

    sufficiency >= threshold

并且：

    marginal information gain < threshold

停止。


============================================================
8. 实验数量
============================================================

核心实验：

    E1  Overall QA Performance
    E2  Evidence Sufficiency Verification
    E3  Adaptive Retrieval Efficiency
    E4  Evidence Acquisition Action Selection
    E5  Component Ablation
    E6  Minimal Sufficient Evidence
    E7  Reasoning Complexity
    E8  Noise / Distractor Robustness

补充实验：

    E9  Controlled Evidence Acquisition
    E10 Generalization Across Graph Backbones


============================================================
9. E1 — Overall QA Performance
============================================================

### Research Question

EA-GraphRAG 是否能够提高最终 QA performance？

### Baselines

至少：

    BM25
    Dense RAG
    GraphRAG
    HippoRAG
    KG²RAG
    GFM-Retriever
    S2G-RAG
    AdaKG-RAG
    ReAct + GraphRAG
    EA-GraphRAG

如果部分 baseline 无法直接复现：

使用官方 implementation / published results。

必须明确区分：

    reproduced result
    reported result

不能混淆。


### Metrics

    Exact Match
    F1
    Answer Accuracy

如果 baseline 支持：

    Retrieval Recall
    Supporting Fact Recall


### Expected Result

EA-GraphRAG：

    Accuracy >= GraphRAG
    Accuracy >= strong GraphRAG baselines

更重要：

    Evidence Cost < GraphRAG


============================================================
10. E2 — Evidence Sufficiency Verification
============================================================

### Research Question

EAC 是否能够判断当前 evidence 是否 sufficient？

### Construct Evidence States

至少四类：

    1. Complete Evidence
    2. Missing Evidence
    3. Irrelevant Evidence
    4. Conflicting Evidence

例如：

Complete:

    A -> B -> C -> Answer

Missing:

    A -> B

Irrelevant:

    X -> Y -> Z

Conflicting:

    A -> B
    A -> C


### Metrics

    Accuracy
    Precision
    Recall
    F1
    AUROC

重点：

    Sufficiency F1


### Baselines

    Semantic-only
    LLM-only Judge
    Random
    EAC

### Expected

    EAC > semantic-only


============================================================
11. E3 — Adaptive Retrieval Efficiency
============================================================

这是主实验之一。

### Research Question

EA-GraphRAG 是否能够用更少 evidence 达到相同或更高的 QA performance？

### Baselines

    Fixed 1-hop
    Fixed 2-hop
    Fixed 3-hop
    Fixed 4-hop
    GraphRAG
    GFM-Retriever
    EA-GraphRAG

### Metrics

    QA F1
    EM

Evidence Cost：

    #nodes
    #edges
    #tokens
    #retrieval calls
    #iterations

### Main Figure

绘制：

    x = Evidence Cost
    y = QA F1

理想结果：

    EA-GraphRAG lies on better Pareto frontier.


### Main Table

| Method | F1 ↑ | EM ↑ | Nodes ↓ | Edges ↓ | Tokens ↓ | Calls ↓ |
|--------|------|------|---------|---------|----------|---------|
| GraphRAG | | | | | | |
| GFM-Retriever | | | | | | |
| S2G-RAG | | | | | | |
| AdaKG-RAG | | | | | | |
| EA-GraphRAG | | | | | | |


============================================================
12. E4 — Action Selection
============================================================

### Research Question

EAC 是否能够选择正确的下一步 evidence acquisition action？

### Action Labels

    RETRIEVE
    EXPAND
    BRIDGE
    VERIFY
    STOP

### Ground Truth

通过 controlled oracle construction 建立。

例如：

Current:

    A -> B

Gold next evidence：

    B -> C

正确 action：

    BRIDGE


### Metrics

    Action Accuracy
    Action Macro-F1

进一步统计：

    Retrieve Accuracy
    Expand Accuracy
    Bridge Accuracy
    Verify Accuracy
    Stop Accuracy


### Baselines

    Random
    Semantic-only
    ReAct
    Oracle
    EAC


### Expected

    EAC >> Random
    EAC > Semantic-only
    EAC approaches Oracle


============================================================
13. E5 — Ablation Study
============================================================

这是论文核心 ablation。

Full：

    Semantic
    +
    Structural
    +
    Reasoning
    +
    Consistency

Variants：

    w/o Semantic
    w/o Structural
    w/o Reasoning
    w/o Consistency

以及：

    Semantic-only
    Semantic + Reasoning
    Semantic + Structural
    Semantic + Reasoning + Consistency
    Full


### Metrics

    QA F1
    Sufficiency F1
    Evidence Cost
    Retrieval Steps
    Action Accuracy


### Expected

Reasoning removal：

    最大程度影响 multi-hop QA

Structural removal：

    增加 unnecessary expansion

Consistency removal：

    增加 conflict errors

Semantic removal：

    增加 irrelevant retrieval


============================================================
14. E6 — Minimal Sufficient Evidence
============================================================

### Research Question

最终 evidence 是否真的接近 minimal sufficient evidence？

### Method

对于最终 evidence：

    G*

执行 leave-one-evidence-out。

对于每个 node / edge：

    G* - e_i

重新进行 QA。

计算：

    performance drop


### Metrics

    Evidence Necessity
    Evidence Precision
    Evidence Recall
    Minimality Ratio

定义：

    Minimality Ratio =
        #necessary evidence
        /
        #retrieved evidence


### Expected

EA-GraphRAG：

    high necessity
    low redundancy


------------------------------------------------------------
14.1 Minimality Test
------------------------------------------------------------

同时进行：

    Random Removal
    Low-score Removal
    High-score Removal
    Gold Evidence Removal

比较 performance degradation。

如果删除 high-criticality evidence
导致最大 performance drop：

证明 controller 选择了关键 evidence。


============================================================
15. E7 — Reasoning Complexity
============================================================

### Research Question

方法是否随着 reasoning complexity 增加而表现更好？

按照：

    1-hop
    2-hop
    3-hop
    4-hop+

分组。

### Metrics

    QA F1
    Evidence Cost
    Sufficiency F1
    Iterations

画：

    x = reasoning hops
    y = QA F1

以及：

    x = reasoning hops
    y = Evidence Cost


### Expected

优势主要出现在：

    3-hop
    4-hop+


============================================================
16. E8 — Noise / Distractor Robustness
============================================================

### Research Question

当 graph 中存在大量 irrelevant evidence 时，
EA-GraphRAG 是否仍能找到 sufficient evidence？

### Noise Ratios

    0%
    25%
    50%
    75%
    90%

### Add

    irrelevant nodes
    irrelevant edges
    semantically similar distractors
    structurally connected distractors


### Metrics

    QA F1
    Evidence Precision
    Evidence Cost
    Stop Accuracy


### Expected

随着 noise 增加：

    GraphRAG performance ↓

EA-GraphRAG：

    degradation slower


============================================================
17. E9 — Controlled Evidence Acquisition
============================================================

这是机制验证实验。

不要使用真实复杂数据。

人工构造 controlled graph。

例如：

    Q
    ↓
    A -> B -> C -> D

同时加入：

    distractor branches
    redundant branches
    missing bridge
    conflicting relation


------------------------------------------------------------
17.1 Variables
------------------------------------------------------------

控制：

    reasoning hops:
        1 / 2 / 3 / 4 / 5

    graph density:
        sparse / medium / dense

    noise ratio:
        0 / 25 / 50 / 75 / 90%

    conflict ratio:
        0 / 10 / 25 / 50%

------------------------------------------------------------
17.2 Metrics
------------------------------------------------------------

    Sufficiency Accuracy
    Action Accuracy
    Stop Accuracy
    Evidence Cost
    Path Recall

这个实验用于证明：

    controller mechanism

而不是证明：

    benchmark performance。


============================================================
18. E10 — Generalization Across Graph Backbones
============================================================

这是非常重要的扩展实验。

目标：

证明 EAC 不是绑定某一个 GraphRAG。

测试：

    GraphRAG-A
    GraphRAG-B
    GraphRAG-C

例如：

    vanilla GraphRAG
    KG²RAG-style retrieval
    HippoRAG-style retrieval

分别：

    baseline

和：

    baseline + EAC


### Metrics

    QA F1
    Evidence Cost
    Retrieval Calls


### Desired Result

在不同 graph retrieval backbone 上：

    + EAC

均能够：

    reduce evidence cost
    maintain / improve QA


这样才能证明：

> EAC 是通用 evidence acquisition controller，
> 而不是特定 GraphRAG 的 engineering trick。


============================================================
19. Structural Information Gain 实验
============================================================

必须单独验证 Structural Signal。

实现三个版本：

    A. No structural signal
    B. Structural entropy
    C. Structural information gain ΔH

比较：

    Stop Accuracy
    Evidence Cost
    QA F1


### 重点 Figure

    x = retrieval iteration
    y = ΔH_struct

同时画：

    x = retrieval iteration
    y = Sufficiency


目标：

证明：

    ΔH_struct ↓

通常对应：

    marginal evidence gain ↓


============================================================
20. Optimal Stopping 实验
============================================================

比较：

    Fixed 1-hop
    Fixed 2-hop
    Fixed 3-hop
    Fixed 4-hop
    Adaptive stopping


Metrics：

    QA F1
    Evidence Cost
    Stop Precision
    Stop Recall
    Over-retrieval Rate
    Under-retrieval Rate


定义：

    Over-retrieval:
        evidence already sufficient
        but agent continues retrieval

    Under-retrieval:
        evidence insufficient
        but agent stops


重点：

    Over-retrieval Rate ↓
    Under-retrieval Rate ↓


============================================================
21. Evidence Acquisition Cost
============================================================

必须统一定义。

至少报告：

    Node Cost
    Edge Cost
    Token Cost
    Retrieval Call Cost
    Iteration Cost


定义：

    C_total =

        λ_v * #nodes
        +
        λ_e * #edges
        +
        λ_t * #tokens
        +
        λ_r * #retrieval_calls


第一阶段可以同时报告 raw metrics。

不要只报告 composite score。


============================================================
22. Pareto Efficiency
============================================================

这是建议加入论文的高级分析。

比较：

    QA F1
    vs
    Evidence Cost

定义：

    Pareto frontier

方法：

    GraphRAG
    GFM-Retriever
    S2G-RAG
    AdaKG-RAG
    EA-GraphRAG


理想结果：

EA-GraphRAG lies closer to:

    high QA
    low evidence cost


============================================================
23. ReAct 对照实验
============================================================

必须做。

因为你的框架有 Agent。

比较：

    ReAct + GraphRAG
    Plan-and-Execute + GraphRAG
    Reflexion + GraphRAG
    EA-GraphRAG


保持：

    same LLM
    same graph
    same retrieval tools
    same maximum iterations


只改变 controller。


### Metrics

    QA F1
    Evidence Cost
    Retrieval Calls
    Sufficiency F1
    Action Accuracy


目标：

证明：

> improvement does not simply come from using an Agent.


而来自：

> sufficiency-aware control。


============================================================
24. LLM 控制器实现
============================================================

第一阶段：

不要 fine-tune LLM。

使用：

    Qwen
    或
    Llama
    或
    GPT API

作为 controller。

固定：

    temperature = 0

要求输出严格 JSON：

{
    "sufficient": true/false,
    "missing_evidence": [],
    "action": "EXPAND",
    "target": [],
    "confidence": 0.87
}

但是：

Semantic / Structural / Reasoning / Consistency
必须作为显式 state 输入。

不能让 LLM 完全自由判断。

例如：

{
    "semantic_score": 0.81,
    "structural_gain": 0.07,
    "reasoning_coverage": 0.67,
    "consistency": 1.0
}


============================================================
25. Oracle Controller
============================================================

必须实现 Oracle。

Oracle 使用：

    gold supporting facts

判断：

    sufficient
    missing evidence
    correct next action


目的：

把系统拆成：

    Retrieval quality
    Controller quality


如果：

    Oracle Controller + Retrieval

明显优于：

    EAC + Retrieval

说明：

    controller 仍然是 bottleneck。


如果：

    EAC ≈ Oracle

说明：

    controller 已经足够准确。


============================================================
26. Retrieval Oracle
============================================================

同样实现：

    Oracle Retrieval

给定：

    current evidence

直接提供：

    gold next evidence


比较：

    EAC Retrieval
    Oracle Retrieval


这可以区分：

    controller error

和：

    retrieval error。


============================================================
27. 必须记录的实验日志
============================================================

每个 query 必须保存：

    query_id
    question
    gold_answer

    gold_supporting_facts

    initial_nodes
    initial_edges

    每一轮：

        iteration
        evidence_nodes
        evidence_edges

        semantic_score
        structural_entropy
        structural_gain

        reasoning_coverage
        consistency_score

        sufficiency

        action
        action_target

        newly_retrieved_nodes
        newly_retrieved_edges

        retrieval_score

        token_count

    final:

        answer
        EM
        F1

        total_nodes
        total_edges
        total_tokens

        retrieval_calls
        iterations

        stop_reason


============================================================
28. 实验结果必须生成的文件
============================================================

results/

├── e1_overall.csv
├── e2_sufficiency.csv
├── e3_efficiency.csv
├── e4_action.csv
├── e5_ablation.csv
├── e6_minimality.csv
├── e7_complexity.csv
├── e8_noise.csv
├── e9_controlled.csv
├── e10_generalization.csv
│
├── trajectories/
│
└── figures/
    ├── overall_performance.pdf
    ├── accuracy_vs_cost.pdf
    ├── sufficiency_curve.pdf
    ├── ablation.pdf
    ├── action_accuracy.pdf
    ├── minimality.pdf
    ├── complexity.pdf
    ├── noise_robustness.pdf
    └── pareto_frontier.pdf


============================================================
29. 第一阶段最小可行实验
============================================================

如果今天只希望快速验证核心 idea：

只实现：

    HotpotQA

    GraphRAG baseline

    Fixed-hop baseline

    ReAct + GraphRAG

    EA-GraphRAG


实现：

    Semantic Score
    Structural Gain
    Reasoning Coverage
    Consistency

实现：

    RETRIEVE
    EXPAND
    BRIDGE
    STOP

先完成：

    E1
    E3
    E5
    E7

即：

    Overall QA
    Efficiency
    Ablation
    Reasoning Complexity


第一阶段不需要：

    multi-agent
    fine-tuning
    LATS
    Tree-of-Thought
    complex RL


============================================================
30. 第一阶段成功标准
============================================================

如果今天实验出现下面结果：

    EA-GraphRAG F1
        >= GraphRAG

同时：

    Evidence Cost
        < GraphRAG

并且：

    w/o Reasoning
        performance ↓

    w/o Structural
        cost ↑

    w/o Semantic
        irrelevant retrieval ↑

    w/o Consistency
        conflict errors ↑

那么：

    核心 idea 得到初步验证。


============================================================
31. 第二阶段实验
============================================================

第一阶段通过以后：

    HotpotQA
        ↓
    2WikiMultiHopQA
        ↓
    MuSiQue

增加：

    E2
    E4
    E6
    E8
    E9


重点验证：

    sufficiency judgment
    action selection
    minimality
    robustness


============================================================
32. 第三阶段实验
============================================================

最后验证：

    E10

在不同 GraphRAG backbone 中加入：

    EAC

如果均有效：

可以提出：

    EAC is a general-purpose
    evidence acquisition controller
    for GraphRAG.


============================================================
33. 最终论文核心实验结论应该回答的问题
============================================================

E1：

    Does EA-GraphRAG improve QA?


E2：

    Can the controller reliably identify
    sufficient evidence?


E3：

    Can it achieve comparable/better QA
    with less evidence?


E4：

    Can it identify what evidence
    should be acquired next?


E5：

    Do semantic / structural /
    reasoning / consistency signals
    each matter?


E6：

    Is the acquired evidence
    actually minimal and necessary?


E7：

    Does the advantage increase
    with reasoning complexity?


E8：

    Is the method robust to
    irrelevant and conflicting evidence?


E9：

    Does the proposed mechanism work
    under controlled conditions?


E10：

    Is the controller generalizable
    across different GraphRAG backbones?


============================================================
34. 最终 Scientific Claim
============================================================

实验最终需要支持：

    Traditional GraphRAG:

        relevance-driven retrieval

    EA-GraphRAG:

        sufficiency-driven evidence acquisition


最终目标：

    not retrieve more evidence

    but retrieve the RIGHT evidence

    until evidence is SUFFICIENT

    and STOP as soon as possible.


核心优化问题：

    min Cost(G_q)

    subject to:

    Sufficiency(Q, G_q) >= τ


最终形成：

    Query
      ↓
    Initial Evidence
      ↓
    Evidence State
      ↓
    Sufficiency Assessment
      ↓
    Gap Identification
      ↓
    Adaptive Action
      ↓
    New Evidence
      ↓
    Evidence State Update
      ↓
    ...
      ↓
    STOP
      ↓
    Minimal Sufficient Evidence
      ↓
    LLM Answer