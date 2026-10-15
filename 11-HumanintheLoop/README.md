# 🛡️ Human-in-the-Loop (HITL) Architecture Guide

This directory provides a production-grade masterclass in **Human-in-the-Loop (HITL)** workflows built with **LangGraph**, **LangChain**, and high-performance open LLMs (via Groq with automatic multi-model fallback).

---

## 🧭 Why Human-in-the-Loop?

In mission-critical enterprise applications, full autonomy introduces unacceptable risk. **Human-in-the-Loop** patterns provide deterministic safety, compliance auditing, and parameter verification.

<div align="center">

![HITL Core Taxonomy](diagrams/0_hitl_taxonomy.png)

</div>

<details><summary>🔍 <b>View Mermaid Architecture Code</b></summary>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#ffffff'}}}%%
flowchart LR
    classDef main fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1B4F72,font-weight:bold;
    classDef p1 fill:#E8F8F5,stroke:#1ABC9C,stroke-width:2px,color:#0E6251;
    classDef p2 fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px,color:#78281F,font-weight:bold;
    classDef p3 fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#7E5109;
    classDef p4 fill:#F4ECF7,stroke:#8E44AD,stroke-width:2px,color:#512E5F;

    HITL[Human-in-the-Loop Core Patterns]:::main
    
    HITL --> B1[1. Static Breakpoints: interrupt_before/after]:::p1
    HITL --> B2[2. Dynamic Interrupts: interrupt & Command]:::p2
    HITL --> B3[3. State Editing: update_state]:::p3
    HITL --> B4[4. Time Travel: get_state_history & rewind]:::p4

    B1 --> Out([Mission-Critical Safety & Oversight]):::main
    B2 --> Out
    B3 --> Out
    B4 --> Out
linkStyle default stroke:#ffffff,stroke-width:2px;
```

</details>

---

## 📂 Notebook Catalog

| Notebook | Focus & Core Primitives | Production Use Case |
| :--- | :--- | :--- |
| **[`0-master_human_in_the_loop.ipynb`](file:///c:/RAG/Code/11-HumanintheLoop/0-master_human_in_the_loop.ipynb)** | **Masterclass Guide**: Breakpoints, Dynamic Interrupts, State Editing, and Time-Travel | Enterprise database migration gates, financial transaction overrides, and multi-turn compliance reviews. |
| **[`1-Humanintheloop.ipynb`](file:///c:/RAG/Code/11-HumanintheLoop/1-Humanintheloop.ipynb)** | **Foundational Guide**: Memory checkpointers, tool binding, state inspection, and streaming values | Step-by-step interactive agent debugging, streaming values, and tool call authorization. |

---

## 🎨 Visual Architecture of the HITL Patterns

### 1. Static Breakpoints (`interrupt_before=["tools"]`)
Halts execution automatically before entering designated nodes (e.g., irreversible tool actions).

<div align="center">

![Static Breakpoints](diagrams/1_breakpoints_flow.png)

</div>

<details><summary>🔍 <b>View Mermaid Architecture Code</b></summary>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#ffffff'}}}%%
flowchart LR
    classDef startNode fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1B4F72;
    classDef agentNode fill:#E8F8F5,stroke:#16A085,stroke-width:2px,color:#0E6251;
    classDef pauseNode fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px,color:#78281F,font-weight:bold;
    classDef humanNode fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#7E5109,font-weight:bold;
    classDef execNode fill:#EAFAF1,stroke:#27AE60,stroke-width:2px,color:#145A32;

    START([User Prompt]):::startNode --> Assistant[Assistant Node: Propose Tool Call]:::agentNode
    Assistant --> Gate{interrupt_before: 'tools'}:::pauseNode
    
    Gate -.->|Paused at Breakpoint| Human([Human Approver]):::humanNode
    Human -.->|Approve / Resume None| Tools[Execute Tools Node]:::execNode
    
    Tools --> Assistant
    Assistant --> END([Final Answer]):::execNode
linkStyle default stroke:#ffffff,stroke-width:2px;
```

</details>

---

### 2. Dynamic In-Node Interrupts (`interrupt()` + `Command(resume=...)`)
Pauses execution dynamically *inside* arbitrary node code, yielding structured payloads and resuming with human decision objects.

<div align="center">

![Dynamic Interrupt Flow](diagrams/2_dynamic_interrupt_flow.png)

</div>

<details><summary>🔍 <b>View Mermaid Architecture Code</b></summary>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#ffffff'}}}%%
flowchart TD
    classDef startNode fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1B4F72;
    classDef prepNode fill:#E8F8F5,stroke:#16A085,stroke-width:2px,color:#0E6251;
    classDef pauseNode fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px,color:#78281F,font-weight:bold;
    classDef humanNode fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#7E5109,font-weight:bold;
    classDef resumeNode fill:#EAFAF1,stroke:#27AE60,stroke-width:2px,color:#145A32;

    START([High-Value Action Request]):::startNode --> Assess[Assess Risk & Prepare Proposal]:::prepNode
    Assess --> Trigger[Call interrupt: Emits Payload & Thread ID]:::pauseNode
    
    Trigger -.->|Execution Suspended| Reviewer([Human Supervisor]):::humanNode
    Reviewer -.->|Invoke Command resume=decision| Resume[State Checkpointer Hydrates Node]:::resumeNode
    
    Resume --> Finalize[Commit Execution or Log Rejection]:::resumeNode
    Finalize --> END([Completed Audit Trail]):::resumeNode
linkStyle default stroke:#ffffff,stroke-width:2px;
```

</details>

---

### 3. State Editing & Parameter Overrides (`update_state`)
Allows supervisors to inspect pending model decisions and overwrite tool arguments or message contents mid-flight.

<div align="center">

![State Edit and Time Travel](diagrams/3_state_edit_time_travel.png)

</div>

<details><summary>🔍 <b>View Mermaid Architecture Code</b></summary>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#ffffff'}}}%%
flowchart LR
    classDef startNode fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1B4F72;
    classDef snapNode fill:#F4ECF7,stroke:#8E44AD,stroke-width:2px,color:#512E5F;
    classDef editNode fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#7E5109,font-weight:bold;
    classDef execNode fill:#EAFAF1,stroke:#27AE60,stroke-width:2px,color:#145A32;

    S1[(Checkpoint 1: Initial)]:::snapNode --> S2[(Checkpoint 2: Pending Tool)]:::snapNode
    S2 --> Inspect[Human inspects state: get_state]:::editNode
    Inspect --> Override[Human modifies payload: update_state]:::editNode
    Override --> S3[(Checkpoint 3: Forked State)]:::snapNode
    S3 --> Resume[Resume Execution with Corrected Parameters]:::execNode
    Resume --> END([Approved Output]):::execNode
linkStyle default stroke:#ffffff,stroke-width:2px;
```

</details>

---

### 4. Interactive Clarification & Disambiguation
Engages human analysts in multi-turn dialogues when ambiguous queries require domain clarification.

<div align="center">

![Multi-Turn Review](diagrams/4_multi_turn_review.png)

</div>

<details><summary>🔍 <b>View Mermaid Architecture Code</b></summary>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#ffffff'}}}%%
flowchart TD
    classDef startNode fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1B4F72;
    classDef agentNode fill:#E8F8F5,stroke:#16A085,stroke-width:2px,color:#0E6251;
    classDef evalNode fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#7E5109;
    classDef humanNode fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px,color:#78281F,font-weight:bold;
    classDef passNode fill:#EAFAF1,stroke:#27AE60,stroke-width:2px,color:#145A32;

    START([Ambiguous Business Query]):::startNode --> Agent[Agent: Identify Ambiguity]:::agentNode
    Agent --> Inter[interrupt: Ask Clarifying Questions]:::evalNode
    Inter -.->|Wait for Response| User([Human User / Analyst]):::humanNode
    User -.->|Command resume: Clarification| Agent
    
    Agent --> Check{Ambiguity Resolved?}
    Check -->|No| Inter
    Check -->|Yes: Clear Spec| Exec[Execute Final Action]:::passNode
    Exec --> END([Accurate Result]):::passNode
linkStyle default stroke:#ffffff,stroke-width:2px;
```

</details>

---

## 📊 Comparison & Decision Guide

| Pattern | Control Granularity | State Changes Allowed? | Best Suited For |
| :--- | :--- | :--- | :--- |
| **Static Breakpoint** | Node boundary (`interrupt_before`) | Yes (via `update_state`) | Database writes, third-party API mutations, destructive commands |
| **Dynamic Interrupt** | Exact code line (`interrupt()`) | Yes (via `Command(resume=...)`) | Context-rich approval, risk evaluations, compliance gates |
| **State Editing** | Any message or tool argument | Yes | Fixing prompt typos, adjusting payout amounts, overriding models |
| **Time Travel** | Checkpoint snapshot history | Yes (can fork/replay past state) | Auditing, incident postmortems, reproducing model bugs |
