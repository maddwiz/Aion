# NovaFinTec Architecture

This document captures the core production flow across Q (research), AION (execution), and NovaSpine (memory/feedback).

## Q Signal Flow

```mermaid
flowchart LR
  P["Market Data (OHLCV, macro, alt, credit)"] --> S["Signal Layer\n(momentum, mean-reversion, carry, overlays)"]
  S --> C["Council Votes"]
  C --> M["Meta-Council / Bandit Weights"]
  M --> G["Governor Stack"]
  G --> F["Final Weights\nruns_plus/portfolio_weights_final.csv"]
  F --> O["Q Overlay Pack\nq_signal_overlay.json"]
```

## Governor Stack (Multiplicative Chain)

```mermaid
flowchart LR
  W0["Base Weights W0"] --> T["Turnover Governor\n[0.00, 1.20]"]
  T --> CD["Confirmation Delay\n[0.00, 1.00]"]
  CD --> HG["Hive Conviction Gate\n[0.30, 1.12]"]
  HG --> CM["Calendar Mask\n[0.75, 1.15]"]
  CM --> ME["Meta Execution Gate\n[0.00, 1.00]"]
  ME --> VT["Vol Target Governor\n[0.40, 1.80]"]
  VT --> RT["Runtime Floor Clamp\n[0.00, 1.00]"]
  RT --> WF["Final Weights Wf"]
```

Effective exposure is the product of enabled scalar governors applied element-wise or per-row against base weights.

## AION Execution Flow

```mermaid
flowchart LR
  QO["Q Overlay Bias + Confidence"] --> FE["AION Feature Engine\n(price, VWAP, momentum, regime)"]
  FE --> CG["Confluence / Entry Gates"]
  CG --> RM["Risk Manager\n(size, stop, partials, trailing)"]
  RM --> EX["Execution Simulator / Paper Loop"]
  EX --> IB["IBKR API (paper/live)"]
  EX --> TM["Telemetry + Decision Logs"]
```

## NovaSpine Feedback Loop

```mermaid
flowchart LR
  Q["Q Metrics + Governor Diagnostics"] --> NS["NovaSpine Memory"]
  A["AION Trade Events + Outcomes"] --> NS
  NS --> FB["Feedback Signals\n(quality, lineage, hive context)"]
  FB --> Q
  FB --> A
```

NovaSpine stores structured lineage and trade outcomes, then feeds summarized context back into both Q and AION to influence gating and confidence scaling.

