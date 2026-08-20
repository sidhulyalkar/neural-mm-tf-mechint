# Control Architecture

Neural MechInt v0.3 treats mechanistic interpretability as an engineering control loop rather than a post-hoc visualization task.

## Observe → Understand → Control → Learn

```text
prompt + trusted context
        │
        ▼
      Gemma
        │ residual activations at selected layers
        ▼
    MechState
        │ calibrated low-cost signals
        ▼
    MechGuard ───── pass ─────► normal generation
        │
        ├──────── intervene ───► steered generation
        │
        └──────── escalate ────► abstain / external policy
        │
        ▼
   state receipt
        │
        ▼
    MechTune
        │ only causally useful cases
        ▼
 targeted data / representation loss / Circuit-LoRA / intervention distillation
        │
        ▼
     MechEval
        └─ behavior + internal state + causal faithfulness + operational cost
```

## Online versus offline

**Online:** selected residual hooks, compact calibrated probes, explicit monitor thresholds, conditional steering, receipts.

**Offline:** activation patching, SAE/transcoder decomposition, attribution graphs, causal scrubbing, necessity/sufficiency curves, model diffs.

The runtime must never require a full attribution graph per token. Expensive interpretation is reserved for discovery, audits, and difficult flagged cases.

## Epistemic boundary

A monitor is not a mechanism merely because its probe is accurate. Deployment signals are admitted only after held-out validation, random/shuffled controls, causal intervention, collateral-damage measurement, and distribution-shift tests.
