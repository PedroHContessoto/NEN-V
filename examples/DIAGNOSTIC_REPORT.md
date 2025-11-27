# Diagnostic Report: Network Death (FR = 0.0)

## Problem Statement

The NEN-V network dies completely (FR = 0.0000, Potential = 0.0000) after extended simulation in `deep_diagnostic`, even with:
- STDP learning enabled
- Homeostatic plasticity active (homeo_eta = 0.1627)
- Continuous external inputs (10% of neurons)
- Adaptive system monitoring

## Root Cause Analysis

### Primary Cause: Weight Decay Overwhelms Learning

**Evidence from `test_input_propagation`:**
```
Step 0: firing=true, weight[0]=0.448032
Step 1: firing=false, weight[0]=0.447987  (-0.000045)
Step 2: firing=false, weight[0]=0.447943  (-0.000044)
Step 3: firing=false, weight[0]=0.447898  (-0.000045)
```

**Weight decay rate** (from `dendritoma.rs:158`):
```rust
weight_decay: 0.0001,
```

**Decay per step**: ~0.000045 (0.01% per step)
**Over 50,000 steps**: Cumulative decay ≈ 39% (assuming no learning)

### Secondary Cause: Insufficient STDP Activity

**STDP only occurs when**:
1. Pre-synaptic neuron fires
2. Post-synaptic neuron fires
3. Both spikes within STDP window (50 timesteps)

**Problem**: Network activity is **sparse** (target FR = 0.14)
- Only ~14% of neurons fire per step
- With refractory period = 5, average firing interval = ~7 steps
- Many synapses receive **no correlated activity** → no STDP

### Contributing Factor: Refractory Period Limits Firing Rate

**From `nenv.rs:158`:**
```rust
refractory_period: 5,
```

**From TESTE 5:**
```
Step  0: fire=true  | last_fire=1   | time_since=0
Step  1: fire=false | last_fire=1   | time_since=1
Step  2: fire=false | last_fire=1   | time_since=2
Step  3: fire=false | last_fire=1   | time_since=3
Step  4: fire=false | last_fire=1   | time_since=4
Step  5: fire=true  | last_fire=6   | time_since=0
```

**Pattern**: Neuron with constant input fires every ~5 steps
- **Max firing rate**: ~20% (1 spike per 5 steps)
- **Target FR**: 14.4%
- **Refractory period blocks** even strong inputs

### Contributing Factor: Homeostasis is Too Slow

**Homeostasis parameters** (from `nenv.rs:164-165`):
```rust
homeo_eta: 0.1627,     // Strength of homeostatic adjustment
homeo_interval: 9,      // Applied every 9 steps
```

**Threshold adjustments observed**:
```
Step 0:  threshold=0.200000
Step 20: threshold=0.185100  (-7.5%)
Step 40: threshold=0.170400  (-14.8%)
Step 60: threshold=0.155500  (-22.3%)
```

**Rate**: ~0.37% decrease per step
**Weight decay**: ~0.01% decrease per step

**Homeostasis helps but can't compensate** for weight decay forever.

## Timeline of Network Death

### Phase 1: Initial Activity (Steps 0-20)
- **FR = 0.417** initially (high because all sensors active)
- Neurons fire, STDP occurs
- Weight decay begins

### Phase 2: Collapse (Steps 20-100)
- **FR drops to 0.000**
- Weight decay dominates learning
- Homeostasis lowers thresholds (0.20 → 0.18)
- Not enough to maintain activity

### Phase 3: Death (Steps 100+)
- Weights approach zero
- Even with low thresholds, potential insufficient
- **FR = 0.000 permanently**
- By step 50,000: weights ~0.0, potential ~0.0

## Why `test_input_propagation` Doesn't Die

Small network (14 neurons) with **constant strong input** (1.0 to same neuron):
- Regular firing every 5 steps
- STDP has opportunity to occur
- Homeostasis keeps threshold adjusted
- Network **survives** but at low activity

Larger network (48 neurons) with **sparse random input** (10%):
- Infrequent firing (14% target)
- Less STDP activity
- Weight decay dominates
- Network **dies**

## Proposed Solutions

### Option 1: Reduce Weight Decay
```rust
// dendritoma.rs
weight_decay: 0.00001,  // Was 0.0001 (10x reduction)
```

**Pros**: Simple fix, allows learning to dominate
**Cons**: May lead to unbounded weight growth

### Option 2: Increase STDP Amplitudes
```rust
// dendritoma.rs
stdp_a_plus: 0.030,    // Was 0.015 (2x increase)
stdp_a_minus: 0.012,   // Was 0.006 (2x increase)
```

**Pros**: Stronger learning signal
**Cons**: May cause instability

### Option 3: Strengthen Homeostasis
```rust
// nenv.rs or AutoConfig
homeo_eta: 0.30,       // Was 0.1627 (2x increase)
homeo_interval: 5,     // Was 9 (more frequent)
```

**Pros**: Faster adaptation to low FR
**Cons**: May cause oscillations

### Option 4: **Adaptive Weight Decay** (RECOMMENDED)
```rust
// In dendritoma.rs, modify apply_weight_maintenance()
let activity_protection = 1.0 - (recent_activity * 0.9);
let effective_decay = self.weight_decay * activity_protection;

for weight in &mut self.weights {
    *weight *= 1.0 - effective_decay;
    // ...
}
```

**Already implemented!** (line 362-363)
**Problem**: `recent_activity` parameter is set to 0.0 in network.rs

**Fix**: Pass actual neuron activity to `apply_weight_maintenance()`

### Option 5: Increase Eligibility Trace Utilization
Eligibility traces are present but **under-utilized**:
- `trace_tau: 200.0` (decays slowly)
- `trace_increment: 0.15` (small boost)

**Current behavior**: Traces build up but reward signal is 0.0
**Suggestion**: Use traces for **unsupervised learning** even without explicit reward

## Recommended Action

**Immediate fix**:
1. Check `recent_activity` parameter passed to `apply_weight_maintenance()`
2. If 0.0, calculate from neuron's `recent_firing_rate`
3. This activates the **already-implemented** activity-based weight protection

**File to check**: `src/network.rs` around line where `apply_weight_maintenance()` is called

**Expected line** (approximate):
```rust
neuron.dendritoma.apply_weight_maintenance(0.0);  // ← PROBLEM: hardcoded 0.0
```

**Should be**:
```rust
neuron.dendritoma.apply_weight_maintenance(neuron.recent_firing_rate);
```

This will protect active weights from decay while allowing unused weights to decay.

---

**Test to confirm**: Run `deep_diagnostic` after fix, expect FR > 0.0 after 50k steps.
