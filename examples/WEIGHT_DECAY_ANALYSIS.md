# Weight Decay Experiment - Análise Detalhada

## Experimento Executado

Comparamos duas redes idênticas rodando por 10,000 steps:
- **Rede A**: STDP ativo
- **Rede B**: Hebbian (sem STDP)

Ambas receberam inputs aleatórios idênticos (10% dos sensores ativos por step).

## Resultados

### Rede A (COM STDP)

| Métrica | Inicial | Final | Mudança |
|---------|---------|-------|---------|
| Soma de pesos | 19.694 | 6.564 | **-66.67%** |
| Peso médio | 0.703 | 0.234 | -66.67% |
| Threshold | 0.200 | 0.005 | -97.5% |
| FR (step 1000) | 0.929 | - | - |
| FR (step 2000+) | - | 0.000 | **MORTA** |
| Disparos totais | - | 158 | 1.58% |

**Observações críticas:**
- Step 1000: Pesos **AUMENTARAM** +182% (55.63), FR = 0.929 ✅
- Step 2000: Pesos **COLAPSARAM** -22% (15.28), FR = 0.000 ❌
- Step 3000+: Decay contínuo até estabilizar em ~6.5

### Rede B (Hebbian, sem STDP)

| Métrica | Inicial | Final | Mudança |
|---------|---------|-------|---------|
| Soma de pesos | 19.862 | 6.191 | **-68.83%** |
| Peso médio | 0.709 | 0.221 | -68.83% |

## Análise Matemática

### Decay Teórico Esperado

```python
decay_rate = 0.0001 (per step)
steps = 10,000
final_ratio = (1 - 0.0001)^10000 = 0.3679
decay_percentage = 63.21%
```

### Decay Observado

- **COM STDP**: 66.67% (3.46% acima do esperado)
- **SEM STDP**: 68.83% (5.62% acima do esperado)

**Conclusão**: Decay observado é **apenas ligeiramente maior** que o esperado. Isso indica que:
1. Weight decay NÃO está sendo aplicado múltiplas vezes de forma severa
2. O problema real é a **falta de compensação por STDP**

## Linha do Tempo da Morte da Rede

### Step 0-1000: Fase de Crescimento
- Pesos aumentam +182% devido a STDP forte
- FR = 0.929 (muito alto!)
- Eligibility traces = 23.5 (alta atividade)
- **158 disparos** do neurônio rastreado

### Step 1000-2000: **COLAPSO CRÍTICO**
- Pesos caem de 55.6 → 15.3 (-72%)
- FR cai para 0.000 (rede morre)
- Eligibility traces = 0.4 (quase zero)
- **Nenhum disparo adicional**

### Step 2000-10000: Morte Estável
- Pesos continuam decaindo lentamente: 15.3 → 6.5 (-57%)
- FR permanece 0.000
- Eligibility traces = 0.000
- Threshold cai para 0.005 (homeostase desesperada)

## Por Que a Rede Morreu?

### 1. Runaway Excitation Inicial (Steps 0-1000)

A rede começou com **atividade excessiva**:
- FR = 0.929 (target = 0.189)
- Pesos aumentaram +182%
- Isso é **runaway LTP** (Long-Term Potentiation)

**Causa**: STDP aplicado sem controle adequado quando há alta correlação.

### 2. Correção Homeostática Excessiva (Steps 1000-2000)

Para combater o runaway, **homeostase interveio**:
- Threshold provavelmente aumentou (dados não coletados)
- Pesos começaram a ser reduzidos
- FR caiu drasticamente

**Problema**: A correção foi **muito forte**, matando a rede.

### 3. Espiral da Morte (Steps 2000+)

Uma vez com FR = 0.000:
- Sem disparos → sem STDP
- Weight decay continua
- Pesos decaem continuamente
- Threshold abaixa mas é inútil (pesos ~0)

**Ciclo vicioso**: Low FR → No STDP → Weight decay → Lower weights → Lower FR → ...

## Análise do Código

### Múltiplos Pontos de Weight Decay

#### 1. `apply_stdp_pair` ([dendritoma.rs:471-472](d:\nenv_visual_sim\src\dendritoma.rs#L471-L472))
```rust
let proportional_decay = self.weights[pre_neuron_id] * 0.0001;
self.weights[pre_neuron_id] -= proportional_decay;
```
**Aplicado**: Por sinapse que recebe STDP

#### 2. `apply_stdp_learning` ([dendritoma.rs:503](d:\nenv_visual_sim\src\dendritoma.rs#L503))
```rust
for weight in &mut self.weights {
    *weight *= 1.0 - self.weight_decay;  // 0.0001
}
```
**Aplicado**: Em TODOS os pesos após STDP

#### 3. `apply_weight_maintenance` ([dendritoma.rs:365](d:\nenv_visual_sim\src\dendritoma.rs#L365))
```rust
let effective_decay = self.weight_decay * activity_protection;
for weight in &mut self.weights {
    *weight *= 1.0 - effective_decay;
}
```
**Aplicado**: Apenas quando **NÃO houve STDP** (ver [network.rs:603-605](d:\nenv_visual_sim\src\network.rs#L603-L605))

### Problema: Decay SEM Proteção Durante STDP

Quando STDP acontece:
- Decay é aplicado em `apply_stdp_pair` (por sinapse)
- Decay é aplicado em `apply_stdp_learning` (global)
- **NENHUM** dos dois usa `activity_protection`!

Quando STDP NÃO acontece:
- Decay é aplicado em `apply_weight_maintenance`
- **USA** `activity_protection` baseado em `recent_firing_rate`

**Resultado**: Neurônios ativos com STDP **não têm proteção** contra decay!

## Soluções Propostas

### Solução 1: Remover Decay Duplo em STDP (RECOMENDADO)

**Problema**: Decay aplicado duas vezes em `apply_stdp_learning`.

**Fix**: Remover o decay global de [dendritoma.rs:501-505](d:\nenv_visual_sim\src\dendritoma.rs#L501-L505):

```rust
// ANTES
pub fn apply_stdp_learning(...) -> usize {
    // ... apply STDP pairs ...

    // Aplica decaimento suave
    for weight in &mut self.weights {
        *weight *= 1.0 - self.weight_decay;  // ← REMOVER ISTO
        *weight = weight.clamp(0.0, self.weight_clamp);
    }

    modified_count
}

// DEPOIS
pub fn apply_stdp_learning(...) -> usize {
    // ... apply STDP pairs ...

    // Decay já foi aplicado em apply_stdp_pair
    // Apenas clamp os pesos
    for weight in &mut self.weights {
        *weight = weight.clamp(0.0, self.weight_clamp);
    }

    modified_count
}
```

**Redução esperada**: De ~66% decay para ~42% em 10k steps.

### Solução 2: Adicionar Activity Protection em STDP

**Problema**: STDP não usa `activity_protection`.

**Fix**: Modificar `apply_stdp_pair` para usar atividade:

```rust
pub fn apply_stdp_pair(&mut self, pre_neuron_id: usize, delta_t: i64, reward: f64) -> bool {
    // ... STDP logic ...

    // Decay proporcional COM PROTEÇÃO
    let proportional_decay = self.weights[pre_neuron_id] * 0.0001;

    // ADICIONAR: proteção baseada em plasticity (proxy para atividade)
    let protection = self.plasticity[pre_neuron_id]; // valores altos = neurônio ativo
    let effective_decay = proportional_decay * (1.0 - protection * 0.5);

    self.weights[pre_neuron_id] -= effective_decay;

    // ...
}
```

### Solução 3: Controlar Runaway LTP

**Problema**: Pesos aumentaram +182% no primeiro 1000 steps.

**Fix**: Adicionar limite dinâmico em `apply_stdp_pair`:

```rust
// Após aplicar STDP weight change
self.weights[pre_neuron_id] += weight_change;

// ADICIONAR: Soft cap para prevenir runaway
let soft_cap = self.weight_clamp * 0.7; // 70% do clamp
if self.weights[pre_neuron_id] > soft_cap {
    let excess = self.weights[pre_neuron_id] - soft_cap;
    self.weights[pre_neuron_id] = soft_cap + excess * 0.3; // Comprime excesso
}

self.weights[pre_neuron_id] = self.weights[pre_neuron_id].clamp(0.0, self.weight_clamp);
```

### Solução 4: Aumentar Amplitudes STDP

**Problema**: STDP muito fraco para compensar decay.

**Fix**: Aumentar `stdp_a_plus` e `stdp_a_minus`:

```rust
// dendritoma.rs:148-149
stdp_a_plus: 0.030,    // Era 0.015 (2x aumento)
stdp_a_minus: 0.012,   // Era 0.006 (2x aumento)
```

**Risco**: Pode causar runaway ainda pior.

## Recomendação Final

**Implementar Solução 1 + Solução 3**:

1. **Remover decay duplo** em `apply_stdp_learning` (linha 503)
2. **Adicionar soft cap** para prevenir runaway LTP
3. **Testar** com o mesmo experimento

**Resultado esperado**:
- Decay reduzido de 66% → ~45%
- Sem runaway inicial
- Rede sobrevive com FR > 0.0

---

**Arquivos para modificar**:
- `src/dendritoma.rs`: Linhas 471-472 (soft cap), 501-505 (remover decay)

**Teste de validação**:
```bash
cargo run --release --example weight_decay_experiment
```

**Critério de sucesso**:
- FR final > 0.05
- Weight decay < 50%
- Sem runaway (max weight_sum < 30.0)
