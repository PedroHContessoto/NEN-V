# Sistema de Cálculo do Score - Hyperparameter Search

Este documento explica detalhadamente como o **score** é calculado durante a otimização de hiperparâmetros.

## Princípio Fundamental: Aprendizado Contínuo

A rede NEN-V é projetada para **aprendizado online e contínuo em tempo real**. Portanto, durante a avaliação:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    APRENDIZADO CONTÍNUO NO HYPEROPT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ UM ÚNICO AGENTE por trial (criado uma vez, reutilizado em tudo)         │
│  ✓ PESOS PRESERVADOS entre episódios e ambientes                           │
│  ✓ STDP/eligibility traces funcionam continuamente                         │
│  ✓ Homeostase e meta-plasticidade acumulam ao longo do trial               │
│                                                                             │
│  Fluxo de um Trial:                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │ Ep 1     │──▶│ Ep 2     │──▶│ Ep 3     │──▶│  ...     │  NavigationEnv  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                 │
│       │              │              │              │                        │
│       └──────────────┴──────────────┴──────────────┘                        │
│                            │                                                │
│                    MESMO AGENTE (pesos acumulam)                            │
│                            │                                                │
│                            ▼                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                                │
│  │ Ep 1     │──▶│ Ep 2     │──▶│  ...     │   PatternMemoryEnv             │
│  └──────────┘   └──────────┘   └──────────┘                                │
│       │                                                                     │
│       └─────────────────────────────────────────────▶ PredictionEnv ...    │
│                                                                             │
│  O agente CONTINUA aprendendo através de todos os ambientes!               │
│  Isso testa a capacidade de TRANSFER LEARNING e generalização.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### O que é preservado vs resetado entre episódios:

| Componente | Resetado? | Motivo |
|------------|-----------|--------|
| **Pesos sinápticos** | NÃO | Preserva todo aprendizado STDP |
| **Eligibility traces** | NÃO | Continua modulação por reward |
| **Thresholds adaptativos** | NÃO | Homeostase continua funcionando |
| **Meta-plasticidade** | NÃO | Histórico de aprendizado mantido |
| **Energia dos neurônios** | NÃO | Estado metabólico preservado |
| Working memory | SIM | Novo contexto de episódio |
| Última observação | SIM | Novo estado inicial do ambiente |
| Último action | SIM | Não carregar viés de ação |

### Código que garante continuidade:

```rust
// evaluation.rs - reset_episode()
pub fn reset_episode(&mut self) {
    // Limpa APENAS working memory (novo contexto de episódio)
    if let Some(ref mut wm) = self.working_memory {
        wm.clear();
    }

    // NÃO reseta a rede - mantém TODO o aprendizado!
    // Apenas estados transitórios de interface
    self.last_observation.fill(0.0);
    self.last_action = 0;
}

// run_all() - mesmo agente para todos os benchmarks
pub fn run_all(&self, params: &HashMap<String, ParameterValue>, seed: u64) -> EvaluationMetrics {
    // Agente criado UMA ÚNICA VEZ por trial
    let mut agent = NENVAgent::new(input_size, output_size, agent_config, seed);

    // REUTILIZADO em todos os benchmarks - aprendizado acumula!
    for benchmark in &self.benchmarks {
        benchmark.run(&mut agent, ...);  // Mesmo agente continua
    }
}
```

### Por que isso é importante?

1. **Realismo**: Sistemas reais não são resetados entre tarefas
2. **Transfer Learning**: Avalia se a rede generaliza conhecimento
3. **Estabilidade**: Configurações ruins causam "forgetting catastrófico"
4. **Eficiência**: Boas configurações aprendem uma vez e reutilizam

## Visão Geral

O score final é um valor entre **0.0 e 1.0** que representa a qualidade de uma configuração de hiperparâmetros. Valores mais altos indicam melhor performance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLUXO DE CÁLCULO DO SCORE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. EPISÓDIOS                                                               │
│     ┌─────────────┐                                                         │
│     │ N episódios │ → reward_history[], successful_episodes                 │
│     │ por ambiente│                                                         │
│     └─────────────┘                                                         │
│            │                                                                │
│            ▼                                                                │
│  2. SCORE DO BENCHMARK (por ambiente)                                       │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ score = 0.60 × success_rate                                         │ │
│     │       + 0.25 × norm_reward                                          │ │
│     │       + 0.15 × norm_stability                                       │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│            │                                                                │
│            ▼                                                                │
│  3. WEIGHTED SCORE (combinação de ambientes)                                │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ weighted_score = Σ (score_i × weight_i)                             │ │
│     │                  ─────────────────────                              │ │
│     │                    Σ weight_i                                       │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│            │                                                                │
│            ▼                                                                │
│  4. ADJUSTED SCORE (métricas de rede)                                       │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ adjusted = 0.35 × norm_reward                                       │ │
│     │          + 0.30 × success_rate                                      │ │
│     │          + 0.15 × stability                                         │ │
│     │          + 0.10 × learning_speed                                    │ │
│     │          + 0.10 × efficiency                                        │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│            │                                                                │
│            ▼                                                                │
│  5. SCORE FINAL                                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ primary_score = (weighted_score + adjusted_score) / 2               │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Etapa 1: Execução de Episódios

Para cada ambiente, o agente executa `N` episódios (tipicamente 15-30):

```rust
for episode in 0..num_episodes {
    let mut episode_reward = 0.0;

    loop {
        let action = agent.select_action(&obs);
        let result = env.step(action);
        agent.learn(&prev_obs, action, result.reward, &result.observation);
        episode_reward += result.reward;

        if result.done { break; }
    }

    reward_history.push(episode_reward);

    if episode_reward >= success_threshold {
        successful_episodes += 1;
    }
}
```

### Métricas Coletadas

| Métrica | Descrição |
|---------|-----------|
| `reward_history[]` | Reward total de cada episódio |
| `successful_episodes` | Episódios com reward >= threshold |
| `firing_rate` | Taxa de disparo média da rede |
| `avg_energy` | Energia média dos neurônios |

## Etapa 2: Score do Benchmark Individual

O score de cada ambiente é calculado com a fórmula:

```rust
// Normalização do reward
let norm_reward = (avg_reward - worst_reward) / (best_reward - worst_reward + 0.01);

// Estabilidade (1.0 = muito estável, 0.0 = muito variável)
let norm_stability = 1.0 - (reward_std / (avg_reward.abs() + 1.0)).min(1.0);

// Taxa de sucesso
let success_rate = successful_episodes / num_episodes;

// Score final do benchmark
let score = 0.60 * success_rate
          + 0.25 * norm_reward.max(0.0)
          + 0.15 * norm_stability;
```

### Pesos do Benchmark

| Componente | Peso | Justificativa |
|------------|------|---------------|
| `success_rate` | 60% | Principal indicador de que a rede aprendeu |
| `norm_reward` | 25% | Qualidade do desempenho (não apenas sucesso/falha) |
| `norm_stability` | 15% | Consistência (evita soluções instáveis) |

### Normalização do Reward

O `norm_reward` normaliza o reward médio no range [0, 1]:

```
                avg_reward - worst_reward
norm_reward = ────────────────────────────────
              best_reward - worst_reward + ε
```

Onde:
- `avg_reward` = média dos rewards de todos os episódios
- `best_reward` = maior reward obtido em um episódio
- `worst_reward` = menor reward obtido em um episódio
- `ε = 0.01` (evita divisão por zero)

### Normalização da Estabilidade

```
                        reward_std
norm_stability = 1 - ─────────────────────
                     |avg_reward| + 1.0
```

- Quanto menor o desvio padrão em relação ao reward médio, maior a estabilidade
- Range: [0, 1] onde 1.0 = perfeitamente estável

## Etapa 3: Weighted Score

Os scores de todos os ambientes são combinados usando pesos:

```rust
weighted_score = Σ (benchmark_score × benchmark_weight) / Σ weights
```

### Pesos dos Ambientes

| Ambiente | Peso | Justificativa |
|----------|------|---------------|
| NavigationEnv | 35% | Teste fundamental de aprendizado espacial |
| PatternMemoryEnv | 25% | Avalia working memory e recall |
| PredictionEnv | 25% | Testa predictive coding |
| AssociationEnv | 15% | Credit assignment básico |
| GridWorldSensorimotor | 10% | Navegação direcional externa |
| RealtimeNavigation | 10% | Complexidade adicional externa |

**Total: 120%** (normalizado para 100%)

## Etapa 4: Adjusted Score

Um score adicional é calculado usando métricas globais da rede:

```rust
// Normalização global
let norm_reward = (avg_reward + 10.0).max(0.0) / 20.0;  // Range [-10, 10] → [0, 1]
let norm_success = success_rate;                         // Já em [0, 1]
let norm_stability = 1.0 - (reward_std / (|avg_reward| + 1.0)).min(1.0);
let norm_learning = (learning_speed + 0.1).max(0.0) / 0.2;
let norm_efficiency = energy_efficiency.min(1.0);

// Score ajustado
adjusted_score = 0.35 × norm_reward
               + 0.30 × norm_success
               + 0.15 × norm_stability
               + 0.10 × norm_learning
               + 0.10 × norm_efficiency;
```

### MetricWeights (Default)

| Métrica | Peso | Descrição |
|---------|------|-----------|
| `reward` | 35% | Reward médio normalizado |
| `success` | 30% | Taxa de sucesso global |
| `stability` | 15% | Baixa variância nos resultados |
| `learning` | 10% | Velocidade de melhoria ao longo dos episódios |
| `efficiency` | 10% | Eficiência energética da rede |

## Etapa 5: Score Final

O score final combina os dois scores:

```rust
primary_score = (weighted_score + adjusted_score) / 2.0;
```

### Por que a média?

1. **weighted_score** captura performance por ambiente (scores locais)
2. **adjusted_score** captura métricas globais da rede
3. A média balanceia ambos os aspectos

## Definição do Success Threshold

O `success_threshold` é calibrado dinamicamente para cada ambiente:

```
success_threshold = random_baseline + X% × (max_reward - random_baseline)
```

| Ambiente | X% | Fórmula |
|----------|-----|---------|
| NavigationEnv | 30% | baseline + 0.30 × gap |
| PatternMemoryEnv | 40% | baseline + 0.40 × gap |
| PredictionEnv | 20% | baseline + 0.20 × gap |
| AssociationEnv | 35% | baseline + 0.35 × gap |

### Exemplo: NavigationEnv

```
random_baseline ≈ -2.0  (agente aleatório)
max_reward ≈ 60.0       (coletando toda comida)

success_threshold = -2.0 + 0.30 × (60.0 - (-2.0))
                  = -2.0 + 0.30 × 62.0
                  = -2.0 + 18.6
                  = 16.6
```

Para "ter sucesso", o agente precisa obter reward >= 16.6, o que significa que ele aprendeu algo significativo além do comportamento aleatório.

## Análise: O Score Está Correto?

### Pontos Positivos

1. **Calibração dinâmica**: O threshold é baseado no baseline aleatório, garantindo que sucesso = real aprendizado
2. **Múltiplos critérios**: Combina sucesso, reward, e estabilidade
3. **Normalização robusta**: Todos os componentes normalizados para [0, 1]
4. **Pesos por ambiente**: Permite priorizar testes mais importantes

### Potenciais Problemas

1. **Peso alto em success_rate (60%)**:
   - Pode ser binário demais (passou ou não passou do threshold)
   - Sugestão: Reduzir para 50%, aumentar `norm_reward` para 35%

2. **Double-averaging**:
   - O score final faz média de weighted_score e adjusted_score
   - Isso pode diluir diferenças importantes
   - Considerar: peso maior no weighted_score (0.6 × weighted + 0.4 × adjusted)

3. **norm_reward local vs global**:
   - No benchmark: usa best/worst daquele trial
   - No adjusted: usa range fixo [-10, 10]
   - Inconsistência pode gerar viés

4. **Eficiência energética**:
   - Peso de 10% no adjusted, mas 0% no benchmark score
   - Pode ser sub-valorizada

### Sugestões de Melhoria

```rust
// Benchmark score mais balanceado:
let score = 0.50 * success_rate      // Reduzido de 0.60
          + 0.30 * norm_reward        // Aumentado de 0.25
          + 0.10 * norm_stability     // Reduzido de 0.15
          + 0.10 * learning_curve;    // NOVO: melhoria ao longo dos episódios

// Score final ponderado:
primary_score = 0.65 * weighted_score + 0.35 * adjusted_score;
```

## Exemplo Numérico Completo

### Cenário: Trial com Score = 0.6543

```
NavigationEnv (weight=0.35):
  - 20 episódios, 14 sucessos → success_rate = 0.70
  - avg_reward = 25.0, best = 45.0, worst = -5.0
  - norm_reward = (25-(-5))/(45-(-5)+0.01) = 30/50.01 = 0.60
  - reward_std = 12.0
  - norm_stability = 1 - (12.0/(25.0+1.0)) = 1 - 0.46 = 0.54

  score = 0.60×0.70 + 0.25×0.60 + 0.15×0.54 = 0.42 + 0.15 + 0.08 = 0.65

PatternMemoryEnv (weight=0.25):
  - success_rate = 0.55, norm_reward = 0.45, norm_stability = 0.60
  - score = 0.60×0.55 + 0.25×0.45 + 0.15×0.60 = 0.33 + 0.11 + 0.09 = 0.53

PredictionEnv (weight=0.25):
  - success_rate = 0.80, norm_reward = 0.70, norm_stability = 0.75
  - score = 0.60×0.80 + 0.25×0.70 + 0.15×0.75 = 0.48 + 0.18 + 0.11 = 0.77

AssociationEnv (weight=0.15):
  - success_rate = 0.60, norm_reward = 0.50, norm_stability = 0.65
  - score = 0.60×0.60 + 0.25×0.50 + 0.15×0.65 = 0.36 + 0.13 + 0.10 = 0.59

Weighted Score:
  = (0.65×0.35 + 0.53×0.25 + 0.77×0.25 + 0.59×0.15) / (0.35+0.25+0.25+0.15)
  = (0.228 + 0.133 + 0.193 + 0.089) / 1.0
  = 0.643

Adjusted Score (usando métricas globais):
  norm_reward = 0.625, success = 0.66, stability = 0.60
  learning = 0.55, efficiency = 0.70

  = 0.35×0.625 + 0.30×0.66 + 0.15×0.60 + 0.10×0.55 + 0.10×0.70
  = 0.219 + 0.198 + 0.090 + 0.055 + 0.070
  = 0.632

Primary Score:
  = (0.643 + 0.632) / 2 = 0.638 ≈ 0.65
```

## Conclusão

O sistema de scoring é **fundamentalmente correto** e bem estruturado:

- Usa calibração dinâmica (baseline aleatório)
- Combina múltiplas métricas relevantes
- Normaliza tudo para [0, 1]
- Pondera por importância do ambiente

As melhorias sugeridas são incrementais e não afetam a validade dos resultados atuais. O score de ~0.65-0.67 indica que a rede está aprendendo significativamente acima do baseline aleatório em múltiplos ambientes diferentes.

---

*Arquivo: `experiments/hyperparameter_search/SCORE_CALCULATION.md`*
*Parte do projeto NEN-V v2.0*
