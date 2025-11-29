# Hyperparameter Search - NEN-V Optimization System

Sistema completo de otimização automática de hiperparâmetros para a rede neural NEN-V, com **avaliação real** usando a rede neural integrada em ambientes de benchmark.

## Visão Geral

O sistema de busca de hiperparâmetros permite encontrar automaticamente a melhor configuração para a rede neural, otimizando **45+ parâmetros** distribuídos em 12 categorias diferentes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER OPTIMIZATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Parameter      │    │    Search       │    │   Real          │        │
│  │  Space          │───▶│    Strategy     │───▶│   Evaluation    │        │
│  │  (45+ params)   │    │  (Bayesian/etc) │    │   (NEN-V Agent) │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                      │                      │                  │
│           ▼                      ▼                      ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                    Experiment Orchestrator                       │      │
│  │  • Runs real NEN-V network in benchmark environments            │      │
│  │  • Collects: reward, success rate, stability, efficiency        │      │
│  │  • Early stopping and checkpointing                             │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Teste rápido (10 trials)
cargo run --release --bin hyperopt -- --quick

# Bayesian Optimization (recomendado)
cargo run --release --bin hyperopt -- --strategy bayesian --trials 100

# Ver todas as opções
cargo run --release --bin hyperopt -- --help

# Após rodar o hyperopt, aplique os resultados ao AutoConfig
cargo run --release --bin apply_hyperopt
```

### Workflow Completo em 4 Passos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. HYPEROPT           2. APPLY             3. REVIEW           4. TEST    │
│  ─────────────────     ─────────────────    ─────────────────   ────────── │
│  cargo run --bin       cargo run --bin      Copie o código      cargo test │
│  hyperopt              apply_hyperopt       para derivation.rs  deep_diag  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Estrutura de Arquivos

```
hyperparameter_search/
├── main.rs                  # CLI principal do hyperopt
├── mod.rs                   # Módulo e re-exports
├── param_space.rs           # Definição do espaço de parâmetros (45+)
├── search.rs                # Algoritmos de busca (4 estratégias)
├── environments.rs          # Ambientes de benchmark parametrizáveis
├── evaluation.rs            # Sistema de avaliação com NEN-V real
├── orchestrator.rs          # Coordenação de experimentos
├── apply_hyperopt.rs        # Aplicador de resultados → AutoConfig
├── apply_hyperopt_main.rs   # CLI do apply_hyperopt
└── README.md                # Este arquivo
```

## Arquitetura do Sistema

### Integração com NEN-V Real

O sistema usa a rede neural NEN-V **real** para avaliação, não simulações sintéticas:

```rust
// NENVAgent usa componentes reais:
pub struct NENVAgent {
    network: Network,                    // Rede NEN-V com STDP, eligibility traces
    working_memory: WorkingMemoryPool,   // Working memory real
    curiosity: CuriosityModule,          // Curiosidade intrínseca
}
```

### Ambientes de Benchmark

Seis ambientes calibrados para avaliar diferentes capacidades:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BENCHMARK ENVIRONMENTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INTERNOS (environments.rs):                                                │
│  ├── NavigationEnv       (35%)  Grid world com comida/perigo               │
│  ├── PatternMemoryEnv    (25%)  Memorização de sequências                  │
│  ├── PredictionEnv       (25%)  Previsão de séries temporais               │
│  └── AssociationEnv      (15%)  Aprendizado estímulo-resposta              │
│                                                                             │
│  EXTERNOS (simulations/):                                                   │
│  ├── GridWorldSensorimotor (10%)  Navegação direcional com sensores        │
│  └── RealtimeNavigation    (10%)  Movimento 8-direcional com comida/perigo │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Ambiente | Origem | Descrição | Testa | Peso |
|----------|--------|-----------|-------|------|
| **NavigationEnv** | interno | Grid world com comida/perigo | Aprendizado espacial, reward seeking | 35% |
| **PatternMemoryEnv** | interno | Memorização de sequências | Working memory, recall | 25% |
| **PredictionEnv** | interno | Previsão de séries temporais | Predictive coding, temporal modeling | 25% |
| **AssociationEnv** | interno | Aprendizado estímulo-resposta | Credit assignment, associação | 15% |
| **GridWorldSensorimotor** | simulations/ | Navegação com 4 sensores direcionais | Mapeamento sensor-motor | 10% |
| **RealtimeNavigation** | simulations/ | Grid complexo com 8 direções | Navegação complexa, evasão | 10% |

### Adicionando Novos Ambientes

Para adicionar uma simulação da pasta `simulations/` ao hyperopt:

1. Crie um adaptador em `external_environments.rs`:
```rust
pub struct MinhaSimulacaoEnv { ... }

impl Environment for MinhaSimulacaoEnv {
    fn reset(&mut self) -> Vec<f64> { ... }
    fn step(&mut self, action: usize) -> StepResult { ... }
    fn observation_size(&self) -> usize { ... }
    fn action_size(&self) -> usize { ... }
    fn name(&self) -> &str { "MinhaSimulacao" }
    fn params(&self) -> &EnvironmentParams { ... }
}
```

2. Adicione o case em `environments.rs` → `create_with_difficulty()`
3. Registre em `default_suite()` com peso apropriado

### Calibração Dinâmica

Os ambientes usam **calibração dinâmica** baseada em configuração:

```rust
// Threshold calculado automaticamente:
// threshold = random_baseline + X% * (max_reward - random_baseline)
//
// NavigationEnv:  30% acima do baseline
// PatternMemory:  40% acima do baseline
// PredictionEnv:  20% acima do baseline
// AssociationEnv: 35% acima do baseline
```

Isso garante que:
- Thresholds são sempre > baseline aleatório
- Dificuldade escala com parâmetros do ambiente
- Métricas são comparáveis entre configurações

### Parametrização dos Ambientes

Cada ambiente é totalmente configurável:

```rust
// NavigationConfig
NavigationConfig {
    width: 12, height: 12,
    num_food: 5, num_danger: 3,
    food_reward: 1.0,
    danger_penalty: -0.5,
    movement_cost: -0.01,
    food_respawn: true,
}

// Presets de dificuldade
NavigationConfig::easy()   // Mais comida, menos perigo
NavigationConfig::hard()   // Grid maior, mais obstáculos

// reward_scale é aplicado em todos os steps
let reward = raw_reward * self.params.reward_scale;
```

### Seeds e Reprodutibilidade

Controle completo de seeds em múltiplos níveis:

```rust
// Nível de ambiente
env.set_seed(seed);

// Nível de episódio
env.reset_with_seed(episode_seed);

// Nível de experimento
ExperimentConfig { seed: 42, .. }
```

## Opções da CLI

```
USAGE:
    cargo run --release --bin hyperopt -- [OPTIONS]

OPTIONS:
    --strategy <NAME>     Estratégia de busca: bayesian, random, evolutionary
                         (default: bayesian)

    --trials <N>         Número máximo de trials (default: 100)

    --population <N>     Tamanho da população para evolutionary (default: 20)

    --importance <F>     Importância mínima de parâmetros [0.0-1.0]
                         (default: 0.6)

    --output <DIR>       Diretório de saída para resultados
                         (default: experiments/results)

    --seed <N>           Seed para reprodutibilidade (default: 42)

    --patience <N>       Early stopping patience (default: 20)

    --name <NAME>        Nome do experimento (default: hyperopt)

    --quick              Teste rápido com 10 trials

    --quiet              Suprime output verboso

    --help               Mostra esta mensagem de ajuda
```

## Estratégias de Busca

### 1. Bayesian Optimization (Recomendado)

```rust
BayesianSearch::new(seed)
    .with_exploration(2.0)  // kappa para UCB
```

**Quando usar:** Poucos trials (<200), função objetivo suave

### 2. Random Search

```rust
RandomSearch::new(seed)
```

**Quando usar:** Baseline, muitos trials, alta dimensão

### 3. Evolutionary Search

```rust
EvolutionarySearch::new(seed, population_size)
```

**Quando usar:** Parâmetros discretos, landscape multimodal

### 4. Grid Search

```rust
GridSearch::new(points_per_param)
```

**Quando usar:** Poucos parâmetros (<5), debugging

## Métricas de Avaliação

### Score Combinado

```rust
MetricWeights {
    reward: 0.35,      // Reward médio por episódio
    success: 0.30,     // Taxa de sucesso (reward > threshold)
    stability: 0.15,   // 1 - coef. variação do reward
    learning: 0.10,    // Slope de melhoria ao longo dos episódios
    efficiency: 0.10,  // Reward / energia consumida
}
```

### Métricas por Ambiente

```rust
EnvironmentMetrics {
    avg_reward: f64,
    reward_std: f64,
    success_rate: f64,
    best_reward: f64,
    worst_reward: f64,
    avg_firing_rate: f64,
    avg_energy: f64,
}
```

## Espaço de Parâmetros

### Categorias Principais

| Categoria | # Params | Importância | Parâmetros Chave |
|-----------|----------|-------------|------------------|
| learning | 7 | Alta | `base_learning_rate`, `stdp_a_plus/minus` |
| homeostasis | 6 | Alta | `target_firing_rate`, `homeo_eta` |
| timing | 6 | Alta | `stdp_window`, `eligibility_trace_tau` |
| network | 4 | Média | `inhibitory_ratio`, `initial_threshold` |
| working_memory | 3 | Média | `capacity`, `recurrent_strength` |
| curiosity | 3 | Média | `scale`, `habituation_rate` |
| energy | 4 | Baixa | `cost_fire_ratio`, `recovery_rate` |

### Mapeamento para NENVAgent

```rust
// AgentConfig::from_params() mapeia hiperparâmetros para a rede:
impl AgentConfig {
    pub fn from_params(params: &HashMap<String, ParameterValue>) -> Self {
        // Arquitetura
        config.hidden_neurons = params["architecture.hidden_neurons"];
        config.inhibitory_ratio = params["architecture.inhibitory_ratio"];

        // Aprendizado
        config.learning_rate = params["learning.base_learning_rate"];
        config.stdp_tau_plus = params["stdp.tau_plus"];

        // Working Memory
        config.wm_capacity = params["working_memory.capacity"];

        // etc...
    }
}
```

## Output e Resultados

### Arquivos Gerados

```
experiments/results/
├── <name>_log.csv           # Log trial-by-trial
├── <name>_results.txt       # Resumo final com best config
└── <name>_checkpoint.json   # Checkpoint para resumir
```

## Aplicando Resultados ao AutoConfig

Após executar o hyperopt, use a ferramenta `apply_hyperopt` para transferir os parâmetros otimizados para o sistema AutoConfig:

```bash
# Usa automaticamente o melhor resultado disponível
cargo run --release --bin apply_hyperopt

# Ou especifique um arquivo de resultados
cargo run --release --bin apply_hyperopt experiments/results/mega_full_results.txt
```

### Como Funciona

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYPEROPT → AUTOCONFIG PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. HYPEROPT (Offline)                                                      │
│     ┌─────────────────┐                                                     │
│     │   Busca nos 45  │ → experiments/results/<name>_results.txt            │
│     │   parâmetros    │                                                     │
│     └─────────────────┘                                                     │
│              │                                                              │
│              ▼                                                              │
│  2. APPLY_HYPEROPT                                                          │
│     ┌─────────────────┐                                                     │
│     │  Lê resultados  │ → Parseia valores otimizados                        │
│     │  Gera código    │ → Funções Rust para derivation.rs                   │
│     │  Mostra diff    │ → Comparação antes/depois                           │
│     └─────────────────┘                                                     │
│              │                                                              │
│              ▼                                                              │
│  3. AUTOCONFIG ATUALIZADO                                                   │
│     ┌─────────────────┐                                                     │
│     │  derivation.rs  │ → Novos defaults otimizados                         │
│     │  (você copia)   │                                                     │
│     └─────────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Exemplo de Output

```
╔══════════════════════════════════════════════════════════════════╗
║           HYPEROPT → AUTOCONFIG PARAMETER MAPPING               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 HOMEOSTASIS                                                  ║
║    refractory_period: 5 → 2                                      ║
║    memory_alpha: 0.02 → 0.0457                                   ║
║    homeo_eta: 0.1627 → 0.2314                                    ║
║                                                                  ║
║  📈 STDP                                                         ║
║    window: 16 → 12                                               ║
║    tau_plus: 12.8 → 44.65                                        ║
║    tau_minus: 4.8 → 18.11                                        ║
║                                                                  ║
║  💾 MEMORY                                                       ║
║    weight_decay: 0.0001 → 0.00467                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
CÓDIGO GERADO (copie para src/autoconfig/derivation.rs)
═══════════════════════════════════════════════════════════════════

pub fn compute_homeostatic_params(target_firing_rate: f64) -> HomeostaticParams {
    HomeostaticParams {
        refractory_period: 2,
        memory_alpha: 0.0457,
        homeo_eta: 0.2314,
        ...
    }
}
```

### Workflow Completo

```bash
# 1. Execute o hyperopt
cargo run --release --bin hyperopt -- --strategy bayesian --trials 200

# 2. Aplique os resultados (gera código)
cargo run --release --bin apply_hyperopt

# 3. Copie o código gerado para derivation.rs
#    (manualmente, revisando as mudanças)

# 4. Teste a rede com os novos parâmetros
cargo test
cargo run --release --bin deep_diagnostic
```

### Por que não atualiza automaticamente?

A ferramenta **gera código** mas não sobrescreve automaticamente porque:

1. **Revisão humana**: Você pode querer ajustar alguns valores
2. **Preservar contexto**: O `derivation.rs` pode ter lógica adicional
3. **Segurança**: Evita regressões acidentais
4. **Flexibilidade**: Permite combinar resultados de múltiplos experimentos

### Parâmetros Mapeados

A ferramenta mapeia **todos os 45 parâmetros** do hyperopt para as funções correspondentes em `derivation.rs`:

| Função em derivation.rs | Parâmetros do Hyperopt |
|-------------------------|------------------------|
| `compute_homeostatic_params` | `homeostasis.*`, `timing.refractory_period` |
| `compute_stdp_params` | `timing.stdp_*`, `learning.stdp_*`, `learning.ltp_ltd_ratio` |
| `compute_memory_params` | `memory.*`, `learning.weight_decay` |
| `compute_energy_params` | `energy.*` |
| `compute_stp_params` | `timing.stp_*`, `stp.use_fraction` |
| `compute_competition_params` | `competition.*` |
| `compute_working_memory_params` | `working_memory.*` |
| `compute_curiosity_params` | `curiosity.*` |
| `compute_eligibility_params` | `timing.eligibility_*`, `learning.trace_*` |

### Exemplo de Execução

```
╔══════════════════════════════════════════════════════════════╗
║          HYPERPARAMETER OPTIMIZATION EXPERIMENT             ║
╠══════════════════════════════════════════════════════════════╣
║ Experiment: hyperopt                                         ║
║ Strategy: BayesianOptimization                               ║
║ Parameters: 35                                               ║
║ Max Trials: 100                                              ║
╚══════════════════════════════════════════════════════════════╝

  ★ Trial    0 | Score: 0.5665 | NEW BEST! | 523ms
  ★ Trial    1 | Score: 0.6439 | NEW BEST! | 487ms
    Trial   10 | Score: 0.6201 | Best: 0.6638 | 512ms
  ★ Trial   15 | Score: 0.6891 | NEW BEST! | 498ms
    ...

╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Trials: 100                                            ║
║ Best Score: 0.7234                                           ║
║ Duration: 52.3s                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Adicionando Novos Ambientes

1. Implemente o trait `Environment` em `environments.rs`:

```rust
pub struct MeuAmbiente {
    config: MeuConfig,
    params: EnvironmentParams,
    // ...
}

impl Environment for MeuAmbiente {
    fn reset(&mut self) -> Vec<f64> { ... }
    fn step(&mut self, action: usize) -> StepResult {
        // Calcule raw_reward
        let raw_reward = ...;

        // IMPORTANTE: Aplique reward_scale
        let reward = raw_reward * self.params.reward_scale;

        StepResult::new(obs, reward, done)
    }
    fn observation_size(&self) -> usize { ... }
    fn action_size(&self) -> usize { ... }
    fn name(&self) -> &str { "MeuAmbiente" }
    fn params(&self) -> &EnvironmentParams { &self.params }
    fn random_baseline(&self) -> f64 {
        // Retorne o reward esperado de um agente aleatório
        ...
    }
}
```

2. Adicione ao `EnvironmentRegistry`:

```rust
pub fn create_with_difficulty(&self, name: &str, seed: u64, difficulty: f64)
    -> Option<Box<dyn Environment>>
{
    match name {
        "MeuAmbiente" => Some(Box::new(MeuAmbiente::new(seed))),
        // ...
    }
}
```

3. Configure threshold dinâmico:

```rust
// No construtor:
let random_baseline = /* cálculo do baseline */;
let max_reward = /* reward máximo teórico */;
let success_threshold = random_baseline + (max_reward - random_baseline) * 0.3;
```

## Testes

```bash
# Todos os testes do módulo
cargo test --bin hyperopt

# Testes específicos
cargo test --bin hyperopt test_success_thresholds
cargo test --bin hyperopt test_random_baselines_calibration
cargo test --bin hyperopt test_nenv_agent_learning
```

## Notas Técnicas

### Calibração de Rewards

- `reward_scale` é aplicado em **todos** os `step()` dos ambientes
- Thresholds são calculados dinamicamente a partir de `config`
- `random_baseline()` retorna valor empírico para agente aleatório
- Testes verificam que `threshold > baseline` para todos os ambientes

### Performance

- ~500ms por trial com rede real (modo release)
- ~50 episódios por benchmark × 4 ambientes = ~200 episódios por trial
- Early stopping reduz tempo total significativamente

---

Parte do projeto **NEN-V v2.0**
