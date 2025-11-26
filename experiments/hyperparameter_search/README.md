# Hyperparameter Search - NEN-V Optimization System

Sistema completo de otimização automática de hiperparâmetros para a rede neural NEN-V.

## 🎯 Visão Geral

O sistema de busca de hiperparâmetros permite encontrar automaticamente a melhor configuração para a rede neural, otimizando **45+ parâmetros** distribuídos em 12 categorias diferentes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER OPTIMIZATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Parameter      │    │    Search       │    │   Evaluation    │        │
│  │  Space          │───▶│    Strategy     │───▶│   System        │        │
│  │  (45+ params)   │    │  (Bayesian/etc) │    │  (Benchmarks)   │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                      │                      │                  │
│           ▼                      ▼                      ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                    Experiment Orchestrator                       │      │
│  │  • Parallel execution        • Early stopping                   │      │
│  │  • Result logging            • Best config tracking             │      │
│  │  • Checkpointing             • Progress visualization           │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Teste rápido (10 trials)
cargo run --release --bin hyperopt -- --quick

# Bayesian Optimization (recomendado)
cargo run --release --bin hyperopt -- --strategy bayesian --trials 100

# Ver todas as opções
cargo run --release --bin hyperopt -- --help
```

## 📁 Estrutura de Arquivos

```
hyperparameter_search/
├── main.rs           # CLI principal e entry point
├── mod.rs            # Módulo e re-exports
├── param_space.rs    # Definição do espaço de parâmetros (45+)
├── search.rs         # Algoritmos de busca (4 estratégias)
├── evaluation.rs     # Sistema de benchmarks e métricas
├── orchestrator.rs   # Coordenação de experimentos
└── README.md         # Este arquivo
```

## 🔧 Uso Detalhado

### Opções da CLI

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

### Exemplos de Uso

```bash
# Otimização completa com Bayesian (melhor qualidade)
cargo run --release --bin hyperopt -- \
    --strategy bayesian \
    --trials 200 \
    --name meu_experimento

# Busca rápida com Random Search
cargo run --release --bin hyperopt -- \
    --strategy random \
    --trials 50 \
    --patience 10

# Evolutionary com população grande
cargo run --release --bin hyperopt -- \
    --strategy evolutionary \
    --trials 500 \
    --population 50

# Otimizar apenas parâmetros mais importantes
cargo run --release --bin hyperopt -- \
    --importance 0.8 \
    --trials 100

# Experimento reprodutível
cargo run --release --bin hyperopt -- \
    --seed 12345 \
    --trials 100
```

## 📊 Estratégias de Busca

### 1. Bayesian Optimization (Recomendado)

```rust
// Usa Gaussian Process como modelo surrogate
// Upper Confidence Bound (UCB) como função de aquisição
BayesianSearch::new(seed)
    .with_exploration(2.0)  // kappa para UCB
```

**Vantagens:**
- Mais eficiente em número de trials
- Aprende com resultados anteriores
- Bom balance exploração/exploitation

**Quando usar:**
- Poucos trials disponíveis (<200)
- Parâmetros contínuos
- Função objetivo suave

### 2. Random Search

```rust
// Amostragem uniforme do espaço de parâmetros
RandomSearch::new(seed)
```

**Vantagens:**
- Simples e rápido
- Paralelizável
- Bom baseline

**Quando usar:**
- Baseline inicial
- Muitos trials disponíveis
- Espaço de alta dimensão

### 3. Evolutionary Search

```rust
// Algoritmo genético com seleção, crossover e mutação
EvolutionarySearch::new(seed, population_size)
    .with_mutation_rate(0.1)
    .with_crossover_rate(0.8)
```

**Vantagens:**
- Bom para espaços discretos
- Mantém diversidade
- Pode escapar de mínimos locais

**Quando usar:**
- Parâmetros categóricos/discretos
- Muitos trials disponíveis
- Landscape multimodal

### 4. Grid Search

```rust
// Busca exaustiva em grade
GridSearch::new(points_per_param)
```

**Vantagens:**
- Cobertura completa
- Determinístico
- Fácil de entender

**Quando usar:**
- Poucos parâmetros (<5)
- Necessário cobertura completa
- Debugging

## 📈 Espaço de Parâmetros

### Categorias e Parâmetros

| Categoria | # Params | Parâmetros Principais |
|-----------|----------|----------------------|
| **timing** | 6 | `stdp_window`, `stdp_tau_plus`, `stdp_tau_minus`, `eligibility_trace_tau`, `refractory_period`, `stp_recovery_tau` |
| **learning** | 7 | `base_learning_rate`, `stdp_a_plus`, `stdp_a_minus`, `ltp_ltd_ratio`, `weight_decay`, `trace_increment`, `istdp_rate` |
| **homeostasis** | 6 | `target_firing_rate`, `homeo_eta`, `homeo_interval`, `memory_alpha`, `meta_threshold`, `meta_alpha` |
| **energy** | 4 | `max_energy`, `cost_fire_ratio`, `recovery_rate`, `plasticity_cost_factor` |
| **memory** | 5 | `weight_clamp`, `tag_decay_rate`, `capture_threshold`, `dopamine_sensitivity`, `consolidation_rate` |
| **curiosity** | 3 | `scale`, `surprise_threshold`, `habituation_rate` |
| **network** | 4 | `inhibitory_ratio`, `initial_threshold`, `initial_exc_weight`, `initial_inh_weight` |
| **working_memory** | 3 | `capacity`, `recurrent_strength`, `decay_rate` |
| **predictive** | 2 | `state_learning_rate`, `inference_iterations` |
| **competition** | 2 | `strength`, `interval` |
| **sleep** | 2 | `interval`, `replay_noise` |
| **stp** | 1 | `use_fraction` |

### Níveis de Importância

Os parâmetros têm scores de importância de 0.0 a 1.0:

```
Importância 0.9+ (Críticos):
  • learning.base_learning_rate
  • learning.stdp_a_plus
  • homeostasis.target_firing_rate
  • timing.stdp_window

Importância 0.7-0.9 (Importantes):
  • timing.eligibility_trace_tau
  • network.inhibitory_ratio
  • memory.weight_clamp
  • curiosity.scale

Importância 0.5-0.7 (Moderados):
  • working_memory.capacity
  • predictive.state_learning_rate
  • competition.strength

Importância <0.5 (Secundários):
  • sleep.replay_noise
  • stp.use_fraction
```

Use `--importance` para filtrar parâmetros por importância mínima.

## 🎯 Sistema de Avaliação

### Benchmarks Disponíveis

#### TaskBenchmark
Avalia performance em tarefas de navegação/RL.
```rust
TaskBenchmark::navigation(episodes: 50, max_steps: 500)
```
- Reward total obtido
- Taxa de sucesso
- Steps médios por episódio

#### ConvergenceBenchmark
Mede velocidade de convergência do aprendizado.
```rust
ConvergenceBenchmark::new(max_steps: 10000, threshold: 0.01)
```
- Steps até convergência
- Estabilidade final

#### StabilityBenchmark
Avalia consistência através de múltiplas execuções.
```rust
StabilityBenchmark::new(num_runs: 5, steps_per_run: 1000)
```
- Coeficiente de variação
- Desvio padrão do reward

#### EfficiencyBenchmark
Mede eficiência energética e computacional.
```rust
EfficiencyBenchmark::new(num_steps: 1000)
```
- Reward por unidade de energia
- Taxa de disparo vs performance

### Pesos dos Benchmarks

```rust
MetricWeights {
    reward: 0.4,      // Performance na tarefa
    success: 0.3,     // Taxa de sucesso
    convergence: 0.1, // Velocidade de aprendizado
    stability: 0.1,   // Consistência
    efficiency: 0.1,  // Eficiência energética
}
```

## 📂 Output e Resultados

### Arquivos Gerados

```
experiments/results/
├── <name>_log.csv           # Log trial-by-trial
├── <name>_results.txt       # Resumo final
└── <name>_checkpoint.json   # Checkpoint periódico
```

### Formato do Log CSV

```csv
trial,score,duration_ms,status,config
0,0.5665,113,Completed,"learning.base_learning_rate=0.01;..."
1,0.6439,44,Completed,"learning.base_learning_rate=0.02;..."
```

### Exemplo de Resultado Final

```
=== HYPERPARAMETER OPTIMIZATION RESULTS ===
Experiment: meu_experimento
Strategy: BayesianOptimization
Trials: 100
Best Score: 0.723456

=== BEST CONFIGURATION ===
learning.base_learning_rate: Float(0.0156)
homeostasis.target_firing_rate: Float(0.1234)
timing.stdp_window: Int(45)
...

=== TOP 10 TRIALS ===
1. Trial 87 - Score: 0.723456
2. Trial 92 - Score: 0.718234
3. Trial 76 - Score: 0.712891
...
```

## ⚡ Early Stopping

O sistema para automaticamente se não houver melhoria:

```rust
ExperimentConfig {
    early_stopping_patience: Some(20),  // Para após 20 trials sem melhoria
    min_improvement: 0.001,             // Threshold mínimo de melhoria
}
```

## 🧪 Testes

```bash
# Todos os testes do módulo
cargo test --bin hyperopt

# Testes específicos
cargo test --bin hyperopt test_bayesian
cargo test --bin hyperopt test_evolutionary
cargo test --bin hyperopt test_early_stopping
```

## 🔌 Integração Programática

### Uso como Biblioteca

```rust
use experiments::hyperparameter_search::{
    ExperimentConfig,
    ExperimentOrchestrator,
    OptimizationObjective,
};

fn main() {
    let config = ExperimentConfig {
        name: "custom_experiment".to_string(),
        max_trials: 50,
        early_stopping_patience: Some(10),
        min_param_importance: 0.7,
        verbose: true,
        ..Default::default()
    };

    let mut experiment = ExperimentOrchestrator::with_bayesian(config);
    let result = experiment.run();

    if let Some(best) = result.best_trial {
        println!("Best score: {}", best.score);
        println!("Best config: {:?}", best.config);
    }
}
```

### Criando Estratégia Custom

```rust
use experiments::hyperparameter_search::search::{SearchStrategy, SearchResult};

struct MyCustomSearch {
    // ...
}

impl SearchStrategy for MyCustomSearch {
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        // Sua lógica de sugestão
    }

    fn register_result(&mut self, result: SearchResult) {
        // Registra resultado para aprendizado
    }

    fn best_result(&self) -> Option<&SearchResult> {
        // Retorna melhor resultado
    }

    fn history(&self) -> &[SearchResult] {
        // Retorna histórico
    }

    fn name(&self) -> &str {
        "MyCustomSearch"
    }
}
```

## 📊 Exemplo de Output

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗███╗   ██╗      ██╗   ██╗    ██╗  ██╗██╗   ██╗██████╗    ║
║   ████╗  ██║██╔════╝████╗  ██║      ██║   ██║    ██║  ██║╚██╗ ██╔╝██╔══██╗   ║
║   ██╔██╗ ██║█████╗  ██╔██╗ ██║█████╗██║   ██║    ███████║ ╚████╔╝ ██████╔╝   ║
║   ██║╚██╗██║██╔══╝  ██║╚██╗██║╚════╝╚██╗ ██╔╝    ██╔══██║  ╚██╔╝  ██╔═══╝    ║
║   ██║ ╚████║███████╗██║ ╚████║       ╚████╔╝     ██║  ██║   ██║   ██║        ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝        ╚═══╝      ╚═╝  ╚═╝   ╚═╝   ╚═╝        ║
║                                                                               ║
║           HYPERPARAMETER OPTIMIZATION FOR NEURAL NETWORKS                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER SPACE                          │
├─────────────────────────────────────────────────────────────┤
│ timing         :   6 parameters                             │
│ learning       :   7 parameters                             │
│ homeostasis    :   6 parameters                             │
│ energy         :   4 parameters                             │
│ memory         :   5 parameters                             │
│ curiosity      :   3 parameters                             │
│ network        :   4 parameters                             │
│ working_memory :   3 parameters                             │
│ predictive     :   2 parameters                             │
│ competition    :   2 parameters                             │
│ sleep          :   2 parameters                             │
│ stp            :   1 parameters                             │
├─────────────────────────────────────────────────────────────┤
│ Total:  45 parameters                                       │
└─────────────────────────────────────────────────────────────┘

>> Using Bayesian Optimization (recommended)

╔══════════════════════════════════════════════════════════════╗
║          HYPERPARAMETER OPTIMIZATION EXPERIMENT             ║
╠══════════════════════════════════════════════════════════════╣
║ Experiment: hyperopt                                         ║
║ Strategy: BayesianOptimization                               ║
║ Parameters: 35                                               ║
║ Max Trials: 100                                              ║
╚══════════════════════════════════════════════════════════════╝

  ★ Trial    0 | Score: +0.5665 | NEW BEST! | 113.6µs
  ★ Trial    1 | Score: +0.6439 | NEW BEST! | 44.6µs
    Trial   10 | Score: +0.6201 | Best: +0.6638 | 52.1µs
  ★ Trial   15 | Score: +0.6891 | NEW BEST! | 61.2µs
    Trial   20 | Score: +0.6445 | Best: +0.6891 | 48.3µs
    ...

╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Trials: 100                                            ║
║ Best Score: 0.7234                                           ║
║ Duration: 1.23s                                              ║
║ Reason: MaxTrialsReached                                     ║
╚══════════════════════════════════════════════════════════════╝
```

## 🔗 Referências

- **Bayesian Optimization**: Snoek, J. et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms.
- **Random Search**: Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization.
- **Evolutionary Strategies**: Hansen, N. (2006). The CMA Evolution Strategy: A Tutorial.

---

<div align="center">

Parte do projeto **NEN-V v2.0**

</div>
