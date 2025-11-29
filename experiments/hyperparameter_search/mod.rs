//! # Sistema de Busca de Hiperparâmetros para NEN-V
//!
//! Framework completo para otimização automática de todos os parâmetros da rede neural.
//!
//! ## Arquitetura
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    HYPERPARAMETER OPTIMIZATION SYSTEM                       │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
//! │  │  Parameter      │    │    Search       │    │   Real          │        │
//! │  │  Space          │───▶│    Strategy     │───▶│   Evaluation    │        │
//! │  │  (45+ params)   │    │  (Bayesian/etc) │    │   (Benchmarks)  │        │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
//! │           │                      │                      │                  │
//! │           ▼                      ▼                      ▼                  │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                    Experiment Orchestrator                           │  │
//! │  │  • Runs neural agent in real environments                           │  │
//! │  │  • Collects: reward, success rate, stability, efficiency            │  │
//! │  │  • Early stopping on score improvement                              │  │
//! │  │  • Logging and checkpointing                                        │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                    Environment Suite                                 │  │
//! │  │  • NavigationEnv     - Grid world navigation                        │  │
//! │  │  • PatternMemoryEnv  - Sequential pattern memorization              │  │
//! │  │  • PredictionEnv     - Time series prediction                       │  │
//! │  │  • AssociationEnv    - Stimulus-response learning                   │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Módulos
//!
//! - `param_space`: Definição do espaço de 45+ parâmetros otimizáveis
//! - `search`: Algoritmos de busca (Grid, Random, Bayesian, Evolutionary)
//! - `environments`: Ambientes reais para avaliação (Navigation, Memory, Prediction)
//! - `evaluation`: Sistema de benchmarks com métricas reais
//! - `orchestrator`: Coordenação de experimentos
//!
//! ## Uso
//!
//! ```bash
//! # Bayesian Optimization (recomendado)
//! cargo run --release --bin hyperopt -- --strategy bayesian --trials 100
//!
//! # Teste rápido
//! cargo run --release --bin hyperopt -- --quick
//!
//! # Evolutionary Search com população grande
//! cargo run --release --bin hyperopt -- --strategy evolutionary --population 30 --trials 200
//! ```
//!
//! ## Adicionando Novos Benchmarks
//!
//! 1. Crie um novo ambiente implementando `trait Environment` em `environments.rs`
//! 2. Registre o ambiente em `EnvironmentBenchmark::create_environment()`
//! 3. Adicione à `BenchmarkSuite::default_suite()` com peso apropriado
//!
//! ```rust
//! // Exemplo de novo ambiente
//! pub struct MeuAmbiente { ... }
//!
//! impl Environment for MeuAmbiente {
//!     fn reset(&mut self) -> Vec<f64> { ... }
//!     fn step(&mut self, action: usize) -> StepResult { ... }
//!     fn observation_size(&self) -> usize { ... }
//!     fn action_size(&self) -> usize { ... }
//!     fn name(&self) -> &str { "MeuAmbiente" }
//! }
//! ```

pub mod param_space;
pub mod search;
pub mod environments;
pub mod evaluation;
pub mod orchestrator;
pub mod apply_hyperopt;
pub mod external_environments;

pub use param_space::{
    ParameterSpace, ParameterDef, ParameterValue, ParameterRange,
    NetworkParameterSpace, create_full_parameter_space,
};
pub use search::{
    SearchStrategy, GridSearch, RandomSearch, BayesianSearch,
    EvolutionarySearch, SearchResult,
};
pub use environments::{
    Environment, StepResult, EnvironmentRegistry,
    NavigationEnv, PatternMemoryEnv, PredictionEnv, AssociationEnv,
};
pub use evaluation::{
    Evaluator, BenchmarkSuite, BenchmarkResult, EvaluationMetrics,
    Benchmark, EnvironmentBenchmark, MetricWeights, EvaluationConfig,
    NENVAgent, AgentConfig, AgentStats, NetworkAgentStats, EpisodeLog,
    EnvironmentMetrics,
};
pub use orchestrator::{
    ExperimentOrchestrator, ExperimentConfig, ExperimentResult,
    TrialResult, OptimizationObjective,
};
pub use external_environments::{
    GridWorldEnv, GridWorldConfig,
    RealtimeEnv, RealtimeEnvConfig,
    ExternalEnvironments,
};
