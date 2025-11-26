//! # Sistema de Busca de Hiperparâmetros para NEN-V
//!
//! Framework completo para otimização automática de todos os parâmetros da rede neural.
//!
//! ## Arquitetura
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    HYPERPARAMETER OPTIMIZATION SYSTEM                   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
//! │  │  Parameter      │    │    Search       │    │   Evaluation    │    │
//! │  │  Space          │───►│    Strategy     │───►│   System        │    │
//! │  │  Definition     │    │  (Bayesian/etc) │    │   (Benchmarks)  │    │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
//! │           │                      │                      │             │
//! │           ▼                      ▼                      ▼             │
//! │  ┌─────────────────────────────────────────────────────────────┐      │
//! │  │                    Experiment Orchestrator                   │      │
//! │  │  - Parallel execution                                        │      │
//! │  │  - Early stopping                                            │      │
//! │  │  - Result logging                                            │      │
//! │  │  - Best config tracking                                      │      │
//! │  └─────────────────────────────────────────────────────────────┘      │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Módulos
//!
//! - `param_space`: Definição do espaço de parâmetros
//! - `search`: Algoritmos de busca (Grid, Random, Bayesian, Evolutionary)
//! - `evaluation`: Benchmarks e métricas de avaliação
//! - `orchestrator`: Coordenação de experimentos
//!
//! ## Uso
//!
//! ```bash
//! cargo run --release --bin hyperopt -- --strategy bayesian --trials 100
//! ```

pub mod param_space;
pub mod search;
pub mod evaluation;
pub mod orchestrator;

pub use param_space::{
    ParameterSpace, ParameterDef, ParameterValue, ParameterRange,
    NetworkParameterSpace, create_full_parameter_space,
};
pub use search::{
    SearchStrategy, GridSearch, RandomSearch, BayesianSearch,
    EvolutionarySearch, SearchResult,
};
pub use evaluation::{
    Evaluator, BenchmarkSuite, BenchmarkResult, EvaluationMetrics,
    TaskBenchmark, ConvergenceBenchmark, StabilityBenchmark,
};
pub use orchestrator::{
    ExperimentOrchestrator, ExperimentConfig, ExperimentResult,
    TrialResult, OptimizationObjective,
};
