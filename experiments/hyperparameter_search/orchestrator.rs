//! # Orquestrador de Experimentos
//!
//! Coordena a execução de experimentos de otimização de hiperparâmetros.
//!
//! ## Funcionalidades
//!
//! - Execução paralela de trials
//! - Early stopping
//! - Logging e checkpointing
//! - Tracking de melhores resultados

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::{self, File};
use std::io::{Write as IoWrite, BufWriter};
use std::path::PathBuf;

use super::param_space::{ParameterSpace, ParameterValue, create_full_parameter_space};
use super::search::{SearchStrategy, SearchResult, RandomSearch, BayesianSearch, EvolutionarySearch};
use super::evaluation::{Evaluator, EvaluationMetrics, BenchmarkSuite};

/// Objetivo de otimização
#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    /// Maximizar (padrão para reward/score)
    Maximize,
    /// Minimizar (para loss/error)
    Minimize,
}

impl Default for OptimizationObjective {
    fn default() -> Self {
        Self::Maximize
    }
}

/// Configuração do experimento
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Nome do experimento
    pub name: String,
    /// Número máximo de trials
    pub max_trials: usize,
    /// Timeout total (segundos)
    pub timeout_secs: Option<u64>,
    /// Timeout por trial (segundos)
    pub trial_timeout_secs: Option<u64>,
    /// Early stopping: para se não melhorar em N trials
    pub early_stopping_patience: Option<usize>,
    /// Threshold mínimo de melhoria para early stopping
    pub min_improvement: f64,
    /// Objetivo de otimização
    pub objective: OptimizationObjective,
    /// Importância mínima de parâmetros a otimizar
    pub min_param_importance: f64,
    /// Diretório para logs
    pub output_dir: PathBuf,
    /// Salvar checkpoints a cada N trials
    pub checkpoint_interval: usize,
    /// Seed base para reprodutibilidade
    pub seed: u64,
    /// Modo verboso
    pub verbose: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "hyperopt_experiment".to_string(),
            max_trials: 100,
            timeout_secs: None,
            trial_timeout_secs: Some(300),
            early_stopping_patience: Some(20),
            min_improvement: 0.001,
            objective: OptimizationObjective::Maximize,
            min_param_importance: 0.5,
            output_dir: PathBuf::from("experiments/results"),
            checkpoint_interval: 10,
            seed: 42,
            verbose: true,
        }
    }
}

/// Resultado de um trial
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Número do trial
    pub trial_number: usize,
    /// Configuração testada
    pub config: HashMap<String, ParameterValue>,
    /// Métricas obtidas
    pub metrics: EvaluationMetrics,
    /// Score principal
    pub score: f64,
    /// Duração do trial
    pub duration: Duration,
    /// Status
    pub status: TrialStatus,
}

/// Status de um trial
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrialStatus {
    Completed,
    Failed,
    TimedOut,
    Skipped,
}

/// Resultado completo do experimento
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Nome do experimento
    pub name: String,
    /// Configuração usada
    pub config: ExperimentConfig,
    /// Melhor resultado encontrado
    pub best_trial: Option<TrialResult>,
    /// Todos os trials
    pub all_trials: Vec<TrialResult>,
    /// Duração total
    pub total_duration: Duration,
    /// Número de trials completados
    pub completed_trials: usize,
    /// Razão de término
    pub termination_reason: TerminationReason,
}

/// Razão de término do experimento
#[derive(Debug, Clone, Copy)]
pub enum TerminationReason {
    MaxTrialsReached,
    Timeout,
    EarlyStopping,
    UserInterrupted,
    Error,
}

/// Orquestrador principal
pub struct ExperimentOrchestrator {
    config: ExperimentConfig,
    parameter_space: ParameterSpace,
    evaluator: Evaluator,
    strategy: Box<dyn SearchStrategy>,
    trials: Vec<TrialResult>,
    best_score: f64,
    trials_without_improvement: usize,
    start_time: Option<Instant>,
    log_file: Option<BufWriter<File>>,
}

impl ExperimentOrchestrator {
    pub fn new(
        config: ExperimentConfig,
        strategy: Box<dyn SearchStrategy>,
    ) -> Self {
        let parameter_space = create_full_parameter_space()
            .filter_by_importance(config.min_param_importance);

        let evaluator = Evaluator::default_evaluator(config.seed);

        let best_score = match config.objective {
            OptimizationObjective::Maximize => f64::NEG_INFINITY,
            OptimizationObjective::Minimize => f64::INFINITY,
        };

        Self {
            config,
            parameter_space,
            evaluator,
            strategy,
            trials: Vec::new(),
            best_score,
            trials_without_improvement: 0,
            start_time: None,
            log_file: None,
        }
    }

    /// Cria orquestrador com Random Search
    pub fn with_random_search(config: ExperimentConfig) -> Self {
        let strategy = Box::new(RandomSearch::new(config.seed));
        Self::new(config, strategy)
    }

    /// Cria orquestrador com Bayesian Optimization
    pub fn with_bayesian(config: ExperimentConfig) -> Self {
        let strategy = Box::new(BayesianSearch::new(config.seed));
        Self::new(config, strategy)
    }

    /// Cria orquestrador com busca evolutiva
    pub fn with_evolutionary(config: ExperimentConfig, population_size: usize) -> Self {
        let strategy = Box::new(EvolutionarySearch::new(config.seed, population_size));
        Self::new(config, strategy)
    }

    /// Inicializa logging
    fn init_logging(&mut self) -> std::io::Result<()> {
        fs::create_dir_all(&self.config.output_dir)?;

        let log_path = self.config.output_dir.join(format!("{}_log.csv", self.config.name));
        let file = File::create(log_path)?;
        let mut writer = BufWriter::new(file);

        // Header CSV
        writeln!(writer, "trial,score,duration_ms,status,config")?;

        self.log_file = Some(writer);
        Ok(())
    }

    /// Log de um trial
    fn log_trial(&mut self, trial: &TrialResult) {
        if let Some(ref mut writer) = self.log_file {
            let config_str = serde_config_to_string(&trial.config);
            let _ = writeln!(
                writer,
                "{},{:.6},{},{:?},\"{}\"",
                trial.trial_number,
                trial.score,
                trial.duration.as_millis(),
                trial.status,
                config_str
            );
            let _ = writer.flush();
        }
    }

    /// Salva checkpoint
    fn save_checkpoint(&self) -> std::io::Result<()> {
        let checkpoint_path = self.config.output_dir.join(format!("{}_checkpoint.json", self.config.name));
        let mut file = File::create(checkpoint_path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"completed_trials\": {},", self.trials.len())?;
        writeln!(file, "  \"best_score\": {},", self.best_score)?;

        if let Some(ref best) = self.best_trial() {
            writeln!(file, "  \"best_config\": \"{}\",", serde_config_to_string(&best.config))?;
        }

        writeln!(file, "  \"strategy\": \"{}\"", self.strategy.name())?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Verifica se deve parar
    fn should_stop(&self) -> Option<TerminationReason> {
        // Max trials
        if self.trials.len() >= self.config.max_trials {
            return Some(TerminationReason::MaxTrialsReached);
        }

        // Timeout total
        if let (Some(timeout), Some(start)) = (self.config.timeout_secs, self.start_time) {
            if start.elapsed().as_secs() >= timeout {
                return Some(TerminationReason::Timeout);
            }
        }

        // Early stopping
        if let Some(patience) = self.config.early_stopping_patience {
            if self.trials_without_improvement >= patience {
                return Some(TerminationReason::EarlyStopping);
            }
        }

        None
    }

    /// Verifica se houve melhoria
    fn is_improvement(&self, score: f64) -> bool {
        match self.config.objective {
            OptimizationObjective::Maximize => score > self.best_score + self.config.min_improvement,
            OptimizationObjective::Minimize => score < self.best_score - self.config.min_improvement,
        }
    }

    /// Executa um trial
    fn run_trial(&mut self, trial_number: usize) -> TrialResult {
        let trial_start = Instant::now();

        // Sugere configuração
        let config = self.strategy.suggest(&self.parameter_space);

        // Avalia
        let metrics = self.evaluator.evaluate(&config, trial_number);
        let score = metrics.primary_score;

        let duration = trial_start.elapsed();

        let status = if let Some(timeout) = self.config.trial_timeout_secs {
            if duration.as_secs() >= timeout {
                TrialStatus::TimedOut
            } else {
                TrialStatus::Completed
            }
        } else {
            TrialStatus::Completed
        };

        TrialResult {
            trial_number,
            config,
            metrics,
            score,
            duration,
            status,
        }
    }

    /// Melhor trial até agora
    pub fn best_trial(&self) -> Option<&TrialResult> {
        match self.config.objective {
            OptimizationObjective::Maximize => {
                self.trials.iter().max_by(|a, b| {
                    a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
                })
            }
            OptimizationObjective::Minimize => {
                self.trials.iter().min_by(|a, b| {
                    a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
                })
            }
        }
    }

    /// Executa o experimento completo
    pub fn run(&mut self) -> ExperimentResult {
        self.start_time = Some(Instant::now());

        // Inicializa logging
        if let Err(e) = self.init_logging() {
            if self.config.verbose {
                eprintln!("Warning: Could not initialize logging: {}", e);
            }
        }

        if self.config.verbose {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║          HYPERPARAMETER OPTIMIZATION EXPERIMENT             ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ Experiment: {:<48} ║", self.config.name);
            println!("║ Strategy: {:<50} ║", self.strategy.name());
            println!("║ Parameters: {:<48} ║", self.parameter_space.len());
            println!("║ Max Trials: {:<48} ║", self.config.max_trials);
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();
        }

        let termination_reason;

        loop {
            // Verifica condições de parada
            if let Some(reason) = self.should_stop() {
                termination_reason = reason;
                break;
            }

            let trial_number = self.trials.len();

            // Executa trial
            let trial_result = self.run_trial(trial_number);

            // Registra na estratégia
            self.strategy.register_result(SearchResult {
                config: trial_result.config.clone(),
                score: trial_result.score,
                metrics: HashMap::new(),
                trial_number,
            });

            // Verifica melhoria
            if self.is_improvement(trial_result.score) {
                self.best_score = trial_result.score;
                self.trials_without_improvement = 0;

                if self.config.verbose {
                    println!(
                        "  ★ Trial {:4} | Score: {:+.4} | NEW BEST! | {:?}",
                        trial_number,
                        trial_result.score,
                        trial_result.duration
                    );
                }
            } else {
                self.trials_without_improvement += 1;

                if self.config.verbose && trial_number % 10 == 0 {
                    println!(
                        "    Trial {:4} | Score: {:+.4} | Best: {:+.4} | {:?}",
                        trial_number,
                        trial_result.score,
                        self.best_score,
                        trial_result.duration
                    );
                }
            }

            // Log
            self.log_trial(&trial_result);

            // Salva trial
            self.trials.push(trial_result);

            // Checkpoint
            if self.trials.len() % self.config.checkpoint_interval == 0 {
                if let Err(e) = self.save_checkpoint() {
                    if self.config.verbose {
                        eprintln!("Warning: Could not save checkpoint: {}", e);
                    }
                }
            }
        }

        let total_duration = self.start_time.map(|s| s.elapsed()).unwrap_or_default();

        if self.config.verbose {
            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    EXPERIMENT COMPLETE                        ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ Total Trials: {:<46} ║", self.trials.len());
            println!("║ Best Score: {:<48.4} ║", self.best_score);
            println!("║ Duration: {:<50} ║", format!("{:.2?}", total_duration));
            println!("║ Reason: {:<52} ║", format!("{:?}", termination_reason));
            println!("╚══════════════════════════════════════════════════════════════╝");
        }

        // Salva resultado final
        let _ = self.save_final_results();

        ExperimentResult {
            name: self.config.name.clone(),
            config: self.config.clone(),
            best_trial: self.best_trial().cloned(),
            all_trials: self.trials.clone(),
            total_duration,
            completed_trials: self.trials.iter().filter(|t| t.status == TrialStatus::Completed).count(),
            termination_reason,
        }
    }

    /// Salva resultados finais
    fn save_final_results(&self) -> std::io::Result<()> {
        let results_path = self.config.output_dir.join(format!("{}_results.txt", self.config.name));
        let mut file = File::create(results_path)?;

        writeln!(file, "=== HYPERPARAMETER OPTIMIZATION RESULTS ===")?;
        writeln!(file, "Experiment: {}", self.config.name)?;
        writeln!(file, "Strategy: {}", self.strategy.name())?;
        writeln!(file, "Trials: {}", self.trials.len())?;
        writeln!(file, "Best Score: {:.6}", self.best_score)?;
        writeln!(file)?;

        if let Some(best) = self.best_trial() {
            writeln!(file, "=== BEST CONFIGURATION ===")?;
            for (name, value) in &best.config {
                writeln!(file, "{}: {:?}", name, value)?;
            }
        }

        writeln!(file)?;
        writeln!(file, "=== TOP 10 TRIALS ===")?;

        let mut sorted_trials = self.trials.clone();
        sorted_trials.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        for (i, trial) in sorted_trials.iter().take(10).enumerate() {
            writeln!(file, "{}. Trial {} - Score: {:.6}", i + 1, trial.trial_number, trial.score)?;
        }

        Ok(())
    }
}

/// Converte config para string (simplificado)
fn serde_config_to_string(config: &HashMap<String, ParameterValue>) -> String {
    config.iter()
        .map(|(k, v)| format!("{}={:?}", k, v))
        .collect::<Vec<_>>()
        .join(";")
}

// =============================================================================
// HELPERS PARA CRIAÇÃO RÁPIDA
// =============================================================================

/// Cria experimento rápido para teste
pub fn quick_experiment(trials: usize) -> ExperimentOrchestrator {
    let config = ExperimentConfig {
        name: "quick_test".to_string(),
        max_trials: trials,
        early_stopping_patience: Some(trials / 2),
        verbose: true,
        ..Default::default()
    };

    ExperimentOrchestrator::with_bayesian(config)
}

/// Cria experimento completo
pub fn full_experiment(name: &str, trials: usize, strategy: &str) -> ExperimentOrchestrator {
    let config = ExperimentConfig {
        name: name.to_string(),
        max_trials: trials,
        min_param_importance: 0.6,
        verbose: true,
        ..Default::default()
    };

    match strategy {
        "bayesian" => ExperimentOrchestrator::with_bayesian(config),
        "evolutionary" => ExperimentOrchestrator::with_evolutionary(config, 20),
        _ => ExperimentOrchestrator::with_random_search(config),
    }
}

// =============================================================================
// TESTES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_config_default() {
        let config = ExperimentConfig::default();
        assert_eq!(config.max_trials, 100);
        assert!(config.early_stopping_patience.is_some());
    }

    #[test]
    fn test_quick_experiment() {
        let mut experiment = quick_experiment(5);
        let result = experiment.run();

        assert!(result.completed_trials <= 5);
        assert!(result.best_trial.is_some());
    }

    #[test]
    fn test_random_search_experiment() {
        let config = ExperimentConfig {
            name: "test_random".to_string(),
            max_trials: 10,
            verbose: false,
            ..Default::default()
        };

        let mut experiment = ExperimentOrchestrator::with_random_search(config);
        let result = experiment.run();

        assert_eq!(result.all_trials.len(), 10);
    }

    #[test]
    fn test_bayesian_experiment() {
        let config = ExperimentConfig {
            name: "test_bayesian".to_string(),
            max_trials: 15,
            verbose: false,
            ..Default::default()
        };

        let mut experiment = ExperimentOrchestrator::with_bayesian(config);
        let result = experiment.run();

        assert_eq!(result.all_trials.len(), 15);
    }

    #[test]
    fn test_evolutionary_experiment() {
        let config = ExperimentConfig {
            name: "test_evolutionary".to_string(),
            max_trials: 20,
            verbose: false,
            ..Default::default()
        };

        let mut experiment = ExperimentOrchestrator::with_evolutionary(config, 5);
        let result = experiment.run();

        assert_eq!(result.all_trials.len(), 20);
    }

    #[test]
    fn test_early_stopping() {
        let config = ExperimentConfig {
            name: "test_early_stop".to_string(),
            max_trials: 100,
            early_stopping_patience: Some(5),
            min_improvement: 10.0, // Muito alto, vai parar cedo
            verbose: false,
            ..Default::default()
        };

        let mut experiment = ExperimentOrchestrator::with_random_search(config);
        let result = experiment.run();

        // Deve parar antes de 100 por early stopping
        assert!(result.all_trials.len() < 100);
        assert!(matches!(result.termination_reason, TerminationReason::EarlyStopping));
    }
}
