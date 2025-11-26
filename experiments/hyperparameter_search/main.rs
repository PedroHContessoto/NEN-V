//! # Hyperparameter Optimization CLI
//!
//! Executa experimentos de otimização de hiperparâmetros para NEN-V.
//!
//! ## Uso
//!
//! ```bash
//! # Busca Bayesiana (recomendado)
//! cargo run --release --bin hyperopt -- --strategy bayesian --trials 100
//!
//! # Busca aleatória rápida
//! cargo run --release --bin hyperopt -- --strategy random --trials 50
//!
//! # Busca evolutiva
//! cargo run --release --bin hyperopt -- --strategy evolutionary --trials 200 --population 30
//!
//! # Experimento rápido para teste
//! cargo run --release --bin hyperopt -- --quick
//! ```

mod param_space;
mod search;
mod environments;
mod evaluation;
mod orchestrator;

use std::env;
use std::path::PathBuf;

use param_space::create_full_parameter_space;
use orchestrator::{ExperimentConfig, ExperimentOrchestrator, OptimizationObjective};

/// Configuração da CLI
struct CliConfig {
    strategy: String,
    trials: usize,
    population: usize,
    min_importance: f64,
    output_dir: PathBuf,
    seed: u64,
    patience: usize,
    quick: bool,
    verbose: bool,
    name: String,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            strategy: "bayesian".to_string(),
            trials: 100,
            population: 20,
            min_importance: 0.6,
            output_dir: PathBuf::from("experiments/results"),
            seed: 42,
            patience: 20,
            quick: false,
            verbose: true,
            name: "hyperopt".to_string(),
        }
    }
}

fn print_help() {
    println!(r#"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    NEN-V HYPERPARAMETER OPTIMIZATION                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    cargo run --release --bin hyperopt -- [OPTIONS]

OPTIONS:
    --strategy <NAME>     Search strategy: bayesian, random, evolutionary
                         (default: bayesian)

    --trials <N>         Maximum number of trials (default: 100)

    --population <N>     Population size for evolutionary search (default: 20)

    --importance <F>     Minimum parameter importance to optimize [0.0-1.0]
                         (default: 0.6)

    --output <DIR>       Output directory for results
                         (default: experiments/results)

    --seed <N>           Random seed for reproducibility (default: 42)

    --patience <N>       Early stopping patience (default: 20)

    --name <NAME>        Experiment name (default: hyperopt)

    --quick              Quick test with 10 trials

    --quiet              Suppress verbose output

    --help               Print this help message

EXAMPLES:
    # Full Bayesian optimization
    cargo run --release --bin hyperopt -- --strategy bayesian --trials 200

    # Quick test
    cargo run --release --bin hyperopt -- --quick

    # Evolutionary with custom population
    cargo run --release --bin hyperopt -- --strategy evolutionary --population 50

PARAMETER SPACE:
    The optimization searches over 40+ parameters including:
    - Timing: STDP windows, eligibility traces, refractory periods
    - Learning: Learning rates, LTP/LTD ratios, weight decay
    - Homeostasis: Target firing rates, adaptation parameters
    - Energy: Metabolic costs, recovery rates
    - Memory: Weight limits, consolidation rates
    - Curiosity: Exploration scales, habituation
    - Network: Inhibitory ratios, initial weights
    - Working Memory: Capacity, recurrence strength
    - Predictive: State learning, inference iterations

OUTPUT:
    Results are saved to the output directory:
    - <name>_log.csv: Trial-by-trial results
    - <name>_results.txt: Final results summary
    - <name>_checkpoint.json: Periodic checkpoints
"#);
}

fn parse_args() -> CliConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = CliConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--strategy" => {
                i += 1;
                if i < args.len() {
                    config.strategy = args[i].clone();
                }
            }
            "--trials" => {
                i += 1;
                if i < args.len() {
                    config.trials = args[i].parse().unwrap_or(100);
                }
            }
            "--population" => {
                i += 1;
                if i < args.len() {
                    config.population = args[i].parse().unwrap_or(20);
                }
            }
            "--importance" => {
                i += 1;
                if i < args.len() {
                    config.min_importance = args[i].parse().unwrap_or(0.6);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    config.output_dir = PathBuf::from(&args[i]);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    config.seed = args[i].parse().unwrap_or(42);
                }
            }
            "--patience" => {
                i += 1;
                if i < args.len() {
                    config.patience = args[i].parse().unwrap_or(20);
                }
            }
            "--name" => {
                i += 1;
                if i < args.len() {
                    config.name = args[i].clone();
                }
            }
            "--quick" => {
                config.quick = true;
                config.trials = 10;
                config.patience = 5;
            }
            "--quiet" => {
                config.verbose = false;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    config
}

fn print_parameter_space_summary() {
    let space = create_full_parameter_space();
    let by_cat = space.by_category();

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                    PARAMETER SPACE                           │");
    println!("├─────────────────────────────────────────────────────────────┤");

    let mut categories: Vec<_> = by_cat.keys().collect();
    categories.sort();

    for cat in categories {
        let params = &by_cat[cat];
        println!("│ {:12} : {:3} parameters                                │", cat, params.len());
    }

    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Total: {:3} parameters                                      │", space.len());
    println!("└─────────────────────────────────────────────────────────────┘");
}

fn main() {
    let cli = parse_args();

    println!(r#"
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
"#);

    if cli.verbose {
        print_parameter_space_summary();
    }

    // Cria configuração do experimento
    let exp_config = ExperimentConfig {
        name: cli.name,
        max_trials: cli.trials,
        timeout_secs: None,
        trial_timeout_secs: Some(300),
        early_stopping_patience: Some(cli.patience),
        min_improvement: 0.001,
        objective: OptimizationObjective::Maximize,
        min_param_importance: cli.min_importance,
        output_dir: cli.output_dir,
        checkpoint_interval: 10,
        seed: cli.seed,
        verbose: cli.verbose,
    };

    // Cria orquestrador com estratégia apropriada
    let mut experiment = match cli.strategy.as_str() {
        "random" => {
            if cli.verbose {
                println!("\n>> Using Random Search strategy");
            }
            ExperimentOrchestrator::with_random_search(exp_config)
        }
        "evolutionary" => {
            if cli.verbose {
                println!("\n>> Using Evolutionary Search (population: {})", cli.population);
            }
            ExperimentOrchestrator::with_evolutionary(exp_config, cli.population)
        }
        "bayesian" | _ => {
            if cli.verbose {
                println!("\n>> Using Bayesian Optimization (recommended)");
            }
            ExperimentOrchestrator::with_bayesian(exp_config)
        }
    };

    println!();

    // Executa experimento
    let result = experiment.run();

    // Imprime resultados finais
    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                    FINAL RESULTS                             │");
    println!("├─────────────────────────────────────────────────────────────┤");

    if let Some(best) = &result.best_trial {
        println!("│ Best Score: {:+.6}                                       │", best.score);
        println!("│ Best Trial: #{}                                            │", best.trial_number);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Best Configuration (top parameters):                        │");

        // Ordena por importância e mostra top 10
        let space = create_full_parameter_space();
        let mut params: Vec<_> = best.config.iter()
            .filter_map(|(name, value)| {
                space.parameters.get(name).map(|def| (name, value, def.importance))
            })
            .collect();

        params.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (name, value, _) in params.iter().take(10) {
            let value_str = match value {
                param_space::ParameterValue::Float(v) => format!("{:.4}", v),
                param_space::ParameterValue::Int(v) => format!("{}", v),
                param_space::ParameterValue::Bool(v) => format!("{}", v),
                param_space::ParameterValue::String(v) => v.clone(),
            };
            println!("│   {}: {}                           │", name, value_str);
        }
    }

    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Total Trials: {:4}                                         │", result.completed_trials);
    println!("│ Duration: {:?}                                    │", result.total_duration);
    println!("│ Termination: {:?}                              │", result.termination_reason);
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\nResults saved to: experiments/results/");
    println!("\nTo use the best configuration, copy the parameters to your network config.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_default() {
        let config = CliConfig::default();
        assert_eq!(config.strategy, "bayesian");
        assert_eq!(config.trials, 100);
    }

    #[test]
    fn test_experiment_runs() {
        let exp_config = ExperimentConfig {
            name: "test".to_string(),
            max_trials: 5,
            verbose: false,
            ..Default::default()
        };

        let mut experiment = ExperimentOrchestrator::with_random_search(exp_config);
        let result = experiment.run();

        assert_eq!(result.all_trials.len(), 5);
    }
}
