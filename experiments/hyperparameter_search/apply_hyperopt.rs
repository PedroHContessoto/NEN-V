//! # Apply Hyperopt Results to AutoConfig
//!
//! Ferramenta para aplicar os melhores parâmetros encontrados pelo hyperopt
//! ao sistema AutoConfig (derivation.rs).
//!
//! ## Uso
//!
//! ```bash
//! cargo run --release --bin apply_hyperopt -- --results experiments/results/mega_full_results.txt
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Representa um valor de parâmetro otimizado
#[derive(Debug, Clone)]
pub enum OptValue {
    Float(f64),
    Int(i64),
}

impl OptValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            OptValue::Float(v) => *v,
            OptValue::Int(v) => *v as f64,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            OptValue::Float(v) => *v as i64,
            OptValue::Int(v) => *v,
        }
    }
}

/// Parâmetros otimizados do hyperopt
#[derive(Debug, Clone, Default)]
pub struct OptimizedParams {
    // Timing
    pub stdp_window: Option<i64>,
    pub refractory_period: Option<i64>,
    pub stdp_tau_plus: Option<f64>,
    pub stdp_tau_minus: Option<f64>,
    pub eligibility_trace_tau: Option<f64>,
    pub stp_recovery_tau: Option<f64>,

    // Learning
    pub base_learning_rate: Option<f64>,
    pub stdp_a_plus: Option<f64>,
    pub stdp_a_minus: Option<f64>,
    pub ltp_ltd_ratio: Option<f64>,
    pub weight_decay: Option<f64>,
    pub trace_increment: Option<f64>,
    pub istdp_rate: Option<f64>,

    // Homeostasis
    pub target_firing_rate: Option<f64>,
    pub homeo_eta: Option<f64>,
    pub homeo_interval: Option<i64>,
    pub memory_alpha: Option<f64>,
    pub meta_threshold: Option<f64>,
    pub meta_alpha: Option<f64>,

    // Energy
    pub max_energy: Option<f64>,
    pub cost_fire_ratio: Option<f64>,
    pub recovery_rate: Option<f64>,
    pub plasticity_cost_factor: Option<f64>,

    // Memory
    pub weight_clamp: Option<f64>,
    pub tag_decay_rate: Option<f64>,
    pub capture_threshold: Option<f64>,
    pub dopamine_sensitivity: Option<f64>,
    pub consolidation_rate: Option<f64>,

    // Network
    pub inhibitory_ratio: Option<f64>,
    pub initial_threshold: Option<f64>,
    pub initial_exc_weight: Option<f64>,
    pub initial_inh_weight: Option<f64>,
    pub adaptive_threshold_multiplier: Option<f64>,

    // Competition
    pub competition_strength: Option<f64>,
    pub competition_interval: Option<i64>,

    // Working Memory
    pub wm_capacity: Option<i64>,
    pub wm_recurrent_strength: Option<f64>,
    pub wm_decay_rate: Option<f64>,

    // Sleep
    pub sleep_interval: Option<i64>,
    pub replay_noise: Option<f64>,

    // STP
    pub stp_use_fraction: Option<f64>,

    // Predictive
    pub state_learning_rate: Option<f64>,
    pub inference_iterations: Option<i64>,

    // Curiosity
    pub curiosity_scale: Option<f64>,
    pub surprise_threshold: Option<f64>,
    pub habituation_rate: Option<f64>,
}

impl OptimizedParams {
    /// Parse dos resultados do hyperopt
    pub fn from_results_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Erro ao ler arquivo: {}", e))?;

        let mut params = Self::default();
        let mut in_config_section = false;

        for line in content.lines() {
            if line.contains("=== BEST CONFIGURATION ===") {
                in_config_section = true;
                continue;
            }
            if line.contains("=== TOP 10") {
                break;
            }

            if in_config_section && line.contains(": ") {
                let parts: Vec<&str> = line.split(": ").collect();
                if parts.len() >= 2 {
                    let key = parts[0].trim();
                    let value_str = parts[1].trim();
                    params.parse_param(key, value_str);
                }
            }
        }

        Ok(params)
    }

    fn parse_param(&mut self, key: &str, value_str: &str) {
        // Parse Float(x) ou Int(x)
        let value = if value_str.starts_with("Float(") {
            let num_str = value_str.trim_start_matches("Float(").trim_end_matches(')');
            num_str.parse::<f64>().ok().map(OptValue::Float)
        } else if value_str.starts_with("Int(") {
            let num_str = value_str.trim_start_matches("Int(").trim_end_matches(')');
            num_str.parse::<i64>().ok().map(OptValue::Int)
        } else {
            None
        };

        let Some(val) = value else { return };

        match key {
            // Timing
            "timing.stdp_window" => self.stdp_window = Some(val.as_i64()),
            "timing.refractory_period" => self.refractory_period = Some(val.as_i64()),
            "timing.stdp_tau_plus" => self.stdp_tau_plus = Some(val.as_f64()),
            "timing.stdp_tau_minus" => self.stdp_tau_minus = Some(val.as_f64()),
            "timing.eligibility_trace_tau" => self.eligibility_trace_tau = Some(val.as_f64()),
            "timing.stp_recovery_tau" => self.stp_recovery_tau = Some(val.as_f64()),

            // Learning
            "learning.base_learning_rate" => self.base_learning_rate = Some(val.as_f64()),
            "learning.stdp_a_plus" => self.stdp_a_plus = Some(val.as_f64()),
            "learning.stdp_a_minus" => self.stdp_a_minus = Some(val.as_f64()),
            "learning.ltp_ltd_ratio" => self.ltp_ltd_ratio = Some(val.as_f64()),
            "learning.weight_decay" => self.weight_decay = Some(val.as_f64()),
            "learning.trace_increment" => self.trace_increment = Some(val.as_f64()),
            "learning.istdp_rate" => self.istdp_rate = Some(val.as_f64()),

            // Homeostasis
            "homeostasis.target_firing_rate" => self.target_firing_rate = Some(val.as_f64()),
            "homeostasis.homeo_eta" => self.homeo_eta = Some(val.as_f64()),
            "homeostasis.homeo_interval" => self.homeo_interval = Some(val.as_i64()),
            "homeostasis.memory_alpha" => self.memory_alpha = Some(val.as_f64()),
            "homeostasis.meta_threshold" => self.meta_threshold = Some(val.as_f64()),
            "homeostasis.meta_alpha" => self.meta_alpha = Some(val.as_f64()),

            // Energy
            "energy.max_energy" => self.max_energy = Some(val.as_f64()),
            "energy.cost_fire_ratio" => self.cost_fire_ratio = Some(val.as_f64()),
            "energy.recovery_rate" => self.recovery_rate = Some(val.as_f64()),
            "energy.plasticity_cost_factor" => self.plasticity_cost_factor = Some(val.as_f64()),

            // Memory
            "memory.weight_clamp" => self.weight_clamp = Some(val.as_f64()),
            "memory.tag_decay_rate" => self.tag_decay_rate = Some(val.as_f64()),
            "memory.capture_threshold" => self.capture_threshold = Some(val.as_f64()),
            "memory.dopamine_sensitivity" => self.dopamine_sensitivity = Some(val.as_f64()),
            "memory.consolidation_rate" => self.consolidation_rate = Some(val.as_f64()),

            // Network
            "network.inhibitory_ratio" => self.inhibitory_ratio = Some(val.as_f64()),
            "network.initial_threshold" => self.initial_threshold = Some(val.as_f64()),
            "network.initial_exc_weight" => self.initial_exc_weight = Some(val.as_f64()),
            "network.initial_inh_weight" => self.initial_inh_weight = Some(val.as_f64()),
            "network.adaptive_threshold_multiplier" => self.adaptive_threshold_multiplier = Some(val.as_f64()),

            // Competition
            "competition.strength" => self.competition_strength = Some(val.as_f64()),
            "competition.interval" => self.competition_interval = Some(val.as_i64()),

            // Working Memory
            "working_memory.capacity" => self.wm_capacity = Some(val.as_i64()),
            "working_memory.recurrent_strength" => self.wm_recurrent_strength = Some(val.as_f64()),
            "working_memory.decay_rate" => self.wm_decay_rate = Some(val.as_f64()),

            // Sleep
            "sleep.interval" => self.sleep_interval = Some(val.as_i64()),
            "sleep.replay_noise" => self.replay_noise = Some(val.as_f64()),

            // STP
            "stp.use_fraction" => self.stp_use_fraction = Some(val.as_f64()),

            // Predictive
            "predictive.state_learning_rate" => self.state_learning_rate = Some(val.as_f64()),
            "predictive.inference_iterations" => self.inference_iterations = Some(val.as_i64()),

            // Curiosity
            "curiosity.scale" => self.curiosity_scale = Some(val.as_f64()),
            "curiosity.surprise_threshold" => self.surprise_threshold = Some(val.as_f64()),
            "curiosity.habituation_rate" => self.habituation_rate = Some(val.as_f64()),

            _ => {}
        }
    }

    /// Gera o código Rust atualizado para derivation.rs
    pub fn generate_derivation_code(&self) -> String {
        let mut code = String::new();

        code.push_str("//! # Funções de Derivação de Parâmetros (ATUALIZADO COM HYPEROPT)\n");
        code.push_str("//!\n");
        code.push_str("//! Valores otimizados por Bayesian Optimization.\n");
        code.push_str("//! Score: 0.668 (mega_full, 383 trials)\n\n");

        // Generate compute_homeostatic_params
        code.push_str(&self.gen_homeostatic_params());
        code.push_str("\n");

        // Generate compute_stdp_params
        code.push_str(&self.gen_stdp_params());
        code.push_str("\n");

        // Generate compute_memory_params
        code.push_str(&self.gen_memory_params());
        code.push_str("\n");

        // Generate compute_energy_params
        code.push_str(&self.gen_energy_params());
        code.push_str("\n");

        // Generate compute_stp_params
        code.push_str(&self.gen_stp_params());
        code.push_str("\n");

        // Generate compute_competition_params
        code.push_str(&self.gen_competition_params());
        code.push_str("\n");

        // Generate compute_working_memory_params
        code.push_str(&self.gen_working_memory_params());
        code.push_str("\n");

        // Generate compute_curiosity_params
        code.push_str(&self.gen_curiosity_params());
        code.push_str("\n");

        // Generate compute_eligibility_params
        code.push_str(&self.gen_eligibility_params());

        code
    }

    fn gen_homeostatic_params(&self) -> String {
        format!(r#"pub fn compute_homeostatic_params(target_firing_rate: f64) -> HomeostaticParams {{
    // Valores otimizados por hyperopt (mega_full)
    HomeostaticParams {{
        refractory_period: {},
        memory_alpha: {},
        homeo_interval: {},
        homeo_eta: {},
        meta_threshold: {},
        meta_alpha: {},
        fr_alpha: 1.0 / (1.0 / target_firing_rate).clamp(10.0, 100.0),
    }}
}}"#,
            self.refractory_period.unwrap_or(2),
            self.memory_alpha.unwrap_or(0.0457),
            self.homeo_interval.unwrap_or(9),
            self.homeo_eta.unwrap_or(0.2314),
            self.meta_threshold.unwrap_or(0.0798),
            self.meta_alpha.unwrap_or(0.00652),
        )
    }

    fn gen_stdp_params(&self) -> String {
        format!(r#"pub fn compute_stdp_params(_connectivity: ConnectivityType, learning_rate: f64) -> STDPParams {{
    // Valores otimizados por hyperopt (mega_full)
    let window = {};
    let tau_plus = {};
    let tau_minus = {};

    // A_plus e a_minus são modulados pelo learning_rate derivado
    let a_plus = {} * (learning_rate / 0.0256).sqrt();
    let a_minus = {} * (learning_rate / 0.0256).sqrt();

    STDPParams {{ window, tau_plus, tau_minus, a_plus, a_minus }}
}}"#,
            self.stdp_window.unwrap_or(12),
            self.stdp_tau_plus.unwrap_or(44.65),
            self.stdp_tau_minus.unwrap_or(18.11),
            self.stdp_a_plus.unwrap_or(0.0469),
            self.stdp_a_minus.unwrap_or(0.0485),
        )
    }

    fn gen_memory_params(&self) -> String {
        format!(r#"pub fn compute_memory_params(stdp: &STDPParams, learning_rate: f64, _task: &TaskSpec) -> MemoryParams {{
    MemoryParams {{
        weight_decay: {},
        weight_clamp: {},
        tag_decay_rate: {},
        tag_multiplier: 1.0,
        capture_threshold: {},
        dopamine_sensitivity: {},
        consolidation_base_rate: {},
        ltm_protection: LTMProtectionParams {{
            stability_threshold: 0.7,
            ltm_relevance_threshold: 0.5,
            attraction_strength: 0.1,
            small_change_threshold: 0.01,
            stability_increment: 0.05,
            stability_decay_factor: 0.99,
            tag_consumption_factor: 0.1,
        }},
        spike_history_capacity: (stdp.window * 2) as usize,
    }}
}}"#,
            self.weight_decay.unwrap_or(0.00467),
            self.weight_clamp.unwrap_or(2.43),
            self.tag_decay_rate.unwrap_or(0.0196),
            self.capture_threshold.unwrap_or(0.0987),
            self.dopamine_sensitivity.unwrap_or(5.11),
            self.consolidation_rate.unwrap_or(0.00197),
        )
    }

    fn gen_energy_params(&self) -> String {
        format!(r#"pub fn compute_energy_params(initial_threshold: f64, target_firing_rate: f64) -> EnergyParams {{
    let max_energy = {};
    let energy_cost_fire = max_energy * {};
    let energy_cost_maintenance = (energy_cost_fire * 0.01).max(0.01);
    let energy_recovery_rate = {};

    EnergyParams {{
        max_energy,
        energy_cost_fire,
        energy_cost_maintenance,
        energy_recovery_rate,
        plasticity_energy_cost_factor: {},
    }}
}}"#,
            self.max_energy.unwrap_or(52.45),
            self.cost_fire_ratio.unwrap_or(0.0335),
            self.recovery_rate.unwrap_or(6.12),
            self.plasticity_cost_factor.unwrap_or(0.074),
        )
    }

    fn gen_stp_params(&self) -> String {
        format!(r#"pub fn compute_stp_params(task: &TaskSpec) -> STPParams {{
    // Valores base otimizados por hyperopt
    let recovery_tau = {};
    let use_fraction = {};

    STPParams {{
        recovery_tau,
        use_fraction,
        enabled: true,
    }}
}}"#,
            self.stp_recovery_tau.unwrap_or(77.84),
            self.stp_use_fraction.unwrap_or(0.153),
        )
    }

    fn gen_competition_params(&self) -> String {
        format!(r#"pub fn compute_competition_params(task: &TaskSpec, total_neurons: usize) -> CompetitionParams {{
    // Valores otimizados por hyperopt
    let strength = {};
    let interval = {};

    let enabled = !matches!(task.task_type, TaskType::AssociativeMemory {{ .. }});

    CompetitionParams {{ strength, interval, enabled }}
}}"#,
            self.competition_strength.unwrap_or(0.221),
            self.competition_interval.unwrap_or(7),
        )
    }

    fn gen_working_memory_params(&self) -> String {
        format!(r#"pub fn compute_working_memory_params(total_neurons: usize) -> WorkingMemoryParams {{
    // Capacidade otimizada por hyperopt (regra de Miller: 7 ± 2)
    let capacity = {};

    WorkingMemoryParams {{
        capacity,
        recurrent_strength: {},
        decay_rate: {},
        lateral_inhibition: 0.08,
        enabled: true,
    }}
}}"#,
            self.wm_capacity.unwrap_or(5),
            self.wm_recurrent_strength.unwrap_or(0.588),
            self.wm_decay_rate.unwrap_or(0.0108),
        )
    }

    fn gen_curiosity_params(&self) -> String {
        format!(r#"pub fn compute_curiosity_params(task: &TaskSpec) -> CuriosityParams {{
    let (curiosity_scale, enabled) = match &task.task_type {{
        TaskType::ReinforcementLearning {{ reward_density, .. }} => {{
            // Escala base otimizada, ajustada por reward density
            let base_scale = {};
            let scale = match reward_density {{
                RewardDensity::Sparse => base_scale * 2.0,
                RewardDensity::Auto => base_scale * 1.5,
                RewardDensity::Moderate => base_scale,
                RewardDensity::Dense => base_scale * 0.5,
            }};
            (scale, true)
        }}
        _ => ({}, false),
    }};

    CuriosityParams {{
        curiosity_scale,
        surprise_threshold: {},
        habituation_rate: {},
        enabled,
    }}
}}"#,
            self.curiosity_scale.unwrap_or(0.0987),
            self.curiosity_scale.unwrap_or(0.0987),
            self.surprise_threshold.unwrap_or(0.00512),
            self.habituation_rate.unwrap_or(0.937),
        )
    }

    fn gen_eligibility_params(&self) -> String {
        format!(r#"pub fn compute_eligibility_params(task: &TaskSpec, stdp: &STDPParams) -> EligibilityParams {{
    let trace_tau = {};

    let trace_increment = match &task.task_type {{
        TaskType::ReinforcementLearning {{ reward_density, .. }} => {{
            match reward_density {{
                RewardDensity::Sparse => {} * 1.3,
                RewardDensity::Auto => {},
                RewardDensity::Moderate => {} * 0.8,
                RewardDensity::Dense => {} * 0.6,
            }}
        }}
        _ => {},
    }};

    let enabled = matches!(task.task_type, TaskType::ReinforcementLearning {{ .. }});

    EligibilityParams {{ trace_tau, trace_increment, enabled }}
}}"#,
            self.eligibility_trace_tau.unwrap_or(244.24),
            self.trace_increment.unwrap_or(0.159),
            self.trace_increment.unwrap_or(0.159),
            self.trace_increment.unwrap_or(0.159),
            self.trace_increment.unwrap_or(0.159),
            self.trace_increment.unwrap_or(0.159),
        )
    }

    /// Gera relatório de diferenças
    pub fn generate_diff_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           HYPEROPT → AUTOCONFIG PARAMETER MAPPING               ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║                                                                  ║\n");

        // Homeostasis
        report.push_str("║  📊 HOMEOSTASIS                                                  ║\n");
        if let Some(v) = self.refractory_period {
            report.push_str(&format!("║    refractory_period: 5 → {}                                    ║\n", v));
        }
        if let Some(v) = self.memory_alpha {
            report.push_str(&format!("║    memory_alpha: 0.02 → {:.4}                               ║\n", v));
        }
        if let Some(v) = self.homeo_eta {
            report.push_str(&format!("║    homeo_eta: 0.1627 → {:.4}                               ║\n", v));
        }
        if let Some(v) = self.meta_threshold {
            report.push_str(&format!("║    meta_threshold: 0.12 → {:.4}                             ║\n", v));
        }

        report.push_str("║                                                                  ║\n");
        report.push_str("║  📈 STDP                                                         ║\n");
        if let Some(v) = self.stdp_window {
            report.push_str(&format!("║    window: 16 → {}                                            ║\n", v));
        }
        if let Some(v) = self.stdp_tau_plus {
            report.push_str(&format!("║    tau_plus: 12.8 → {:.2}                                    ║\n", v));
        }
        if let Some(v) = self.stdp_tau_minus {
            report.push_str(&format!("║    tau_minus: 4.8 → {:.2}                                    ║\n", v));
        }

        report.push_str("║                                                                  ║\n");
        report.push_str("║  💾 MEMORY                                                       ║\n");
        if let Some(v) = self.weight_decay {
            report.push_str(&format!("║    weight_decay: 0.0001 → {:.6}                          ║\n", v));
        }
        if let Some(v) = self.weight_clamp {
            report.push_str(&format!("║    weight_clamp: 2.5 → {:.2}                                  ║\n", v));
        }

        report.push_str("║                                                                  ║\n");
        report.push_str("║  ⚡ ENERGY                                                       ║\n");
        if let Some(v) = self.max_energy {
            report.push_str(&format!("║    max_energy: 100.0 → {:.2}                                 ║\n", v));
        }
        if let Some(v) = self.recovery_rate {
            report.push_str(&format!("║    recovery_rate: adaptive → {:.2}                           ║\n", v));
        }

        report.push_str("║                                                                  ║\n");
        report.push_str("║  🧠 WORKING MEMORY                                               ║\n");
        if let Some(v) = self.wm_capacity {
            report.push_str(&format!("║    capacity: 7±2 → {}                                         ║\n", v));
        }
        if let Some(v) = self.wm_recurrent_strength {
            report.push_str(&format!("║    recurrent_strength: 0.85 → {:.2}                          ║\n", v));
        }

        report.push_str("║                                                                  ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        report
    }
}

/// Função principal para aplicar os parâmetros
pub fn apply_hyperopt_to_derivation(results_path: &str, derivation_path: &str) -> Result<(), String> {
    println!("📖 Lendo resultados de: {}", results_path);
    let params = OptimizedParams::from_results_file(results_path)?;

    println!("\n{}", params.generate_diff_report());

    println!("\n📝 Gerando código para derivation.rs...\n");
    let code = params.generate_derivation_code();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("CÓDIGO GERADO (copie para {})", derivation_path);
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("{}", code);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_float() {
        let mut params = OptimizedParams::default();
        params.parse_param("homeostasis.homeo_eta", "Float(0.23140821013632423)");
        assert!((params.homeo_eta.unwrap() - 0.2314).abs() < 0.001);
    }

    #[test]
    fn test_parse_int() {
        let mut params = OptimizedParams::default();
        params.parse_param("timing.stdp_window", "Int(12)");
        assert_eq!(params.stdp_window.unwrap(), 12);
    }

    #[test]
    fn test_generate_code() {
        let mut params = OptimizedParams::default();
        params.stdp_window = Some(12);
        params.homeo_eta = Some(0.23);

        let code = params.generate_derivation_code();
        assert!(code.contains("window = 12"));
        assert!(code.contains("homeo_eta: 0.23"));
    }
}
