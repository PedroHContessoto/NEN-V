//! # Funções de Derivação de Parâmetros
//!
//! Contém todas as fórmulas para calcular os 80+ parâmetros da rede
//! a partir da arquitetura e especificação da tarefa.
//!
//! ## Valores Otimizados por Hyperopt
//!
//! Parâmetros atualizados com resultados do mega_full (383 trials, Score: 0.668)

use crate::network::ConnectivityType;
use super::architecture::DerivedArchitecture;
use super::task::{TaskSpec, TaskType, RewardDensity};
use super::params::*;

/// Deriva todos os parâmetros
pub fn derive_all_params(arch: &DerivedArchitecture, task: &TaskSpec) -> NetworkParams {
    let target_firing_rate = compute_target_firing_rate(arch.total_neurons, arch.connectivity);
    let avg_connections = compute_avg_connections(arch.total_neurons, arch.connectivity);
    let learning_rate = compute_learning_rate(avg_connections);
    let (initial_excitatory_weight, initial_inhibitory_weight) =
        compute_initial_weights(arch.inhibitory_ratio, target_firing_rate);

    let energy = compute_energy_params(arch.initial_threshold, target_firing_rate);
    let stdp = compute_stdp_params(arch.connectivity, learning_rate);
    let istdp = compute_istdp_params(learning_rate, target_firing_rate);
    let plasticity = compute_plasticity_params();
    let homeostatic = compute_homeostatic_params(target_firing_rate);
    let input = compute_input_params(target_firing_rate, arch.total_neurons, arch.connectivity);
    let memory = compute_memory_params(&stdp, learning_rate, task);
    let novelty = compute_novelty_params(target_firing_rate, homeostatic.memory_alpha);
    let sleep = compute_sleep_params(memory.consolidation_base_rate, task);

    let rl = match &task.task_type {
        TaskType::ReinforcementLearning { .. } => {
            Some(compute_rl_params(task.num_actuators, target_firing_rate, &stdp))
        }
        _ => None,
    };

    let eligibility = compute_eligibility_params(task, &stdp);
    let stp = compute_stp_params(task);
    let competition = compute_competition_params(task, arch.total_neurons);
    let working_memory = compute_working_memory_params(arch.total_neurons);
    let curiosity = compute_curiosity_params(task);

    NetworkParams {
        target_firing_rate,
        learning_rate,
        avg_connections,
        initial_excitatory_weight,
        initial_inhibitory_weight,
        energy,
        stdp,
        istdp,
        plasticity,
        homeostatic,
        input,
        memory,
        novelty,
        sleep,
        rl,
        eligibility,
        stp,
        competition,
        working_memory,
        curiosity,
    }
}

// ============================================================================
// FUNÇÕES DE CÁLCULO INDIVIDUAIS
// ============================================================================

pub fn compute_target_firing_rate(total_neurons: usize, connectivity: ConnectivityType) -> f64 {
    let effective_fan_in = compute_avg_connections(total_neurons, connectivity);
    let base_fr = 1.0 / (effective_fan_in as f64).sqrt();
    base_fr.clamp(0.03, 0.25)
}

pub fn compute_avg_connections(total_neurons: usize, connectivity: ConnectivityType) -> usize {
    match connectivity {
        ConnectivityType::FullyConnected => total_neurons,
        ConnectivityType::Grid2D => 8,  // Moore neighborhood
        ConnectivityType::Isolated => 1,
    }
}

pub fn compute_learning_rate(avg_connections: usize) -> f64 {
    let base_lr = 0.15 / (avg_connections as f64).sqrt();
    base_lr.clamp(0.002, 0.08)
}

pub fn compute_initial_weights(inhibitory_ratio: f64, _target_firing_rate: f64) -> (f64, f64) {
    // Calibrado para inputs ESPARSOS (típico: 10-30% dos sensores ativos)
    // Com input esparso, cada conexão ativa precisa contribuir mais
    //
    // Cenário MUITO ESPARSO (pior caso):
    //   - 10 sensores, apenas 1 ativo por step
    //   - Threshold ~0.15
    //   - Queremos: 1 input × peso >= threshold
    //   - Logo: peso >= 0.5 para robustez
    //
    // Usamos peso ALTO para garantir disparo mesmo com 1 único input ativo
    let excitatory_base = 0.5;  // MUITO robusto - funciona mesmo com 1 input

    // Inibição proporcional para manter balanço E/I
    let excitatory_ratio = 1.0 - inhibitory_ratio;
    let ei_balance = excitatory_ratio / inhibitory_ratio;  // ~4.0 para 20% inib
    let inhibitory_base = (excitatory_base * ei_balance * 0.8).clamp(0.5, 2.0);

    (excitatory_base, inhibitory_base)
}

pub fn compute_energy_params(_initial_threshold: f64, _target_firing_rate: f64) -> EnergyParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    let max_energy: f64 = 52.45;
    let energy_cost_fire: f64 = max_energy * 0.0335;  // cost_fire_ratio otimizado
    let energy_cost_maintenance: f64 = (energy_cost_fire * 0.01).max(0.01);
    let energy_recovery_rate: f64 = 6.12;

    EnergyParams {
        max_energy,
        energy_cost_fire,
        energy_cost_maintenance,
        energy_recovery_rate,
        plasticity_energy_cost_factor: 0.074,
    }
}

pub fn compute_stdp_params(_connectivity: ConnectivityType, learning_rate: f64) -> STDPParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    let window = 12;
    let tau_plus = 44.65;
    let tau_minus = 18.11;

    // a_plus e a_minus modulados pelo learning_rate derivado
    // Valores base: a_plus=0.0469, a_minus=0.0485, lr_base=0.0256
    let lr_factor = (learning_rate / 0.0256).sqrt();
    let a_plus = 0.0469 * lr_factor;
    let a_minus = 0.0485 * lr_factor;  // LTP/LTD ratio ~0.97 (quase simétrico)

    STDPParams { window, tau_plus, tau_minus, a_plus, a_minus }
}

pub fn compute_istdp_params(learning_rate: f64, target_firing_rate: f64) -> ISTDPParams {
    ISTDPParams {
        learning_rate: learning_rate * 0.1,
        target_rate: target_firing_rate,
    }
}

pub fn compute_plasticity_params() -> PlasticityParams {
    PlasticityParams {
        base_gain: 1.0,
        min_gain: 0.1,
        energy_threshold_for_full_plasticity: 0.5,
    }
}

pub fn compute_homeostatic_params(target_firing_rate: f64) -> HomeostaticParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    HomeostaticParams {
        refractory_period: 2,      // era 5
        memory_alpha: 0.0457,      // era 0.02
        homeo_interval: 9,
        homeo_eta: 0.2314,         // era 0.1627
        meta_threshold: 0.0798,    // era 0.12
        meta_alpha: 0.00652,       // era 0.005
        fr_alpha: 1.0 / (1.0 / target_firing_rate).clamp(10.0, 100.0),
    }
}

pub fn compute_input_params(
    target_firing_rate: f64,
    total_neurons: usize,
    _connectivity: ConnectivityType,
) -> DerivedInputParams {
    DerivedInputParams {
        recommended_input_density: target_firing_rate * 2.0,
        recommended_input_amplitude: 1.0 / (total_neurons as f64).sqrt(),
    }
}

pub fn compute_memory_params(stdp: &STDPParams, _learning_rate: f64, _task: &TaskSpec) -> MemoryParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    MemoryParams {
        weight_decay: 0.00467,     // era 0.0001
        weight_clamp: 2.43,        // era 2.5
        tag_decay_rate: 0.0196,    // era 0.008
        tag_multiplier: 1.0,
        capture_threshold: 0.0987, // era 0.15
        dopamine_sensitivity: 5.11,// era 5.0
        consolidation_base_rate: 0.00197,  // valor fixo otimizado
        ltm_protection: LTMProtectionParams {
            stability_threshold: 0.7,
            ltm_relevance_threshold: 0.5,
            attraction_strength: 0.1,
            small_change_threshold: 0.01,
            stability_increment: 0.05,
            stability_decay_factor: 0.99,
            tag_consumption_factor: 0.1,
        },
        spike_history_capacity: (stdp.window * 2) as usize,
    }
}

pub fn compute_novelty_params(target_firing_rate: f64, memory_alpha: f64) -> NoveltyParams {
    NoveltyParams {
        alert_decay_rate: memory_alpha / 5.0,
        novelty_alert_threshold: target_firing_rate * 0.5,
        alert_sensitivity: 1.0,
        sleep_alert_level: 0.3,
        initial_priority: 1.0,
    }
}

pub fn compute_sleep_params(consolidation_base_rate: f64, task: &TaskSpec) -> SleepParams {
    let reward_density = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            match reward_density {
                RewardDensity::Dense => 0.15,
                RewardDensity::Moderate => 0.05,
                RewardDensity::Sparse | RewardDensity::Auto => 0.01,
            }
        }
        _ => 0.1,
    };

    let sleep_interval = if reward_density < 0.01 { 5000 }
        else if reward_density < 0.1 { 3000 }
        else { 1000 };

    SleepParams {
        sleep_interval,
        sleep_duration: (1.6 / consolidation_base_rate) as usize,
        sleep_replay_noise: 0.05,
        min_selectivity_to_sleep: 0.03,
        sleep_learning_rate_factor: 0.0,
        sleep_metabolic_factor: 1.5,
    }
}

pub fn compute_rl_params(num_actuators: usize, target_firing_rate: f64, stdp: &STDPParams) -> RLParams {
    let action_complexity = (num_actuators as f64).ln();
    let initial_exploration_rate = (0.3 * action_complexity).clamp(0.1, 0.5);

    RLParams {
        initial_exploration_rate,
        exploration_decay_rate: 0.995 - (target_firing_rate * 0.1),
        eligibility_trace_window: stdp.window * 10,
        spike_classification: SpikeClassificationParams {
            feedback_excess_factor: 2.0,
        },
    }
}

pub fn compute_eligibility_params(task: &TaskSpec, _stdp: &STDPParams) -> EligibilityParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    let trace_tau = 244.24;  // era stdp.window * 8.0

    let trace_increment = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            // Valor base: 0.159, ajustado por reward density
            match reward_density {
                RewardDensity::Sparse => 0.159 * 1.3,   // ~0.207
                RewardDensity::Auto => 0.159,
                RewardDensity::Moderate => 0.159 * 0.8, // ~0.127
                RewardDensity::Dense => 0.159 * 0.6,    // ~0.095
            }
        }
        _ => 0.159,
    };

    let enabled = matches!(task.task_type, TaskType::ReinforcementLearning { .. });

    EligibilityParams { trace_tau, trace_increment, enabled }
}

pub fn compute_stp_params(_task: &TaskSpec) -> STPParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    STPParams {
        recovery_tau: 77.84,   // era 120-200 dependente de task
        use_fraction: 0.153,   // era 0.12
        enabled: true,
    }
}

pub fn compute_competition_params(task: &TaskSpec, _total_neurons: usize) -> CompetitionParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    let strength = 0.221;  // era 0.15-0.35 dependente de neurons
    let interval = 7;      // era 5-15 dependente de task

    let enabled = !matches!(task.task_type, TaskType::AssociativeMemory { .. });

    CompetitionParams { strength, interval, enabled }
}

pub fn compute_working_memory_params(_total_neurons: usize) -> WorkingMemoryParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    WorkingMemoryParams {
        capacity: 5,                    // era 5-9 dependente de neurons
        recurrent_strength: 0.588,      // era 0.85
        decay_rate: 0.0108,             // era 0.02
        lateral_inhibition: 0.08,
        enabled: true,
    }
}

pub fn compute_curiosity_params(task: &TaskSpec) -> CuriosityParams {
    // Valores otimizados por hyperopt (mega_full, Score: 0.668)
    let (curiosity_scale, enabled) = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            // Escala base: 0.0987, ajustada por reward density
            let base_scale = 0.0987;
            let scale = match reward_density {
                RewardDensity::Sparse => base_scale * 2.0,   // ~0.197
                RewardDensity::Auto => base_scale * 1.5,     // ~0.148
                RewardDensity::Moderate => base_scale,       // ~0.099
                RewardDensity::Dense => base_scale * 0.5,    // ~0.049
            };
            (scale, true)
        }
        _ => (0.0987, false),
    };

    CuriosityParams {
        curiosity_scale,
        surprise_threshold: 0.00512,  // era 0.01
        habituation_rate: 0.937,      // era 0.995
        enabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_firing_rate_scaling() {
        // Redes maiores devem ter menor firing rate
        let small_fr = compute_target_firing_rate(10, ConnectivityType::FullyConnected);
        let large_fr = compute_target_firing_rate(100, ConnectivityType::FullyConnected);

        assert!(small_fr > large_fr);
    }

    #[test]
    fn test_stdp_params() {
        let stdp = compute_stdp_params(ConnectivityType::FullyConnected, 0.01);

        // tau_plus > tau_minus (preserva causalidade)
        assert!(stdp.tau_plus > stdp.tau_minus);

        // a_plus e a_minus devem ser positivos
        assert!(stdp.a_plus > 0.0);
        assert!(stdp.a_minus > 0.0);

        // Após hyperopt: LTP/LTD ratio ~0.97 (quase simétrico)
        let ratio = stdp.a_plus / stdp.a_minus;
        assert!(ratio > 0.8 && ratio < 1.2);
    }

    #[test]
    fn test_energy_balance() {
        let energy = compute_energy_params(0.3, 0.15);

        // Recuperação deve ser maior que custo médio
        let avg_cost = energy.energy_cost_fire * 0.15;
        let avg_gain = energy.energy_recovery_rate * 0.85;

        assert!(avg_gain > avg_cost);
    }
}
