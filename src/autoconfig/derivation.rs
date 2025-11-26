//! # Funções de Derivação de Parâmetros
//!
//! Contém todas as fórmulas para calcular os 80+ parâmetros da rede
//! a partir da arquitetura e especificação da tarefa.

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

pub fn compute_initial_weights(inhibitory_ratio: f64, target_firing_rate: f64) -> (f64, f64) {
    let excitatory_base = 0.06;
    let excitatory_ratio = 1.0 - inhibitory_ratio;
    let expected_excitation = excitatory_ratio * target_firing_rate;
    let inhibitory_base = (expected_excitation / inhibitory_ratio).clamp(0.15, 1.2);
    (excitatory_base, inhibitory_base)
}

pub fn compute_energy_params(initial_threshold: f64, target_firing_rate: f64) -> EnergyParams {
    let max_energy = 100.0;
    let energy_cost_fire = (initial_threshold * max_energy * 0.1).clamp(max_energy * 0.01, max_energy * 0.15);
    let energy_cost_maintenance = (energy_cost_fire * 0.01).max(0.01);
    let equilibrium_recovery = energy_cost_fire * target_firing_rate / (1.0 - target_firing_rate);
    let energy_recovery_rate = (equilibrium_recovery * 1.3).clamp(1.0, 25.0);

    EnergyParams {
        max_energy,
        energy_cost_fire,
        energy_cost_maintenance,
        energy_recovery_rate,
        plasticity_energy_cost_factor: 0.08,
    }
}

pub fn compute_stdp_params(_connectivity: ConnectivityType, learning_rate: f64) -> STDPParams {
    let window = 16;
    // ASSIMÉTRICO: tau_plus > tau_minus para favorecer padrões causais
    let tau_plus = (window as f64) * 0.8;
    let tau_minus = (window as f64) * 0.3;
    let a_plus = learning_rate * 2.5;
    let a_minus = a_plus / 2.5;  // LTP/LTD ratio de 2.5:1

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
    HomeostaticParams {
        refractory_period: 5,
        memory_alpha: 0.02,
        homeo_interval: 9,
        homeo_eta: 0.1627,  // Otimizado por grid-search
        meta_threshold: 0.12,
        meta_alpha: 0.005,
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

pub fn compute_memory_params(stdp: &STDPParams, learning_rate: f64, _task: &TaskSpec) -> MemoryParams {
    MemoryParams {
        weight_decay: 0.0001,
        weight_clamp: 2.5,
        tag_decay_rate: 0.008,
        tag_multiplier: 1.0,
        capture_threshold: 0.15,
        dopamine_sensitivity: 5.0,
        consolidation_base_rate: learning_rate * 0.1,
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

pub fn compute_eligibility_params(task: &TaskSpec, stdp: &STDPParams) -> EligibilityParams {
    let trace_tau = (stdp.window as f64) * 8.0;

    let trace_increment = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            match reward_density {
                RewardDensity::Sparse => 0.20,   // Mais forte para reward esparso
                RewardDensity::Auto => 0.15,
                RewardDensity::Moderate => 0.12,
                RewardDensity::Dense => 0.10,
            }
        }
        _ => 0.12,
    };

    let enabled = matches!(task.task_type, TaskType::ReinforcementLearning { .. });

    EligibilityParams { trace_tau, trace_increment, enabled }
}

pub fn compute_stp_params(task: &TaskSpec) -> STPParams {
    let recovery_tau = match &task.task_type {
        TaskType::ReinforcementLearning { temporal_horizon, .. } => {
            match temporal_horizon {
                Some(h) if *h > 200 => 200.0,
                Some(h) if *h > 100 => 150.0,
                _ => 120.0,
            }
        }
        _ => 150.0,
    };

    STPParams {
        recovery_tau,
        use_fraction: 0.12,
        enabled: true,
    }
}

pub fn compute_competition_params(task: &TaskSpec, total_neurons: usize) -> CompetitionParams {
    let strength = if total_neurons > 50 { 0.35 }
        else if total_neurons > 20 { 0.25 }
        else { 0.15 };

    let interval = match &task.task_type {
        TaskType::ReinforcementLearning { .. } => 10,
        TaskType::SupervisedClassification { .. } => 5,
        TaskType::AssociativeMemory { .. } => 15,
    };

    let enabled = !matches!(task.task_type, TaskType::AssociativeMemory { .. });

    CompetitionParams { strength, interval, enabled }
}

pub fn compute_working_memory_params(total_neurons: usize) -> WorkingMemoryParams {
    // Regra de Miller: 7 ± 2
    let capacity = if total_neurons > 50 { 9 }
        else if total_neurons > 20 { 7 }
        else { 5 };

    WorkingMemoryParams {
        capacity,
        recurrent_strength: 0.85,
        decay_rate: 0.02,
        lateral_inhibition: 0.08,
        enabled: true,
    }
}

pub fn compute_curiosity_params(task: &TaskSpec) -> CuriosityParams {
    let (curiosity_scale, enabled) = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            let scale = match reward_density {
                RewardDensity::Sparse => 0.2,   // Mais curiosidade para reward esparso
                RewardDensity::Auto => 0.15,
                RewardDensity::Moderate => 0.1,
                RewardDensity::Dense => 0.05,
            };
            (scale, true)
        }
        _ => (0.1, false),
    };

    CuriosityParams {
        curiosity_scale,
        surprise_threshold: 0.01,
        habituation_rate: 0.995,
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
    fn test_asymmetric_stdp() {
        let stdp = compute_stdp_params(ConnectivityType::FullyConnected, 0.01);

        // tau_plus > tau_minus
        assert!(stdp.tau_plus > stdp.tau_minus);

        // a_plus > a_minus
        assert!(stdp.a_plus > stdp.a_minus);
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
