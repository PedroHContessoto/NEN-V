//! Cálculo automático de todos os parâmetros da rede
//!
//! NÍVEIS 1-5: Deriva 60+ parâmetros baseado na arquitetura

use super::*;

impl NetworkParams {
    /// Calcula todos os parâmetros a partir da arquitetura derivada
    pub fn from_architecture(arch: &DerivedArchitecture, task: &TaskSpec) -> Self {
        // NÍVEL 1: Estruturais
        let target_firing_rate = compute_target_firing_rate(
            arch.total_neurons,
            arch.connectivity,
        );

        let avg_connections = compute_avg_connections(
            arch.total_neurons,
            arch.connectivity,
        );

        let learning_rate = compute_learning_rate(avg_connections);

        let (initial_excitatory_weight, initial_inhibitory_weight) =
            compute_initial_weights(arch.inhibitory_ratio, target_firing_rate);

        // NÍVEL 2: Metabólicos
        let energy = compute_energy_params(
            arch.initial_threshold,
            target_firing_rate,
        );

        // NÍVEL 3: Plasticidade
        let stdp = compute_stdp_params(arch.connectivity, learning_rate);
        let istdp = compute_istdp_params(learning_rate, target_firing_rate);
        let plasticity = compute_plasticity_params();

        // NÍVEL 4: Homeostase
        let homeostatic = compute_homeostatic_params(
            target_firing_rate,
            arch.connectivity,
        );

        // NÍVEL 4.5: Input (derivado do target FR e conectividade)
        let input = compute_input_params(
            target_firing_rate,
            arch.total_neurons,
            arch.connectivity,
        );

        // NÍVEL 5: Memória
        let memory = compute_memory_params(
            &stdp,
            learning_rate,
            task,
        );

        // Novidade/Alerta
        let novelty = compute_novelty_params(
            target_firing_rate,
            homeostatic.memory_alpha,
        );

        // Sono/Consolidação
        let sleep = compute_sleep_params(
            memory.consolidation_base_rate,
            task,
        );

        // RL-específico
        let rl = match &task.task_type {
            TaskType::ReinforcementLearning { .. } => {
                Some(compute_rl_params(
                    task.num_actuators,
                    target_firing_rate,
                    &stdp,
                    arch.initial_threshold,
                ))
            }
            _ => None,
        };

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
        }
    }
}

// ============================================================================
// NÍVEL 1: PARÂMETROS ESTRUTURAIS
// ============================================================================

/// Calcula target firing rate baseado no fan-in EFETIVO (correção do Furo 1)
///
/// # Regra
/// FR ∝ 1/√(fan_in_efetivo)
///
/// Grid2D: fan_in sempre 8 → FR não muda com tamanho da rede ✓
/// FullyConnected: fan_in = N → FR decresce com N ✓
pub fn compute_target_firing_rate(
    total_neurons: usize,
    connectivity: ConnectivityType,
) -> f64 {
    let effective_fan_in = compute_avg_connections(total_neurons, connectivity);

    let base_fr = 1.0 / (effective_fan_in as f64).sqrt();

    // Clamp para valores biologicamente razoáveis
    base_fr.clamp(0.03, 0.25)
}

/// Calcula número médio de conexões (fan-in efetivo)
pub fn compute_avg_connections(
    total_neurons: usize,
    connectivity: ConnectivityType,
) -> usize {
    match connectivity {
        ConnectivityType::FullyConnected => total_neurons,
        ConnectivityType::Grid2D => 8,  // Moore neighborhood (fixo)
        ConnectivityType::Isolated => 1,
    }
}

/// Calcula learning rate baseado no fan-in
///
/// # Regra
/// LR ∝ 1/√(avg_connections)
///
/// Evita saturação quando muitos inputs competem
pub fn compute_learning_rate(avg_connections: usize) -> f64 {
    let base_lr = 0.1 / (avg_connections as f64).sqrt();

    base_lr.clamp(0.001, 0.05)
}

/// Calcula pesos iniciais (excitatory, inhibitory)
pub fn compute_initial_weights(
    inhibitory_ratio: f64,
    target_firing_rate: f64,
) -> (f64, f64) {
    // Excitatory: pequeno e uniforme (tabula rasa)
    let excitatory_base = 0.05;

    // Inhibitory: balanceia excitação esperada
    let excitatory_ratio = 1.0 - inhibitory_ratio;
    let expected_excitation = excitatory_ratio * target_firing_rate;
    let inhibitory_base = expected_excitation / inhibitory_ratio;

    let inhibitory_base = inhibitory_base.clamp(0.1, 1.0);

    (excitatory_base, inhibitory_base)
}

// ============================================================================
// NÍVEL 2: PARÂMETROS METABÓLICOS
// ============================================================================

/// Calcula parâmetros de energia
pub fn compute_energy_params(
    initial_threshold: f64,
    target_firing_rate: f64,
) -> EnergyParams {
    let max_energy = 100.0;

    // Custo proporcional ao threshold
    let energy_cost_fire = (initial_threshold * max_energy * 0.1)
        .clamp(max_energy * 0.01, max_energy * 0.15);

    // Manutenção = 1% do custo de disparo
    let energy_cost_maintenance = (energy_cost_fire * 0.01).max(0.01);

    // Recovery: balanceia gasto com margem de 20%
    let equilibrium_recovery = energy_cost_fire * target_firing_rate
                               / (1.0 - target_firing_rate);
    let safety_margin = 1.2;
    let energy_recovery_rate = (equilibrium_recovery * safety_margin).clamp(1.0, 20.0);

    // Custo de plasticidade
    let plasticity_energy_cost_factor = 0.1;

    EnergyParams {
        max_energy,
        energy_cost_fire,
        energy_cost_maintenance,
        energy_recovery_rate,
        plasticity_energy_cost_factor,
    }
}

// ============================================================================
// NÍVEL 3: PARÂMETROS DE PLASTICIDADE
// ============================================================================

/// Calcula parâmetros STDP
pub fn compute_stdp_params(
    connectivity: ConnectivityType,
    learning_rate: f64,
) -> STDPParams {
    // Refractory period REDUZIDO (5 → 2) para permitir FR alvo ~0.22
    // Ainda biologicamente plausível (~2ms em escala real)
    let refractory_period = 2i64;

    // Window ampliada: 8× refractory para capturar padrões com atraso
    let window = refractory_period * 8;

    // Janela assimétrica: LTP com tau maior que LTD para favorecer causalidade tardia
    let tau_plus = (window as f64) * 0.8; // 80% da janela
    let tau_minus = (window as f64) * 0.3; // 30% da janela

    // Amplitudes: STDP 2× mais forte que Hebbian
    let stdp_strength = 2.0;
    let a_plus = learning_rate * stdp_strength;

    // Ratio LTP:LTD = 2:1
    let ltp_ltd_ratio = 2.0;
    let a_minus = a_plus / ltp_ltd_ratio;

    STDPParams {
        window,
        tau_plus,
        tau_minus,
        a_plus,
        a_minus,
    }
}

/// Calcula parâmetros iSTDP
pub fn compute_istdp_params(
    learning_rate: f64,
    target_firing_rate: f64,
) -> iSTDPParams {
    // iSTDP 10× mais lento que STDP excitatório
    let istdp_learning_rate = learning_rate * 0.1;

    // Target DEVE ser igual ao target_firing_rate (correção do Furo original)
    let target_rate = target_firing_rate;

    iSTDPParams {
        learning_rate: istdp_learning_rate,
        target_rate,
    }
}

/// Calcula parâmetros de plasticidade geral
pub fn compute_plasticity_params() -> PlasticityParams {
    PlasticityParams {
        base_gain: 1.0,
        min_gain: 0.1,  // 10% sob energia crítica
        energy_threshold_for_full_plasticity: 0.5,  // 50%
    }
}

// ============================================================================
// NÍVEL 4: PARÂMETROS HOMEOSTÁTICOS
// ============================================================================

/// Calcula parâmetros homeostáticos COM DERIVAÇÃO CIENTÍFICA
///
/// **MUDANÇA FUNDAMENTAL:** Parâmetros agora são derivados do target_firing_rate,
/// não mais fixos. Isso permite verdadeira auto-regulação emergente.
pub fn compute_homeostatic_params(
    target_firing_rate: f64,
    _connectivity: ConnectivityType,
) -> HomeostaticParams {
    // === DERIVAÇÃO CIENTÍFICA BASEADA EM TARGET FR ===

    // 1. REFRACTORY PERIOD: Derivado do target FR
    // refractory_max = 1 / target_fr (hard cap físico)
    // Usamos ~40% desse limite para dar margem de regulação
    let refractory_max = 1.0 / target_firing_rate;
    let refractory_period = (0.4 * refractory_max).ceil() as i64;
    // Ex: target_fr = 0.22 → refractory = (0.4/0.22) ≈ 1.8 ≈ 2
    //     target_fr = 0.15 → refractory = (0.4/0.15) ≈ 2.7 ≈ 3

    // 2. MEMORY ALPHA: Independente de FR (janela ~50 steps)
    let memory_alpha = 0.02;

    // 3. FR ALPHA: Proporcional ao target (FR alto → tracking mais rápido)
    let base_fr_alpha = 0.01;
    let fr_alpha = base_fr_alpha * (target_firing_rate / 0.15).sqrt();
    // Ex: target=0.22 → fr_alpha ≈ 0.012
    //     target=0.10 → fr_alpha ≈ 0.008

    // 4. HOMEOSTASE INTERVAL: Inversamente proporcional ao target
    // FR alto → ajustes mais frequentes
    let base_interval = 150i64;
    let mut homeo_interval =
        (base_interval as f64 / (target_firing_rate / 0.15)).ceil();
    // Ajuste do grid-search: interval_multiplier = 0.858 (W65T35_eta3.3x_int0.858x)
    homeo_interval *= 0.858;
    let homeo_interval = homeo_interval.clamp(2.0, 300.0) as i64;
    // Ex: target=0.22 → interval ≈ 118 steps
    //     target=0.10 → interval ≈ 229 steps

    // 5. HOMEOSTASE ETA: Proporcional ao target
    // FR alto → correções mais agressivas
    let base_eta = 0.03;
    // Ajuste do grid-search: eta_multiplier = 3.253
    let homeo_eta = base_eta * (target_firing_rate / 0.15).sqrt() * 3.253;
    // Ex: target=0.22 → eta ≈ 0.120
    //     target=0.10 → eta ≈ 0.079

    // 6. BCM meta-threshold: quadrado do target FR (inalterado)
    let meta_threshold = target_firing_rate * target_firing_rate;

    // 7. BCM alpha: Proporcional ao FR alpha
    let meta_alpha = fr_alpha * 0.5;

    HomeostaticParams {
        refractory_period,
        memory_alpha,
        homeo_interval,
        homeo_eta,
        meta_threshold,
        meta_alpha,
        fr_alpha,
    }
}

// ============================================================================
// NÍVEL 4.5: PARÂMETROS DE ENTRADA (INPUT)
// ============================================================================

/// Calcula parâmetros de entrada (input strength) DERIVADOS do target FR
///
/// ## Lógica Científica:
///
/// 1. **Input Density**: Proporcional ao target FR, inversamente à conectividade
///    - Rede densa precisa menos input externo (self-sustaining)
///    - FR alto requer mais drive externo
///
/// 2. **Input Amplitude**: Compensa densidade
///    - Densidade baixa → amplitude alta (poucos neurônios, forte estímulo)
///    - Densidade alta → amplitude moderada (muitos neurônios, estímulo distribuído)
///
/// ## Exemplo:
/// ```
/// target_fr = 0.22, connectivity = 0.08 → density ≈ 0.15, amplitude = 1.5
/// target_fr = 0.10, connectivity = 0.15 → density ≈ 0.08, amplitude = 2.0
/// ```
pub fn compute_input_params(
    target_firing_rate: f64,
    total_neurons: usize,
    connectivity_type: ConnectivityType,
) -> DerivedInputParams {
    // Calcula conectividade como fração (avg_connections / total_neurons)
    let avg_connections = compute_avg_connections(total_neurons, connectivity_type);
    let connectivity = avg_connections as f64 / total_neurons as f64;
    // 1. Densidade base: normalizada por FR de referência (0.15)
    let base_density = target_firing_rate / 0.15;

    // 2. Fator de conectividade: quanto mais densa, menos input externo precisa
    let connectivity_factor = 1.0 / connectivity.max(0.05);

    // 3. Densidade final: produto dos fatores, com baseline de 15%
    let input_density = (base_density * connectivity_factor * 0.15)
        .clamp(0.05, 0.30); // Min 5%, Max 30%

    // 4. Amplitude compensa densidade
    // Lógica: poucos neurônios → estímulo forte; muitos neurônios → distribuído
    let input_amplitude = if input_density < 0.1 {
        2.0  // Densidade baixa: amplitude alta
    } else if input_density < 0.2 {
        1.5  // Densidade moderada: amplitude moderada
    } else {
        1.2  // Densidade alta: amplitude mais suave
    };

    DerivedInputParams {
        recommended_input_density: input_density,
        recommended_input_amplitude: input_amplitude,
    }
}

// ============================================================================
// NÍVEL 5: PARÂMETROS DE MEMÓRIA
// ============================================================================

/// Calcula parâmetros de memória e consolidação
pub fn compute_memory_params(
    stdp: &STDPParams,
    learning_rate: f64,
    task: &TaskSpec,
) -> MemoryParams {
    // Weight decay: meia-vida de 5000 steps
    let half_life_steps = 5000.0;
    let weight_decay = 1.0 - 0.5_f64.powf(1.0 / half_life_steps);

    // Weight clamp
    let weight_clamp = 2.5;

    // Tag decay: 50× mais rápido que weight decay
    let tag_decay_rate = weight_decay * 50.0;

    // Tag multiplier (do código atual)
    let tag_multiplier = 10.0;

    // Capture threshold: requer ~3-5 eventos STDP
    let avg_stdp_amplitude = (stdp.a_plus + stdp.a_minus) / 2.0;
    let events_required = 4.0;
    let capture_threshold = avg_stdp_amplitude * events_required * tag_multiplier * 0.5;

    // Dopamine sensitivity
    let dopamine_sensitivity = 5.0;

    // Consolidation rate: converg 80% em 500 steps de sono
    let sleep_duration = 500.0;
    let time_constants = 2.0;
    let consolidation_base_rate = time_constants / sleep_duration;

    // LTM protection
    let ltm_protection = compute_ltm_protection_params(consolidation_base_rate);

    // Spike history capacity
    let spike_history_capacity = compute_spike_history_capacity(stdp.window);

    MemoryParams {
        weight_decay,
        weight_clamp,
        tag_decay_rate,
        tag_multiplier,
        capture_threshold,
        dopamine_sensitivity,
        consolidation_base_rate,
        ltm_protection,
        spike_history_capacity,
    }
}

/// Calcula parâmetros de proteção de LTM
fn compute_ltm_protection_params(consolidation_base_rate: f64) -> LTMProtectionParams {
    LTMProtectionParams {
        stability_threshold: 0.8,
        ltm_relevance_threshold: 0.1,
        attraction_strength: 0.5,
        small_change_threshold: 1e-4,
        stability_increment: consolidation_base_rate * 2.0,
        stability_decay_factor: 1.0 - consolidation_base_rate,
        tag_consumption_factor: 0.5,
    }
}

/// Calcula capacidade do histórico de spikes
fn compute_spike_history_capacity(stdp_window: i64) -> usize {
    let capacity = (stdp_window * 2) as usize;
    capacity.max(10)
}

// ============================================================================
// NOVIDADE / ALERTA
// ============================================================================

/// Calcula parâmetros de novidade e alerta
pub fn compute_novelty_params(
    target_firing_rate: f64,
    memory_alpha: f64,
) -> NoveltyParams {
    // Alert decay: 5× mais lento que memória
    let alert_decay_rate = memory_alpha / 5.0;

    // Threshold de novidade: 50% do FR alvo
    let novelty_alert_threshold = target_firing_rate * 0.5;

    // Sensibilidade linear
    let alert_sensitivity = 1.0;

    // Alert durante sono: 30%
    let sleep_alert_level = 0.3;

    // Priority inicial neutro
    let initial_priority = 1.0;

    NoveltyParams {
        alert_decay_rate,
        novelty_alert_threshold,
        alert_sensitivity,
        sleep_alert_level,
        initial_priority,
    }
}

// ============================================================================
// SONO / CONSOLIDAÇÃO
// ============================================================================

/// Calcula parâmetros de sono
pub fn compute_sleep_params(
    consolidation_base_rate: f64,
    task: &TaskSpec,
) -> SleepParams {
    // Plasticidade durante sono: 0% (só consolidação)
    let sleep_learning_rate_factor = 0.0;

    // Ajuste metabólico: 150% recovery, 50% custo
    let sleep_metabolic_factor = 1.5;

    // Sleep interval baseado em reward density estimada
    let reward_density = match &task.task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            match reward_density {
                RewardDensity::Dense => 0.15,
                RewardDensity::Moderate => 0.05,
                RewardDensity::Sparse | RewardDensity::Auto => 0.01,
            }
        }
        _ => 0.1,  // Default
    };

    let sleep_interval = if reward_density < 0.01 {
        5000
    } else if reward_density < 0.1 {
        3000
    } else {
        1000
    } as u64;

    // Sleep duration: tempo para consolidar ~80%
    let convergence_time = 1.6 / consolidation_base_rate;
    let sleep_duration = convergence_time as usize;

    // Replay noise: 5% (constante biológica)
    let sleep_replay_noise = 0.05;

    // Min selectivity: 3%
    let min_selectivity_to_sleep = 0.03;

    SleepParams {
        sleep_interval,
        sleep_duration,
        sleep_replay_noise,
        min_selectivity_to_sleep,
        sleep_learning_rate_factor,
        sleep_metabolic_factor,
    }
}

// ============================================================================
// RL-ESPECÍFICO
// ============================================================================

/// Calcula parâmetros de Reinforcement Learning
pub fn compute_rl_params(
    num_actuators: usize,
    target_firing_rate: f64,
    stdp: &STDPParams,
    initial_threshold: f64,
) -> RLParams {
    // Exploration inicial: 30% (fase de descoberta)
    let base_exploration = 0.3;
    let action_complexity = (num_actuators as f64).ln();
    let initial_exploration_rate = (base_exploration * action_complexity).clamp(0.1, 0.5);

    // Exploration decay: mais lento se FR baixo
    let exploration_decay_rate = 0.995 - (target_firing_rate * 0.1);

    // Eligibility trace: 7× janela STDP
    let eligibility_trace_window = stdp.window * 7;

    // Spike classification
    let spike_classification = SpikeClassificationParams {
        feedback_excess_factor: 2.0,
    };

    RLParams {
        initial_exploration_rate,
        exploration_decay_rate,
        eligibility_trace_window,
        spike_classification,
    }
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_firing_rate_grid2d_consistent() {
        // Grid2D: FR deve ser igual para N=100 e N=10000 (fan_in=8 sempre)
        let fr_100 = compute_target_firing_rate(100, ConnectivityType::Grid2D);
        let fr_10000 = compute_target_firing_rate(10000, ConnectivityType::Grid2D);

        assert_eq!(fr_100, fr_10000);
        assert_eq!(fr_100, 0.25);  // 1/sqrt(8) ≈ 0.35 → clamped=0.25
    }

    #[test]
    fn test_target_firing_rate_fully_connected_scales() {
        // FullyConnected: FR deve decrescer com N
        let fr_20 = compute_target_firing_rate(20, ConnectivityType::FullyConnected);
        let fr_100 = compute_target_firing_rate(100, ConnectivityType::FullyConnected);

        assert!(fr_20 > fr_100);
    }

    #[test]
    fn test_energy_balance_positive() {
        // Balanço energético deve ser positivo
        let threshold = 0.3;
        let target_fr = 0.15;

        let energy = compute_energy_params(threshold, target_fr);

        let avg_cost = energy.energy_cost_fire * target_fr;
        let avg_gain = energy.energy_recovery_rate * (1.0 - target_fr);
        let balance = avg_gain - avg_cost;

        assert!(balance > 0.0, "Balanço energético negativo: {}", balance);
    }

    #[test]
    fn test_istdp_aligned_with_target_fr() {
        // iSTDP target DEVE ser igual ao target_firing_rate
        let target_fr = 0.15;
        let lr = 0.01;

        let istdp = compute_istdp_params(lr, target_fr);

        assert_eq!(istdp.target_rate, target_fr);
    }
}
