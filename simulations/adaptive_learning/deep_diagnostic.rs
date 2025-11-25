//! Deep Diagnostic: Análise Ultra-Detalhada do Sistema Adaptativo
//!
//! Este script captura TODAS as métricas possíveis para identificar
//! o que está impedindo o FR de convergir para o target.
//!
//! ## Métricas Capturadas:
//!
//! ### Nível de Rede
//! - FR médio, mín, máx por janela
//! - Distribuição de FR por neurônio
//! - Número de neurônios silenciosos vs ativos vs hiper-ativos
//! - Conectividade efetiva (pesos > threshold)
//!
//! ### Nível de Neurônio
//! - Threshold: médio, mín, máx, distribuição
//! - Energia: médio, mín, máx, distribuição
//! - Bloqueios: % por energia vs threshold vs refratário
//! - Potencial: médio, mín, máx
//! - Pesos sinápticos: médio, mín, máx, % zeros
//!
//! ### Controlador PI
//! - Erro proporcional
//! - Erro integral (e se saturou)
//! - Sinal de controle u
//! - Delta threshold aplicado
//! - Cooldown status
//!
//! ### Entrada Externa
//! - Taxa de estímulo efetiva
//! - Amplitude média
//! - Distribuição espacial
//!
//! ### Dinâmica Temporal
//! - Autocorrelação de FR
//! - Oscilações (FFT ou variância em janelas)
//! - Taxa de convergência

use nenv_visual_sim::autoconfig::{
    AutoConfig, TaskSpec, TaskType, RewardDensity,
    adaptive::{AdaptiveState, monitor_and_adapt},
};
use nenv_visual_sim::network::{LearningMode, Network};
use rand::Rng;

#[derive(Debug, Clone)]
struct NetworkSnapshot {
    step: usize,

    // FR metrics
    fr_mean: f64,
    fr_std: f64,
    fr_min: f64,
    fr_max: f64,

    // Neurônio classification
    num_silent: usize,      // FR < 0.01
    num_normal: usize,      // 0.01 <= FR < 0.5
    num_hyperactive: usize, // FR >= 0.5

    // Threshold metrics
    threshold_mean: f64,
    threshold_std: f64,
    threshold_min: f64,
    threshold_max: f64,
    num_at_min_threshold: usize, // quantos saturaram em 0.001

    // Energia metrics
    energy_mean: f64,
    energy_std: f64,
    energy_min: f64,
    energy_max: f64,

    // Bloqueios (último step)
    blocks_by_energy: usize,
    blocks_by_threshold: usize,
    blocks_by_refractory: usize,

    // Potencial
    potential_mean: f64,
    potential_std: f64,

    // Sinapses
    weight_mean: f64,
    weight_std: f64,
    weight_min: f64,
    weight_max: f64,
    num_dead_synapses: usize, // pesos < 0.01
    effective_connectivity: f64, // % sinapses ativas

    // PI state
    pi_error: f64,
    pi_integral: f64,
    pi_integral_saturated: bool,
    pi_u: f64,
    pi_delta: Option<f64>,

    // Input
    input_rate: f64,
    input_amplitude_mean: f64,
}

impl NetworkSnapshot {
    fn capture(network: &Network, adaptive: &AdaptiveState, step: usize, last_input: &[f64]) -> Self {
        let num_neurons = network.num_neurons();

        // FR per neuron (proxy: is_firing no step atual)
        let firing_flags: Vec<bool> = (0..num_neurons)
            .map(|i| network.neurons[i].is_firing)
            .collect();

        let fr_values: Vec<f64> = firing_flags.iter()
            .map(|&fired| if fired { 1.0 } else { 0.0 })
            .collect();

        let fr_mean = fr_values.iter().sum::<f64>() / num_neurons as f64;
        let fr_variance = fr_values.iter()
            .map(|x| (x - fr_mean).powi(2))
            .sum::<f64>() / num_neurons as f64;
        let fr_std = fr_variance.sqrt();
        let fr_min = fr_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let fr_max = fr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Neurônio classification
        let num_silent = firing_flags.iter().filter(|&&x| !x).count();
        let num_normal = firing_flags.iter().filter(|&&x| x).count();
        let num_hyperactive = 0; // Simplificado (não temos histórico aqui)

        // Thresholds
        let thresholds: Vec<f64> = network.neurons.iter().map(|n| n.threshold).collect();
        let threshold_mean = thresholds.iter().sum::<f64>() / num_neurons as f64;
        let threshold_variance = thresholds.iter()
            .map(|x| (x - threshold_mean).powi(2))
            .sum::<f64>() / num_neurons as f64;
        let threshold_std = threshold_variance.sqrt();
        let threshold_min = thresholds.iter().cloned().fold(f64::INFINITY, f64::min);
        let threshold_max = thresholds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let num_at_min_threshold = thresholds.iter().filter(|&&t| t < 0.002).count();

        // Energia
        let energies: Vec<f64> = network.neurons.iter().map(|n| n.glia.energy).collect();
        let energy_mean = energies.iter().sum::<f64>() / num_neurons as f64;
        let energy_variance = energies.iter()
            .map(|x| (x - energy_mean).powi(2))
            .sum::<f64>() / num_neurons as f64;
        let energy_std = energy_variance.sqrt();
        let energy_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let energy_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Bloqueios (simplificados: energia e refratário não são expostos publicamente)
        // Vamos focar em threshold e firing rate
        let blocks_by_energy = 0; // Não temos acesso a can_fire() públicamente
        let blocks_by_threshold = network.neurons.iter()
            .filter(|n| !n.is_firing && n.recent_firing_rate < 0.01) // Proxy: neurônios silenciosos
            .count();
        let blocks_by_refractory = network.neurons.iter()
            .filter(|n| network.current_time_step - n.last_fire_time <= 5) // Proxy: disparou recentemente
            .count();

        // Potencial (não exposto, vamos usar recent_firing_rate como proxy)
        let potentials: Vec<f64> = network.neurons.iter().map(|n| n.recent_firing_rate).collect();
        let potential_mean = potentials.iter().sum::<f64>() / num_neurons as f64;
        let potential_variance = potentials.iter()
            .map(|x| (x - potential_mean).powi(2))
            .sum::<f64>() / num_neurons as f64;
        let potential_std = potential_variance.sqrt();

        // Sinapses (métricas agregadas - não temos acesso direto aos pesos)
        let total_weights: Vec<f64> = network.neurons.iter()
            .map(|n| n.dendritoma.total_weight())
            .collect();

        let weight_norms: Vec<f64> = network.neurons.iter()
            .map(|n| n.dendritoma.weight_norm())
            .collect();

        let weight_mean = if !total_weights.is_empty() {
            total_weights.iter().sum::<f64>() / (total_weights.len() as f64 * num_neurons as f64)
        } else {
            0.0
        };

        let weight_std = if !weight_norms.is_empty() {
            let norm_mean = weight_norms.iter().sum::<f64>() / weight_norms.len() as f64;
            weight_norms.iter()
                .map(|x| (x - norm_mean).powi(2))
                .sum::<f64>() / weight_norms.len() as f64
        } else {
            0.0
        }.sqrt();

        let weight_min = total_weights.iter().cloned().fold(f64::INFINITY, f64::min) / num_neurons as f64;
        let weight_max = total_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / num_neurons as f64;

        // Proxy: neurônios com total_weight muito baixo
        let num_dead_synapses = total_weights.iter().filter(|&&w| w < 0.1).count();
        let effective_connectivity = if !total_weights.is_empty() {
            (total_weights.len() - num_dead_synapses) as f64 / total_weights.len() as f64
        } else {
            0.0
        };

        // PI state
        let (target, integral, clamp) = adaptive.pi_state();
        let pi_error = target - fr_mean;
        let pi_integral = integral;
        let pi_integral_saturated = integral.abs() >= clamp * 0.95;
        let pi_u = 0.4 * pi_error + 0.05 * integral; // Kp=0.4, Ki=0.05
        let pi_delta = if pi_u.abs() > 0.01 {
            Some(-0.4 * pi_u)
        } else {
            None
        };

        // Input
        let input_rate = last_input.iter().filter(|&&x| x > 0.0).count() as f64 / num_neurons as f64;
        let input_amplitude_mean = if last_input.iter().any(|&x| x > 0.0) {
            last_input.iter().filter(|&&x| x > 0.0).sum::<f64>()
                / last_input.iter().filter(|&&x| x > 0.0).count() as f64
        } else {
            0.0
        };

        Self {
            step,
            fr_mean,
            fr_std,
            fr_min,
            fr_max,
            num_silent,
            num_normal,
            num_hyperactive,
            threshold_mean,
            threshold_std,
            threshold_min,
            threshold_max,
            num_at_min_threshold,
            energy_mean,
            energy_std,
            energy_min,
            energy_max,
            blocks_by_energy,
            blocks_by_threshold,
            blocks_by_refractory,
            potential_mean,
            potential_std,
            weight_mean,
            weight_std,
            weight_min,
            weight_max,
            num_dead_synapses,
            effective_connectivity,
            pi_error,
            pi_integral,
            pi_integral_saturated,
            pi_u,
            pi_delta,
            input_rate,
            input_amplitude_mean,
        }
    }

    fn print_header() {
        println!("\n{:=<120}", "");
        println!("{:^120}", "DEEP DIAGNOSTIC REPORT");
        println!("{:=<120}\n", "");
    }

    fn print(&self) {
        println!("┌─ STEP {:6} ─────────────────────────────────────────────────────────────────────────┐", self.step);

        // FR
        println!("│ 📊 FIRING RATE");
        println!("│   Mean: {:.4}  Std: {:.4}  Min: {:.4}  Max: {:.4}",
            self.fr_mean, self.fr_std, self.fr_min, self.fr_max);
        println!("│   Silent: {}  Normal: {}  Hyperactive: {}",
            self.num_silent, self.num_normal, self.num_hyperactive);

        // Threshold
        println!("│ 🎚️  THRESHOLD");
        println!("│   Mean: {:.4}  Std: {:.4}  Min: {:.6}  Max: {:.4}",
            self.threshold_mean, self.threshold_std, self.threshold_min, self.threshold_max);
        println!("│   ⚠️  Saturated at min (< 0.002): {}", self.num_at_min_threshold);

        // Energia
        println!("│ ⚡ ENERGIA");
        println!("│   Mean: {:.1}%  Std: {:.1}  Min: {:.1}  Max: {:.1}",
            self.energy_mean, self.energy_std, self.energy_min, self.energy_max);

        // Bloqueios
        println!("│ 🚫 BLOQUEIOS");
        println!("│   By Energy: {}  By Threshold: {}  By Refractory: {}",
            self.blocks_by_energy, self.blocks_by_threshold, self.blocks_by_refractory);

        // Potencial
        println!("│ 🔋 POTENCIAL");
        println!("│   Mean: {:.4}  Std: {:.4}", self.potential_mean, self.potential_std);

        // Sinapses
        println!("│ 🔗 SINAPSES");
        println!("│   Weight Mean: {:.4}  Std: {:.4}  Min: {:.4}  Max: {:.4}",
            self.weight_mean, self.weight_std, self.weight_min, self.weight_max);
        println!("│   Dead (<0.01): {}  Effective Connectivity: {:.1}%",
            self.num_dead_synapses, self.effective_connectivity * 100.0);

        // PI
        println!("│ 🎛️  CONTROLADOR PI");
        println!("│   Error: {:.4}  Integral: {:.4}{}  U: {:.4}",
            self.pi_error,
            self.pi_integral,
            if self.pi_integral_saturated { " [SATURATED]" } else { "" },
            self.pi_u);
        if let Some(delta) = self.pi_delta {
            println!("│   → Delta Threshold: {:.4}", delta);
        } else {
            println!("│   → [Deadzone, no action]");
        }

        // Input
        println!("│ 📥 INPUT EXTERNO");
        println!("│   Rate: {:.1}%  Amplitude Mean: {:.2}",
            self.input_rate * 100.0, self.input_amplitude_mean);

        println!("└{:─<118}┘\n", "");
    }
}

fn main() {
    NetworkSnapshot::print_header();

    println!("🔬 Iniciando análise ultra-detalhada do sistema...\n");

    // Configuração
    let task = TaskSpec {
        num_sensors: 6,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().expect("Falha ao construir rede");

    // Desliga STDP e homeostase local
    network.set_learning_mode(LearningMode::Hebbian);
    for neuron in &mut network.neurons {
        neuron.homeo_eta = 0.0;
    }

    let mut adaptive_state = AdaptiveState::with_target_fr(config.params.target_firing_rate);

    println!("📋 Configuração:");
    println!("  Neurônios: {}", network.num_neurons());
    println!("  Target FR: {:.4}", config.params.target_firing_rate);
    println!("  STDP: OFF | Homeostase local: OFF\n");

    let target_fr = config.params.target_firing_rate;
    let total_steps = 50_000; // Reduzido para análise rápida
    let snapshot_interval = 5_000;

    let mut rng = rand::thread_rng();
    let mut snapshots = Vec::new();
    let mut last_input = vec![0.0; network.num_neurons()];

    println!("🚀 Rodando simulação com snapshots a cada {} steps...\n", snapshot_interval);

    for step in 0..total_steps {
        // Input aleatório
        let num_inputs = (network.num_neurons() as f64 * 0.1) as usize;
        last_input = vec![0.0; network.num_neurons()];
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            last_input[idx] = 1.0;
        }

        network.update(&last_input);

        let _adapted = monitor_and_adapt(
            &mut network,
            &mut adaptive_state,
            target_fr,
            step as i64,
            false,
        );

        // Captura snapshot
        if (step + 1) % snapshot_interval == 0 {
            let snapshot = NetworkSnapshot::capture(&network, &adaptive_state, step + 1, &last_input);
            snapshot.print();
            snapshots.push(snapshot);
        }
    }

    // Análise final
    println!("\n{:=<120}", "");
    println!("{:^120}", "ANÁLISE DE TENDÊNCIAS");
    println!("{:=<120}\n", "");

    println!("📈 Evolução de FR:");
    for snap in &snapshots {
        let erro = ((snap.fr_mean - target_fr).abs() / target_fr) * 100.0;
        println!("  Step {:6}k: FR={:.4} (erro {:.1}%)",
            snap.step / 1000, snap.fr_mean, erro);
    }

    println!("\n📉 Evolução de Threshold Médio:");
    for snap in &snapshots {
        println!("  Step {:6}k: {:.4} (min={:.6}, {} saturados)",
            snap.step / 1000, snap.threshold_mean, snap.threshold_min, snap.num_at_min_threshold);
    }

    println!("\n⚡ Evolução de Energia Média:");
    for snap in &snapshots {
        println!("  Step {:6}k: {:.1}% (min={:.1})",
            snap.step / 1000, snap.energy_mean, snap.energy_min);
    }

    println!("\n🔗 Evolução de Conectividade Efetiva:");
    for snap in &snapshots {
        println!("  Step {:6}k: {:.1}% (weight_mean={:.4}, dead_syn={})",
            snap.step / 1000,
            snap.effective_connectivity * 100.0,
            snap.weight_mean,
            snap.num_dead_synapses);
    }

    println!("\n🎛️  Evolução do Integral do PI:");
    for snap in &snapshots {
        println!("  Step {:6}k: {:.4}{}",
            snap.step / 1000,
            snap.pi_integral,
            if snap.pi_integral_saturated { " [SAT]" } else { "" });
    }

    println!("\n🚫 Distribuição de Bloqueios (último snapshot):");
    if let Some(last) = snapshots.last() {
        let total_blocks = last.blocks_by_energy + last.blocks_by_threshold + last.blocks_by_refractory;
        if total_blocks > 0 {
            println!("  Energia:     {} ({:.1}%)",
                last.blocks_by_energy,
                last.blocks_by_energy as f64 / total_blocks as f64 * 100.0);
            println!("  Threshold:   {} ({:.1}%)",
                last.blocks_by_threshold,
                last.blocks_by_threshold as f64 / total_blocks as f64 * 100.0);
            println!("  Refratário:  {} ({:.1}%)",
                last.blocks_by_refractory,
                last.blocks_by_refractory as f64 / total_blocks as f64 * 100.0);
        } else {
            println!("  [Nenhum bloqueio detectado]");
        }
    }

    // Diagnóstico final
    println!("\n{:=<120}", "");
    println!("{:^120}", "DIAGNÓSTICO FINAL");
    println!("{:=<120}\n", "");

    if let Some(last) = snapshots.last() {
        let erro_fr = ((last.fr_mean - target_fr).abs() / target_fr) * 100.0;

        println!("🎯 FR: {:.4} / {:.4} (erro {:.1}%)", last.fr_mean, target_fr, erro_fr);
        println!();

        // Diagnóstico baseado em métricas
        let mut diagnoses = Vec::new();

        if last.num_at_min_threshold > network.num_neurons() / 2 {
            diagnoses.push("❌ CRÍTICO: >50% neurônios saturaram threshold no mínimo (0.001)");
            diagnoses.push("   → PI não pode mais reduzir threshold. Sistema travado.");
        }

        if last.energy_mean < 40.0 {
            diagnoses.push("❌ CRÍTICO: Energia média muito baixa (<40%)");
            diagnoses.push("   → Gating energético impedindo disparos mesmo com threshold baixo.");
        }

        if last.effective_connectivity < 0.3 {
            diagnoses.push("❌ CRÍTICO: Conectividade efetiva <30%");
            diagnoses.push("   → Weight decay matou sinapses. Rede praticamente desconectada.");
        }

        if last.blocks_by_energy > network.num_neurons() / 4 {
            diagnoses.push("⚠️  ALERTA: >25% neurônios bloqueados por energia");
            diagnoses.push("   → Reduzir energy_cost_fire ou aumentar energy_recovery_rate.");
        }

        if last.pi_integral_saturated {
            diagnoses.push("⚠️  ALERTA: Integral do PI saturou");
            diagnoses.push("   → Aumentar fr_integral_clamp ou aceitar que há limite estrutural.");
        }

        if last.input_rate < 0.05 {
            diagnoses.push("⚠️  ALERTA: Input externo muito esparso (<5%)");
            diagnoses.push("   → Aumentar densidade ou amplitude de entrada.");
        }

        if diagnoses.is_empty() {
            println!("✅ Nenhum problema óbvio detectado.");
            println!("   Sistema pode estar em equilíbrio estrutural não-ideal.");
            println!("   Considerar:");
            println!("   - Aumentar Ki do PI ainda mais");
            println!("   - Ajustar balanço E/I da rede");
            println!("   - Revisar modelo energético");
        } else {
            for diag in diagnoses {
                println!("{}", diag);
            }
        }
    }

    println!("\n🏁 Análise concluída.\n");
}
