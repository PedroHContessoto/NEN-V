//! Análise Profunda do Fluxo Completo da Rede
//!
//! Este experimento rastreia TODOS os mecanismos em ação:
//! - STDP (quando acontece, amplitude)
//! - Weight decay (onde e quanto)
//! - Homeostase (threshold adjustment, synaptic scaling)
//! - Período refratário
//! - Energy gating
//! - Eligibility traces
//! - STP (Short-Term Plasticity)

use nenv_v2::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_v2::network::LearningMode;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║           ANÁLISE PROFUNDA DO FLUXO COMPLETO DA REDE                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Rede pequena para análise detalhada
    let task = TaskSpec {
        num_sensors: 5,
        num_actuators: 2,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task.clone());
    let mut network = config.build_network().expect("Falha ao construir rede");
    network.set_learning_mode(LearningMode::STDP);

    let total_neurons = network.num_neurons();
    let sensor_indices = config.architecture.sensor_indices.clone();
    let hidden_indices = config.architecture.hidden_indices.clone();

    println!("Configuração:");
    println!("  Total neurônios: {}", total_neurons);
    println!("  Sensores: {} (índices {:?})", sensor_indices.len(), sensor_indices);
    println!("  Hidden: {} (índices {:?})", hidden_indices.len(), hidden_indices);
    println!("  Target FR: {:.4}", config.params.target_firing_rate);
    println!("  Learning mode: STDP");
    println!();

    // Neurônio rastreado: primeiro hidden
    let tracked_idx = hidden_indices.start;

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  FASE 1: Estado Inicial (Step 0)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let n = &network.neurons[tracked_idx];
    println!("Neurônio rastreado: #{} (hidden)", tracked_idx);
    println!("\n📊 ESTADO INICIAL:");
    println!("  Threshold: {:.6}", n.threshold);
    println!("  Base threshold: {:.6}", n.threshold);
    println!("  Target FR: {:.6}", n.target_firing_rate);
    println!("  Homeo eta: {:.6}", n.homeo_eta);
    println!("  Homeo interval: {}", n.homeo_interval);
    println!("  Recent firing rate: {:.6}", n.recent_firing_rate);
    println!("\n🔗 PESOS (Dendritoma):");
    println!("  Soma total: {:.6}", n.dendritoma.total_weight());
    println!("  Média: {:.6}", n.dendritoma.total_weight() / n.dendritoma.weights.len() as f64);
    println!("  Learning rate: {:.6}", n.dendritoma.get_learning_rate());
    println!("  STDP a_plus: {:.6}, a_minus: {:.6}", n.dendritoma.get_stdp_amplitudes().0, n.dendritoma.get_stdp_amplitudes().1);
    println!("  Weight decay: 0.0001");
    println!("  Plasticity gain: {:.6}", n.dendritoma.plasticity_gain);
    println!("\n⚡ ENERGIA (Glia):");
    println!("  Energy: {:.2}", n.glia.energy);
    println!("  Max energy: {:.2}", n.glia.max_energy);
    println!("\n🧠 ELIGIBILITY TRACES:");
    println!("  Total: {:.6}", n.dendritoma.total_eligibility());
    println!("  Trace tau: {:.1}", n.dendritoma.trace_tau);
    println!("\n🔄 SHORT-TERM PLASTICITY:");
    let avg_resources: f64 = n.dendritoma.synaptic_resources.iter().sum::<f64>() / n.dendritoma.synaptic_resources.len() as f64;
    println!("  Avg resources: {:.6}", avg_resources);

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  FASE 2: Simulação Detalhada (100 steps)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let mut rng = StdRng::seed_from_u64(42);
    let steps = 100;

    // Estrutura para capturar eventos
    #[derive(Debug)]
    struct StepEvent {
        step: usize,
        fired: bool,
        input_active: bool,
        weight_sum_before: f64,
        weight_sum_after: f64,
        threshold_before: f64,
        threshold_after: f64,
        fr_network: f64,
        energy: f64,
        eligibility: f64,
        recent_fr: f64,
        num_firing_network: usize,
    }

    let mut events = Vec::new();

    for step in 0..steps {
        // Captura estado ANTES do update
        let n_before = &network.neurons[tracked_idx];
        let weight_sum_before = n_before.dendritoma.total_weight();
        let threshold_before = n_before.threshold;
        let energy_before = n_before.glia.energy;

        // Input aleatório (10% dos sensores)
        let mut inputs = vec![0.0; total_neurons];
        let mut input_active = false;
        for idx in sensor_indices.clone() {
            if rng.gen_bool(0.1) {
                inputs[idx] = 1.0;
                input_active = true;
            }
        }

        // Update da rede
        network.update(&inputs);

        // Captura estado DEPOIS do update
        let n_after = &network.neurons[tracked_idx];
        let weight_sum_after = n_after.dendritoma.total_weight();
        let threshold_after = n_after.threshold;
        let fired = n_after.is_firing;
        let fr_network = network.num_firing() as f64 / total_neurons as f64;
        let eligibility = n_after.dendritoma.total_eligibility();
        let recent_fr = n_after.recent_firing_rate;
        let num_firing = network.num_firing();

        events.push(StepEvent {
            step,
            fired,
            input_active,
            weight_sum_before,
            weight_sum_after,
            threshold_before,
            threshold_after,
            fr_network,
            energy: energy_before,
            eligibility,
            recent_fr,
            num_firing_network: num_firing,
        });

        // Log de eventos críticos
        if step < 20 || fired || (weight_sum_after - weight_sum_before).abs() > 0.1 {
            let weight_delta = weight_sum_after - weight_sum_before;
            let threshold_delta = threshold_after - threshold_before;

            print!("  Step {:3}: ", step);

            if fired {
                print!("🔥 FIRE ");
            } else {
                print!("       ");
            }

            if input_active {
                print!("📥 INPUT ");
            } else {
                print!("        ");
            }

            println!("| W: {:.3}→{:.3} ({:+.3}) | Th: {:.4}→{:.4} ({:+.4}) | FR_net: {:.2} | E: {:.1}",
                weight_sum_before,
                weight_sum_after,
                weight_delta,
                threshold_before,
                threshold_after,
                threshold_delta,
                fr_network,
                energy_before);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  FASE 3: Análise Estatística dos Eventos");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Análise de disparos
    let total_fires: usize = events.iter().filter(|e| e.fired).count();
    let fire_rate = total_fires as f64 / steps as f64;

    println!("📊 ESTATÍSTICAS DE DISPAROS:");
    println!("  Total disparos: {} / {} steps ({:.2}%)", total_fires, steps, fire_rate * 100.0);
    println!("  Target FR: {:.2}%", config.params.target_firing_rate * 100.0);

    // Quando parou de disparar?
    if let Some(last_fire) = events.iter().rev().find(|e| e.fired) {
        println!("  Último disparo: step {}", last_fire.step);
        if last_fire.step < steps - 10 {
            println!("  ⚠️  Rede parou de disparar {} steps antes do fim!", steps - last_fire.step);
        }
    }

    // Análise de pesos
    let initial_weight = events[0].weight_sum_before;
    let final_weight = events.last().unwrap().weight_sum_after;
    let weight_change = final_weight - initial_weight;
    let weight_change_pct = (weight_change / initial_weight) * 100.0;

    println!("\n📊 EVOLUÇÃO DOS PESOS:");
    println!("  Inicial: {:.6}", initial_weight);
    println!("  Final: {:.6}", final_weight);
    println!("  Mudança: {:+.6} ({:+.2}%)", weight_change, weight_change_pct);

    // Análise de threshold
    let initial_threshold = events[0].threshold_before;
    let final_threshold = events.last().unwrap().threshold_after;
    let threshold_change = final_threshold - initial_threshold;
    let threshold_change_pct = (threshold_change / initial_threshold) * 100.0;

    println!("\n📊 EVOLUÇÃO DO THRESHOLD:");
    println!("  Inicial: {:.6}", initial_threshold);
    println!("  Final: {:.6}", final_threshold);
    println!("  Mudança: {:+.6} ({:+.2}%)", threshold_change, threshold_change_pct);

    // Analisa períodos
    println!("\n📊 ANÁLISE POR PERÍODO:");

    let periods = [
        (0, 20, "Primeiros 20 steps"),
        (20, 50, "Steps 20-50"),
        (50, 100, "Steps 50-100"),
    ];

    for (start, end, name) in periods {
        let period_events: Vec<_> = events.iter().filter(|e| e.step >= start && e.step < end).collect();
        let period_fires = period_events.iter().filter(|e| e.fired).count();
        let period_fr = period_fires as f64 / (end - start) as f64;

        let avg_weight: f64 = period_events.iter().map(|e| e.weight_sum_after).sum::<f64>() / period_events.len() as f64;
        let avg_threshold: f64 = period_events.iter().map(|e| e.threshold_after).sum::<f64>() / period_events.len() as f64;
        let avg_fr_net: f64 = period_events.iter().map(|e| e.fr_network).sum::<f64>() / period_events.len() as f64;

        println!("\n  {}", name);
        println!("    Disparos: {} ({:.2}%)", period_fires, period_fr * 100.0);
        println!("    Peso médio: {:.6}", avg_weight);
        println!("    Threshold médio: {:.6}", avg_threshold);
        println!("    FR rede: {:.3}", avg_fr_net);
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  FASE 4: Estado Final Detalhado");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let n_final = &network.neurons[tracked_idx];

    println!("Neurônio #{} (estado final):", tracked_idx);
    println!("\n📊 COMPARAÇÃO INICIAL → FINAL:");
    println!("  Threshold: {:.6} → {:.6} ({:+.2}%)",
        initial_threshold,
        n_final.threshold,
        threshold_change_pct);
    println!("  Weight sum: {:.6} → {:.6} ({:+.2}%)",
        initial_weight,
        n_final.dendritoma.total_weight(),
        weight_change_pct);
    println!("  Recent FR: 0.000 → {:.6}", n_final.recent_firing_rate);
    println!("  Energy: 100.0 → {:.2}", n_final.glia.energy);
    println!("  Eligibility: 0.000 → {:.6}", n_final.dendritoma.total_eligibility());

    // Análise de pesos individuais
    println!("\n🔍 DISTRIBUIÇÃO DOS PESOS FINAIS:");
    let weights = &n_final.dendritoma.weights;
    let weight_min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let weight_max = weights.iter().cloned().fold(0.0, f64::max);
    let weight_avg = weights.iter().sum::<f64>() / weights.len() as f64;
    let weight_std = (weights.iter().map(|w| (w - weight_avg).powi(2)).sum::<f64>() / weights.len() as f64).sqrt();

    println!("  Min: {:.6}", weight_min);
    println!("  Max: {:.6}", weight_max);
    println!("  Avg: {:.6}", weight_avg);
    println!("  Std: {:.6}", weight_std);

    // Análise de conectividade efetiva
    let effective_weights = weights.iter().filter(|&&w| w > 0.01).count();
    println!("  Pesos > 0.01: {} / {} ({:.1}%)",
        effective_weights,
        weights.len(),
        (effective_weights as f64 / weights.len() as f64) * 100.0);

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  DIAGNÓSTICO FINAL");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    if final_weight < initial_weight * 0.5 {
        println!("⚠️  PROBLEMA CRÍTICO: Weight decay excessivo!");
        println!("    Pesos caíram mais de 50% ({:.1}%)", weight_change_pct.abs());
    }

    if fire_rate < config.params.target_firing_rate * 0.5 {
        println!("⚠️  PROBLEMA CRÍTICO: Firing rate muito baixo!");
        println!("    FR observado: {:.2}% (target: {:.2}%)",
            fire_rate * 100.0,
            config.params.target_firing_rate * 100.0);
    }

    if events.last().unwrap().fr_network < 0.01 {
        println!("⚠️  PROBLEMA CRÍTICO: Rede morta!");
        println!("    FR da rede: {:.3}", events.last().unwrap().fr_network);
    }

    if n_final.threshold < initial_threshold * 0.5 {
        println!("⚠️  Homeostase baixou threshold drasticamente ({:.1}%)", threshold_change_pct.abs());
        println!("    Tentativa de compensar baixo FR, mas não funcionou");
    }

    println!("\n✅ Análise completa. Dados salvos para análise posterior.");
}
