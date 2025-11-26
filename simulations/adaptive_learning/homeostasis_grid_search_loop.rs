//! Busca Adaptativa: Encontrando o Sweet Spot da Homeostase
//!
//! ## Estrategia de Otimizacao
//!
//! ### Fase 1: Amostragem Global (Latin Hypercube Sampling)
//! - 32 amostras bem distribuidas no espaco 3D de parametros
//! - Cobre todo o espaco de busca uniformemente
//! - Identifica regioes promissoras
//!
//! ### Fase 2: Refino Local Adaptativo
//! - Toma os TOP 3 resultados da Fase 1
//! - Amostra 8 pontos ao redor de cada um (perturbacao gaussiana)
//! - Total: 24 amostras adicionais focadas nas regioes boas
//!
//! ## Espaco de Parametros
//!
//! 1. **weight_ratio**: [0.30, 0.70] (amplia busca)
//! 2. **eta_multiplier**: [1.5, 3.5] (mais contido para evitar overshoot)
//! 3. **interval_multiplier**: [0.50, 1.00] (intervalos bem lentos para estabilidade)
//!
//! ## Metrica de Otimizacao
//!
//! **Score = -erro^2** (minimizamos erro quadratico)
//! - Penaliza erros grandes mais fortemente
//! - Convergencia tambem considerada (bonus se < 20%)
//! - Penaliza overshoot: FR > 2x target recebe multa pesada

use nenv_visual_sim::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
use nenv_visual_sim::network::LearningMode;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::time::Instant;

#[derive(Debug, Clone)]
struct HomeoConfig {
    weight_ratio: f64,
    threshold_ratio: f64,
    eta_multiplier: f64,
    interval_multiplier: f64,
}

impl HomeoConfig {
    fn name(&self) -> String {
        format!(
            "W{:.0}T{:.0}_eta{:.1}x_int{:.3}x",
            self.weight_ratio * 100.0,
            self.threshold_ratio * 100.0,
            self.eta_multiplier,
            self.interval_multiplier,
        )
    }
}

#[derive(Debug, Clone)]
struct ExperimentResult {
    config: HomeoConfig,
    final_fr: f64,
    final_error_pct: f64,
    convergence_step: Option<usize>,
    mean_fr_last_20k: f64,
    std_dev_last_20k: f64,
    min_threshold: f64,
    max_threshold: f64,
    final_threshold: f64,
    runtime_ms: u128,
    max_fr_seen: f64,
    overshoot: bool,
}

fn main() {
    println!("===============================================================");
    println!("  FASE 3: Busca Focada no Sweet Spot");
    println!("===============================================================");
    println!("\nRanges Ajustados (edge effects):");
    println!("   - Weight/Threshold: 0.30-0.70 (ampliado)");
    println!("   - Eta multiplier: 1.5-3.5x (estabilidade)");
    println!("   - Interval multiplier: 0.50-1.00x (mais estável)\n");

    let mut rng = StdRng::seed_from_u64(43);
    let max_loops = 5;
    let mut loop_idx = 0;
    let mut found_stable = false;

    println!("FASE 3.1: Amostragem Global Focada (Latin Hypercube)");
    println!("   Testando 32 configuracoes na regiao promissora...\n");

    let phase1_configs = latin_hypercube_sampling(&mut rng, 32);
    let mut phase1_results = Vec::new();

    for (i, config) in phase1_configs.iter().enumerate() {
        println!("[Fase 3.1: {}/32] {}", i + 1, config.name());

        let result = run_experiment(config.clone());

        println!(
            "   FR: {:.4} | Erro: {:.1}% | Score: {:.3} | Tempo: {}ms\n",
            result.final_fr,
            result.final_error_pct,
            compute_score(&result),
            result.runtime_ms
        );

        phase1_results.push(result);
    }

    let mut sorted_phase1 = phase1_results.clone();
    sorted_phase1.sort_by(|a, b| compute_score(b).partial_cmp(&compute_score(a)).unwrap());

    println!("\nTOP 3 da Fase 3.1:");
    for (i, result) in sorted_phase1.iter().take(3).enumerate() {
        println!(
            "  {}. {} | Erro: {:.1}% | Score: {:.3}",
            i + 1,
            result.config.name(),
            result.final_error_pct,
            compute_score(result)
        );
    }

    println!("\nFASE 3.2: Refino Local Ultra-Preciso");
    println!("   Refinando ao redor dos 3 melhores (8 amostras cada)...\n");

    let mut phase2_results = Vec::new();

    for (top_idx, top_result) in sorted_phase1.iter().take(3).enumerate() {
        println!("Refinando ao redor de TOP {}: {}", top_idx + 1, top_result.config.name());

        let local_configs = local_refinement(&mut rng, &top_result.config, 8);

        for (i, config) in local_configs.iter().enumerate() {
            println!("   [{}/8] {}", i + 1, config.name());

            let result = run_experiment(config.clone());

            println!(
                "      Erro: {:.1}% | Score: {:.3}",
                result.final_error_pct,
                compute_score(&result)
            );

            phase2_results.push(result);
        }
        println!();
    }

    let mut all_results = phase1_results;
    all_results.extend(phase2_results);

    println!("\n===============================================================");
    println!("  ANALISE FINAL ({} experimentos)", all_results.len());
    println!("===============================================================\n");

    analyze_results(&all_results);
    // Checa se algum resultado ficou sem overshoot
    found_stable = all_results.iter().any(|r| !r.overshoot);
    if found_stable {
        println!("✅ Encontrou configuração sem overshoot na primeira rodada.");
        return;
    }

    // Se não achou resultado sem overshoot, tenta mais loops de busca focada
    while !found_stable && loop_idx < max_loops {
        loop_idx += 1;
        println!("\n⚠️ Nenhum resultado sem overshoot. Reamostrando (loop {})...\n", loop_idx);

        let phase1_configs = latin_hypercube_sampling(&mut rng, 48);
        let mut extra_results = Vec::new();

        for (i, config) in phase1_configs.iter().enumerate() {
            println!("[Loop {} - Exploração {}/{}] {}", loop_idx, i + 1, phase1_configs.len(), config.name());
            let result = run_experiment(config.clone());
            println!(
                "   FR: {:.4} | Erro: {:.2}% | Score: {:.3} | overshoot: {}\n",
                result.final_fr,
                result.final_error_pct,
                compute_score(&result),
                result.overshoot
            );
            extra_results.push(result);
        }

        all_results.extend(extra_results);
        analyze_results(&all_results);
        found_stable = all_results.iter().any(|r| !r.overshoot);

        if found_stable {
            println!("✅ Encontrou configuração sem overshoot no loop {}.", loop_idx);
            break;
        }
    }

    println!("\nBusca adaptativa concluida.\n");
}

fn compute_score(result: &ExperimentResult) -> f64 {
    let base_score = -(result.final_error_pct * result.final_error_pct);
    let convergence_bonus = if result.convergence_step.is_some() { 100.0 } else { 0.0 };
    let stability_penalty = -result.std_dev_last_20k * 50.0;
    let overshoot_penalty = if result.overshoot { 5000.0 } else { 0.0 };

    // Penaliza thresholds que colam no piso (instabilidade/colapso)
    let collapse_penalty = if result.min_threshold < 0.05 { 1000.0 } else { 0.0 };

    base_score + convergence_bonus + stability_penalty - overshoot_penalty - collapse_penalty
}

fn latin_hypercube_sampling(rng: &mut StdRng, n_samples: usize) -> Vec<HomeoConfig> {
    let mut configs = Vec::new();

    let weight_range = (0.30, 0.70);
    let eta_range = (1.5, 3.5);
    let interval_range = (0.50, 1.00);

    let mut weight_perm: Vec<usize> = (0..n_samples).collect();
    let mut eta_perm: Vec<usize> = (0..n_samples).collect();
    let mut interval_perm: Vec<usize> = (0..n_samples).collect();

    use rand::seq::SliceRandom;
    weight_perm.shuffle(rng);
    eta_perm.shuffle(rng);
    interval_perm.shuffle(rng);

    let uniform = Uniform::new(0.0, 1.0);

    for i in 0..n_samples {
        let weight_cell = weight_perm[i] as f64 + uniform.sample(rng);
        let eta_cell = eta_perm[i] as f64 + uniform.sample(rng);
        let interval_cell = interval_perm[i] as f64 + uniform.sample(rng);

        let weight_norm = weight_cell / n_samples as f64;
        let eta_norm = eta_cell / n_samples as f64;
        let interval_norm = interval_cell / n_samples as f64;

        let weight_ratio = weight_range.0 + weight_norm * (weight_range.1 - weight_range.0);
        let threshold_ratio = 1.0 - weight_ratio;

        let eta_multiplier = eta_range.0 + eta_norm * (eta_range.1 - eta_range.0);
        let interval_multiplier = interval_range.0 + interval_norm * (interval_range.1 - interval_range.0);

        configs.push(HomeoConfig {
            weight_ratio,
            threshold_ratio,
            eta_multiplier,
            interval_multiplier,
        });
    }

    configs
}

fn local_refinement(rng: &mut StdRng, center: &HomeoConfig, n_samples: usize) -> Vec<HomeoConfig> {
    let mut configs = Vec::new();

    let weight_std = 0.05;
    let eta_std = 0.2;
    let interval_std = 0.08;

    for _ in 0..n_samples {
        let weight_delta = Normal::new(0.0, weight_std).unwrap().sample(rng);
        let eta_delta = Normal::new(0.0, eta_std).unwrap().sample(rng);
        let interval_delta = Normal::new(0.0, interval_std).unwrap().sample(rng);

        let weight_ratio = (center.weight_ratio + weight_delta).clamp(0.35, 0.65);
        let threshold_ratio = 1.0 - weight_ratio;

        let eta_multiplier = (center.eta_multiplier + eta_delta).clamp(1.5, 3.5);
        let interval_multiplier = (center.interval_multiplier + interval_delta).clamp(0.50, 1.00);

        configs.push(HomeoConfig {
            weight_ratio,
            threshold_ratio,
            eta_multiplier,
            interval_multiplier,
        });
    }

    configs
}

fn run_experiment(config: HomeoConfig) -> ExperimentResult {
    let start_time = Instant::now();

    let task = TaskSpec {
        num_sensors: 6,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let autoconfig = AutoConfig::from_task(task);
    let mut network = autoconfig.build_network().expect("Falha ao construir rede");

    // Liga STDP para estressar homeostase
    network.set_learning_mode(LearningMode::STDP);

    let base_eta = autoconfig.params.homeostatic.homeo_eta;
    let base_interval = autoconfig.params.homeostatic.homeo_interval;

    for neuron in &mut network.neurons {
        neuron.homeo_eta = base_eta * config.eta_multiplier;
        neuron.homeo_interval = (base_interval as f64 * config.interval_multiplier)
            .max(1.0) as i64;

        neuron.homeo_weight_ratio = config.weight_ratio;
        neuron.homeo_threshold_ratio = config.threshold_ratio;
    }

    let target_fr = autoconfig.params.target_firing_rate;
    let total_steps = 100_000;

    let input_density = autoconfig.params.input.recommended_input_density;
    let input_amplitude = autoconfig.params.input.recommended_input_amplitude;

    let mut rng = rand::thread_rng();
    let mut fr_samples = Vec::new();
    let mut threshold_samples = Vec::new();
    let mut convergence_step = None;
    let mut max_fr_seen = 0.0;

    for step in 0..total_steps {
        let num_inputs = (network.num_neurons() as f64 * input_density) as usize;
        let mut inputs = vec![0.0; network.num_neurons()];
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            inputs[idx] = input_amplitude;
        }

        network.update(&inputs);

        let current_fr = network.num_firing() as f64 / network.num_neurons() as f64;
        fr_samples.push(current_fr);
        if current_fr > max_fr_seen {
            max_fr_seen = current_fr;
        }

        let avg_threshold: f64 = network.neurons.iter()
            .map(|n| n.threshold)
            .sum::<f64>() / network.num_neurons() as f64;
        threshold_samples.push(avg_threshold);

        if convergence_step.is_none() && step > 1000 {
            let recent_fr: f64 = fr_samples.iter()
                .rev()
                .take(1000)
                .sum::<f64>() / 1000.0;

            let error = ((recent_fr - target_fr).abs() / target_fr) * 100.0;

            if error < 20.0 {
                convergence_step = Some(step);
            }
        }
    }

    let runtime_ms = start_time.elapsed().as_millis();

    let last_20k: Vec<f64> = fr_samples.iter()
        .rev()
        .take(20_000)
        .copied()
        .collect();

    let mean_fr_last_20k = last_20k.iter().sum::<f64>() / last_20k.len() as f64;

    let variance = last_20k.iter()
        .map(|x| (x - mean_fr_last_20k).powi(2))
        .sum::<f64>() / last_20k.len() as f64;
    let std_dev_last_20k = variance.sqrt();

    let final_error_pct = ((mean_fr_last_20k - target_fr).abs() / target_fr) * 100.0;

    let min_threshold = threshold_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_threshold = threshold_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let final_threshold = *threshold_samples.last().unwrap();
    let overshoot = max_fr_seen > target_fr * 2.0;

    ExperimentResult {
        config,
        final_fr: mean_fr_last_20k,
        final_error_pct,
        convergence_step,
        mean_fr_last_20k,
        std_dev_last_20k,
        min_threshold,
        max_threshold,
        final_threshold,
        runtime_ms,
        max_fr_seen,
        overshoot,
    }
}

fn analyze_results(results: &[ExperimentResult]) {
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| compute_score(b).partial_cmp(&compute_score(a)).unwrap());

    println!("TOP 10 Melhores Configuracoes (por score):\n");

    for (i, result) in sorted.iter().take(10).enumerate() {
        let score = compute_score(result);

        println!("{}. {} | Score: {:.1}", i + 1, result.config.name(), score);
        println!(
            "   FR: {:.4} | Erro: {:.2}% | sigma: {:.4} | overshoot: {}",
            result.final_fr,
            result.final_error_pct,
            result.std_dev_last_20k,
            if result.overshoot { "Sim" } else { "Nao" }
        );
        println!(
            "   Threshold: {:.3} -> {:.3} (min: {:.3}, max: {:.3})",
            0.3,
            result.final_threshold,
            result.min_threshold,
            result.max_threshold
        );
        println!(
            "   Convergiu: {} | Tempo: {}ms",
            if result.convergence_step.is_some() { "Sim" } else { "Nao" },
            result.runtime_ms
        );
        println!();
    }

    let best = &sorted[0];
    let worst = sorted.last().unwrap();

    let avg_error: f64 = results.iter()
        .map(|r| r.final_error_pct)
        .sum::<f64>() / results.len() as f64;

    let convergence_rate = results.iter()
        .filter(|r| r.convergence_step.is_some())
        .count() as f64 / results.len() as f64 * 100.0;

    println!("\nEstatisticas Globais:");
    println!(
        "   Melhor erro: {:.2}% | Pior: {:.2}% | Medio: {:.2}%",
        best.final_error_pct, worst.final_error_pct, avg_error
    );
    println!("   Taxa de convergencia (< 20%): {:.1}%", convergence_rate);
    println!();

    println!("Configuracao Recomendada:");
    println!("   {}", best.config.name());
    println!("   weight_ratio = {:.3}", best.config.weight_ratio);
    println!("   threshold_ratio = {:.3}", best.config.threshold_ratio);
    println!("   eta_multiplier = {:.3}", best.config.eta_multiplier);
    println!("   interval_multiplier = {:.3}", best.config.interval_multiplier);
}
