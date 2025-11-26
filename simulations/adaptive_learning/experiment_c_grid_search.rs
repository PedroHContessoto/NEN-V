//! Busca rápida de parâmetros para o Experimento C (estabilidade + aprendizado)
//!
//! Varia:
//! - input_density (ruído baseline)
//! - pattern_prob (probabilidade do padrão na fase de treino)
//! - input_amplitude (amplitude do estímulo)
//! - learning_rate (STDP)
//! - threshold_floor_factor (piso do threshold relativo ao inicial)
//!
//! Protocolo moderado: 60k steps (10k/40k/10k) para triagem.

use nenv_visual_sim::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_visual_sim::network::LearningMode;
use nenv_visual_sim::NeuronType;
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

#[derive(Debug, Clone)]
struct SearchConfig {
    input_density: f64,
    pattern_prob: f64,
    input_amplitude: f64,
    learning_rate: f64,
    threshold_floor_factor: f64,
}

#[derive(Debug, Clone)]
struct SearchResult {
    cfg: SearchConfig,
    fr_mean: f64,
    fr_max: f64,
    gap_w: f64,
    min_threshold: f64,
    pattern_ltp_events: usize,
    pattern_ltd_events: usize,
    noise_ltp_events: usize,
    noise_ltd_events: usize,
    score: f64,
    overshoot: bool,
    runtime_ms: u128,
}

fn main() {
    println!("===============================================================");
    println!("  GRID RÁPIDO: Experimento C (estabilidade + aprendizado)");
    println!("===============================================================\n");

    let mut rng = StdRng::seed_from_u64(999);
    let samples = latin_hypercube(&mut rng, 32);
    let mut results = Vec::new();

    for (i, cfg) in samples.iter().enumerate() {
        println!(
            "[{}/{}] dens={:.3} patt={:.2} amp={:.2} lr={:.4} floor={:.3}",
            i + 1,
            samples.len(),
            cfg.input_density,
            cfg.pattern_prob,
            cfg.input_amplitude,
            cfg.learning_rate,
            cfg.threshold_floor_factor
        );
        let res = run_experiment(cfg.clone());
        println!(
            "   FR_mean={:.3} max={:.3} gap_w={:.3} min_th={:.3} overshoot={} score={:.1} time={}ms",
            res.fr_mean,
            res.fr_max,
            res.gap_w,
            res.min_threshold,
            res.overshoot,
            res.score,
            res.runtime_ms
        );
        results.push(res);
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    println!("\nTOP 5 por score:\n");
    for (i, r) in results.iter().take(5).enumerate() {
        println!(
            "{}. dens={:.3} patt={:.2} amp={:.2} lr={:.4} floor={:.3} | FRm={:.3} max={:.3} gap_w={:.3} min_th={:.3} overshoot={} score={:.1} | LTPp/LTDp={}/{} LTPn/LTDn={}/{}",
            i + 1,
            r.cfg.input_density,
            r.cfg.pattern_prob,
            r.cfg.input_amplitude,
            r.cfg.learning_rate,
            r.cfg.threshold_floor_factor,
            r.fr_mean,
            r.fr_max,
            r.gap_w,
            r.min_threshold,
            r.overshoot,
            r.score,
            r.pattern_ltp_events,
            r.pattern_ltd_events,
            r.noise_ltp_events,
            r.noise_ltd_events
        );
    }
}

fn run_experiment(cfg: SearchConfig) -> SearchResult {
    let start = Instant::now();

    let task = TaskSpec {
        num_sensors: 10,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task.clone());
    let mut network = config.build_network().expect("Falha ao construir rede");
    let base_thr = config.architecture.initial_threshold;

    // Ajusta learning rate de STDP
    for n in &mut network.neurons {
        n.dendritoma.set_learning_rate(cfg.learning_rate);
    }
    network.set_learning_mode(LearningMode::STDP);

    // Padrão: sensores pares ativos
    let sensor_indices = config.architecture.sensor_indices.clone();
    let hid_start = config.architecture.hidden_indices.start;
    let hid_end = config.architecture.hidden_indices.end;
    let mut pattern_input = vec![0.0; network.num_neurons()];
    for i in sensor_indices.clone() {
        if i % 2 == 0 {
            pattern_input[i] = 1.0;
        }
    }

    // Protocolo moderado
    let total_steps = 60_000;
    let phase_1_end = 10_000;
    let phase_2_end = 50_000;
    // Mantém o padrão ativo por alguns passos consecutivos para favorecer correlação temporal
    let pattern_period = 25;
    let pattern_duration = 5;

    let mut rng = StdRng::seed_from_u64(42);
    let mut inputs = vec![0.0; network.num_neurons()];

    let mut fr_sum = 0.0;
    let mut fr_max = 0.0;
    let mut min_threshold = f64::MAX;
    let mut steps_executed = 0.0;
    let target_fr = config.params.target_firing_rate;

    // Para métricas temporais (LTP/LTD contados incrementalmente)
    let mut pattern_ltp_events = 0usize;
    let mut pattern_ltd_events = 0usize;
    let mut noise_ltp_events = 0usize;
    let mut noise_ltd_events = 0usize;

    // Snapshot inicial dos pesos relevantes
    let mut prev_pattern_weights = Vec::new();
    let mut prev_noise_weights = Vec::new();
    for n in &network.neurons {
        if n.id >= hid_start && n.id < hid_end && n.neuron_type == NeuronType::Excitatory {
            for (idx, &w) in n.dendritoma.weights.iter().enumerate() {
                if sensor_indices.contains(&idx) {
                    if idx % 2 == 0 {
                        prev_pattern_weights.push(w);
                    } else {
                        prev_noise_weights.push(w);
                    }
                }
            }
        }
    }

    for step in 0..total_steps {
        inputs.fill(0.0);

        if step < phase_1_end {
            for i in sensor_indices.clone() {
                if rng.gen_bool(cfg.input_density) {
                    inputs[i] = 1.0 * cfg.input_amplitude;
                }
            }
        } else if step < phase_2_end {
            let in_pattern_window = step % pattern_period < pattern_duration;
            if in_pattern_window && rng.gen_bool(cfg.pattern_prob) {
                for (i, &val) in pattern_input.iter().enumerate() {
                    if val > 0.0 {
                        inputs[i] = val * cfg.input_amplitude;
                    }
                }
            } else {
                for i in sensor_indices.clone() {
                    if rng.gen_bool(cfg.input_density) {
                        inputs[i] = 1.0 * cfg.input_amplitude;
                    }
                }
            }
        } else {
            for i in sensor_indices.clone() {
                if rng.gen_bool(cfg.input_density) {
                        inputs[i] = 1.0 * cfg.input_amplitude;
                }
            }
        }

        network.update(&inputs);

        // Impõe piso de threshold adaptado
        for n in &mut network.neurons {
            let min_thr = (base_thr * cfg.threshold_floor_factor).max(0.001);
            n.threshold = n.threshold.max(min_thr);
            if n.threshold < min_threshold {
                min_threshold = n.threshold;
            }
        }

        let fr = network.num_firing() as f64 / network.num_neurons() as f64;
        fr_sum += fr;
        steps_executed += 1.0;
        if fr > fr_max {
            fr_max = fr;
        }

        // A cada 1000 steps, mede deltas de pesos para contar LTP/LTD
        if step % 1000 == 0 {
            let mut idx_pattern = 0usize;
            let mut idx_noise = 0usize;
            for n in &network.neurons {
                if n.id >= hid_start && n.id < hid_end && n.neuron_type == NeuronType::Excitatory {
                    for (idx, &w) in n.dendritoma.weights.iter().enumerate() {
                        if sensor_indices.contains(&idx) {
                            if idx % 2 == 0 {
                                if let Some(prev) = prev_pattern_weights.get_mut(idx_pattern) {
                                    let delta = w - *prev;
                                    if delta > 1e-4 {
                                        pattern_ltp_events += 1;
                                    } else if delta < -1e-4 {
                                        pattern_ltd_events += 1;
                                    }
                                    *prev = w;
                                }
                                idx_pattern += 1;
                            } else {
                                if let Some(prev) = prev_noise_weights.get_mut(idx_noise) {
                                    let delta = w - *prev;
                                    if delta > 1e-4 {
                                        noise_ltp_events += 1;
                                    } else if delta < -1e-4 {
                                        noise_ltd_events += 1;
                                    }
                                    *prev = w;
                                }
                                idx_noise += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Pesos: média das conexões de sensores pares vs ímpares (hidden only)
    let mut w_pattern_sum = 0.0;
    let mut w_pattern_count = 0;
    let mut w_noise_sum = 0.0;
    let mut w_noise_count = 0;

    for n in &network.neurons {
        if n.id >= hid_start && n.id < hid_end && n.neuron_type == NeuronType::Excitatory {
            for (idx, &w) in n.dendritoma.weights.iter().enumerate() {
                if sensor_indices.contains(&idx) {
                    if idx % 2 == 0 {
                        w_pattern_sum += w;
                        w_pattern_count += 1;
                    } else {
                        w_noise_sum += w;
                        w_noise_count += 1;
                    }
                }
            }
        }
    }

    let avg_w_pattern = if w_pattern_count > 0 {
        w_pattern_sum / w_pattern_count as f64
    } else {
        0.0
    };
    let avg_w_noise = if w_noise_count > 0 {
        w_noise_sum / w_noise_count as f64
    } else {
        0.0
    };
    let gap_w = avg_w_pattern - avg_w_noise;

    let fr_mean = if steps_executed > 0.0 {
        fr_sum / steps_executed
    } else {
        0.0
    };
    // Marca overshoot mas permite continuar para ver recuperação
    let overshoot = fr_max > 1.0;

    // Score: penaliza overshoot/runaway, FR muito baixo/alto, premia gap_w e proximidade ao target
    let mut score = 0.0;
    if overshoot {
        score -= 20_000.0;
    }
    let fr_error = ((fr_mean - target_fr).abs() / target_fr) * 100.0;
    score -= fr_error * fr_error; // erro quadrático

    // Penaliza silencioso (<50% do target) ou hiperativo (>200% do target)
    if fr_mean < target_fr * 0.5 || fr_mean > target_fr * 2.0 {
        score -= 7_500.0;
    }

    // Penaliza gap negativo (não aprendeu padrão)
    if gap_w <= 0.0 {
        score -= 50_000.0;
    }

    score -= (min_threshold < cfg.threshold_floor_factor * 0.8) as i32 as f64 * 2000.0;

    // Bônus pela seletividade (gap positivo) e proximidade ao target
    if gap_w > 0.0 {
        score += gap_w * 500.0;
    }

    let runtime_ms = start.elapsed().as_millis();

    SearchResult {
        cfg,
        fr_mean,
        fr_max,
        gap_w,
        min_threshold,
        pattern_ltp_events,
        pattern_ltd_events,
        noise_ltp_events,
        noise_ltd_events,
        score,
        overshoot,
        runtime_ms,
    }
}

fn latin_hypercube(rng: &mut StdRng, n: usize) -> Vec<SearchConfig> {
    let mut configs = Vec::new();

    let ranges = (
        (0.01, 0.08),     // input_density
        (0.5, 0.8),       // pattern_prob
        (2.0, 4.0),       // input_amplitude
        (0.001, 0.010),   // learning_rate
        (0.10, 0.30),     // threshold_floor_factor
    );

    let mut idxs: Vec<usize> = (0..n).collect();
    use rand::seq::SliceRandom;
    let mut perm1 = idxs.clone();
    let mut perm2 = idxs.clone();
    let mut perm3 = idxs.clone();
    let mut perm4 = idxs.clone();
    let mut perm5 = idxs.clone();
    perm1.shuffle(rng);
    perm2.shuffle(rng);
    perm3.shuffle(rng);
    perm4.shuffle(rng);
    perm5.shuffle(rng);

    let uni = Uniform::new(0.0, 1.0);

    for i in 0..n {
        let d_cell = perm1[i] as f64 + uni.sample(rng);
        let p_cell = perm2[i] as f64 + uni.sample(rng);
        let amp_cell = perm3[i] as f64 + uni.sample(rng);
        let lr_cell = perm4[i] as f64 + uni.sample(rng);
        let floor_cell = perm5[i] as f64 + uni.sample(rng);

        let d_norm = d_cell / n as f64;
        let p_norm = p_cell / n as f64;
        let amp_norm = amp_cell / n as f64;
        let lr_norm = lr_cell / n as f64;
        let floor_norm = floor_cell / n as f64;

        let input_density = ranges.0 .0 + d_norm * (ranges.0 .1 - ranges.0 .0);
        let pattern_prob = ranges.1 .0 + p_norm * (ranges.1 .1 - ranges.1 .0);
        let input_amplitude = ranges.2 .0 + amp_norm * (ranges.2 .1 - ranges.2 .0);
        let learning_rate = ranges.3 .0 + lr_norm * (ranges.3 .1 - ranges.3 .0);
        let threshold_floor_factor = ranges.4 .0 + floor_norm * (ranges.4 .1 - ranges.4 .0);

        configs.push(SearchConfig {
            input_density,
            pattern_prob,
            input_amplitude,
            learning_rate,
            threshold_floor_factor,
        });
    }

    configs
}
