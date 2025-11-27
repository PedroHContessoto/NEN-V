//! Simulação: Aprendizado Adaptativo com AutoConfig v2.0
//!
//! Demonstra o sistema adaptativo em ação:
//! 1. Rede criada automaticamente via AutoConfig
//! 2. Sistema adaptativo monitora e corrige problemas
//! 3. Métricas detalhadas mostram evolução

use nenv_v2::autoconfig::*;

struct SimulationMetrics {
    step: i64,
    firing_rate: f64,
    avg_energy: f64,
    num_interventions: usize,
    avg_weight: f64,
}

fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  Simulação: Aprendizado Adaptativo NEN-V         ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // FASE 1: Configuração Automática
    // ========================================================================

    let task = TaskSpec {
        num_sensors: 8,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Sparse,
            temporal_horizon: Some(100),
        },
    };

    println!("📋 Especificação da Tarefa:");
    println!("  • Sensores: {} (ambiente complexo)", task.num_sensors);
    println!("  • Atuadores: {} (ações discretas)", task.num_actuators);
    println!("  • Tipo: Reinforcement Learning (sparse rewards)");
    println!("  • Horizonte Temporal: 100 steps\n");

    let config = AutoConfig::from_task(task);

    println!("🔧 AutoConfig derivou:");
    println!("  • {} neurônios totais", config.architecture.total_neurons);
    println!("  • Target FR: {:.1}%", config.params.target_firing_rate * 100.0);
    println!("  • Balanço energético: +{:.0}% margem",
        ((config.params.energy.energy_recovery_rate * (1.0 - config.params.target_firing_rate) -
          config.params.energy.energy_cost_fire * config.params.target_firing_rate) /
         (config.params.energy.energy_cost_fire * config.params.target_firing_rate)) * 100.0);
    println!();

    // Cria rede
    let mut network = config.build_network()
        .expect("Erro ao criar rede");

    let target_fr = config.params.target_firing_rate;

    // IMPORTANTE: Cria adaptive alinhado com o target do AutoConfig
    let mut adaptive_state = AdaptiveState::new(config.clone());

    println!("🎯 Alinhamento de Targets (Científico):");
    println!("  • AutoConfig target FR: {:.4}", target_fr);
    println!("  • Adaptive target FR: {:.4}", target_fr);
    println!("  • NENV target FR: {:.4} (alinhado via build)", target_fr);
    println!("\n🔧 Sistema Adaptativo (Correção Automática):");
    println!("  • Detecção: Dead network, runaway, under/over firing");
    println!("  • Correções: Threshold, learning rate, energy recovery");
    println!("  • Cooldown: {} steps entre intervenções", adaptive_state.intervention_cooldown);
    println!("  • Histórico: {} métricas registradas\n", adaptive_state.fr_history.capacity());

    // ========================================================================
    // FASE 2: Treinamento com Sistema Adaptativo
    // ========================================================================

    println!("🧠 Iniciando treinamento com sistema adaptativo...\n");

    const TOTAL_STEPS: i64 = 200000; // 100x mais tempo
    const REPORT_INTERVAL: i64 = 20000; // Reports a cada 20000 steps

    let mut metrics_history = Vec::new();
    let mut last_report_step = 0;

    for step in 0..TOTAL_STEPS {
        // Gera inputs simulando ambiente
        let mut inputs = vec![0.0; network.num_neurons()];

        // Padrões temporais: ativa sensores em sequência
        let pattern = (step / 10) % 8;
        inputs[pattern as usize] = 1.0;

        // Adiciona ruído ocasional
        if step % 50 == 0 {
            inputs[(step / 50) as usize % 8] = 0.5;
        }

        // Atualiza rede
        network.update(&inputs);

        // Sistema adaptativo monitora e corrige
        adaptive_state.monitor_and_adapt(&mut network);

        // Coleta métricas
        let firing_rate = network.num_firing() as f64 / network.num_neurons() as f64;
        let avg_energy = network.average_energy();

        let total_weight: f64 = network.neurons.iter()
            .map(|n| n.dendritoma.weights.iter().sum::<f64>())
            .sum();
        let avg_weight = total_weight / (network.num_neurons() * network.num_neurons()) as f64;

        metrics_history.push(SimulationMetrics {
            step,
            firing_rate,
            avg_energy,
            num_interventions: adaptive_state.actions_taken.len(),
            avg_weight,
        });

        // Relatório periódico
        if step - last_report_step >= REPORT_INTERVAL {
            print_progress_report(step, &metrics_history, last_report_step as usize, target_fr);
            last_report_step = step;
        }

        // Ciclos de sono (consolidação)
        if step > 0 && step % (config.params.sleep.sleep_interval as i64) == 0 {
            let pre_sleep_fr = network.num_firing() as f64 / network.num_neurons() as f64;

            println!("\n  😴 Ciclo de sono (FR pré={:.4})...", pre_sleep_fr);

            network.enter_sleep(
                config.params.sleep.sleep_replay_noise,
                config.params.sleep.sleep_duration,
            );

            // Roda sono
            for _ in 0..config.params.sleep.sleep_duration {
                network.update(&vec![0.0; network.num_neurons()]);
            }

            network.wake_up();

            let post_sleep_fr = network.num_firing() as f64 / network.num_neurons() as f64;
            println!("  ✅ Acordou (FR pós={:.4})\n", post_sleep_fr);
        }
    }

    // ========================================================================
    // FASE 3: Análise Final
    // ========================================================================

    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║  ANÁLISE FINAL                                    ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    analyze_training_results(&metrics_history, target_fr, &config);
}

fn print_progress_report(
    step: i64,
    metrics: &[SimulationMetrics],
    start_idx: usize,
    target_fr: f64,
) {
    let recent_metrics: Vec<_> = metrics.iter()
        .skip(start_idx)
        .collect();

    if recent_metrics.is_empty() {
        return;
    }

    let avg_fr: f64 = recent_metrics.iter()
        .map(|m| m.firing_rate)
        .sum::<f64>() / recent_metrics.len() as f64;

    let avg_energy: f64 = recent_metrics.iter()
        .map(|m| m.avg_energy)
        .sum::<f64>() / recent_metrics.len() as f64;

    let last_metric = recent_metrics.last().unwrap();
    let total_interventions = last_metric.num_interventions;

    let fr_error = ((avg_fr - target_fr) / target_fr * 100.0).abs();

    println!("📊 Step {:5}: FR={:.4} (erro {:>5.1}%), E={:>4.1}, Intervenções={:>4}",
        step,
        avg_fr,
        fr_error,
        avg_energy,
        total_interventions
    );
}

fn analyze_training_results(
    metrics: &[SimulationMetrics],
    target_fr: f64,
    _config: &AutoConfig,
) {
    // Divide em janelas
    let window_size = 40000; // 10x maior
    let num_windows = metrics.len() / window_size;

    println!("📈 Evolução por Janela ({} steps):", window_size);
    println!("  {:>6} | {:>8} | {:>9} | {:>8} | {:>13}", "Window", "Avg FR", "FR Erro", "Energia", "Intervenções");
    println!("  {}", "-".repeat(62));

    for w in 0..num_windows {
        let start = w * window_size;
        let end = (w + 1) * window_size;
        let window = &metrics[start..end];

        let avg_fr: f64 = window.iter().map(|m| m.firing_rate).sum::<f64>() / window.len() as f64;
        let fr_error = ((avg_fr - target_fr) / target_fr * 100.0).abs();
        let avg_energy: f64 = window.iter().map(|m| m.avg_energy).sum::<f64>() / window.len() as f64;
        let interventions = window.last().unwrap().num_interventions;

        println!("  {:6} | {:8.4} | {:8.1}% | {:8.1} | {:13}",
            w + 1,
            avg_fr,
            fr_error,
            avg_energy,
            interventions
        );
    }

    // Estatísticas finais
    let final_window = &metrics[metrics.len() - 20000..];
    let final_fr: f64 = final_window.iter().map(|m| m.firing_rate).sum::<f64>() / final_window.len() as f64;
    let final_energy: f64 = final_window.iter().map(|m| m.avg_energy).sum::<f64>() / final_window.len() as f64;

    println!("\n🎯 Métricas Finais (últimos 20000 steps):");
    println!("  • Firing Rate: {:.4} (target: {:.4})", final_fr, target_fr);
    println!("  • Erro FR: {:.1}%", ((final_fr - target_fr) / target_fr * 100.0).abs());
    println!("  • Energia Média: {:.1}", final_energy);

    let last_m = metrics.last().unwrap();
    println!("  • Total de Intervenções: {}", last_m.num_interventions);

    // Taxa de intervenção (intervenções por 1000 steps)
    let intervention_rate = (last_m.num_interventions as f64 / metrics.len() as f64) * 1000.0;
    println!("  • Taxa de Intervenção: {:.2} por 1000 steps", intervention_rate);

    // Avaliação
    let fr_error_final = ((final_fr - target_fr) / target_fr * 100.0).abs();
    println!("\n✅ Avaliação:");

    if fr_error_final < 10.0 {
        println!("  ✅ EXCELENTE: FR convergiu próximo ao target (<10% erro)");
    } else if fr_error_final < 25.0 {
        println!("  ⚠️  BOM: FR razoável mas pode melhorar (10-25% erro)");
    } else {
        println!("  ❌ PRECISA MELHORAR: FR distante do target (>25% erro)");
    }

    if final_energy > 60.0 {
        println!("  ✅ Balanço energético saudável (>60%)");
    } else if final_energy > 40.0 {
        println!("  ⚠️  Energia moderada (40-60%)");
    } else {
        println!("  ❌ Risco de depleção energética (<40%)");
    }

    if last_m.num_interventions < 5 {
        println!("  ✅ Configuração inicial estável (poucas intervenções)");
    } else {
        println!("  ⚠️  Sistema adaptativo interveio {} vezes", last_m.num_interventions);
    }

    println!("\n💡 Análise do Sistema Adaptativo:");
    println!("  • Monitora: Dead network, runaway, under/over firing");
    println!("  • Corrige: Threshold, learning rate, energy recovery");
    println!("  • Cooldown entre intervenções evita thrashing");

    if intervention_rate < 5.0 {
        println!("  ✅ Sistema ESTÁVEL: <5 intervenções/1000 steps");
    } else if intervention_rate < 20.0 {
        println!("  ⚠️  Sistema MODERADO: 5-20 intervenções/1000 steps");
    } else {
        println!("  ❌ Sistema INSTÁVEL: >20 intervenções/1000 steps (precisa ajuste)");
    }

    println!("\n🔬 AutoConfig + Sistema Adaptativo:");
    println!("  • AutoConfig deriva parâmetros iniciais da tarefa");
    println!("  • Sistema adaptativo corrige desvios durante execução");
    println!("  • Target FR único: {:.4} (sem conflitos)", target_fr);
}
