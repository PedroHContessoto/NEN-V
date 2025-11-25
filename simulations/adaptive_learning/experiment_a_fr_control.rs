//! Experimento A: Controle Homeostático Puro de Firing Rate
//!
//! ## Objetivo Científico
//!
//! Validar que o controlador PI + homeostase local conseguem regular
//! o firing rate médio da rede para o target biologicamente plausível,
//! **SEM** interferência de aprendizado por reforço ou sono.
//!
//! ## Hipótese
//!
//! O sistema de controle homeostático (PI global + synaptic scaling local)
//! deve levar o FR médio → target_FR com erro relativo < 10% em estado
//! estacionário, demonstrando robustez biológica do mecanismo.
//!
//! ## Protocolo Experimental
//!
//! 1. **Entrada**: Estímulo aleatório moderado (~10% neurônios ativos por step)
//! 2. **Plasticidade**: DESLIGADA (sem STDP, apenas homeostase)
//! 3. **Sono**: DESLIGADO
//! 4. **Reward**: NÃO USADO (adaptive.avg_reward = 0)
//! 5. **Duração**: 100k steps (suficiente para steady-state do PI)
//! 6. **Métrica de sucesso**: |FR_final - FR_target| / FR_target < 0.10

use nenv_visual_sim::autoconfig::{
    AutoConfig, TaskSpec, TaskType, RewardDensity,
    adaptive::{AdaptiveState, monitor_and_adapt},
};
use nenv_visual_sim::network::LearningMode;
use rand::Rng;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  EXPERIMENTO A: Controle Homeostático Puro de Firing Rate");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Configuração da rede via AutoConfig
    // Usa um task spec minimalista apenas para derivar parâmetros
    let task = TaskSpec {
        num_sensors: 6,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);

    println!("📋 Configuração:");
    println!("  • Neurônios: {}", config.architecture.total_neurons);
    println!("  • Target FR: {:.4}", config.params.target_firing_rate);
    println!("  • Plasticidade: DESLIGADA (apenas homeostase)");
    println!("  • Sono: DESLIGADO");
    println!("  • Reward: NÃO USADO\n");

    // Constrói rede
    let mut network = config.build_network()
        .expect("Falha ao construir rede");

    // IMPORTANTE: Desliga STDP para isolar controle homeostático
    network.set_learning_mode(LearningMode::Hebbian); // Hebbian = sem STDP

    // CRÍTICO: Desliga homeostase LOCAL para evitar conflito com PI global
    // No cenário real, ambos coexistem, mas aqui queremos isolar o PI
    for neuron in &mut network.neurons {
        neuron.homeo_eta = 0.0; // Desliga synaptic scaling local
        // Refratário agora é 2 (via params.rs), não precisa mexer aqui
    }

    // Cria estado adaptativo alinhado com config
    let mut adaptive_state = AdaptiveState::with_target_fr(config.params.target_firing_rate);

    println!("🎛️  Parâmetros do Sistema (AJUSTADOS):");
    println!("  • Kp (proporcional): 0.4");
    println!("  • Ki (integral): 0.05");
    println!("  • Integral clamp: 5.0");
    println!("  • Threshold range: [0.001, 5.0] ← AMPLIADO");
    println!("  • Refractory period: 2 steps ← REDUZIDO (era 5)");
    println!("  • Input: 15% neurônios @ 1.5 amplitude\n");

    let target_fr = config.params.target_firing_rate;
    let total_steps = 100_000;
    let report_interval = 10_000;

    let mut rng = rand::thread_rng();

    println!("🚀 Iniciando simulação ({} steps)...\n", total_steps);

    // Métricas de análise
    let mut fr_samples = Vec::new();

    for step in 0..total_steps {
        // Entrada aleatória MODERADAMENTE FORTE (15% neurônios, amplitude 1.5)
        let num_inputs = (network.num_neurons() as f64 * 0.15) as usize;
        let mut inputs = vec![0.0; network.num_neurons()];
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            inputs[idx] = 1.5; // Estímulo mais forte (antes 1.0)
        }

        // Processa step (Network usa update(), não process_inputs/step)
        network.update(&inputs);

        // Monitora e adapta usando controlador PI
        let _adapted = monitor_and_adapt(
            &mut network,
            &mut adaptive_state,
            target_fr,
            step as i64,
            false, // verbose=false (só imprime relatórios)
        );

        // Coleta amostra de FR
        let current_fr = network.num_firing() as f64 / network.num_neurons() as f64;
        fr_samples.push(current_fr);

        // Relatório periódico
        if (step + 1) % report_interval == 0 {
            let recent_fr: f64 = fr_samples.iter()
                .rev()
                .take(1000)
                .sum::<f64>() / 1000.min(fr_samples.len()) as f64;

            let fr_error = ((recent_fr - target_fr).abs() / target_fr) * 100.0;

            println!("📊 Step {:6}k | FR={:.4} | Target={:.4} | Erro={:5.1}% | Adaptações={} | Stable={} steps",
                (step + 1) / 1000,
                recent_fr,
                target_fr,
                fr_error,
                adaptive_state.adaptation_count,
                adaptive_state.stable_steps,
            );
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ANÁLISE DE RESULTADOS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Análise da convergência
    analyze_convergence(&fr_samples, target_fr, &adaptive_state);

    println!("\n🏁 Experimento concluído.\n");
}

fn analyze_convergence(fr_samples: &[f64], target_fr: f64, adaptive_state: &AdaptiveState) {
    // Calcula métricas em diferentes janelas temporais
    let windows = [
        (0, 20_000, "Inicial (0-20k)"),
        (20_000, 50_000, "Transiente (20-50k)"),
        (50_000, 80_000, "Convergência (50-80k)"),
        (80_000, fr_samples.len(), "Steady-State (80-100k)"),
    ];

    println!("📈 Análise por Janela Temporal:\n");

    for (start, end, label) in windows {
        let window_samples: Vec<f64> = fr_samples[start..end].to_vec();
        let avg_fr = window_samples.iter().sum::<f64>() / window_samples.len() as f64;

        let variance = window_samples.iter()
            .map(|x| (x - avg_fr).powi(2))
            .sum::<f64>() / window_samples.len() as f64;
        let std_dev = variance.sqrt();

        let fr_error = ((avg_fr - target_fr).abs() / target_fr) * 100.0;

        let status = if fr_error < 10.0 {
            "✅ EXCELENTE"
        } else if fr_error < 20.0 {
            "⚠️  ACEITÁVEL"
        } else {
            "❌ FORA DO ALVO"
        };

        println!("  {} steps: FR={:.4} ± {:.4} | Erro={:5.1}% | {}",
            label, avg_fr, std_dev, fr_error, status);
    }

    // Veredito final
    let final_window: Vec<f64> = fr_samples.iter()
        .rev()
        .take(20_000)
        .copied()
        .collect();
    let final_avg_fr = final_window.iter().sum::<f64>() / final_window.len() as f64;
    let final_error = ((final_avg_fr - target_fr).abs() / target_fr) * 100.0;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  VEREDITO FINAL");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  • FR final (últimos 20k steps): {:.4}", final_avg_fr);
    println!("  • FR alvo: {:.4}", target_fr);
    println!("  • Erro relativo: {:.2}%", final_error);
    println!("  • Adaptações aplicadas: {}", adaptive_state.adaptation_count);
    println!("  • Steps estáveis finais: {}", adaptive_state.stable_steps);

    if final_error < 10.0 {
        println!("\n  ✅ HIPÓTESE CONFIRMADA");
        println!("     Controlador PI + homeostase regularam FR com erro < 10%.");
        println!("     Sistema demonstra robustez homeostática biologicamente plausível.");
    } else if final_error < 20.0 {
        println!("\n  ⚠️  HIPÓTESE PARCIALMENTE CONFIRMADA");
        println!("     FR próximo ao alvo, mas ainda há viés estacionário.");
        println!("     Sugestão: aumentar Ki ou verificar conflito com homeostase local.");
    } else {
        println!("\n  ❌ HIPÓTESE REJEITADA");
        println!("     FR distante do alvo. Sistema de controle precisa ser reajustado.");
        println!("     Possíveis causas:");
        println!("       - Ganhos do PI mal ajustados (Kp/Ki)");
        println!("       - Homeostase local conflitando com controle global");
        println!("       - Saturação de energia ou thresholds");
    }

    println!();
}
