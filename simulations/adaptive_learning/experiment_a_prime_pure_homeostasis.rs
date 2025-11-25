//! Experimento A-Prime: Auto-Regulação Homeostática PURA
//!
//! ## Objetivo Científico
//!
//! Validar que **APENAS homeostase local** (synaptic scaling + BCM) conseguem
//! regular o firing rate médio da rede para o target biologicamente plausível,
//! **SEM QUALQUER controlador externo (PI)**.
//!
//! ## Diferença do Experimento A (Original)
//!
//! | Aspecto           | A (Engineering)  | A-Prime (Biology)  |
//! |-------------------|------------------|--------------------|
//! | Homeostase Local  | ❌ OFF           | ✅ ON              |
//! | Controlador PI    | ✅ ON            | ❌ OFF             |
//! | Input             | Manual (15%@1.5) | ✅ Derivado        |
//! | Paradigma         | Eng. controle    | Bio. emergente     |
//!
//! ## Hipótese
//!
//! Homeostase local (synaptic scaling guiado por `homeo_eta` + BCM) deve
//! regular FR médio → target_FR com erro < 15%, demonstrando capacidade
//! de **auto-regulação emergente** sem controle centralizado.
//!
//! ## Protocolo Experimental
//!
//! 1. **Entrada**: Derivada automaticamente (via `compute_input_params()`)
//! 2. **Plasticidade**: DESLIGADA (apenas homeostase, sem STDP)
//! 3. **Sono**: DESLIGADO
//! 4. **Reward**: NÃO USADO
//! 5. **Homeostase Local**: ✅ LIGADA (eta derivado do target FR)
//! 6. **PI Global**: ❌ DESLIGADO (monitor_and_adapt NÃO é chamado)
//! 7. **Duração**: 100k steps (suficiente para steady-state homeostático)
//! 8. **Métrica de sucesso**: |FR_final - FR_target| / FR_target < 0.15
//!
//! ## Parâmetros AUTO-DERIVADOS
//!
//! Tudo é calculado a partir do `target_firing_rate`:
//! - `refractory_period = (0.4 / target_fr).ceil()`
//! - `homeo_eta = base_eta * (target_fr / 0.15).sqrt()`
//! - `homeo_interval = base_interval / (target_fr / 0.15)`
//! - `input_density = f(target_fr, connectivity)`
//! - `input_amplitude = f(input_density)`
//!
//! ## Expectativa
//!
//! - **Erro esperado: 10-15%** (menos preciso que PI, mas biologicamente válido)
//! - **Convergência mais lenta** (homeostase local é gradual)
//! - **Sem saturação de parâmetros** (sistema auto-limita)
//! - **Demonstra emergência**: regulação surge da interação local, não controle global

use nenv_visual_sim::autoconfig::{
    AutoConfig, TaskSpec, TaskType, RewardDensity,
};
use nenv_visual_sim::network::LearningMode;
use rand::Rng;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  EXPERIMENTO A-PRIME: Auto-Regulação Homeostática PURA");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Configuração da rede via AutoConfig
    let task = TaskSpec {
        num_sensors: 6,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);

    println!("📋 Configuração AUTO-DERIVADA:");
    println!("  • Neurônios: {}", config.architecture.total_neurons);
    println!("  • Target FR: {:.4}", config.params.target_firing_rate);
    println!("  • Refractory: {} steps (derivado)", config.params.homeostatic.refractory_period);
    println!("  • Homeo eta: {:.5} (derivado)", config.params.homeostatic.homeo_eta);
    println!("  • Homeo interval: {} (derivado)", config.params.homeostatic.homeo_interval);
    println!("  • Input density: {:.1}% (derivado)",
        config.params.input.recommended_input_density * 100.0);
    println!("  • Input amplitude: {:.2} (derivado)",
        config.params.input.recommended_input_amplitude);
    println!("  • Plasticidade: DESLIGADA (apenas homeostase)");
    println!("  • PI Global: DESLIGADO (sem controle externo)");
    println!("  • Sono: DESLIGADO\n");

    // Constrói rede
    let mut network = config.build_network()
        .expect("Falha ao construir rede");

    // IMPORTANTE: Desliga STDP para isolar homeostase
    network.set_learning_mode(LearningMode::Hebbian); // Hebbian = sem STDP

    // CRÍTICO: Homeostase LOCAL está LIGADA (não mexemos em homeo_eta)
    // Os valores já foram derivados automaticamente por compute_homeostatic_params()
    println!("🧬 Homeostase Local:");
    println!("  • Status: ✅ ATIVA");
    println!("  • Eta médio: {:.5}",
        network.neurons.iter().map(|n| n.homeo_eta).sum::<f64>() / network.num_neurons() as f64);
    println!("  • FR alpha: {:.4}", config.params.homeostatic.fr_alpha);
    println!("  • BCM threshold: {:.4}\n", config.params.homeostatic.meta_threshold);

    let target_fr = config.params.target_firing_rate;
    let total_steps = 100_000;
    let report_interval = 10_000;

    let mut rng = rand::thread_rng();

    // Usa parâmetros de input DERIVADOS do AutoConfig
    let input_density = config.params.input.recommended_input_density;
    let input_amplitude = config.params.input.recommended_input_amplitude;

    println!("🚀 Iniciando simulação ({} steps)...\n", total_steps);
    println!("⚠️  SEM CONTROLADOR PI - apenas homeostase local emergente\n");

    // Métricas de análise
    let mut fr_samples = Vec::new();

    for step in 0..total_steps {
        // Entrada usando parâmetros AUTO-DERIVADOS
        let num_inputs = (network.num_neurons() as f64 * input_density) as usize;
        let mut inputs = vec![0.0; network.num_neurons()];
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            inputs[idx] = input_amplitude;
        }

        // Processa step
        network.update(&inputs);

        // CRÍTICO: NÃO chamamos monitor_and_adapt()
        // A rede se auto-regula VIA HOMEOSTASE LOCAL APENAS

        // Coleta amostra de FR
        let current_fr = network.num_firing() as f64 / network.num_neurons() as f64;
        fr_samples.push(current_fr);

        // Relatório periódico (sem métricas de PI)
        if (step + 1) % report_interval == 0 {
            let recent_fr: f64 = fr_samples.iter()
                .rev()
                .take(1000)
                .sum::<f64>() / 1000.min(fr_samples.len()) as f64;

            let fr_error = ((recent_fr - target_fr).abs() / target_fr) * 100.0;

            // Calcula estatísticas de energia
            let avg_energy: f64 = network.neurons.iter()
                .map(|n| n.glia.energy)
                .sum::<f64>() / network.num_neurons() as f64;

            // Calcula estatísticas de threshold
            let avg_threshold: f64 = network.neurons.iter()
                .map(|n| n.threshold)
                .sum::<f64>() / network.num_neurons() as f64;

            println!("📊 Step {:6}k | FR={:.4} | Target={:.4} | Erro={:5.1}% | E={:.2} | Th={:.3}",
                (step + 1) / 1000,
                recent_fr,
                target_fr,
                fr_error,
                avg_energy,
                avg_threshold,
            );
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ANÁLISE DE RESULTADOS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Análise da convergência
    analyze_convergence(&fr_samples, target_fr);

    println!("\n🏁 Experimento concluído.\n");
}

fn analyze_convergence(fr_samples: &[f64], target_fr: f64) {
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

        // Critério mais relaxado para homeostase pura (< 15% vs < 10%)
        let status = if fr_error < 10.0 {
            "✅ EXCELENTE"
        } else if fr_error < 15.0 {
            "✅ BOM (homeostase emergente)"
        } else if fr_error < 25.0 {
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
    println!("  • Paradigma: BIOLÓGICO (homeostase local emergente)");
    println!("  • Controlador PI: NÃO USADO");

    if final_error < 10.0 {
        println!("\n  ✅ HIPÓTESE FORTEMENTE CONFIRMADA");
        println!("     Homeostase local pura regulou FR com erro < 10%.");
        println!("     Sistema demonstra auto-regulação emergente EXCEPCIONAL.");
        println!("     AutoConfig está VERDADEIRAMENTE funcionando como prometido.");
    } else if final_error < 15.0 {
        println!("\n  ✅ HIPÓTESE CONFIRMADA");
        println!("     Homeostase local pura regulou FR com erro < 15%.");
        println!("     Sistema demonstra auto-regulação emergente biologicamente válida.");
        println!("     AutoConfig entrega o que promete: rede se auto-regula.");
    } else if final_error < 25.0 {
        println!("\n  ⚠️  HIPÓTESE PARCIALMENTE CONFIRMADA");
        println!("     FR próximo ao alvo, mas ainda há desvio significativo.");
        println!("     Homeostase local funciona, mas pode precisar ajustes na derivação:");
        println!("       - Verificar se homeo_eta está muito baixo");
        println!("       - Verificar se homeo_interval está muito alto");
        println!("       - Considerar aumentar input_density");
    } else {
        println!("\n  ❌ HIPÓTESE REJEITADA");
        println!("     FR distante do alvo. Homeostase local não convergiu.");
        println!("     Possíveis causas:");
        println!("       - Derivação de homeo_eta incorreta (muito baixo?)");
        println!("       - Input insuficiente para manter atividade");
        println!("       - Refractory period ainda limitando FR máximo");
        println!("       - BCM conflitando com synaptic scaling");
    }

    println!();
}
