use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity, AdaptiveState};
use nenv_v2::network::LearningMode;
use rand::Rng;

fn main() {
    let task = TaskSpec {
        num_sensors: 6,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().expect("Falha");
    let mut adaptive_state = AdaptiveState::new(config.clone());

    network.set_learning_mode(LearningMode::STDP);

    println!("=== TESTE DE AUTO-REGULAÇÃO ===");
    println!("Target FR: {:.4}\n", config.params.target_firing_rate);

    // Estatísticas
    let mut total_firings = 0u64;
    let mut rng = rand::thread_rng();
    let total_steps = 50_000;

    // Contadores para média móvel
    let mut fr_sum = 0.0;
    let mut fr_count = 0;

    for step in 0..total_steps {
        // Input aleatório (10%)
        let num_inputs = (network.num_neurons() as f64 * 0.1) as usize;
        let mut last_input = vec![0.0; network.num_neurons()];
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            last_input[idx] = 1.0;
        }

        network.update(&last_input);
        adaptive_state.monitor_and_adapt(&mut network);

        let firing = network.num_firing();
        total_firings += firing as u64;

        // Média móvel da FR
        let fr = firing as f64 / network.num_neurons() as f64;
        fr_sum += fr;
        fr_count += 1;

        // Output a cada 5000 steps
        if (step + 1) % 5000 == 0 {
            let avg_fr = fr_sum / fr_count as f64;
            let avg_threshold: f64 = network.neurons.iter().map(|n| n.threshold).sum::<f64>() / network.num_neurons() as f64;
            let avg_weight: f64 = network.neurons.iter()
                .map(|n| n.dendritoma.total_weight() / n.dendritoma.num_inputs() as f64)
                .sum::<f64>() / network.num_neurons() as f64;

            println!("Step {:5}: avg_FR={:.4} | threshold={:.4} | weight={:.4} | total_fires={}",
                step + 1, avg_fr, avg_threshold, avg_weight, total_firings);

            // Reset contadores
            fr_sum = 0.0;
            fr_count = 0;
        }
    }

    // Resultado final
    let overall_fr = total_firings as f64 / (total_steps as f64 * network.num_neurons() as f64);
    println!("\n=== RESULTADO FINAL ===");
    println!("FR Geral: {:.4}", overall_fr);
    println!("Target:   {:.4}", config.params.target_firing_rate);
    println!("Erro:     {:.2}%", (overall_fr - config.params.target_firing_rate).abs() / config.params.target_firing_rate * 100.0);

    // Teste se dispara
    let mut test_input = vec![0.0; network.num_neurons()];
    test_input[0] = 1.0;
    test_input[1] = 1.0;
    let integrated = {
        let mut temp = network.neurons[0].dendritoma.clone();
        temp.integrate(&test_input)
    };
    let modulated = network.neurons[0].glia.modulate(integrated);
    let threshold = network.neurons[0].threshold;
    println!("\nTeste disparo: potencial={:.4} vs threshold={:.4} → {}",
        modulated, threshold, if modulated > threshold { "DISPARA" } else { "NAO DISPARA" });
}
