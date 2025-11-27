//! Teste para diagnosticar propagação de inputs externos

use nenv_v2::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};

fn main() {
    println!("=== TESTE: Propagação de Inputs Externos ===\n");

    // Cria uma rede pequena
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

    println!("Rede criada:");
    println!("  Total de neurônios: {}", network.num_neurons());
    println!("  Sensores: {:?}", config.architecture.sensor_indices);
    println!("  Hidden: {:?}", config.architecture.hidden_indices);
    println!("  Actuators: {:?}\n", config.architecture.actuator_indices);

    // Prepara um input simples: ativa APENAS o neurônio 0
    let mut inputs = vec![0.0; network.num_neurons()];
    inputs[0] = 1.0;

    println!("Input: neurônio 0 recebe 1.0, todos os outros 0.0\n");

    // Antes do update: vamos inspecionar o neurônio 0
    let neuron_0 = &network.neurons[0];
    println!("ANTES DO UPDATE:");
    println!("  Neurônio 0:");
    println!("    Threshold: {:.6}", neuron_0.threshold);
    println!("    Número de pesos: {}", neuron_0.dendritoma.weights.len());
    println!("    Peso[0] (auto-conexão): {:.6}", neuron_0.dendritoma.weights[0]);
    println!("    Soma de todos os pesos: {:.6}", neuron_0.dendritoma.total_weight());
    println!("    Energia: {:.1}", neuron_0.glia.energy);

    // Executa um update
    network.update(&inputs);

    // Depois do update
    let neuron_0 = &network.neurons[0];
    println!("\nDEPOIS DO UPDATE:");
    println!("  Neurônio 0:");
    println!("    Is firing: {}", neuron_0.is_firing);
    println!("    Output signal: {:.1}", neuron_0.output_signal);
    println!("    Threshold: {:.6}", neuron_0.threshold);
    println!("    Energia: {:.1}", neuron_0.glia.energy);

    // Vamos tentar calcular manualmente o potencial esperado
    println!("\nCÁLCULO MANUAL:");
    println!("  Input externo[0] = 1.0");
    println!("  Peso[0] (auto-conexão) = {:.6}", neuron_0.dendritoma.weights[0]);
    println!("  Potencial esperado ≈ 1.0 × {:.6} = {:.6}",
        neuron_0.dendritoma.weights[0],
        1.0 * neuron_0.dendritoma.weights[0]);
    println!("  Threshold = {:.6}", neuron_0.threshold);
    println!("  Deveria disparar: {}",
        (1.0 * neuron_0.dendritoma.weights[0]) > neuron_0.threshold);

    // Vamos verificar a conectividade
    println!("\nCONECTIVIDADE:");
    println!("  Matriz de conectividade para neurônio 0:");
    let connections: Vec<usize> = network.connectivity_matrix[0]
        .iter()
        .enumerate()
        .filter(|(_, &val)| val == 1)
        .map(|(idx, _)| idx)
        .collect();
    println!("    Conectado aos neurônios: {:?}", connections);
    println!("    Total de conexões: {}", connections.len());
    println!("    Tem auto-conexão (0→0): {}",
        network.connectivity_matrix[0][0] == 1);

    // Testa com múltiplos steps
    println!("\n=== TESTE COM 10 STEPS ===");
    for step in 0..10 {
        inputs = vec![0.0; network.num_neurons()];
        inputs[0] = 1.0; // Sempre ativa neurônio 0

        network.update(&inputs);

        let n0 = &network.neurons[0];
        println!("Step {}: firing={}, threshold={:.6}, energy={:.1}",
            step, n0.is_firing, n0.threshold, n0.glia.energy);
    }

    println!("\n=== FIM DO TESTE ===");
}
