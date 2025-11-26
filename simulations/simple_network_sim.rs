//! Simple simulation using AutoConfig
//! - Builds the network from a default TaskSpec
//! - Runs 30k steps with sparse noise using recommended density/amplitude
//! - Enables STDP to stress homeostasis and prints basic metrics

use nenv_visual_sim::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_visual_sim::network::LearningMode;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("===============================================================");
    println!("  QUICK SIM: AutoConfig network with STDP + Homeostasis");
    println!("===============================================================\n");

    let task = TaskSpec {
        num_sensors: 10,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().expect("Failed to build network");
    network.set_learning_mode(LearningMode::STDP);

    let input_density = config.params.input.recommended_input_density;
    let input_amplitude = config.params.input.recommended_input_amplitude;
    let target_fr = config.params.target_firing_rate;

    let total_steps = 30_000;
    let report_every = 5_000;

    let mut rng = rand::thread_rng();
    let mut inputs = vec![0.0; network.num_neurons()];

    let mut fr_sum = 0.0;
    let mut fr_max = 0.0;

    let start = Instant::now();

    for step in 0..total_steps {
        inputs.fill(0.0);
        let num_inputs = (network.num_neurons() as f64 * input_density) as usize;
        for _ in 0..num_inputs {
            let idx = rng.gen_range(0..network.num_neurons());
            inputs[idx] = input_amplitude;
        }

        network.update(&inputs);

        let fr = network.num_firing() as f64 / network.num_neurons() as f64;
        fr_sum += fr;
        if fr > fr_max {
            fr_max = fr;
        }

        if step % report_every == 0 {
            let avg_thresh: f64 = network
                .neurons
                .iter()
                .map(|n| n.threshold)
                .sum::<f64>()
                / network.num_neurons() as f64;
            println!(
                "Step {:5} | FR={:.3} (target {:.3}) | Th={:.3}",
                step, fr, target_fr, avg_thresh
            );
        }
    }

    let runtime_ms = start.elapsed().as_millis();
    let fr_mean = fr_sum / total_steps as f64;
    let avg_thresh: f64 = network
        .neurons
        .iter()
        .map(|n| n.threshold)
        .sum::<f64>()
        / network.num_neurons() as f64;

    println!("\nDone in {} ms", runtime_ms);
    println!(
        "FR mean = {:.4} | FR max = {:.4} | Final avg threshold = {:.4}",
        fr_mean, fr_max, avg_thresh
    );
    println!("Target FR = {:.3}", target_fr);
}
