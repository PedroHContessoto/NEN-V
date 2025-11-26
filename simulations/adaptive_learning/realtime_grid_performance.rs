//! GridWorld sensorimotor using AutoConfig (NENV)
//! 4 directional sensors -> 4 motors, learned via STDP + homeostasis.

use nenv_visual_sim::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_visual_sim::network::LearningMode;
use rand::Rng;
use std::time::Instant;

struct Agent { x: i32, y: i32 }
struct Food { x: i32, y: i32 }

struct Environment {
    grid_size: i32,
    agent: Agent,
    food: Food,
}

impl Environment {
    fn new(grid_size: i32) -> Self {
        let mut rng = rand::thread_rng();
        let agent = Agent { x: grid_size / 2, y: grid_size / 2 };
        let mut food = Food { x: rng.gen_range(0..grid_size), y: rng.gen_range(0..grid_size) };
        while food.x == agent.x && food.y == agent.y {
            food.x = rng.gen_range(0..grid_size);
            food.y = rng.gen_range(0..grid_size);
        }
        Self { grid_size, agent, food }
    }

    fn get_sensor_inputs(&self) -> [f64; 4] {
        let dx = self.food.x - self.agent.x;
        let dy = self.food.y - self.agent.y;
        let mut sensors = [0.0; 4];
        if dy < 0 { sensors[0] = 2.0; }
        if dy > 0 { sensors[1] = 2.0; }
        if dx < 0 { sensors[2] = 2.0; }
        if dx > 0 { sensors[3] = 2.0; }
        sensors
    }

    fn execute_motor(&mut self, motor_rel_idx: usize) -> bool {
        match motor_rel_idx {
            0 => self.agent.y = (self.agent.y - 1).max(0),
            1 => self.agent.y = (self.agent.y + 1).min(self.grid_size - 1),
            2 => self.agent.x = (self.agent.x - 1).max(0),
            3 => self.agent.x = (self.agent.x + 1).min(self.grid_size - 1),
            _ => {}
        }
        self.agent.x == self.food.x && self.agent.y == self.food.y
    }

    fn respawn_food(&mut self) {
        let mut rng = rand::thread_rng();
        self.food.x = rng.gen_range(0..self.grid_size);
        self.food.y = rng.gen_range(0..self.grid_size);
        while self.food.x == self.agent.x && self.food.y == self.agent.y {
            self.food.x = rng.gen_range(0..self.grid_size);
            self.food.y = rng.gen_range(0..self.grid_size);
        }
    }

    fn expand_grid(&mut self) {
        self.grid_size += 1;
        self.agent.x = self.grid_size / 2;
        self.agent.y = self.grid_size / 2;
        self.respawn_food();
    }
}

struct Metrics {
    total_steps: u64,
    score: u32,
    steps_since_last_meal: u32,
    steps_history: Vec<u32>,
    motor_fires: [u32; 4],
}

impl Metrics {
    fn new() -> Self {
        Self { total_steps: 0, score: 0, steps_since_last_meal: 0, steps_history: Vec::new(), motor_fires: [0; 4] }
    }
    fn record_success(&mut self) {
        self.steps_history.push(self.steps_since_last_meal);
        if self.steps_history.len() > 50 { self.steps_history.remove(0); }
        self.steps_since_last_meal = 0;
        self.score += 1;
    }
    fn average_steps(&self) -> f64 {
        if self.steps_history.is_empty() { 0.0 } else { self.steps_history.iter().sum::<u32>() as f64 / self.steps_history.len() as f64 }
    }
    fn step(&mut self) {
        self.total_steps += 1;
        self.steps_since_last_meal += 1;
    }
}

fn main() {
    println!("NENV GridWorld - STDP + homeostasis (AutoConfig)\n");

    let task = TaskSpec {
        num_sensors: 10,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };
    let config = AutoConfig::from_task(task);
    let mut net = config.build_network().expect("Falha ao construir rede");
    net.set_learning_mode(LearningMode::STDP);

    let sensor_ids: Vec<usize> = config.architecture.sensor_indices.clone().take(4).collect();
    let motor_ids: Vec<usize> = config.architecture.actuator_indices.clone().take(4).collect();
    assert!(sensor_ids.len() == 4 && motor_ids.len() == 4, "Arquitetura insuficiente");

    let input_density = config.params.input.recommended_input_density;
    let input_amplitude = config.params.input.recommended_input_amplitude;

    let mut env = Environment::new(5);
    let mut metrics = Metrics::new();
    let mut rng = rand::thread_rng();
    let mut food_count_level = 0;
    const FOODS_PER_LEVEL: u32 = 10;

    let mut batch_timer = Instant::now();
    const REPORT_INTERVAL: u64 = 1000;

    loop {
        let sensors = env.get_sensor_inputs();
        let mut inputs = vec![0.0; net.num_neurons()];

        let noise_spikes = (net.num_neurons() as f64 * input_density) as usize;
        for _ in 0..noise_spikes {
            let idx = rng.gen_range(0..net.num_neurons());
            inputs[idx] = input_amplitude * 0.5;
        }

        inputs[sensor_ids[0]] = sensors[0] * input_amplitude;
        inputs[sensor_ids[1]] = sensors[1] * input_amplitude;
        inputs[sensor_ids[2]] = sensors[2] * input_amplitude;
        inputs[sensor_ids[3]] = sensors[3] * input_amplitude;

        let exploration_rate = if metrics.score < 20 { 0.15 } else { 0.05 };
        for &mid in &motor_ids {
            if rng.gen_bool(exploration_rate) {
                inputs[mid] += input_amplitude;
            }
        }

        net.update(&inputs);

        let mut best_motor: Option<usize> = None;
        let mut best_score = f64::MIN;
        for (rel_idx, &mid) in motor_ids.iter().enumerate() {
            let neuron = &net.neurons[mid];
            if neuron.is_firing && neuron.recent_firing_rate > best_score {
                best_score = neuron.recent_firing_rate;
                best_motor = Some(rel_idx);
            }
        }

        let mut collected_food = false;
        if let Some(rel_idx) = best_motor {
            collected_food = env.execute_motor(rel_idx);
            metrics.motor_fires[rel_idx] += 1;
        }

        if collected_food {
            net.global_reward_signal = 1.0;
            metrics.record_success();
            food_count_level += 1;
            if food_count_level >= FOODS_PER_LEVEL {
                env.expand_grid();
                food_count_level = 0;
                println!("   Nivel aumentado! Grid: {}x{}", env.grid_size, env.grid_size);
            }
            env.respawn_food();
            for neuron in &mut net.neurons {
                neuron.glia.energy = neuron.glia.max_energy;
            }
        } else {
            net.global_reward_signal = 0.0;
        }

        metrics.step();

        if metrics.total_steps % REPORT_INTERVAL == 0 {
            let elapsed = batch_timer.elapsed();
            let steps_per_sec = REPORT_INTERVAL as f64 / elapsed.as_secs_f64();
            let avg_energy = net.average_energy();
            let avg_steps = metrics.average_steps();
            let total_motor_fires: u32 = metrics.motor_fires.iter().sum();
            let motor_fire_rate = total_motor_fires as f64 / REPORT_INTERVAL as f64;

            println!(
                "Step:{:7} | Score:{:4} | Grid:{:2}x{:2} | Avg:{:5.1} | E:{:4.1}% | MFR:{:.2} | Speed:{:.0} s/s",
                metrics.total_steps,
                metrics.score,
                env.grid_size,
                env.grid_size,
                avg_steps,
                avg_energy,
                motor_fire_rate,
                steps_per_sec
            );

            metrics.motor_fires = [0; 4];
            batch_timer = Instant::now();
        }
    }
}
