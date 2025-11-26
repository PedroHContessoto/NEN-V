//! Módulo que implementa a rede NEN-V
//!
//! A Network orquestra a simulação, gerindo os neurônios e suas conexões.
//!
//! ## Novidades v2.0
//!
//! - **Competição Lateral**: Winner-take-all suave para especialização
//! - **Reward Propagation**: Propaga reward para eligibility traces
//! - **Métricas de Seletividade**: Monitora gap entre padrão e ruído

use crate::nenv::{NeuronType, SpikeOrigin, NENV};
use crate::neuromodulation::{NeuromodulationSystem, NeuromodulatorType};

/// Tipo de topologia de rede
#[derive(Debug, Clone, Copy)]
pub enum ConnectivityType {
    /// Todos os neurônios conectados a todos
    FullyConnected,
    /// Grade 2D com vizinhança de Moore (8 vizinhos)
    Grid2D,
    /// Neurônios isolados (sem conexões)
    Isolated,
}

/// Modo de aprendizado sináptico
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningMode {
    /// Aprendizado Hebbiano clássico
    Hebbian,
    /// STDP (Spike-Timing-Dependent Plasticity)
    STDP,
}

/// Estado global da rede
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkState {
    /// Estado normal: recebe inputs externos
    Awake,
    /// Estado de consolidação: replay espontÃ¢neo
    Sleep {
        replay_noise: f64,
        consolidation_steps: usize,
        max_sleep_duration: usize,
    },
}

/// Estrutura principal da rede NEN-V
#[derive(Debug)]
pub struct Network {
    /// Vetor de todos os neurônios
    pub neurons: Vec<NENV>,

    /// Matriz de conectividade
    pub connectivity_matrix: Vec<Vec<u8>>,

    /// Passo de tempo atual
    pub current_time_step: i64,

    /// Dimensões da grade
    pub grid_width: usize,
    pub grid_height: usize,

    /// Nível de alerta global
    pub alert_level: f64,

    /// Taxa de decaimento do alert_level
    pub alert_decay_rate: f64,

    /// Novidade média atual
    current_avg_novelty: f64,

    /// Threshold de novidade para alerta
    novelty_alert_threshold: f64,

    /// Sensibilidade do alerta
    alert_sensitivity: f64,

    /// Modo de aprendizado
    pub learning_mode: LearningMode,

    /// Buffer de spikes para STDP
    spike_buffer: Vec<(i64, usize, SpikeOrigin)>,

    /// Janela temporal STDP
    pub stdp_window: i64,

    /// Estado atual
    pub state: NetworkState,

    /// Fator de aprendizado durante sono
    pub sleep_learning_rate_factor: f64,

    /// Sinal de reward global
    pub global_reward_signal: f64,

    // ========== COMPETIÃ‡ÃƒO LATERAL (NOVO v2.0) ==========
    /// Habilita competição lateral
    pub lateral_competition_enabled: bool,

    /// Força da competição [0.0, 1.0]
    pub competition_strength: f64,

    /// Intervalo para aplicar competição
    pub competition_interval: i64,

    /// Contador de competição
    competition_counter: i64,

    // ========== NEUROMODULAÃ‡ÃƒO (NOVO v2.0) ==========
    /// Sistema de neuromodulação
    pub neuromodulation: NeuromodulationSystem,

    /// Habilita neuromodulação
    pub neuromodulation_enabled: bool,

    // ========== MÃ‰TRICAS DE APRENDIZADO (NOVO v2.0) ==========
    /// Ãndices dos neurônios "sensores" (recebem input externo)
    pub sensor_indices: Vec<usize>,

    /// Ãndices dos neurônios "hidden"
    pub hidden_indices: Vec<usize>,

    /// Ãndices dos neurônios "atuadores"
    pub actuator_indices: Vec<usize>,

    /// Histórico de firing rate médio
    fr_history: Vec<f64>,

    /// Histórico de reward
    reward_history: Vec<f64>,
}

impl Network {
    /// Cria uma nova rede
    pub fn new(
        num_neurons: usize,
        connectivity_type: ConnectivityType,
        inhibitory_ratio: f64,
        initial_threshold: f64,
    ) -> Self {
        let (grid_width, grid_height) = match connectivity_type {
            ConnectivityType::Grid2D => {
                let side = (num_neurons as f64).sqrt().ceil() as usize;
                (side, side)
            }
            ConnectivityType::FullyConnected | ConnectivityType::Isolated => (0, 0),
        };

        let connectivity_matrix =
            Self::generate_connectivity(num_neurons, connectivity_type, grid_width);

        let mut neurons = Vec::with_capacity(num_neurons);
        let num_inhibitory = (num_neurons as f64 * inhibitory_ratio).floor() as usize;

        for i in 0..num_neurons {
            let neuron_type = if i < num_inhibitory {
                NeuronType::Inhibitory
            } else {
                NeuronType::Excitatory
            };

            let mut neuron = NENV::new(i, num_neurons, initial_threshold, neuron_type);

            // TABULA RASA: Pesos iniciais pequenos e aleatórios
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for w in &mut neuron.dendritoma.weights {
                *w = rng.gen_range(0.04..0.06);
            }

            // Pesos inibitórios iniciais
            for source_id in 0..num_neurons {
                if source_id < neuron.dendritoma.weights.len() {
                    if connectivity_matrix[i][source_id] == 1 {
                        if source_id < num_inhibitory && source_id != i {
                            neuron.dendritoma.weights[source_id] = 0.3;
                        }
                    }
                }
            }
            neurons.push(neuron);
        }

        Self {
            neurons,
            connectivity_matrix,
            current_time_step: 0,
            grid_width,
            grid_height,
            alert_level: 0.0,
            alert_decay_rate: 0.05,
            current_avg_novelty: 0.0,
            novelty_alert_threshold: 0.05,
            alert_sensitivity: 1.0,
            learning_mode: LearningMode::Hebbian,
            spike_buffer: Vec::new(),
            stdp_window: 50, // AUMENTADO v2.0 (era 20)
            state: NetworkState::Awake,
            sleep_learning_rate_factor: 0.0,
            global_reward_signal: 0.0,

            // Competição lateral v2.0
            lateral_competition_enabled: true,
            competition_strength: 0.3,
            competition_interval: 10,
            competition_counter: 0,

            // Neuromodulação v2.0
            neuromodulation: NeuromodulationSystem::new(),
            neuromodulation_enabled: true,

            // Métricas v2.0
            sensor_indices: Vec::new(),
            hidden_indices: Vec::new(),
            actuator_indices: Vec::new(),
            fr_history: Vec::new(),
            reward_history: Vec::new(),
        }
    }

    /// Define o modo de aprendizado
    pub fn set_learning_mode(&mut self, mode: LearningMode) {
        self.learning_mode = mode;
    }

    /// Define os índices das camadas
    pub fn set_layer_indices(
        &mut self,
        sensors: Vec<usize>,
        hidden: Vec<usize>,
        actuators: Vec<usize>,
    ) {
        self.sensor_indices = sensors;
        self.hidden_indices = hidden;
        self.actuator_indices = actuators;
    }

    /// Gera matriz de conectividade
    fn generate_connectivity(
        num_neurons: usize,
        connectivity_type: ConnectivityType,
        grid_width: usize,
    ) -> Vec<Vec<u8>> {
        match connectivity_type {
            ConnectivityType::FullyConnected => {
                vec![vec![1; num_neurons]; num_neurons]
            }
            ConnectivityType::Grid2D => {
                Self::generate_2d_grid_connectivity(num_neurons, grid_width)
            }
            ConnectivityType::Isolated => {
                vec![vec![0; num_neurons]; num_neurons]
            }
        }
    }

    fn generate_2d_grid_connectivity(num_neurons: usize, width: usize) -> Vec<Vec<u8>> {
        let mut matrix = vec![vec![0; num_neurons]; num_neurons];

        for i in 0..num_neurons {
            let (row, col) = (i / width, i % width);

            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }

                    let new_row = row as i32 + dr;
                    let new_col = col as i32 + dc;

                    if new_row >= 0
                        && new_row < width as i32
                        && new_col >= 0
                        && new_col < width as i32
                    {
                        let j = (new_row as usize) * width + (new_col as usize);
                        if j < num_neurons {
                            matrix[i][j] = 1;
                        }
                    }
                }
            }
        }

        matrix
    }

    /// Coleta inputs para um neurônio
    fn gather_inputs(
        &self,
        neuron_idx: usize,
        all_outputs: &[f64],
        external_inputs: &[f64],
    ) -> Vec<f64> {
        let mut inputs = vec![0.0; self.neurons.len()];

        for j in 0..self.neurons.len() {
            if self.connectivity_matrix[neuron_idx][j] == 1 {
                inputs[j] = all_outputs[j];
            }
        }

        if neuron_idx < external_inputs.len() {
            inputs[neuron_idx] += external_inputs[neuron_idx];
        }

        inputs
    }

    // ========================================================================
    // COMPETIÃ‡ÃƒO LATERAL (NOVO v2.0)
    // ========================================================================

    /// Aplica competição lateral suave (soft winner-take-all)
    ///
    /// Neurônios mais ativos suprimem parcialmente os menos ativos,
    /// promovendo especialização e seletividade.
    pub fn apply_lateral_competition(&mut self, layer_indices: &[usize]) {
        if layer_indices.is_empty() {
            return;
        }

        // Coleta ativações da camada
        let activations: Vec<f64> = layer_indices
            .iter()
            .map(|&i| self.neurons[i].recent_firing_rate)
            .collect();

        // Encontra máximo e média
        let max_activation = activations.iter().cloned().fold(0.0, f64::max);
        let mean_activation: f64 = activations.iter().sum::<f64>() / activations.len() as f64;

        if max_activation < 0.01 {
            return; // Camada inativa
        }

        // Aplica supressão aos "perdedores"
        for (local_idx, &global_idx) in layer_indices.iter().enumerate() {
            let relative_activation = activations[local_idx] / max_activation;

            // Supressão proporcional Ã  distÃ¢ncia do máximo
            let suppression = (1.0 - relative_activation) * self.competition_strength;

            // Reduz plasticidade dos perdedores
            let new_gain = 1.0 - suppression;
            self.neurons[global_idx].dendritoma.set_plasticity_gain(new_gain.max(0.1));

            // Aumenta levemente threshold dos muito ativos (homeostase competitiva)
            if activations[local_idx] > mean_activation * 1.5 {
                self.neurons[global_idx].threshold *= 1.001;
            }
        }
    }

    /// Aplica competição em todas as camadas definidas
    fn apply_all_lateral_competition(&mut self) {
        if !self.lateral_competition_enabled {
            return;
        }

        self.competition_counter += 1;
        if self.competition_counter < self.competition_interval {
            return;
        }
        self.competition_counter = 0;

        // Aplica na camada hidden
        if !self.hidden_indices.is_empty() {
            let indices = self.hidden_indices.clone();
            self.apply_lateral_competition(&indices);
        }

        // Aplica na camada de atuadores
        if !self.actuator_indices.is_empty() {
            let indices = self.actuator_indices.clone();
            self.apply_lateral_competition(&indices);
        }
    }

    // ========================================================================
    // REWARD PROPAGATION (NOVO v2.0)
    // ========================================================================

    /// Propaga sinal de reward para todos os neurônios
    ///
    /// Usa eligibility traces para atribuir crédito Ã s sinapses
    /// que contribuíram para o resultado.
    pub fn propagate_reward(&mut self, reward: f64) {
        self.global_reward_signal = reward;

        // Processa pelo sistema de neuromodulação
        if self.neuromodulation_enabled {
            self.neuromodulation.process_reward(reward);
        }

        // Aplica reward a todos os neurônios via eligibility traces
        let plasticity_mod = if self.neuromodulation_enabled {
            self.neuromodulation.plasticity_modulation()
        } else {
            1.0
        };

        for neuron in &mut self.neurons {
            neuron.dendritoma.apply_reward_modulated_learning(reward, plasticity_mod);
        }

        // Armazena no histórico
        self.reward_history.push(reward);
        if self.reward_history.len() > 1000 {
            self.reward_history.remove(0);
        }
    }

    // ========================================================================
    // UPDATE PRINCIPAL
    // ========================================================================

    /// Executa um passo de atualização da rede
    pub fn update(&mut self, external_inputs: &[f64]) {
        self.current_time_step += 1;

        let has_external_input = external_inputs.iter().any(|&x| x.abs() > 1e-6);

        // Verifica se deve acordar
        if let NetworkState::Sleep { consolidation_steps, max_sleep_duration, .. } = self.state {
            if consolidation_steps >= max_sleep_duration {
                self.wake_up();
            }
        }

        // Atualiza contadores de sono
        if let NetworkState::Sleep { ref mut consolidation_steps, .. } = self.state {
            *consolidation_steps += 1;

            for neuron in &mut self.neurons {
                neuron.dendritoma.consolidate_memory_tagged(0.01);
            }
        }

        // Atualiza alert_level
        self.update_alert_level();

        // Atualiza neuromodulação
        if self.neuromodulation_enabled {
            self.neuromodulation.update();
        }

        // Coleta saídas anteriores
        let all_neuron_outputs: Vec<f64> = self.neurons.iter().map(|n| n.output_signal).collect();

        // Precompute inputs to avoid mutable/immutable borrow conflicts
        let saved_awake: Vec<f64> = self.neurons.iter().map(|n| n.saved_awake_activity).collect();
        let mut gathered_inputs = Vec::with_capacity(self.neurons.len());
        for idx in 0..self.neurons.len() {
            let inputs = match self.state {
                NetworkState::Awake => {
                    self.gather_inputs(idx, &all_neuron_outputs, external_inputs)
                },
                NetworkState::Sleep { replay_noise, .. } => {
                    let mut sleep_inputs =
                        self.gather_inputs(idx, &all_neuron_outputs, &vec![0.0; external_inputs.len()]);

                    let noise_prob = replay_noise + (saved_awake[idx] * 1.0);

                    if rand::random::<f64>() < noise_prob {
                        let connected_active: Vec<(usize, f64)> = self.connectivity_matrix[idx]
                            .iter()
                            .enumerate()
                            .filter(|&(_, &connected)| connected == 1)
                            .map(|(j, _)| (j, saved_awake[j]))
                            .filter(|(_, activity)| *activity > 0.01)
                            .collect();

                        if !connected_active.is_empty() {
                            let best_input = connected_active
                                .iter()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(idx, _)| *idx)
                                .unwrap();

                            sleep_inputs[best_input] += 1.5;
                        } else {
                            let input_idx = rand::random::<usize>() % sleep_inputs.len();
                            sleep_inputs[input_idx] += 1.5;
                        }
                    }

                    sleep_inputs
                }
            };
            gathered_inputs.push(inputs);
        }

        // Temporary vectors
        let mut integrated_potentials = Vec::with_capacity(self.neurons.len());
        let mut modulated_potentials = Vec::with_capacity(self.neurons.len());

        // Phase 1-2: potentials using precomputed inputs
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let inputs = &gathered_inputs[idx];

            // Integration with STP
            let integrated = neuron.dendritoma.integrate(inputs);
            let modulated = neuron.glia.modulate(integrated);

            integrated_potentials.push(integrated);
            modulated_potentials.push(modulated);
        }

        // Fase 3: Decisão de disparo
        for ((neuron, &modulated_potential), inputs) in
            self.neurons.iter_mut()
                .zip(modulated_potentials.iter())
                .zip(gathered_inputs.iter())
        {
            let has_external_input = inputs.iter().any(|&inp| inp > 0.5);
            neuron.decide_to_fire(modulated_potential, self.current_time_step, has_external_input);
        }

        // Fase 4: Aprendizado
        let mut total_novelty = 0.0;

        let neuron_types: Vec<NeuronType> = self.neurons.iter()
            .map(|n| n.neuron_type)
            .collect();
        let neuron_firing_states: Vec<bool> = self.neurons.iter()
            .map(|n| n.is_firing)
            .collect();

        // Obtém modulação de plasticidade do sistema de neuromodulação
        let nm_plasticity_mod = if self.neuromodulation_enabled {
            self.neuromodulation.plasticity_modulation()
        } else {
            1.0
        };

        for (idx, (neuron, inputs)) in self.neurons.iter_mut().zip(gathered_inputs.iter()).enumerate() {
            let novelty = neuron.compute_novelty(inputs);
            total_novelty += novelty;
            neuron.update_priority(novelty, 2.0);

            // Atualiza eligibility traces
            let pre_active: Vec<f64> = inputs.clone();
            neuron.dendritoma.update_eligibility_traces(&pre_active, neuron.is_firing);

            // Atualiza STP
            neuron.dendritoma.update_stp();

            let mut learning_happened = false;

            if neuron.is_firing {
                // Aplica modulação de plasticidade da neuromodulação
                let current_gain = neuron.dendritoma.plasticity_gain;
                neuron.dendritoma.set_plasticity_gain(current_gain * nm_plasticity_mod);

                match self.learning_mode {
                    LearningMode::Hebbian => {
                        neuron.dendritoma.apply_learning(inputs);
                        learning_happened = true;
                    }
                    LearningMode::STDP => {
                        for &(spike_time, spike_neuron_id, pre_origin) in &self.spike_buffer {
                            if spike_neuron_id == idx { continue; }

                            let post_origin = neuron.spike_origin;

                            let should_apply_stdp = !matches!(
                                (pre_origin, post_origin),
                                (SpikeOrigin::Feedback, _) | (_, SpikeOrigin::Feedback)
                            );

                            if should_apply_stdp {
                                let delta_t = self.current_time_step - spike_time;

                                let pre_neuron_type = neuron_types[spike_neuron_id];
                                let pre_neuron_firing = neuron_firing_states[spike_neuron_id];

                                match pre_neuron_type {
                                    NeuronType::Excitatory => {
                                        neuron.dendritoma.apply_stdp_pair(spike_neuron_id, delta_t, self.global_reward_signal);
                                        learning_happened = true;
                                    },
                                    NeuronType::Inhibitory => {
                                        neuron.dendritoma.apply_istdp(
                                            spike_neuron_id,
                                            neuron.recent_firing_rate,
                                            pre_neuron_firing,
                                            neuron.is_firing,
                                        );
                                        learning_happened = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // Restaura ganho de plasticidade
                neuron.dendritoma.set_plasticity_gain(current_gain);
            }

            if !learning_happened {
                neuron.dendritoma.apply_weight_maintenance(neuron.recent_firing_rate);
            }

            neuron.dendritoma.decay_tags();

            neuron.glia.update_state(neuron.is_firing);
            neuron.update_memory(inputs);

            let firing_value = if neuron.is_firing { 1.0 } else { 0.0 };
            neuron.recent_firing_rate = 0.99 * neuron.recent_firing_rate + 0.01 * firing_value;
            neuron.update_meta_threshold(neuron.is_firing);
            neuron.apply_homeostatic_plasticity(self.current_time_step, has_external_input);
        }

        // Atualiza buffer STDP
        if self.learning_mode == LearningMode::STDP {
            for (idx, neuron) in self.neurons.iter().enumerate() {
                if neuron.is_firing {
                    self.spike_buffer.push((self.current_time_step, idx, neuron.spike_origin));
                }
            }
            self.spike_buffer.retain(|(time, _, _)| self.current_time_step - time <= self.stdp_window);
        }

        // Fase 5: Novelty-Alert
        self.current_avg_novelty = total_novelty / self.neurons.len() as f64;

        if self.current_avg_novelty > self.novelty_alert_threshold {
            let alert_boost = (self.current_avg_novelty - self.novelty_alert_threshold) * self.alert_sensitivity;
            self.boost_alert_level(alert_boost);

            // Processa novidade na neuromodulação
            if self.neuromodulation_enabled {
                self.neuromodulation.process_novelty(self.current_avg_novelty);
            }
        }

        // Fase 6: Competição Lateral
        self.apply_all_lateral_competition();

        // Armazena FR no histórico
        let current_fr = self.num_firing() as f64 / self.num_neurons() as f64;
        self.fr_history.push(current_fr);
        if self.fr_history.len() > 1000 {
            self.fr_history.remove(0);
        }
    }

    // ========================================================================
    // MÃ‰TRICAS E ESTATÃSTICAS
    // ========================================================================

    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn num_firing(&self) -> usize {
        self.neurons.iter().filter(|n| n.is_firing).count()
    }

    pub fn average_energy(&self) -> f64 {
        let total_energy: f64 = self.neurons.iter().map(|n| n.glia.energy).sum();
        total_energy / self.neurons.len() as f64
    }

    pub fn get_firing_states(&self) -> Vec<bool> {
        self.neurons.iter().map(|n| n.is_firing).collect()
    }

    pub fn get_energy_levels(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.glia.energy).collect()
    }

    /// Calcula gap de pesos entre um conjunto de sinapses vs resto
    ///
    /// Ãštil para medir seletividade (padrão vs ruído)
    pub fn compute_weight_gap(&self, pattern_sources: &[usize], target_neurons: &[usize]) -> f64 {
        let mut pattern_weight_sum = 0.0;
        let mut pattern_count = 0;
        let mut other_weight_sum = 0.0;
        let mut other_count = 0;

        for &target_idx in target_neurons {
            if target_idx >= self.neurons.len() { continue; }

            for (source_idx, &weight) in self.neurons[target_idx].dendritoma.weights.iter().enumerate() {
                if self.connectivity_matrix[target_idx][source_idx] == 1 {
                    if pattern_sources.contains(&source_idx) {
                        pattern_weight_sum += weight;
                        pattern_count += 1;
                    } else {
                        other_weight_sum += weight;
                        other_count += 1;
                    }
                }
            }
        }

        let avg_pattern = if pattern_count > 0 { pattern_weight_sum / pattern_count as f64 } else { 0.0 };
        let avg_other = if other_count > 0 { other_weight_sum / other_count as f64 } else { 0.0 };

        avg_pattern - avg_other
    }

    /// Retorna média de firing rate dos últimos N steps
    pub fn average_recent_fr(&self, n: usize) -> f64 {
        if self.fr_history.is_empty() { return 0.0; }

        let count = n.min(self.fr_history.len());
        self.fr_history.iter().rev().take(count).sum::<f64>() / count as f64
    }

    /// Retorna estatísticas completas da rede
    pub fn get_stats(&self) -> NetworkStats {
        let firing_rate = self.num_firing() as f64 / self.num_neurons() as f64;
        let avg_energy = self.average_energy();

        let avg_threshold: f64 = self.neurons.iter()
            .map(|n| n.threshold)
            .sum::<f64>() / self.num_neurons() as f64;

        let avg_eligibility: f64 = self.neurons.iter()
            .map(|n| n.dendritoma.total_eligibility())
            .sum::<f64>() / self.num_neurons() as f64;

        let nm_stats = self.neuromodulation.get_stats();

        NetworkStats {
            time_step: self.current_time_step,
            firing_rate,
            avg_energy,
            avg_threshold,
            avg_novelty: self.current_avg_novelty,
            alert_level: self.alert_level,
            avg_eligibility,
            dopamine: nm_stats.dopamine,
            norepinephrine: nm_stats.norepinephrine,
            state: self.state,
        }
    }

    // ========================================================================
    // CONTROLE DE ESTADO
    // ========================================================================

    pub fn index_to_coords(&self, index: usize) -> Option<(usize, usize)> {
        if self.grid_width > 0 && index < self.neurons.len() {
            Some((index / self.grid_width, index % self.grid_width))
        } else {
            None
        }
    }

    pub fn coords_to_index(&self, row: usize, col: usize) -> Option<usize> {
        if self.grid_width > 0 && row < self.grid_height && col < self.grid_width {
            let index = row * self.grid_width + col;
            if index < self.neurons.len() {
                return Some(index);
            }
        }
        None
    }

    pub fn set_alert_level(&mut self, level: f64) {
        self.alert_level = level.clamp(0.0, 1.0);

        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    pub fn boost_alert_level(&mut self, boost: f64) {
        self.alert_level = (self.alert_level + boost).min(1.0);

        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    fn update_alert_level(&mut self) {
        self.alert_level *= 1.0 - self.alert_decay_rate;

        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    pub fn average_novelty(&self) -> f64 {
        self.current_avg_novelty
    }

    pub fn set_novelty_alert_params(&mut self, threshold: f64, sensitivity: f64) {
        self.novelty_alert_threshold = threshold.max(0.0);
        self.alert_sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    pub fn enter_sleep(&mut self, replay_noise: f64, duration: usize) {
        if let NetworkState::Awake = self.state {
            self.state = NetworkState::Sleep {
                replay_noise,
                consolidation_steps: 0,
                max_sleep_duration: duration,
            };

            for neuron in &mut self.neurons {
                neuron.saved_awake_activity = neuron.recent_firing_rate;
                neuron.dendritoma.scale_learning_rate(self.sleep_learning_rate_factor);
                neuron.enter_sleep_mode();
            }

            self.alert_level = 0.3;
        }
    }

    pub fn wake_up(&mut self) {
        if let NetworkState::Sleep { .. } = self.state {
            self.state = NetworkState::Awake;

            if self.sleep_learning_rate_factor > 0.0 {
                for neuron in &mut self.neurons {
                    neuron.dendritoma.scale_learning_rate(1.0 / self.sleep_learning_rate_factor);
                    neuron.exit_sleep_mode();
                }
            } else {
                for neuron in &mut self.neurons {
                    neuron.dendritoma.reset_learning_params();
                    neuron.exit_sleep_mode();
                }
            }
        }
    }

    pub fn get_state(&self) -> NetworkState {
        self.state
    }

    pub fn set_weight_decay(&mut self, decay: f64) {
        for neuron in &mut self.neurons {
            neuron.dendritoma.set_weight_decay(decay);
        }
    }
}

/// Estatísticas da rede
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub time_step: i64,
    pub firing_rate: f64,
    pub avg_energy: f64,
    pub avg_threshold: f64,
    pub avg_novelty: f64,
    pub alert_level: f64,
    pub avg_eligibility: f64,
    pub dopamine: f64,
    pub norepinephrine: f64,
    pub state: NetworkState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let network = Network::new(100, ConnectivityType::Grid2D, 0.2, 0.5);

        assert_eq!(network.num_neurons(), 100);
        assert_eq!(network.current_time_step, 0);

        let num_inhibitory = network
            .neurons
            .iter()
            .filter(|n| n.neuron_type == NeuronType::Inhibitory)
            .count();
        assert_eq!(num_inhibitory, 20);
    }

    #[test]
    fn test_lateral_competition() {
        let mut network = Network::new(10, ConnectivityType::FullyConnected, 0.2, 0.5);

        // Configura camada hidden
        network.hidden_indices = vec![2, 3, 4, 5, 6, 7];

        // Simula diferentes ativações
        network.neurons[2].recent_firing_rate = 0.5;
        network.neurons[3].recent_firing_rate = 0.1;
        network.neurons[4].recent_firing_rate = 0.3;

        // Aplica competição
        let indices = network.hidden_indices.clone();
        network.apply_lateral_competition(&indices);

        // Neurônio mais ativo deve ter plasticidade maior
        assert!(network.neurons[2].dendritoma.plasticity_gain >= network.neurons[3].dendritoma.plasticity_gain);
    }

    #[test]
    fn test_reward_propagation() {
        let mut network = Network::new(5, ConnectivityType::FullyConnected, 0.2, 0.5);

        // Configura eligibility traces
        for neuron in &mut network.neurons {
            neuron.dendritoma.eligibility_trace = vec![0.5; 5];
        }

        let initial_weights: Vec<f64> = network.neurons[0].dendritoma.weights.clone();

        // Propaga reward
        network.propagate_reward(1.0);

        // Pesos devem ter mudado
        let changed = network.neurons[0].dendritoma.weights.iter()
            .zip(initial_weights.iter())
            .any(|(new, old)| (new - old).abs() > 1e-6);

        assert!(changed);
    }

    #[test]
    fn test_weight_gap_calculation() {
        let mut network = Network::new(10, ConnectivityType::FullyConnected, 0.0, 0.5);

        // Aumenta pesos de algumas fontes (padrão)
        for neuron in &mut network.neurons {
            neuron.dendritoma.weights[0] = 0.5;
            neuron.dendritoma.weights[1] = 0.5;
            // Outras ficam em 0.05
        }

        let pattern_sources = vec![0, 1];
        let targets: Vec<usize> = (2..10).collect();

        let gap = network.compute_weight_gap(&pattern_sources, &targets);

        // Gap deve ser positivo (padrão > ruído)
        assert!(gap > 0.0);
    }
}