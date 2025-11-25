/// Módulo que implementa a rede NEN-V
///
/// A Network orquestra a simulação, gerindo os neurónios e suas conexões.

use crate::nenv::{NeuronType, SpikeOrigin, NENV};

/// Tipo de topologia de rede
#[derive(Debug, Clone, Copy)]
pub enum ConnectivityType {
    /// Todos os neurónios conectados a todos
    FullyConnected,
    /// Grade 2D onde cada neurónio conecta aos 8 vizinhos (Moore neighborhood)
    Grid2D,
    /// Neurónios isolados (sem conexões entre si) - usado para testes de validação
    Isolated,
}

/// Modo de aprendizado sináptico
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningMode {
    /// Aprendizado Hebbiano clássico: "neurons that fire together, wire together"
    Hebbian,
    /// STDP (Spike-Timing-Dependent Plasticity): aprendizado temporal baseado em causalidade
    STDP,
}

/// Estado global da rede
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkState {
    /// Estado normal: recebe inputs externos, plasticidade alta
    Awake,
    /// Estado de consolidação: sem inputs externos, replay espontâneo, plasticidade reduzida
    Sleep {
        /// Probabilidade de um neurônio disparar espontaneamente
        replay_noise: f64,
        /// Quantos passos a rede já dormiu
        consolidation_steps: usize,
        /// Duração máxima do ciclo de sono
        max_sleep_duration: usize,
    },
}

/// Estrutura principal da rede NEN-V
#[derive(Debug)]
pub struct Network {
    /// Vetor de todos os neurónios na rede
    pub neurons: Vec<NENV>,

    /// Matriz de conectividade (1 se conectado, 0 se não)
    /// connectivity_matrix[i][j] = 1 significa que neurónio i recebe input de neurónio j
    pub connectivity_matrix: Vec<Vec<u8>>,

    /// Passo de tempo atual da simulação
    pub current_time_step: i64,

    /// Dimensões da grade (para topologia Grid2D)
    pub grid_width: usize,
    pub grid_height: usize,

    /// Nível de alerta global da rede [0.0, 1.0]
    /// 0.0 = estado normal, 1.0 = alerta máximo
    /// Afeta a recuperação de energia de todos os neurónios
    pub alert_level: f64,

    /// Taxa de decaimento do alert_level (retorna gradualmente ao baseline)
    pub alert_decay_rate: f64,

    /// Novidade média atual da rede (calculada no último update)
    current_avg_novelty: f64,

    /// Threshold de novidade para ativar alert_level automaticamente
    novelty_alert_threshold: f64,

    /// Sensibilidade do boost de alert baseado em novidade
    alert_sensitivity: f64,

    /// Modo de aprendizado sináptico (Hebbian ou STDP)
    pub learning_mode: LearningMode,

    /// Buffer de spikes para STDP baseado em pares
    /// Cada entrada é (tempo, id_neurônio)
    spike_buffer: Vec<(i64, usize, SpikeOrigin)>,

    /// Janela temporal máxima para STDP (em passos de tempo)
    pub stdp_window: i64,

    /// Estado atual da rede (Vigília ou Sono)
    pub state: NetworkState,

    /// Fator de redução da taxa de aprendizado durante o sono (ex: 0.1)
    pub sleep_learning_rate_factor: f64,

    /// Sinal de recompensa global para modular STDP (dopamine-like)
    /// - reward > 0: reforça sinapses ativas (ação boa)
    /// - reward < 0: enfraquece sinapses ativas (ação ruim)
    /// - reward = 0: aprendizado neutro (baseline)
    pub global_reward_signal: f64,
}

impl Network {
    /// Cria uma nova rede
    ///
    /// # Argumentos
    /// * `num_neurons` - Número total de neurónios
    /// * `connectivity_type` - Tipo de topologia
    /// * `inhibitory_ratio` - Proporção de neurónios inibitórios (0.0 a 1.0)
    /// * `initial_threshold` - Limiar de disparo inicial para todos os neurónios
    pub fn new(
        num_neurons: usize,
        connectivity_type: ConnectivityType,
        inhibitory_ratio: f64,
        initial_threshold: f64,
    ) -> Self {
        // ... (código de grid/matrix mantido igual) ...
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

            // 1. TABULA RASA: Pesos iniciais pequenos e ligeiramente aleatórios
            // Aleatorização quebra simetria e permite diferenciação natural
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for w in &mut neuron.dendritoma.weights {
                *w = rng.gen_range(0.04..0.06);  // Pequena variação em torno de 0.05
            }

            // 2. INIBIÇÃO INICIAL: Pequeno bias inicial para sinapses inibitórias
            // iSTDP vai aprender os pesos corretos baseado em E/I balance
            for source_id in 0..num_neurons {
                if source_id < neuron.dendritoma.weights.len() {
                    if connectivity_matrix[i][source_id] == 1 {
                        if source_id < num_inhibitory && source_id != i {
                            // Peso inicial modesto - iSTDP vai ajustar
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
            // CORREÇÃO: Janela menor (20) evita interferência entre testes repetidos
            stdp_window: 20,
            state: NetworkState::Awake,
            sleep_learning_rate_factor: 0.0,
            global_reward_signal: 0.0, // Neutro por padrão
        }
    }

    /// Define o modo de aprendizado da rede
    pub fn set_learning_mode(&mut self, mode: LearningMode) {
        self.learning_mode = mode;
    }

    /// Gera a matriz de conectividade baseada no tipo
    fn generate_connectivity(
        num_neurons: usize,
        connectivity_type: ConnectivityType,
        grid_width: usize,
    ) -> Vec<Vec<u8>> {
        match connectivity_type {
            ConnectivityType::FullyConnected => {
                // Todos conectados a todos (exceto si mesmo)
                vec![vec![1; num_neurons]; num_neurons]
            }
            ConnectivityType::Grid2D => {
                Self::generate_2d_grid_connectivity(num_neurons, grid_width)
            }
            ConnectivityType::Isolated => {
                // Nenhuma conexão entre neurónios (matriz de zeros)
                vec![vec![0; num_neurons]; num_neurons]
            }
        }
    }

    /// Gera conectividade de grade 2D (Moore neighborhood - 8 vizinhos)
    fn generate_2d_grid_connectivity(num_neurons: usize, width: usize) -> Vec<Vec<u8>> {
        let mut matrix = vec![vec![0; num_neurons]; num_neurons];

        for i in 0..num_neurons {
            let (row, col) = (i / width, i % width);

            // Conecta aos 8 vizinhos (Moore neighborhood)
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue; // Não conecta a si mesmo
                    }

                    let new_row = row as i32 + dr;
                    let new_col = col as i32 + dc;

                    // Verifica limites
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

    /// Coleta os inputs para um neurónio específico baseado nas conexões
    ///
    /// # Argumentos
    /// * `neuron_idx` - Índice do neurónio alvo
    /// * `all_outputs` - Vetor com saídas de todos os neurónios
    /// * `external_inputs` - Vetor com inputs externos (opcional)
    ///
    /// # Retorna
    /// Vetor de inputs combinados (rede + externos)
    fn gather_inputs(
        &self,
        neuron_idx: usize,
        all_outputs: &[f64],
        external_inputs: &[f64],
    ) -> Vec<f64> {
        let mut inputs = vec![0.0; self.neurons.len()];

        // Coleta inputs da rede baseado na matriz de conectividade
        for j in 0..self.neurons.len() {
            if self.connectivity_matrix[neuron_idx][j] == 1 {
                inputs[j] = all_outputs[j];
            }
        }

        // Adiciona input externo apenas para o neurónio correspondente
        if neuron_idx < external_inputs.len() {
            // CORREÇÃO: Input externo do neurónio i vai para inputs[i], não para todos
            // Mas como inputs[j] representa o input do neurónio j para o neurónio atual,
            // o input externo deve ir para inputs[neuron_idx] (auto-conexão virtual)
            inputs[neuron_idx] += external_inputs[neuron_idx];
        }

        inputs
    }

    /// Executa um passo de atualização da rede
    ///
    /// Este é o coração da simulação, implementando o algoritmo do guia v2:
    /// 1. Coleta saídas do passo anterior
    /// 2. Para cada neurónio: integra, modula, decide disparar
    /// 3. Para cada neurónio: aplica aprendizado e atualiza estado
    ///
    /// # Argumentos
    /// * `external_inputs` - Vetor de inputs externos (um valor por neurónio)
    pub fn update(&mut self, external_inputs: &[f64]) {
        self.current_time_step += 1;

        // CORREÇÃO: Detecta se há input externo (para controle de homeostase)
        let has_external_input = external_inputs.iter().any(|&x| x.abs() > 1e-6);

        // Verifica se deve acordar automaticamente
        if let NetworkState::Sleep { consolidation_steps, max_sleep_duration, .. } = self.state {
            if consolidation_steps >= max_sleep_duration {
                self.wake_up();
            }
        }

        // Atualiza contadores de sono e aplica consolidação
        if let NetworkState::Sleep { ref mut consolidation_steps, .. } = self.state {
            *consolidation_steps += 1;

            // SYNAPTIC TAGGING AND CAPTURE: Consolidação seletiva baseada em tags
            // Durante o sono, apenas sinapses com tags fortes (relevantes) consolidam.
            // Isso filtra o ruído e implementa "consolidação dirigida por relevância".
            //
            // Taxa base: 1% por passo (mais lenta que a antiga, mas dinâmica via tags)
            // Tags fortes (tag=2.0) consolidam até 20x mais rápido que tags fracas (tag=0.2)
            for neuron in &mut self.neurons {
                neuron.dendritoma.consolidate_memory_tagged(0.01);
            }
        }

        // Fase 0: Atualiza alert_level (decaimento gradual)
        self.update_alert_level();

        // Coleta todas as saídas do passo anterior
        let all_neuron_outputs: Vec<f64> = self.neurons.iter().map(|n| n.output_signal).collect();

        // Cria vetores temporários para armazenar resultados da Fase 1-3
        let mut integrated_potentials = Vec::with_capacity(self.neurons.len());
        let mut modulated_potentials = Vec::with_capacity(self.neurons.len());
        let mut gathered_inputs = Vec::with_capacity(self.neurons.len());

        // Fase 1-2: Calcular potenciais para todos os neurónios
        for (idx, neuron) in self.neurons.iter().enumerate() {
            // Determina inputs baseado no estado
            let inputs = match self.state {
                NetworkState::Awake => {
                    self.gather_inputs(idx, &all_neuron_outputs, external_inputs)
                },
                NetworkState::Sleep { replay_noise, .. } => {
                    // No sono, ignora inputs externos e adiciona ruído de replay
                    // Ruído é proporcional à atividade SALVA da vigília (replay dirigido)
                    let mut sleep_inputs = self.gather_inputs(idx, &all_neuron_outputs, &vec![0.0; external_inputs.len()]);

                    // CORREÇÃO CRÍTICA: Replay dirigido aos neurônios mais ativos
                    // Probabilidade de ruído aumenta com saved_awake_activity
                    let noise_prob = replay_noise + (neuron.saved_awake_activity * 1.0); // REFATORADO: Replay proporcional à atividade

                    if rand::random::<f64>() < noise_prob {
                        // Em vez de escolher input ALEATÓRIO, escolher input de neurônio com alta atividade salva
                        // Pegar neurônios conectados
                        let connected_active: Vec<(usize, f64)> = self.connectivity_matrix[idx]
                            .iter()
                            .enumerate()
                            .filter(|&(_, &connected)| connected == 1)
                            .map(|(j, _)| (j, self.neurons[j].saved_awake_activity))
                            .filter(|(_, activity)| *activity > 0.01) // Apenas neurônios que estavam ativos
                            .collect();

                        if !connected_active.is_empty() {
                            // Escolher neurônio conectado com maior atividade salva (replay dirigido)
                            let best_input = connected_active.iter()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(idx, _)| *idx)
                                .unwrap();

                            sleep_inputs[best_input] += 1.5; // Reativa neurônio mais ativo
                        } else {
                            // Fallback: Se nenhum neurônio conectado estava ativo, usa aleatório
                            let input_idx = rand::random::<usize>() % sleep_inputs.len();
                            sleep_inputs[input_idx] += 1.5;
                        }
                    }

                    sleep_inputs
                }
            };

            let integrated = neuron.dendritoma.integrate(&inputs);
            let modulated = neuron.glia.modulate(integrated);

            integrated_potentials.push(integrated);
            modulated_potentials.push(modulated);
            gathered_inputs.push(inputs);
        }

        // Fase 3: Decisão de disparo para todos os neurónios
        for ((neuron, &modulated_potential), inputs) in
            self.neurons.iter_mut()
                .zip(modulated_potentials.iter())
                .zip(gathered_inputs.iter())
        {
            // Determina se há input externo significativo (threshold: 0.5)
            let has_external_input = inputs.iter().any(|&inp| inp > 0.5);
            neuron.decide_to_fire(modulated_potential, self.current_time_step, has_external_input);
        }

        // Fase 4: Aprendizado e atualização de estado
        let mut total_novelty = 0.0;

        // Coleta tipos de neurônios antes do loop mutável (para iSTDP)
        let neuron_types: Vec<NeuronType> = self.neurons.iter()
            .map(|n| n.neuron_type)
            .collect();
        let neuron_firing_states: Vec<bool> = self.neurons.iter()
            .map(|n| n.is_firing)
            .collect();

        for (idx, (neuron, inputs)) in self.neurons.iter_mut().zip(gathered_inputs.iter()).enumerate() {
            let novelty = neuron.compute_novelty(inputs);
            total_novelty += novelty;
            neuron.update_priority(novelty, 2.0);

            let mut learning_happened = false;

            if neuron.is_firing {
                match self.learning_mode {
                    LearningMode::Hebbian => {
                        neuron.dendritoma.apply_learning(inputs);
                        learning_happened = true;
                    }
                    LearningMode::STDP => {
                        // CORREÇÃO STDP: Lê origem correta do buffer
                        for &(spike_time, spike_neuron_id, pre_origin) in &self.spike_buffer {
                            if spike_neuron_id == idx { continue; }

                            let post_origin = neuron.spike_origin;

                            // Gating robusto usando a origem GRAVADA no momento do spike
                            let should_apply_stdp = !matches!(
                                (pre_origin, post_origin),
                                (SpikeOrigin::Feedback, _) | (_, SpikeOrigin::Feedback)
                            );

                            if should_apply_stdp {
                                let delta_t = self.current_time_step - spike_time;

                                // Usa informação pre-coletada do neurônio pré-sináptico
                                let pre_neuron_type = neuron_types[spike_neuron_id];
                                let pre_neuron_firing = neuron_firing_states[spike_neuron_id];

                                // Aplica STDP ou iSTDP baseado no tipo do neurônio pré-sináptico
                                match pre_neuron_type {
                                    NeuronType::Excitatory => {
                                        // STDP regular para sinapses excitatórias, modulado por reward
                                        neuron.dendritoma.apply_stdp_pair(spike_neuron_id, delta_t, self.global_reward_signal);
                                        learning_happened = true;
                                    },
                                    NeuronType::Inhibitory => {
                                        // iSTDP para sinapses inibitórias (E/I balance)
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
            }

            // CORREÇÃO: Decay seletivo - aplica apenas em modo Hebbian
            // Em modo STDP, o decay dentro de apply_stdp_pair() já é suficiente
            // Em modo Hebbian, neurônios inativos precisam de decay adicional
            // if !learning_happened && self.learning_mode == LearningMode::Hebbian {
            if !learning_happened {
                neuron.dendritoma.apply_weight_maintenance(neuron.recent_firing_rate);
            }

            // SYNAPTIC TAGGING: Aplica decaimento das tags a cada passo
            // Tags são temporárias e decaem naturalmente se não reforçadas
            neuron.dendritoma.decay_tags();

            neuron.glia.update_state(neuron.is_firing);
            neuron.update_memory(inputs);

            let firing_value = if neuron.is_firing { 1.0 } else { 0.0 };
            neuron.recent_firing_rate = 0.99 * neuron.recent_firing_rate + 0.01 * firing_value;
            neuron.update_meta_threshold(neuron.is_firing);
            neuron.apply_homeostatic_plasticity(self.current_time_step, has_external_input);
        }

        // Atualização do Buffer com ORIGEM
        if self.learning_mode == LearningMode::STDP {
            for (idx, neuron) in self.neurons.iter().enumerate() {
                if neuron.is_firing {
                    // CORREÇÃO: Armazena a origem (Exogenous/Endogenous) junto com o tempo
                    self.spike_buffer.push((self.current_time_step, idx, neuron.spike_origin));
                }
            }
            // Limpa buffer antigo (usando stdp_window 20 definido no new)
            // Note: self.stdp_window é i64
            self.spike_buffer.retain(|(time, _, _)| self.current_time_step - time <= self.stdp_window);
        }

        // Fase 5: Integração Novelty-Alert (v0.3.0)
        // Calcula novidade média da rede
        self.current_avg_novelty = total_novelty / self.neurons.len() as f64;

        // Se novidade excede threshold, boost alert_level automaticamente
        if self.current_avg_novelty > self.novelty_alert_threshold {
            // Alert boost baseado no EXCESSO acima do threshold
            let alert_boost = (self.current_avg_novelty - self.novelty_alert_threshold) * self.alert_sensitivity;
            self.boost_alert_level(alert_boost);
        }
    }

    /// Retorna o número de neurónios na rede
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Retorna o número de neurónios que estão disparando no momento
    pub fn num_firing(&self) -> usize {
        self.neurons.iter().filter(|n| n.is_firing).count()
    }

    /// Retorna a energia média da rede
    pub fn average_energy(&self) -> f64 {
        let total_energy: f64 = self.neurons.iter().map(|n| n.glia.energy).sum();
        total_energy / self.neurons.len() as f64
    }

    /// Retorna vetor com estado de disparo de todos os neurónios
    pub fn get_firing_states(&self) -> Vec<bool> {
        self.neurons.iter().map(|n| n.is_firing).collect()
    }

    /// Retorna vetor com níveis de energia de todos os neurónios
    pub fn get_energy_levels(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.glia.energy).collect()
    }

    /// Converte índice linear para coordenadas (row, col) na grade
    pub fn index_to_coords(&self, index: usize) -> Option<(usize, usize)> {
        if self.grid_width > 0 && index < self.neurons.len() {
            Some((index / self.grid_width, index % self.grid_width))
        } else {
            None
        }
    }

    /// Converte coordenadas (row, col) para índice linear
    pub fn coords_to_index(&self, row: usize, col: usize) -> Option<usize> {
        if self.grid_width > 0 && row < self.grid_height && col < self.grid_width {
            let index = row * self.grid_width + col;
            if index < self.neurons.len() {
                return Some(index);
            }
        }
        None
    }

    /// Define o nível de alerta global da rede
    ///
    /// O alert_level afeta a recuperação de energia de todos os neurónios.
    /// Valores altos fazem a rede responder mais rapidamente a eventos.
    ///
    /// # Argumentos
    /// * `level` - Nível de alerta [0.0, 1.0]
    pub fn set_alert_level(&mut self, level: f64) {
        self.alert_level = level.clamp(0.0, 1.0);

        // Propaga alert_level para todos os neurónios
        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    /// Aumenta o alert_level baseado na atividade global da rede
    ///
    /// Chamado automaticamente quando detecta alta atividade ou novidade.
    /// O alert_level decai gradualmente a cada passo de simulação.
    ///
    /// # Argumentos
    /// * `boost` - Quantidade para aumentar o alert_level
    pub fn boost_alert_level(&mut self, boost: f64) {
        self.alert_level = (self.alert_level + boost).min(1.0);

        // Propaga para todos os neurónios
        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    /// Atualiza o alert_level (decaimento gradual para baseline)
    ///
    /// Chamado automaticamente a cada passo de update()
    fn update_alert_level(&mut self) {
        // Decai gradualmente para zero (estado normal)
        self.alert_level *= 1.0 - self.alert_decay_rate;

        // Propaga para neurónios
        for neuron in &mut self.neurons {
            neuron.glia.alert_level = self.alert_level;
        }
    }

    /// Retorna a novidade média da rede (calculada no último update)
    ///
    /// A novidade é a diferença média entre inputs atuais e memória contextual
    /// de todos os neurônios. Valores altos indicam eventos inesperados.
    ///
    /// # Retorna
    /// Novidade média [0.0, ∞), calculada automaticamente durante update()
    pub fn average_novelty(&self) -> f64 {
        self.current_avg_novelty
    }

    pub fn set_novelty_alert_params(&mut self, threshold: f64, sensitivity: f64) {
        self.novelty_alert_threshold = threshold.max(0.0);
        self.alert_sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    /// Entra no estado de sono
    ///
    /// # Argumentos
    /// * `replay_noise` - Probabilidade base de disparo espontâneo
    /// * `duration` - Duração do sono em passos
    pub fn enter_sleep(&mut self, replay_noise: f64, duration: usize) {
        if let NetworkState::Awake = self.state {
            self.state = NetworkState::Sleep {
                replay_noise,
                consolidation_steps: 0,
                max_sleep_duration: duration,
            };

            // Salva atividade de vigília ANTES de reduzir taxa de aprendizado
            // Ajusta metabolismo para sono (recuperação rápida, baixo consumo)
            for neuron in &mut self.neurons {
                neuron.saved_awake_activity = neuron.recent_firing_rate;
                neuron.dendritoma.scale_learning_rate(self.sleep_learning_rate_factor);
                neuron.enter_sleep_mode(); // NOVO: ajusta metabolismo
            }

            // Reduz alert_level durante sono
            self.alert_level = 0.3;
        }
    }

    /// Acorda a rede
    pub fn wake_up(&mut self) {
        if let NetworkState::Sleep { .. } = self.state {
            self.state = NetworkState::Awake;

            // Restaura taxa de aprendizado e metabolismo
            // Se sleep_learning_rate_factor == 0, precisamos restaurar manualmente
            if self.sleep_learning_rate_factor > 0.0 {
                for neuron in &mut self.neurons {
                    neuron.dendritoma.scale_learning_rate(1.0 / self.sleep_learning_rate_factor);
                    neuron.exit_sleep_mode(); // NOVO: restaura metabolismo
                }
            } else {
                // Taxa estava zerada, restaurar todos os parâmetros
                for neuron in &mut self.neurons {
                    neuron.dendritoma.reset_learning_params();
                    neuron.exit_sleep_mode(); // NOVO: restaura metabolismo
                }
            }
        }
    }

    /// Retorna o estado atual da rede
    pub fn get_state(&self) -> NetworkState {
        self.state
    }

    /// Define a taxa de decaimento dos pesos STM para todos os neurônios
    ///
    /// # Argumentos
    /// * `decay` - Taxa de decaimento (0.0 = sem decaimento, 1.0 = decaimento total)
    pub fn set_weight_decay(&mut self, decay: f64) {
        for neuron in &mut self.neurons {
            neuron.dendritoma.set_weight_decay(decay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let network = Network::new(100, ConnectivityType::Grid2D, 0.2, 0.5);

        assert_eq!(network.num_neurons(), 100);
        assert_eq!(network.current_time_step, 0);

        // Verifica proporção de inibitórios
        let num_inhibitory = network
            .neurons
            .iter()
            .filter(|n| n.neuron_type == NeuronType::Inhibitory)
            .count();
        assert_eq!(num_inhibitory, 20);
    }

    #[test]
    fn test_grid_dimensions() {
        let network = Network::new(100, ConnectivityType::Grid2D, 0.2, 0.5);

        assert_eq!(network.grid_width, 10);
        assert_eq!(network.grid_height, 10);
    }

    #[test]
    fn test_coords_conversion() {
        let network = Network::new(100, ConnectivityType::Grid2D, 0.2, 0.5);

        // Testa conversão de ida e volta
        let index = network.coords_to_index(5, 5).unwrap();
        assert_eq!(index, 55);

        let (row, col) = network.index_to_coords(55).unwrap();
        assert_eq!(row, 5);
        assert_eq!(col, 5);
    }

    #[test]
    fn test_2d_grid_connectivity() {
        let network = Network::new(9, ConnectivityType::Grid2D, 0.0, 0.5);

        // Grade 3x3: neurónio central (idx 4) deve ter 8 conexões
        let connections: usize = network.connectivity_matrix[4].iter().map(|&x| x as usize).sum();
        assert_eq!(connections, 8);

        // Neurónio de canto (idx 0) deve ter 3 conexões
        let connections: usize = network.connectivity_matrix[0].iter().map(|&x| x as usize).sum();
        assert_eq!(connections, 3);
    }

    #[test]
    fn test_fully_connected() {
        let network = Network::new(10, ConnectivityType::FullyConnected, 0.0, 0.5);

        // Cada neurónio deve conectar a todos os outros
        for i in 0..10 {
            let connections: usize = network.connectivity_matrix[i]
                .iter()
                .map(|&x| x as usize)
                .sum();
            assert_eq!(connections, 10); // Conecta a todos (incluindo si mesmo na matriz)
        }
    }

    #[test]
    fn test_network_update_increments_time() {
        let mut network = Network::new(10, ConnectivityType::Grid2D, 0.2, 0.5);
        let external_inputs = vec![0.0; 10];

        assert_eq!(network.current_time_step, 0);

        network.update(&external_inputs);
        assert_eq!(network.current_time_step, 1);

        network.update(&external_inputs);
        assert_eq!(network.current_time_step, 2);
    }

    #[test]
    fn test_network_stats() {
        let network = Network::new(10, ConnectivityType::Grid2D, 0.2, 0.5);

        // Energia inicial deve ser 100% (MAX_ENERGY)
        assert_eq!(network.average_energy(), 100.0);

        // Nenhum neurónio deve estar disparando inicialmente
        assert_eq!(network.num_firing(), 0);
    }
}
