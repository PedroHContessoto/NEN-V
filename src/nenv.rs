/// Módulo que implementa o neurónio NENV (Neurónio-Entrada-Núcleo-Vasos)
///
/// O NENV é a unidade central da arquitetura, integrando o Dendritoma (entrada),
/// a Glia (modulação metabólica) e memória contextual.

use crate::dendritoma::Dendritoma;
use crate::glia::Glia;
use std::collections::VecDeque;

/// Tipo de neurónio: Excitatório ou Inibitório
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    /// Neurónios excitatórios emitem sinais positivos (+1.0)
    Excitatory,
    /// Neurónios inibitórios emitem sinais negativos (-1.0)
    Inhibitory,
}

/// Origem do spike: usado para STDP gated (3-factor STDP)
///
/// Distingue entre spikes genuínos (threshold-crossing) e spikes
/// forçados por input externo ou gerados por feedback recorrente.
/// Isso previne que loops de feedback sejam reforçados por STDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikeOrigin {
    /// Spike endógeno: gerado naturalmente ao ultrapassar threshold
    /// Estes são spikes "legítimos" que devem participar de STDP
    Endogenous,

    /// Spike exógeno: forçado por input externo (estímulo experimental)
    /// Usados para treinar padrões específicos via STDP
    Exogenous,

    /// Spike de feedback: causado por recorrência na rede
    /// NÃO devem participar de STDP para evitar runaway LTP
    Feedback,

    /// Nenhum spike ocorreu
    None,
}

/// Estrutura principal do neurónio NENV
#[derive(Debug, Clone)]
pub struct NENV {
    /// Identificador único do neurónio
    pub id: usize,

    /// Tipo do neurónio (excitatório ou inibitório)
    pub neuron_type: NeuronType,

    /// Componente de entrada e aprendizado sináptico
    pub dendritoma: Dendritoma,

    /// Componente de modulação metabólica
    pub glia: Glia,

    /// Traço de memória contextual (média móvel exponencial dos inputs)
    pub memory_trace: Vec<f64>,

    /// Passo de tempo do último disparo
    pub last_fire_time: i64,

    /// Histórico de tempos de spikes recentes (para STDP)
    /// Mantém apenas os últimos N spikes dentro da janela temporal relevante
    spike_history: VecDeque<i64>,

    /// Limiar de disparo
    pub threshold: f64,

    /// Limiar de disparo inicial (referência para clamps adaptativos)
    base_threshold: f64,

    /// Contador de overshoot para modular piso dinâmico
    overshoot_count: f64,

    /// Estado de disparo atual
    pub is_firing: bool,

    /// Origem do último spike (para STDP gated/3-factor)
    /// Permite distinguir spikes endógenos, exógenos e de feedback
    pub spike_origin: SpikeOrigin,

    /// Sinal de saída (+1.0 para excitatório, -1.0 para inibitório, 0.0 se não disparou)
    pub output_signal: f64,

    // Parâmetros de dinâmica
    refractory_period: i64,
    memory_alpha: f64,

    /// Taxa de disparo recente (média móvel exponencial)
    /// Usada para guiar o replay durante o sono
    pub recent_firing_rate: f64,

    /// Taxa de disparo salva antes de entrar no sono
    /// Preserva a atividade durante vigília para replay mais preciso
    pub saved_awake_activity: f64,

    // Parâmetros de plasticidade homeostática
    /// Taxa de disparo alvo para homeostase (default: 0.1 = 10% de atividade)
    pub target_firing_rate: f64,

    /// Taxa de ajuste homeostático (default: 0.01)
    pub homeo_eta: f64,

    /// Intervalo entre aplicações de homeostase (em passos)
    pub homeo_interval: i64,

    /// Último passo em que homeostase foi aplicada
    last_homeo_update: i64,

    /// Proporção do esforço homeostático em ajuste de pesos (0.0-1.0)
    /// Default: 0.7 (70% em pesos, 30% em threshold)
    pub homeo_weight_ratio: f64,

    /// Proporção do esforço homeostático em ajuste de threshold (0.0-1.0)
    /// Default: 0.3 (30% threshold, 70% pesos)
    /// Nota: weight_ratio + threshold_ratio devem somar 1.0
    pub homeo_threshold_ratio: f64,

    // Parâmetros de metaplasticidade BCM
    /// Limiar metaplástico dinâmico (θ_M na teoria BCM)
    /// Ajusta-se baseado na atividade quadrática média
    pub meta_threshold: f64,

    /// Taxa de atualização do meta_threshold (alpha para EMA)
    pub meta_alpha: f64,
}

impl NENV {
    /// Cria um novo neurónio NENV
    ///
    /// # Argumentos
    /// * `id` - Identificador único
    /// * `num_inputs` - Número de conexões de entrada
    /// * `initial_threshold` - Limiar de disparo inicial
    /// * `neuron_type` - Tipo do neurónio (excitatório ou inibitório)
    pub fn new(
        id: usize,
        num_inputs: usize,
        initial_threshold: f64,
        neuron_type: NeuronType,
    ) -> Self {
        Self {
            id,
            neuron_type,
            dendritoma: Dendritoma::new(num_inputs),
            glia: Glia::new(),
            memory_trace: vec![0.0; num_inputs],
            last_fire_time: -1,
            spike_history: VecDeque::with_capacity(10),
            threshold: initial_threshold,
            base_threshold: initial_threshold,
            overshoot_count: 0.0,
            is_firing: false,
            spike_origin: SpikeOrigin::None,
            output_signal: 0.0,
            // RESTAURADO: Período refratário biológico (5ms)
            refractory_period: 5,
            memory_alpha: 0.02,
            recent_firing_rate: 0.0,
            saved_awake_activity: 0.0,
            target_firing_rate: 0.15,  // Será sobrescrito pelo AutoConfig
            // Parâmetros homeostáticos ajustados via grid-search (W65T35_eta3.3x_int0.858x)
            homeo_eta: 0.1627,     // 0.05 * 3.253 (grid)
            homeo_interval: 9,     // 10 * 0.858 (grid, arredondado)
            last_homeo_update: -1,
            homeo_weight_ratio: 0.650,    // Grid-search
            homeo_threshold_ratio: 0.350, // Grid-search
            meta_threshold: 0.12,  // OTIMIZADO: BCM threshold
            meta_alpha: 0.005,     // OTIMIZADO: BCM learning rate
        }
    }

    /// Cria um neurónio excitatório
    pub fn excitatory(id: usize, num_inputs: usize, initial_threshold: f64) -> Self {
        Self::new(id, num_inputs, initial_threshold, NeuronType::Excitatory)
    }

    /// Cria um neurónio inibitório
    pub fn inhibitory(id: usize, num_inputs: usize, initial_threshold: f64) -> Self {
        Self::new(id, num_inputs, initial_threshold, NeuronType::Inhibitory)
    }

    /// Decide se o neurónio deve disparar baseado no potencial modulado
    ///
    /// # Argumentos
    /// * `modulated_potential` - Potencial após modulação glial
    /// * `current_time` - Passo de tempo atual da simulação
    /// * `has_external_input` - True se há input externo direto (para STDP gated)
    pub fn decide_to_fire(&mut self, modulated_potential: f64, current_time: i64, has_external_input: bool) {
        // Verifica período refratário
        // Neurônio nunca disparado (last_fire_time = -1) não está em refratário
        let is_in_refractory = if self.last_fire_time < 0 {
            false
        } else {
            (current_time - self.last_fire_time) < self.refractory_period
        };

        // Reset do estado de disparo
        self.is_firing = false;
        self.spike_origin = SpikeOrigin::None;
        self.output_signal = 0.0;

        // SPARSE CODING: Adaptive threshold aumenta com firing rate
        // Se neurônio dispara muito → threshold sobe → mais difícil disparar
        let adaptive_threshold = self.threshold * (1.0 + self.recent_firing_rate * 3.0);

        // HARD ENERGY GATING: Só dispara se tem energia mínima
        // (energy_level já foi reduzido por glia.modulate)
        let has_energy = self.glia.energy > 5.0;  // Reserva mínima

        // Dispara se: potencial > adaptive_threshold E não refratário E tem energia
        if modulated_potential > adaptive_threshold && !is_in_refractory && has_energy {
            self.is_firing = true;
            self.last_fire_time = current_time;

            // Overshoot tracking: incrementa se muito acima do target recente
            if self.recent_firing_rate > self.target_firing_rate * 2.0 {
                self.overshoot_count = (self.overshoot_count + 1.0).min(100.0);
            }

            // Determina a origem do spike (STDP gated / 3-factor)
            // Regra: Se há input externo direto → Exogenous
            //        Se não → pode ser Endogenous (genuíno) ou Feedback (recorrente)
            //        Para distinguir Endogenous de Feedback, verificamos se o potencial
            //        é muito maior que o threshold (indica feedback intenso)
            if has_external_input {
                self.spike_origin = SpikeOrigin::Exogenous;
            } else {
                // Heurística: potencial >> threshold sugere feedback recorrente
                // Se potencial é apenas ligeiramente acima do threshold → Endogenous
                let excess = modulated_potential - self.threshold;
                if excess > self.threshold * 2.0 {
                    // Potencial muito alto → provavelmente feedback recorrente
                    self.spike_origin = SpikeOrigin::Feedback;
                } else {
                    // Potencial moderado → disparo genuíno/endógeno
                    self.spike_origin = SpikeOrigin::Endogenous;
                }
            }

            // Registra spike no histórico para STDP
            self.spike_history.push_back(current_time);

            // Limita histórico aos últimos 10 spikes
            if self.spike_history.len() > 10 {
                self.spike_history.pop_front();
            }

            // O sinal de saída depende do tipo de neurónio
            self.output_signal = match self.neuron_type {
                NeuronType::Excitatory => 1.0,
                NeuronType::Inhibitory => -1.0,
            };
        }
    }

    /// Atualiza a memória contextual do neurónio
    ///
    /// Implementa uma média móvel exponencial dos padrões de entrada,
    /// permitindo que o neurónio "lembre" inputs recentes.
    ///
    /// # Argumentos
    /// * `inputs` - Vetor de sinais de entrada atual
    pub fn update_memory(&mut self, inputs: &[f64]) {
        assert_eq!(
            inputs.len(),
            self.memory_trace.len(),
            "Número de inputs deve ser igual ao tamanho da memória"
        );

        for i in 0..self.memory_trace.len() {
            self.memory_trace[i] =
                (1.0 - self.memory_alpha) * self.memory_trace[i] + self.memory_alpha * inputs[i];
        }
    }

    /// Calcula a novidade do padrão de entrada atual
    ///
    /// Novidade é medida como a diferença absoluta média entre o input atual
    /// e a memória contextual (padrões recentes). Valores altos indicam
    /// padrões inesperados ou não familiares.
    ///
    /// # Argumentos
    /// * `inputs` - Vetor de sinais de entrada atual
    ///
    /// # Retorna
    /// Valor de novidade [0.0, ∞), onde 0 = completamente familiar
    pub fn compute_novelty(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.memory_trace.len(),
            "Número de inputs deve ser igual ao tamanho da memória"
        );

        // Calcula diferença absoluta média entre input e memória
        let total_diff: f64 = inputs
            .iter()
            .zip(self.memory_trace.iter())
            .map(|(input, memory)| (input - memory).abs())
            .sum();

        // Normaliza pelo número de inputs para manter escala consistente
        total_diff / inputs.len() as f64
    }

    /// Atualiza o priority da Glia baseado na novidade do input
    ///
    /// Priority aumenta com novidade, tornando o neurónio mais sensível
    /// a padrões inesperados (mecanismo de atenção emergente).
    ///
    /// Fórmula: priority = 1.0 + novelty * sensitivity_factor
    ///
    /// # Argumentos
    /// * `novelty` - Valor de novidade calculado
    /// * `sensitivity_factor` - Multiplicador de sensibilidade (padrão: 1.0)
    pub fn update_priority(&mut self, novelty: f64, sensitivity_factor: f64) {
        // Priority base é 1.0, aumenta proporcionalmente à novidade
        self.glia.priority = 1.0 + novelty * sensitivity_factor;

        // Limita priority a um máximo razoável para evitar instabilidade
        self.glia.priority = self.glia.priority.min(3.0);
    }

    /// Calcula o ganho de plasticidade baseado na energia disponível
    ///
    /// Implementa energy-gated learning: neurônios com baixa energia têm
    /// plasticidade fortemente reduzida, enquanto neurônios com alta energia
    /// mantêm plasticidade plena.
    ///
    /// CORRIGIDO: Modulação mais forte e gradual
    /// Fórmula: gain = (e^2) para e < 0.5, depois suaviza para e > 0.5
    ///
    /// Janela plástica:
    /// - e = 0.1: gain = 0.01 (1%, quase nulo)
    /// - e = 0.2: gain = 0.04 (4%)
    /// - e = 0.5: gain = 0.25 (25%)
    /// - e = 0.8: gain = 0.70 (70%)
    /// - e = 1.0: gain = 1.00 (100%)
    ///
    /// # Retorna
    /// Ganho de plasticidade [0.0, 1.0]
    fn compute_plasticity_gain(&self) -> f64 {
        let e = self.glia.energy_fraction();

        if e < 0.5 {
            // Região crítica: penalização quadrática
            // e=0.1 → gain=0.01 (1%)
            // e=0.2 → gain=0.04 (4%)
            // e=0.3 → gain=0.09 (9%)
            // e=0.4 → gain=0.16 (16%)
            e * e
        } else {
            // Região estável: aprendizado pleno (≥50% energia)
            // Transição suave para 100%
            0.25 + 1.5 * (e - 0.5)
        }
    }

    /// Atualiza o limiar metaplástico BCM
    ///
    /// O BCM (Bienenstock-Cooper-Munro) ajusta o limiar de plasticidade
    /// baseado na atividade quadrática média do neurônio. Isso implementa
    /// metaplasticidade: neurônios muito ativos aumentam seu limiar,
    /// tornando-se mais seletivos; neurônios pouco ativos diminuem o limiar,
    /// tornando-se mais sensíveis.
    ///
    /// θ_M = (1-α) × θ_M + α × y²
    ///
    /// # Argumentos
    /// * `fired` - Se o neurônio disparou neste passo
    pub fn update_meta_threshold(&mut self, fired: bool) {
        let y = if fired { 1.0 } else { 0.0 };
        let y_sq = y * y;

        // Atualização EMA do limiar metaplástico
        self.meta_threshold =
            (1.0 - self.meta_alpha) * self.meta_threshold + self.meta_alpha * y_sq;
    }

    /// Calcula o ganho de plasticidade BCM
    ///
    /// Modula a plasticidade baseado na relação entre atividade recente
    /// e o limiar metaplástico. Neurônios muito acima do limiar têm
    /// plasticidade reduzida (evita saturação), neurônios abaixo têm
    /// plasticidade aumentada (facilita aprendizado).
    ///
    /// # Retorna
    /// Ganho de plasticidade [0.5, 1.5]
    fn compute_bcm_gain(&self) -> f64 {
        let y_bar = self.recent_firing_rate;
        let theta = self.meta_threshold.sqrt().max(1e-6); // Volta para escala de y

        // Diferença entre atividade e limiar
        let diff = y_bar - theta;

        // Ganho modulado: atividade alta → ganho menor
        // Limites conservadores para não quebrar STDP
        (1.0 - 0.5 * diff).clamp(0.5, 1.5)
    }

    /// Aplica plasticidade homeostática periodicamente
    ///
    /// Implementa DOIS mecanismos biológicos simultâneos:
    /// 1. Synaptic Scaling (Peso): Ajusta "volume" das entradas
    /// 2. Intrinsic Plasticity (Threshold): Ajusta "sensibilidade" do neurônio
    ///
    /// # Argumentos
    /// * `current_time` - Passo de tempo atual
    /// * `has_external_input` - Se há input externo ativo (>0.5)
    pub fn apply_homeostatic_plasticity(&mut self, current_time: i64, has_external_input: bool) {
        // Verifica se chegou a hora de aplicar homeostase
        if current_time - self.last_homeo_update < self.homeo_interval {
            return;
        }
        self.last_homeo_update = current_time;

        // 🔥 CORREÇÃO: Permite homeostase mesmo sem input externo se FR for 0
        // (Neurônios "mortos" precisam baixar threshold para procurar sinal)
        if !has_external_input && self.recent_firing_rate > 0.01 {
            return;
        }

        // Calcula erro de firing rate (positivo = hiperativo, negativo = hipoativo)
        let rate_error = self.recent_firing_rate - self.target_firing_rate;

        // Só aplica se erro for significativo (> 1%)
        if rate_error.abs() < 0.01 {
            return;
        }

        // Modula erro pela energia
        let energy = self.glia.energy_fraction();
        let energy_weight = if energy < 0.3 { 0.3 } else { energy };
        let effective_error = rate_error * energy_weight;

        // MECANISMO 1: Synaptic Scaling (proporção configurável do esforço homeostático)
        // Ajusta os pesos para tentar compensar o erro
        self.dendritoma.apply_synaptic_scaling(effective_error, self.homeo_eta * self.homeo_weight_ratio);

        // ✨ MECANISMO 2: Intrinsic Plasticity (proporção configurável do esforço homeostático) ✨
        // Ajusta o threshold.
        // Se rate_error < 0 (hipoativo) → threshold DEVE CAIR (ficar mais sensível)
        // Se rate_error > 0 (hiperativo) → threshold DEVE SUBIR (ficar menos sensível)
        let threshold_delta = effective_error * self.homeo_eta * self.homeo_threshold_ratio;

        self.threshold += threshold_delta;

        // Clamp adaptativo: evita colapso/negativação sem matar autorregulação
        let energy_frac = self.glia.energy_fraction();
        // Overshoot aumenta o piso; decai lentamente a cada aplicação
        self.overshoot_count *= 0.99;
        let min_threshold = (self.base_threshold * 0.02
            + (1.0 - energy_frac) * self.base_threshold * 0.3)
            * (1.0 + 0.1 * self.overshoot_count)
            .max(0.001);
        self.threshold = self.threshold.clamp(min_threshold, 5.0);
    }

    /// Processa um passo completo de atualização do neurónio
    ///
    /// Esta função encapsula o fluxo completo:
    /// 1. Integração de sinais (Dendritoma)
    /// 2. Modulação (Glia)
    /// 3. Decisão de disparo
    /// 4. Energy-gated learning (plasticidade controlada por energia)
    /// 5. Aprendizado (se disparou E tem energia suficiente)
    /// 6. Atualização de estado
    ///
    /// # Argumentos
    /// * `inputs` - Vetor de sinais de entrada
    /// * `current_time` - Passo de tempo atual
    ///
    /// # Retorna
    /// O sinal de saída do neurónio
    pub fn step(&mut self, inputs: &[f64], current_time: i64) -> f64 {
        // Fase 1: Integração
        let integrated_potential = self.dendritoma.integrate(inputs);

        // Fase 2: Modulação glial
        let modulated_potential = self.glia.modulate(integrated_potential);

        // Determina se há input externo direto (para STDP gated)
        // Input externo = qualquer elemento > threshold de significância (0.5)
        let has_external_input = inputs.iter().any(|&inp| inp > 0.5);

        // Fase 3: Decisão de disparo
        self.decide_to_fire(modulated_potential, current_time, has_external_input);

        // Fase 4: Energy-gated + BCM learning - energia e metaplasticidade governam plasticidade
        let energy_gain = self.compute_plasticity_gain();
        let bcm_gain = self.compute_bcm_gain();

        // Combinação multiplicativa: energia controla se aprende, BCM modula quanto
        let plasticity_gain = energy_gain * bcm_gain;
        self.dendritoma.set_plasticity_gain(plasticity_gain);

        // Fase 5: Aprendizado (apenas se disparou E tem energia/plasticidade)
        if self.is_firing && plasticity_gain > 0.0 {
            // Captura pesos antes do aprendizado para calcular custo
            let weights_before = self.dendritoma.weights.clone();

            self.dendritoma.apply_learning(inputs);

            // Calcula magnitude da mudança (custo metabólico de plasticidade)
            let plasticity_cost: f64 = self.dendritoma.weights
                .iter()
                .zip(weights_before.iter())
                .map(|(w_new, w_old)| (w_new - w_old).abs())
                .sum();

            // Aprendizado consome energia proporcional às mudanças
            self.glia.consume_plasticity_energy(plasticity_cost);
        }

        // Fase 6: Atualização de estado
        self.glia.update_state(self.is_firing);
        self.update_memory(inputs);

        // Atualiza taxa de disparo recente (alpha = 0.01 para média de ~100 passos)
        let firing_value = if self.is_firing { 1.0 } else { 0.0 };
        self.recent_firing_rate = 0.99 * self.recent_firing_rate + 0.01 * firing_value;

        // Atualiza limiar metaplástico BCM
        self.update_meta_threshold(self.is_firing);

        // Fase 7: Plasticidade homeostática (aplicada periodicamente)
        // CORRIGIDO: Passa has_external_input para evitar homeostase durante inatividade
        self.apply_homeostatic_plasticity(current_time, has_external_input);

        self.output_signal
    }

    /// Útil para debugging e visualização
    pub fn get_modulated_potential(&self, inputs: &[f64]) -> f64 {
        let integrated = self.dendritoma.integrate(inputs);
        self.glia.modulate(integrated)
    }

    /// Define o período refratário
    pub fn set_refractory_period(&mut self, period: i64) {
        self.refractory_period = period;
    }

    /// Retorna o limiar base configurado na criação
    pub fn base_threshold(&self) -> f64 {
        self.base_threshold
    }

    /// Define a taxa de atualização da memória
    pub fn set_memory_alpha(&mut self, alpha: f64) {
        self.memory_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Retorna o tempo do último spike (para STDP)
    ///
    /// # Retorna
    /// `Some(time)` se o neurônio já disparou, `None` caso contrário
    pub fn get_last_spike_time(&self) -> Option<i64> {
        if self.last_fire_time >= 0 {
            Some(self.last_fire_time)
        } else {
            None
        }
    }

    /// Retorna uma referência ao histórico de spikes (para análise)
    pub fn spike_history(&self) -> &VecDeque<i64> {
        &self.spike_history
    }

    /// Ajusta metabolismo para modo sono (baixo consumo, alta recuperação)
    pub fn enter_sleep_mode(&mut self) {
        self.glia.enter_sleep_mode();
    }

    /// Restaura metabolismo para modo vigília (consumo normal)
    pub fn exit_sleep_mode(&mut self) {
        self.glia.exit_sleep_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nenv_initialization() {
        let neuron = NENV::excitatory(0, 10, 0.5);

        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.neuron_type, NeuronType::Excitatory);
        assert_eq!(neuron.threshold, 0.5);
        assert_eq!(neuron.memory_trace.len(), 10);
        assert!(!neuron.is_firing);
        assert_eq!(neuron.output_signal, 0.0);
    }

    #[test]
    fn test_excitatory_neuron_output() {
        let mut neuron = NENV::excitatory(0, 2, 1.5); // Limiar ajustado

        // Configura pesos não normalizados para garantir disparo
        // potencial = 1.0*1.0 + 1.0*1.0 = 2.0
        // modulado = 2.0 * 1.0 (energia_max) * 1.0 (priority) = 2.0 > 1.5
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;

        let inputs = vec![1.0, 1.0];
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 0);

        assert!(neuron.is_firing);
        assert_eq!(neuron.output_signal, 1.0);
    }

    #[test]
    fn test_inhibitory_neuron_output() {
        let mut neuron = NENV::inhibitory(0, 2, 1.5); // Limiar ajustado

        // Configura pesos não normalizados para garantir disparo
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;

        let inputs = vec![1.0, 1.0];
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 0, false);

        assert!(neuron.is_firing);
        assert_eq!(neuron.output_signal, -1.0);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = NENV::excitatory(0, 2, 1.5); // Limiar ajustado
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;
        neuron.set_refractory_period(5);

        let inputs = vec![1.0, 1.0];

        // Primeiro disparo no tempo 0
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 0, false);
        assert!(neuron.is_firing);

        // Tentativa de disparo no tempo 2 (dentro do período refratário)
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 2, false);
        assert!(!neuron.is_firing);

        // Tentativa de disparo no tempo 6 (fora do período refratário)
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 6, false);
        assert!(neuron.is_firing);
    }

    #[test]
    fn test_memory_update() {
        let mut neuron = NENV::excitatory(0, 3, 0.5);
        neuron.set_memory_alpha(0.5); // Alta taxa para teste rápido

        let inputs1 = vec![1.0, 0.0, 0.0];
        neuron.update_memory(&inputs1);

        // Após uma atualização, memória deve ser 0.5 * inputs1
        assert_relative_eq!(neuron.memory_trace[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(neuron.memory_trace[1], 0.0, epsilon = 1e-10);

        let inputs2 = vec![0.0, 1.0, 0.0];
        neuron.update_memory(&inputs2);

        // Memória do primeiro canal decai, segundo canal aumenta
        assert_relative_eq!(neuron.memory_trace[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(neuron.memory_trace[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_threshold_prevents_firing() {
        let mut neuron = NENV::excitatory(0, 2, 10.0); // Limiar muito alto
        neuron.dendritoma.weights = vec![0.5, 0.5];

        let inputs = vec![1.0, 1.0];
        let potential = neuron.get_modulated_potential(&inputs);

        neuron.decide_to_fire(potential, 0, false);
        assert!(!neuron.is_firing);
    }

    #[test]
    fn test_energy_depletion_prevents_firing() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.energy = 0.0; // Sem energia

        let inputs = vec![1.0, 1.0];
        let potential = neuron.get_modulated_potential(&inputs);

        // Potencial integrado é alto, mas modulação reduz a zero
        assert_relative_eq!(potential, 0.0, epsilon = 1e-10);

        neuron.decide_to_fire(potential, 0, false);
        assert!(!neuron.is_firing);
    }

    // === Testes v0.2.0: Priority & Alert Level ===

    #[test]
    fn test_compute_novelty_zero_for_familiar() {
        let mut neuron = NENV::excitatory(0, 3, 0.5);

        // Define memória como um padrão específico
        neuron.memory_trace = vec![0.5, 0.3, 0.2];

        // Input idêntico à memória
        let inputs = vec![0.5, 0.3, 0.2];
        let novelty = neuron.compute_novelty(&inputs);

        // Novidade deve ser zero (completamente familiar)
        assert_relative_eq!(novelty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_novelty_high_for_novel() {
        let mut neuron = NENV::excitatory(0, 3, 0.5);

        // Memória com zeros (nenhum input recente)
        neuron.memory_trace = vec![0.0, 0.0, 0.0];

        // Input forte e completamente novo
        let inputs = vec![1.0, 1.0, 1.0];
        let novelty = neuron.compute_novelty(&inputs);

        // Novidade deve ser 1.0 (média de diferenças absolutas)
        assert_relative_eq!(novelty, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_novelty_partial() {
        let mut neuron = NENV::excitatory(0, 4, 0.5);

        neuron.memory_trace = vec![0.5, 0.5, 0.5, 0.5];
        let inputs = vec![1.0, 0.0, 1.0, 0.0];

        let novelty = neuron.compute_novelty(&inputs);

        // Diferenças: |1.0-0.5| + |0.0-0.5| + |1.0-0.5| + |0.0-0.5| = 2.0
        // Média: 2.0 / 4 = 0.5
        assert_relative_eq!(novelty, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_update_priority_increases_with_novelty() {
        let mut neuron = NENV::excitatory(0, 2, 0.5);

        // Priority inicial deve ser 1.0
        assert_eq!(neuron.glia.priority, 1.0);

        // Atualiza com novelty=0.5 e sensitivity_factor=1.0
        neuron.update_priority(0.5, 1.0);

        // Priority = 1.0 + 0.5*1.0 = 1.5
        assert_relative_eq!(neuron.glia.priority, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_update_priority_sensitivity_factor() {
        let mut neuron = NENV::excitatory(0, 2, 0.5);

        // Sensitivity factor = 2.0 (mais sensível)
        neuron.update_priority(0.5, 2.0);

        // Priority = 1.0 + 0.5*2.0 = 2.0
        assert_relative_eq!(neuron.glia.priority, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_update_priority_clamps_at_max() {
        let mut neuron = NENV::excitatory(0, 2, 0.5);

        // Novelty muito alto com sensitivity alto
        neuron.update_priority(10.0, 1.0);

        // Priority deve ser limitado a 3.0
        assert_eq!(neuron.glia.priority, 3.0);
    }

    #[test]
    fn test_priority_modulates_potential() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![0.7071067811865475, 0.7071067811865475];
        neuron.glia.energy = 100.0; // Energia máxima
        neuron.glia.priority = 2.0; // Priority dobrado

        let inputs = vec![1.0, 1.0];
        let potential = neuron.get_modulated_potential(&inputs);

        // Sem priority: ~1.41
        // Com priority=2.0: ~2.82
        assert!(potential > 2.5);
        assert!(potential < 3.0);
    }

    #[test]
    fn test_priority_enables_firing() {
        let mut neuron = NENV::excitatory(0, 2, 1.5); // Limiar alto
        neuron.dendritoma.weights = vec![0.7071067811865475, 0.7071067811865475];
        neuron.glia.priority = 1.0; // Priority normal

        let inputs = vec![1.0, 1.0];

        // Sem priority alto, não dispara
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 0, false);
        assert!(!neuron.is_firing);

        // Com priority alto, dispara
        neuron.glia.priority = 2.0;
        let potential_boosted = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential_boosted, 1, false);
        assert!(neuron.is_firing);
    }

    // === Testes v0.4.0: Spike History & STDP Support ===

    #[test]
    fn test_get_last_spike_time_none_initially() {
        let neuron = NENV::excitatory(0, 2, 0.5);

        // Neurônio nunca disparou, deve retornar None
        assert_eq!(neuron.get_last_spike_time(), None);
    }

    #[test]
    fn test_get_last_spike_time_after_firing() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;

        let inputs = vec![1.0, 1.0];

        // Dispara no tempo 100
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 100, false);

        assert!(neuron.is_firing);
        assert_eq!(neuron.get_last_spike_time(), Some(100));
    }

    #[test]
    fn test_spike_history_updates_on_firing() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;

        let inputs = vec![1.0, 1.0];

        // Inicialmente vazio
        assert_eq!(neuron.spike_history().len(), 0);

        // Dispara no tempo 10
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 10, false);
        assert_eq!(neuron.spike_history().len(), 1);
        assert_eq!(neuron.spike_history()[0], 10);

        // Dispara no tempo 20
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 20, false);
        assert_eq!(neuron.spike_history().len(), 2);
        assert_eq!(neuron.spike_history()[1], 20);
    }

    #[test]
    fn test_spike_history_limits_to_10_spikes() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;
        neuron.set_refractory_period(0); // Remove período refratário para teste

        let inputs = vec![1.0, 1.0];

        // Dispara 15 vezes
        for t in 0..15 {
            let potential = neuron.get_modulated_potential(&inputs);
            neuron.decide_to_fire(potential, t * 10, false);
        }

        // Deve manter apenas os últimos 10 spikes
        assert_eq!(neuron.spike_history().len(), 10);

        // O primeiro spike deve ser do tempo 50 (índice 5)
        assert_eq!(neuron.spike_history()[0], 50);

        // O último spike deve ser do tempo 140 (índice 14)
        assert_eq!(neuron.spike_history()[9], 140);
    }

    #[test]
    fn test_spike_history_empty_when_no_firing() {
        let mut neuron = NENV::excitatory(0, 2, 10.0); // Limiar muito alto
        neuron.dendritoma.weights = vec![0.1, 0.1];

        let inputs = vec![1.0, 1.0];

        // Tenta disparar mas não consegue
        for t in 0..5 {
            let potential = neuron.get_modulated_potential(&inputs);
            neuron.decide_to_fire(potential, t * 10, false);
        }

        // Histórico deve permanecer vazio
        assert_eq!(neuron.spike_history().len(), 0);
    }

    #[test]
    fn test_spike_history_respects_refractory_period() {
        let mut neuron = NENV::excitatory(0, 2, 0.1);
        neuron.dendritoma.weights = vec![1.0, 1.0];
        neuron.glia.priority = 1.0;
        neuron.set_refractory_period(5);

        let inputs = vec![1.0, 1.0];

        // Dispara no tempo 0
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 0, false);
        assert_eq!(neuron.spike_history().len(), 1);

        // Tenta disparar no tempo 2 (dentro do período refratário)
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 2, false);
        assert_eq!(neuron.spike_history().len(), 1); // Não deve adicionar

        // Dispara no tempo 6 (fora do período refratário)
        let potential = neuron.get_modulated_potential(&inputs);
        neuron.decide_to_fire(potential, 6, false);
        assert_eq!(neuron.spike_history().len(), 2); // Deve adicionar
        assert_eq!(neuron.spike_history()[1], 6);
    }
}
