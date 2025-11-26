//! # STDP (Spike-Timing-Dependent Plasticity)
//!
//! Implementa aprendizado STDP com janela assimétrica otimizada.
//!
//! ## Novidades v2.0
//!
//! - tau_plus > tau_minus para favorecer padrões causais
//! - Modulação por reward (3-factor learning)
//! - Integração com synaptic tagging
//!
//! ## Referências
//!
//! - Bi & Poo (1998) - Synaptic modification by correlated activity
//! - Markram et al. (1997) - Regulation of synaptic efficacy

use crate::constants::timing;
use crate::constants::learning;

/// Configuração do algoritmo STDP
#[derive(Debug, Clone)]
pub struct STDPConfig {
    /// Amplitude de LTP (potenciação)
    pub a_plus: f64,

    /// Amplitude de LTD (depressão)
    pub a_minus: f64,

    /// Constante de tempo para janela de potenciação (ms)
    pub tau_plus: f64,

    /// Constante de tempo para janela de depressão (ms)
    pub tau_minus: f64,

    /// Janela temporal para STDP (ms)
    pub window: i64,

    /// Habilita modulação por reward
    pub reward_modulation: bool,

    /// Ganho de plasticidade global
    pub plasticity_gain: f64,

    /// Clamp máximo para pesos
    pub weight_clamp: f64,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: learning::STDP_A_PLUS,
            a_minus: learning::STDP_A_MINUS,
            tau_plus: timing::STDP_TAU_PLUS,
            tau_minus: timing::STDP_TAU_MINUS,
            window: timing::STDP_WINDOW,
            reward_modulation: true,
            plasticity_gain: 1.0,
            weight_clamp: learning::WEIGHT_CLAMP,
        }
    }
}

impl STDPConfig {
    /// Cria configuração para tarefa de RL (mais sensível a reward)
    pub fn for_reinforcement_learning() -> Self {
        Self {
            a_plus: learning::STDP_A_PLUS * 1.2,
            a_minus: learning::STDP_A_MINUS * 1.2,
            tau_plus: timing::STDP_TAU_PLUS * 1.5,  // Janela mais longa para crédito temporal
            tau_minus: timing::STDP_TAU_MINUS,
            window: timing::STDP_WINDOW * 2,
            reward_modulation: true,
            plasticity_gain: 1.0,
            weight_clamp: learning::WEIGHT_CLAMP,
        }
    }

    /// Cria configuração para classificação (mais rápida)
    pub fn for_classification() -> Self {
        Self {
            a_plus: learning::STDP_A_PLUS * 1.5,
            a_minus: learning::STDP_A_MINUS * 1.5,
            tau_plus: timing::STDP_TAU_PLUS * 0.8,
            tau_minus: timing::STDP_TAU_MINUS * 0.8,
            window: timing::STDP_WINDOW,
            reward_modulation: false,
            plasticity_gain: 1.0,
            weight_clamp: learning::WEIGHT_CLAMP,
        }
    }

    /// Retorna ratio LTP/LTD
    pub fn ltp_ltd_ratio(&self) -> f64 {
        self.a_plus / self.a_minus
    }
}

/// Trait para algoritmos de aprendizado STDP
pub trait STDPLearning {
    /// Aplica STDP para um par de spikes
    ///
    /// # Argumentos
    /// * `synapse_idx` - Índice da sinapse
    /// * `delta_t` - Diferença temporal (post_time - pre_time)
    /// * `reward` - Sinal de reward [-1.0, 1.0]
    /// * `plasticity` - Fator de plasticidade local da sinapse
    ///
    /// # Retorna
    /// Mudança de peso aplicada
    fn apply_stdp_pair(
        &mut self,
        synapse_idx: usize,
        delta_t: i64,
        reward: f64,
        plasticity: f64,
    ) -> f64;

    /// Aplica STDP para múltiplos pares de spikes
    fn apply_stdp_batch(
        &mut self,
        pre_spike_times: &[Option<i64>],
        post_spike_time: i64,
    ) -> usize;

    /// Retorna configuração atual
    fn config(&self) -> &STDPConfig;

    /// Modifica configuração
    fn set_config(&mut self, config: STDPConfig);
}

/// Implementação de STDP assimétrico
#[derive(Debug, Clone)]
pub struct AsymmetricSTDP {
    /// Configuração
    config: STDPConfig,

    /// Pesos sinápticos
    pub weights: Vec<f64>,

    /// Fator de plasticidade por sinapse
    pub plasticity: Vec<f64>,

    /// Synaptic tags para consolidação
    pub synaptic_tags: Vec<f64>,

    /// Taxa de decaimento das tags
    pub tag_decay_rate: f64,

    /// Sensibilidade à dopamina
    pub dopamine_sensitivity: f64,

    /// Estatísticas
    total_ltp: f64,
    total_ltd: f64,
    update_count: u64,
}

impl AsymmetricSTDP {
    /// Cria novo sistema STDP
    pub fn new(num_synapses: usize) -> Self {
        Self {
            config: STDPConfig::default(),
            weights: vec![0.05; num_synapses],
            plasticity: vec![1.0; num_synapses],
            synaptic_tags: vec![0.0; num_synapses],
            tag_decay_rate: learning::WEIGHT_DECAY * 80.0,  // ~0.008
            dopamine_sensitivity: 5.0,
            total_ltp: 0.0,
            total_ltd: 0.0,
            update_count: 0,
        }
    }

    /// Cria com configuração personalizada
    pub fn with_config(num_synapses: usize, config: STDPConfig) -> Self {
        let mut stdp = Self::new(num_synapses);
        stdp.config = config;
        stdp
    }

    /// Retorna número de sinapses
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Verifica se está vazio
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Calcula parâmetros STDP efetivos
    fn effective_params(&self) -> (f64, f64) {
        (
            self.config.a_plus * self.config.plasticity_gain,
            self.config.a_minus * self.config.plasticity_gain,
        )
    }

    /// Decaimento das synaptic tags
    pub fn decay_tags(&mut self) {
        for tag in &mut self.synaptic_tags {
            *tag *= 1.0 - self.tag_decay_rate;
            if *tag < 1e-3 {
                *tag = 0.0;
            }
        }
    }

    /// Retorna estatísticas de aprendizado
    pub fn get_learning_stats(&self) -> STDPStats {
        STDPStats {
            total_ltp: self.total_ltp,
            total_ltd: self.total_ltd,
            ltp_ltd_ratio: if self.total_ltd > 0.0 {
                self.total_ltp / self.total_ltd
            } else {
                f64::INFINITY
            },
            update_count: self.update_count,
            avg_tag: self.synaptic_tags.iter().sum::<f64>() / self.synaptic_tags.len() as f64,
        }
    }

    /// Define amplitudes STDP
    pub fn set_amplitudes(&mut self, a_plus: f64, a_minus: f64) {
        self.config.a_plus = a_plus.max(0.0);
        self.config.a_minus = a_minus.max(0.0);
    }

    /// Retorna amplitudes atuais
    pub fn get_amplitudes(&self) -> (f64, f64) {
        (self.config.a_plus, self.config.a_minus)
    }

    /// Define ganho de plasticidade
    pub fn set_plasticity_gain(&mut self, gain: f64) {
        self.config.plasticity_gain = gain.clamp(0.0, 2.0);
    }
}

impl STDPLearning for AsymmetricSTDP {
    fn apply_stdp_pair(
        &mut self,
        synapse_idx: usize,
        delta_t: i64,
        reward: f64,
        plasticity: f64,
    ) -> f64 {
        if synapse_idx >= self.weights.len() || delta_t == 0 {
            return 0.0;
        }

        let (a_plus_eff, a_minus_eff) = self.effective_params();

        // Garante aprendizado mínimo mesmo com baixa energia
        let min_gain = 0.1;
        let final_a_plus = a_plus_eff.max(self.config.a_plus * min_gain);
        let final_a_minus = a_minus_eff.max(self.config.a_minus * min_gain);

        let weight_change = if delta_t > 0 {
            // Causal (pré antes de pós): LTP
            if reward >= 0.0 || !self.config.reward_modulation {
                let reward_modulation = if self.config.reward_modulation {
                    1.0 + reward * 2.0
                } else {
                    1.0
                };
                let change = final_a_plus
                    * plasticity
                    * (-delta_t as f64 / self.config.tau_plus).exp()
                    * reward_modulation;
                self.total_ltp += change.abs();
                change
            } else {
                // Reward negativo: inverte para LTD
                let punishment_strength = -reward;
                let change = -final_a_plus
                    * plasticity
                    * (-delta_t as f64 / self.config.tau_plus).exp()
                    * punishment_strength;
                self.total_ltd += change.abs();
                change
            }
        } else {
            // Anti-causal (pós antes de pré): LTD
            let reward_modulation = if self.config.reward_modulation {
                (1.0 + reward).max(0.0)
            } else {
                1.0
            };
            let change = -final_a_minus
                * plasticity
                * (delta_t.abs() as f64 / self.config.tau_minus).exp()
                * reward_modulation;
            self.total_ltd += change.abs();
            change
        };

        // Aplica mudança
        self.weights[synapse_idx] += weight_change;

        // Synaptic tagging
        let magnitude = weight_change.abs();
        if magnitude > 1e-4 {
            let relevance_factor = 1.0 + (reward.abs() * self.dopamine_sensitivity);
            self.synaptic_tags[synapse_idx] += magnitude * relevance_factor * 10.0;
            self.synaptic_tags[synapse_idx] = self.synaptic_tags[synapse_idx].min(2.0);
        }

        // Decay proporcional
        let proportional_decay = self.weights[synapse_idx] * 0.0001;
        self.weights[synapse_idx] -= proportional_decay;

        // Clamp
        self.weights[synapse_idx] =
            self.weights[synapse_idx].clamp(0.0, self.config.weight_clamp);

        self.update_count += 1;

        weight_change
    }

    fn apply_stdp_batch(
        &mut self,
        pre_spike_times: &[Option<i64>],
        post_spike_time: i64,
    ) -> usize {
        assert_eq!(pre_spike_times.len(), self.weights.len());

        let mut modified_count = 0;

        for i in 0..self.weights.len() {
            if let Some(pre_time) = pre_spike_times[i] {
                let delta_t = post_spike_time - pre_time;

                if delta_t.abs() <= self.config.window {
                    let plasticity = self.plasticity.get(i).copied().unwrap_or(1.0);
                    self.apply_stdp_pair(i, delta_t, 0.0, plasticity);
                    modified_count += 1;
                }
            }
        }

        // Aplica decaimento suave
        for weight in &mut self.weights {
            *weight *= 1.0 - learning::WEIGHT_DECAY;
            *weight = weight.clamp(0.0, self.config.weight_clamp);
        }

        modified_count
    }

    fn config(&self) -> &STDPConfig {
        &self.config
    }

    fn set_config(&mut self, config: STDPConfig) {
        self.config = config;
    }
}

/// Estatísticas de STDP
#[derive(Debug, Clone)]
pub struct STDPStats {
    pub total_ltp: f64,
    pub total_ltd: f64,
    pub ltp_ltd_ratio: f64,
    pub update_count: u64,
    pub avg_tag: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_config_defaults() {
        let config = STDPConfig::default();

        // tau_plus deve ser maior que tau_minus (assimetria)
        assert!(config.tau_plus > config.tau_minus);

        // a_plus deve ser maior que a_minus
        assert!(config.a_plus > config.a_minus);
    }

    #[test]
    fn test_stdp_ltp() {
        let mut stdp = AsymmetricSTDP::new(5);

        // delta_t > 0 (causal) deve resultar em LTP
        let change = stdp.apply_stdp_pair(0, 10, 0.0, 1.0);

        assert!(change > 0.0);
        assert!(stdp.weights[0] > 0.05);  // Peso aumentou
    }

    #[test]
    fn test_stdp_ltd() {
        let mut stdp = AsymmetricSTDP::new(5);

        // delta_t < 0 (anti-causal) deve resultar em LTD
        let change = stdp.apply_stdp_pair(0, -10, 0.0, 1.0);

        assert!(change < 0.0);
    }

    #[test]
    fn test_stdp_reward_modulation() {
        let mut stdp = AsymmetricSTDP::new(5);

        // Com reward positivo, LTP deve ser mais forte
        let change_no_reward = stdp.apply_stdp_pair(0, 10, 0.0, 1.0);
        stdp.weights[0] = 0.05;  // Reset

        let change_with_reward = stdp.apply_stdp_pair(0, 10, 1.0, 1.0);

        assert!(change_with_reward > change_no_reward);
    }

    #[test]
    fn test_synaptic_tagging() {
        let mut stdp = AsymmetricSTDP::new(5);

        // STDP deve criar tags
        stdp.apply_stdp_pair(0, 10, 1.0, 1.0);

        assert!(stdp.synaptic_tags[0] > 0.0);
    }
}
