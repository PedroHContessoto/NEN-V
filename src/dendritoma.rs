//! Módulo responsável pela integração de sinais de entrada e aprendizado sináptico
//!
//! O Dendritoma recebe e pondera os sinais de entrada, aplicando aprendizado
//! STDP com eligibility traces, short-term plasticity e normalização competitiva.
//!
//! ## Novidades v2.0
//!
//! - **Eligibility Traces**: Permite crédito temporal tardio (3-factor learning)
//! - **Short-Term Plasticity (STP)**: Facilitação/Depressão de curto prazo
//! - **STDP Assimétrico**: tau_plus > tau_minus para favorecer padrões causais
//! - **Normalização Competitiva**: Orçamento sináptico limitado

#[derive(Debug, Clone)]
pub struct Dendritoma {
    /// Pesos sinápticos de curto prazo (STM - Short-Term Memory)
    pub weights: Vec<f64>,

    /// Pesos sinápticos de longo prazo (LTM - Long-Term Memory)
    pub weights_ltm: Vec<f64>,

    /// Fator de plasticidade para cada peso (modula a taxa de aprendizado)
    pub plasticity: Vec<f64>,

    // ========== ELIGIBILITY TRACES (NOVO v2.0) ==========
    /// Eligibility trace para cada sinapse [0.0, 1.0]
    /// Decai exponencialmente, aumenta com atividade pré-pós correlacionada
    /// Permite aprendizado com reward tardio (3-factor learning)
    pub eligibility_trace: Vec<f64>,

    /// Constante de tempo do eligibility trace (em timesteps)
    /// Valores típicos: 100-500 (representa ~100-500ms em escala biológica)
    pub trace_tau: f64,

    /// Incremento do trace quando há correlação pré-pós
    pub trace_increment: f64,

    // ========== SHORT-TERM PLASTICITY (NOVO v2.0) ==========
    /// Recursos sinápticos disponíveis [0.0, 1.0]
    /// Simula vesículas de neurotransmissor disponíveis
    pub synaptic_resources: Vec<f64>,

    /// Taxa de recuperação de recursos (timesteps para 63% de recuperação)
    pub stp_recovery_tau: f64,

    /// Fração de recursos usados por spike pré-sináptico
    pub stp_use_fraction: f64,

    /// Fator de facilitação (aumenta com uso repetido em curto prazo)
    pub stp_facilitation: Vec<f64>,

    /// Taxa de decaimento da facilitação
    pub stp_facilitation_decay: f64,

    // ========== PARÃ‚METROS DE APRENDIZADO ==========
    learning_rate: f64,

    // ParÃ¢metros de STDP (Spike-Timing-Dependent Plasticity)
    /// Amplitude de potenciação (LTP) quando préâ†’post
    stdp_a_plus: f64,

    /// Amplitude de depressão (LTD) quando postâ†’pre
    stdp_a_minus: f64,

    /// Constante de tempo para janela de potenciação (ms)
    /// ASSIMÃ‰TRICO v2.0: tau_plus > tau_minus
    stdp_tau_plus: f64,

    /// Constante de tempo para janela de depressão (ms)
    stdp_tau_minus: f64,

    /// Janela temporal para STDP (ms)
    stdp_window: i64,

    /// ParÃ¢metros de iSTDP (Inhibitory STDP) - Vogels et al. 2011
    pub istdp_learning_rate: f64,
    pub istdp_target_rate: f64,

    /// Taxa de decaimento dos pesos STM
    weight_decay: f64,

    /// Ganho de plasticidade global (modulado por energia, BCM, etc.)
    pub plasticity_gain: f64,

    /// Estabilidade de cada peso LTM [0.0, 1.0]
    pub ltm_stability: Vec<f64>,

    /// Limite máximo para pesos (clamp superior)
    pub weight_clamp: f64,

    // ========== SYNAPTIC TAGGING AND CAPTURE ==========
    pub synaptic_tags: Vec<f64>,
    pub tag_decay_rate: f64,
    pub capture_threshold: f64,
    pub dopamine_sensitivity: f64,

    // ========== NORMALIZAÃ‡ÃƒO COMPETITIVA (NOVO v2.0) ==========
    /// Soma alvo dos pesos (orçamento sináptico)
    pub target_weight_sum: f64,

    /// Habilita normalização competitiva
    pub competitive_normalization_enabled: bool,

    /// Intervalo para aplicar normalização (em timesteps)
    pub normalization_interval: i64,

    /// Contador para normalização
    normalization_counter: i64,
}

impl Dendritoma {
    /// Cria um novo Dendritoma com pesos aleatórios iniciais
    pub fn new(num_inputs: usize) -> Self {
        let weights: Vec<f64> = vec![0.05; num_inputs];
        let plasticity = vec![1.0; num_inputs];
        let weights_ltm = vec![0.0; num_inputs];
        let ltm_stability = vec![0.0; num_inputs];
        let synaptic_tags = vec![0.0; num_inputs];

        // Inicializa eligibility traces zerados
        let eligibility_trace = vec![0.0; num_inputs];

        // Inicializa recursos sinápticos cheios
        let synaptic_resources = vec![1.0; num_inputs];
        let stp_facilitation = vec![1.0; num_inputs];

        Self {
            weights,
            weights_ltm,
            plasticity,

            // Eligibility Traces v2.0
            eligibility_trace,
            trace_tau: 200.0,      // ~200ms de memória
            trace_increment: 0.15, // Incremento por correlação

            // Short-Term Plasticity v2.0
            synaptic_resources,
            stp_recovery_tau: 77.84214867927184,   // OTIMIZADO via hyperparameter_search (range: 50-300)
            stp_use_fraction: 0.15,   // 15% dos recursos por spike
            stp_facilitation,
            stp_facilitation_decay: 0.95,

            // ParÃ¢metros de aprendizado otimizados
            learning_rate: 0.008,  // Aumentado para aprendizado mais rápido

            // STDP ASSIMÃ‰TRICO v2.0
            // tau_plus > tau_minus favorece padrões causais
            stdp_a_plus: 0.015,    // LTP mais forte
            stdp_a_minus: 0.006,   // LTD mais fraco (ratio 2.5:1)
            stdp_tau_plus: 40.0,   // Janela LTP MAIOR (era 20)
            stdp_tau_minus: 15.0,  // Janela LTD menor
            stdp_window: 50,       // Janela total maior (era 20)

            // iSTDP
            istdp_learning_rate: 0.001,
            istdp_target_rate: 0.15,

            weight_decay: 0.0001,
            plasticity_gain: 1.0,
            ltm_stability,
            weight_clamp: 2.5,

            // Synaptic Tagging
            synaptic_tags,
            tag_decay_rate: 0.008,      // Decaimento mais lento
            capture_threshold: 0.15,     // Threshold mais acessível
            dopamine_sensitivity: 5.0,

            // Normalização Competitiva v2.0
            target_weight_sum: num_inputs as f64 * 0.1, // 0.1 por sinapse em média
            competitive_normalization_enabled: true,
            normalization_interval: 100,
            normalization_counter: 0,
        }
    }

    /// Cria um Dendritoma com parÃ¢metros personalizados
    pub fn with_params(num_inputs: usize, learning_rate: f64) -> Self {
        let mut dendritoma = Self::new(num_inputs);
        dendritoma.learning_rate = learning_rate;
        dendritoma
    }

    /// Configura parÃ¢metros STDP personalizados
    pub fn set_stdp_params(
        &mut self,
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
    ) {
        self.stdp_a_plus = a_plus.max(0.0);
        self.stdp_a_minus = a_minus.max(0.0);
        self.stdp_tau_plus = tau_plus.max(1.0);
        self.stdp_tau_minus = tau_minus.max(1.0);
    }

    /// Configura parÃ¢metros de Eligibility Trace
    pub fn set_trace_params(&mut self, tau: f64, increment: f64) {
        self.trace_tau = tau.max(10.0);
        self.trace_increment = increment.clamp(0.01, 0.5);
    }

    /// Configura parÃ¢metros de Short-Term Plasticity
    pub fn set_stp_params(&mut self, recovery_tau: f64, use_fraction: f64) {
        self.stp_recovery_tau = recovery_tau.max(10.0);
        self.stp_use_fraction = use_fraction.clamp(0.05, 0.5);
    }

    // ========================================================================
    // INTEGRAÃ‡ÃƒO COM STP (Short-Term Plasticity)
    // ========================================================================

    /// Integra os sinais de entrada com Short-Term Plasticity
    ///
    /// STP modula a eficácia sináptica baseado em uso recente:
    /// - Depressão: Inputs repetitivos perdem eficácia (recursos esgotados)
    /// - Facilitação: Inputs em sequÃªncia rápida podem ter boost temporário
    ///
    /// Isso naturalmente favorece padrões estruturados sobre ruído aleatório.
    ///
    /// CORREÇÃO: Piso mínimo de recursos STP para evitar silenciamento completo
    pub fn integrate(&mut self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Número de inputs deve ser igual ao número de pesos"
        );

        let mut potential = 0.0;

        // Piso mínimo de recursos para garantir transmissão basal
        let min_resources = 0.2;

        for i in 0..self.weights.len() {
            // Peso efetivo = (STM + LTM) * recursos * facilitação
            let base_weight = self.weights[i] + self.weights_ltm[i];
            // CORREÇÃO: Recursos nunca vão abaixo do mínimo
            let effective_resources = self.synaptic_resources[i].max(min_resources);
            let stp_modulation = effective_resources * self.stp_facilitation[i];
            let effective_weight = base_weight * stp_modulation;

            potential += inputs[i] * effective_weight;

            // Atualiza STP se input ativo
            if inputs[i].abs() > 0.1 {
                // Consome recursos (depressão) - respeitando piso
                self.synaptic_resources[i] *= 1.0 - self.stp_use_fraction;
                self.synaptic_resources[i] = self.synaptic_resources[i].max(min_resources);

                // Aumenta facilitação temporariamente
                self.stp_facilitation[i] += 0.1;
                self.stp_facilitation[i] = self.stp_facilitation[i].min(2.0);
            }
        }

        potential
    }

    /// Atualiza o estado do STP (chamado a cada timestep)
    pub fn update_stp(&mut self) {
        for i in 0..self.synaptic_resources.len() {
            // Recuperação de recursos (exponencial)
            let recovery = (1.0 - self.synaptic_resources[i]) / self.stp_recovery_tau;
            self.synaptic_resources[i] += recovery;
            self.synaptic_resources[i] = self.synaptic_resources[i].min(1.0);

            // Decaimento da facilitação
            self.stp_facilitation[i] = 1.0 + (self.stp_facilitation[i] - 1.0) * self.stp_facilitation_decay;
        }
    }

    // ========================================================================
    // ELIGIBILITY TRACES (3-Factor Learning)
    // ========================================================================

    /// Atualiza eligibility traces a cada timestep
    ///
    /// Traces "lembram" quais sinapses estavam ativas recentemente,
    /// permitindo que reward tardio ainda afete as sinapses corretas.
    ///
    /// # Argumentos
    /// * `pre_active` - Quais inputs pré-sinápticos estavam ativos
    /// * `post_fired` - Se o neurônio pós-sináptico disparou
    pub fn update_eligibility_traces(&mut self, pre_active: &[f64], post_fired: bool) {
        assert_eq!(pre_active.len(), self.eligibility_trace.len());

        for i in 0..self.eligibility_trace.len() {
            // Decaimento exponencial do trace
            self.eligibility_trace[i] *= (-1.0 / self.trace_tau).exp();

            // Incrementa trace se pré estava ativo E pós disparou (correlação Hebbiana)
            if pre_active[i].abs() > 0.1 && post_fired {
                self.eligibility_trace[i] += self.trace_increment;
                self.eligibility_trace[i] = self.eligibility_trace[i].min(1.0);
            }

            // Incremento menor se só pré ativo (permite algum crédito sem pós)
            if pre_active[i].abs() > 0.5 && !post_fired {
                self.eligibility_trace[i] += self.trace_increment * 0.1;
                self.eligibility_trace[i] = self.eligibility_trace[i].min(1.0);
            }
        }
    }

    /// Aplica aprendizado modulado por reward usando eligibility traces
    ///
    /// Este é o coração do 3-factor learning:
    /// Î”w = Î· * reward * eligibility_trace
    ///
    /// Permite que eventos distantes no tempo (padrão â†’ delay â†’ reward)
    /// ainda resultem em aprendizado correto.
    ///
    /// # Argumentos
    /// * `reward` - Sinal de reward [-1.0, 1.0]
    /// * `modulation` - Fator de modulação adicional (energia, etc.)
    pub fn apply_reward_modulated_learning(&mut self, reward: f64, modulation: f64) {
        let effective_lr = self.learning_rate * self.plasticity_gain * modulation;

        for i in 0..self.weights.len() {
            if self.eligibility_trace[i] > 0.01 {
                // Î”w = Î· * reward * trace * plasticidade_local
                let delta_w = effective_lr * reward * self.eligibility_trace[i] * self.plasticity[i];

                self.weights[i] += delta_w;
                self.weights[i] = self.weights[i].clamp(0.0, self.weight_clamp);

                // Atualiza synaptic tag baseado na mudança
                if delta_w.abs() > 1e-5 {
                    let relevance = 1.0 + reward.abs() * self.dopamine_sensitivity;
                    self.synaptic_tags[i] += delta_w.abs() * relevance * 5.0;
                    self.synaptic_tags[i] = self.synaptic_tags[i].min(2.0);
                }

                // Consome parcialmente o trace (evita re-uso infinito)
                self.eligibility_trace[i] *= 0.5;
            }
        }
    }

    /// Retorna a soma dos eligibility traces (para diagnóstico)
    pub fn total_eligibility(&self) -> f64 {
        self.eligibility_trace.iter().sum()
    }

    // ========================================================================
    // APRENDIZADO HEBBIANO E STDP
    // ========================================================================

    /// Aplica aprendizado Hebbiano com normalização
    pub fn apply_learning(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.weights.len());
        let lr = self.effective_lr();

        for i in 0..self.weights.len() {
            let input_magnitude = inputs[i].abs();

            if input_magnitude > 0.0 {
                let hebbian_update = lr * self.plasticity[i] * input_magnitude;
                self.weights[i] += hebbian_update;
            }
        }

        // Aplica manutenção
        self.apply_weight_maintenance(0.0);
    }

    /// Aplica manutenção dos pesos (decaimento e clamp)
    ///
    /// CORREÇÃO: O decay agora é proporcional à atividade para evitar
    /// que pesos morram durante períodos de inatividade (ciclo vicioso).
    pub fn apply_weight_maintenance(&mut self, recent_activity: f64) {
        // Proteção baseada em atividade: quanto mais ativo, menos decay
        let activity_protection = 1.0 - (recent_activity * 0.9);

        // NOVO: Fator de resgate - se atividade muito baixa, reduz decay drasticamente
        // Isso evita que pesos morram quando a rede está silenciosa
        let rescue_factor = if recent_activity < 0.01 {
            0.1  // Apenas 10% do decay quando inativo
        } else if recent_activity < 0.05 {
            0.3  // 30% do decay quando pouco ativo
        } else {
            1.0  // Decay normal quando ativo
        };

        let effective_decay = self.weight_decay * activity_protection * rescue_factor;

        // NOVO: Piso mínimo de peso para evitar morte sináptica completa
        let min_weight = 0.02;  // Piso que permite recuperação

        for weight in &mut self.weights {
            *weight *= 1.0 - effective_decay;
            *weight = weight.clamp(min_weight, self.weight_clamp);
        }

        self.apply_ltm_protection();

        // Aplica normalização competitiva periodicamente
        self.normalization_counter += 1;
        if self.competitive_normalization_enabled &&
           self.normalization_counter >= self.normalization_interval {
            self.apply_competitive_normalization();
            self.normalization_counter = 0;
        }
    }

    /// Proteção anti-regressão para memórias consolidadas
    fn apply_ltm_protection(&mut self) {
        for i in 0..self.weights.len() {
            let stability = self.ltm_stability[i];

            if stability > 0.8 {
                let ltm = self.weights_ltm[i];

                if ltm.abs() > 0.1 {
                    let attraction = 0.5 * stability;
                    self.weights[i] = self.weights[i] * (1.0 - attraction) + ltm * attraction;
                }
            }
        }
    }

    // ========================================================================
    // STDP ASSIMÃ‰TRICO v2.0
    // ========================================================================

    /// Aplica STDP com janela assimétrica otimizada
    ///
    /// NOVIDADE v2.0: tau_plus > tau_minus
    /// Isso dá mais tempo para padrões causais receberem LTP,
    /// enquanto mantém LTD rápida para conexões anti-causais.
    pub fn apply_stdp_pair(&mut self, pre_neuron_id: usize, delta_t: i64, reward: f64) -> bool {
        if pre_neuron_id >= self.weights.len() || delta_t == 0 {
            return false;
        }

        // ParÃ¢metros STDP modulados por ganho de plasticidade
        let (a_plus_eff, a_minus_eff) = self.effective_stdp_params();

        // Garante aprendizado mínimo mesmo com baixa energia
        let min_gain = 0.1;
        let final_a_plus = a_plus_eff.max(self.stdp_a_plus * min_gain);
        let final_a_minus = a_minus_eff.max(self.stdp_a_minus * min_gain);

        // REWARD MODULATION
        let weight_change = if delta_t > 0 {
            // Causal (pré antes de pós): LTP
            // Usa tau_plus MAIOR para dar mais tempo
            if reward >= 0.0 {
                let reward_modulation = 1.0 + reward * 2.0; // Boost maior com reward
                final_a_plus
                    * self.plasticity[pre_neuron_id]
                    * (-delta_t as f64 / self.stdp_tau_plus).exp()
                    * reward_modulation
            } else {
                // Reward negativo: inverte para LTD
                let punishment_strength = -reward;
                -final_a_plus
                    * self.plasticity[pre_neuron_id]
                    * (-delta_t as f64 / self.stdp_tau_plus).exp()
                    * punishment_strength
            }
        } else {
            // Anti-causal (pós antes de pré): LTD
            // Usa tau_minus MENOR para ser mais seletivo
            if reward >= 0.0 {
                let reward_modulation = 1.0 + reward;
                -final_a_minus
                    * self.plasticity[pre_neuron_id]
                    * (delta_t.abs() as f64 / self.stdp_tau_minus).exp()
                    * reward_modulation
            } else {
                let reward_modulation = 1.0 + reward;
                -final_a_minus
                    * self.plasticity[pre_neuron_id]
                    * (delta_t.abs() as f64 / self.stdp_tau_minus).exp()
                    * reward_modulation.max(0.0)
            }
        };

        // Aplica mudança com soft saturation para prevenir runaway LTP
        // CORREÇÃO: Saturação mais agressiva e limite absoluto por update
        let max_change_per_update = 0.05;  // Máximo 5% do peso por update

        if weight_change > 0.0 {
            // LTP: aplica soft cap baseado em proximidade do weight_clamp
            // Usa saturação quadrática para ser mais agressivo perto do limite
            let relative_weight = self.weights[pre_neuron_id] / self.weight_clamp;
            let saturation_factor = (1.0 - relative_weight).powi(2);  // Quadrático

            let adjusted_change = weight_change * saturation_factor;
            // Limita mudança máxima por update
            let capped_change = adjusted_change.min(max_change_per_update);
            self.weights[pre_neuron_id] += capped_change;
        } else {
            // LTD: limita também para evitar colapso rápido
            let capped_change = weight_change.max(-max_change_per_update);
            self.weights[pre_neuron_id] += capped_change;
        }

        // SYNAPTIC TAGGING
        let magnitude = weight_change.abs();
        if magnitude > 1e-4 {
            let relevance_factor = 1.0 + (reward.abs() * self.dopamine_sensitivity);
            self.synaptic_tags[pre_neuron_id] += magnitude * relevance_factor * 10.0;
            self.synaptic_tags[pre_neuron_id] = self.synaptic_tags[pre_neuron_id].min(2.0);

            // Também atualiza eligibility trace com STDP
            self.eligibility_trace[pre_neuron_id] += magnitude * 2.0;
            self.eligibility_trace[pre_neuron_id] = self.eligibility_trace[pre_neuron_id].min(1.0);
        }

        // Decay proporcional reduzido (já temos mecanismos de controle)
        let proportional_decay = self.weights[pre_neuron_id] * 0.00005;
        self.weights[pre_neuron_id] -= proportional_decay;

        // CORREÇÃO: Usa mesmo piso mínimo que weight_maintenance
        let min_weight = 0.02;
        self.weights[pre_neuron_id] = self.weights[pre_neuron_id].clamp(min_weight, self.weight_clamp);

        true
    }

    /// Aplica STDP para múltiplos pares de spikes
    pub fn apply_stdp_learning(
        &mut self,
        pre_spike_times: &[Option<i64>],
        post_spike_time: i64,
    ) -> usize {
        assert_eq!(pre_spike_times.len(), self.weights.len());

        let mut modified_count = 0;

        for i in 0..self.weights.len() {
            if let Some(pre_time) = pre_spike_times[i] {
                let delta_t = post_spike_time - pre_time;

                // Só processa se dentro da janela STDP
                if delta_t.abs() <= self.stdp_window {
                    self.apply_stdp_pair(i, delta_t, 0.0);
                    modified_count += 1;
                }
            }
        }

        // Aplica decaimento suave
        for weight in &mut self.weights {
            *weight *= 1.0 - self.weight_decay;
            *weight = weight.clamp(0.0, self.weight_clamp);
        }

        modified_count
    }

    /// Aplica iSTDP (Inhibitory STDP) - Vogels et al. 2011
    pub fn apply_istdp(
        &mut self,
        pre_neuron_id: usize,
        post_firing_rate: f64,
        pre_fired: bool,
        post_fired: bool,
    ) {
        if pre_neuron_id >= self.weights.len() {
            return;
        }

        let rate_error = post_firing_rate - self.istdp_target_rate;

        let min_weight = 0.02;

        if pre_fired && post_fired {
            let weight_change = self.istdp_learning_rate * rate_error;
            self.weights[pre_neuron_id] += weight_change;
            self.weights[pre_neuron_id] = self.weights[pre_neuron_id].clamp(min_weight, self.weight_clamp);
        }

        self.weights[pre_neuron_id] *= 0.999;
        self.weights[pre_neuron_id] = self.weights[pre_neuron_id].max(min_weight);
    }

    // ========================================================================
    // NORMALIZAÃ‡ÃƒO COMPETITIVA v2.0
    // ========================================================================

    /// Aplica normalização competitiva dos pesos
    ///
    /// Mantém a soma total dos pesos constante (orçamento sináptico),
    /// forçando competição entre sinapses.
    ///
    /// CORREÇÃO: Respeita piso mínimo de peso
    pub fn apply_competitive_normalization(&mut self) {
        let current_sum: f64 = self.weights.iter().sum();
        let min_weight = 0.02;

        if current_sum > 0.0 && (current_sum - self.target_weight_sum).abs() > 0.01 {
            let scale = self.target_weight_sum / current_sum;

            // Aplica escala suavizada para evitar mudanças bruscas
            let smooth_scale = 1.0 + (scale - 1.0) * 0.3; // 30% da correção por vez

            for w in &mut self.weights {
                *w *= smooth_scale;
                *w = w.clamp(min_weight, self.weight_clamp);
            }
        }
    }

    /// Normalização competitiva com proteção de pesos fortes
    ///
    /// Pesos acima de um threshold são parcialmente protegidos da normalização,
    /// permitindo que "memórias" importantes sobrevivam.
    ///
    /// CORREÇÃO: Respeita piso mínimo de peso
    pub fn apply_competitive_normalization_with_protection(&mut self, protection_threshold: f64) {
        let current_sum: f64 = self.weights.iter().sum();
        let min_weight = 0.02;

        if current_sum > 0.0 && (current_sum - self.target_weight_sum).abs() > 0.01 {
            let scale = self.target_weight_sum / current_sum;

            for i in 0..self.weights.len() {
                // Pesos fortes são parcialmente protegidos
                let protection = if self.weights[i] > protection_threshold {
                    0.5 // 50% de proteção
                } else {
                    0.0
                };

                let effective_scale = 1.0 + (scale - 1.0) * (1.0 - protection) * 0.3;
                self.weights[i] *= effective_scale;
                self.weights[i] = self.weights[i].clamp(min_weight, self.weight_clamp);
            }
        }
    }

    // ========================================================================
    // CONSOLIDAÃ‡ÃƒO DE MEMÃ“RIA
    // ========================================================================

    /// Consolida memórias usando synaptic tagging
    pub fn consolidate_memory_tagged(&mut self, base_consolidation_rate: f64) {
        for i in 0..self.weights.len() {
            let tag = self.synaptic_tags[i];

            if tag > self.capture_threshold {
                let stm = self.weights[i];
                let ltm = self.weights_ltm[i];

                // Velocidade proporcional Ã  tag
                let dynamic_rate = base_consolidation_rate * tag;

                self.weights_ltm[i] += dynamic_rate * (stm - ltm);
                self.weights_ltm[i] = self.weights_ltm[i].clamp(0.0, self.weight_clamp * 2.0);

                // Consome tag
                self.synaptic_tags[i] *= 0.5;

                // Atualiza estabilidade
                let abs_delta = (dynamic_rate * (stm - ltm)).abs();
                if abs_delta < 1e-4 {
                    self.ltm_stability[i] = (self.ltm_stability[i] + 0.02).min(1.0);
                } else {
                    self.ltm_stability[i] = (self.ltm_stability[i] * 0.98).max(0.0);
                }
            }
        }
    }

    /// Consolida memória (versão simples sem tagging)
    pub fn consolidate_memory(&mut self, consolidation_rate: f64, consolidation_factor: f64) {
        for i in 0..self.weights.len() {
            let stm = self.weights[i];
            let target_ltm = stm * consolidation_factor;

            let delta = consolidation_rate * (target_ltm - self.weights_ltm[i]);
            self.weights_ltm[i] += delta;
            self.weights_ltm[i] = self.weights_ltm[i].clamp(0.0, self.weight_clamp * 2.0);

            let abs_delta = delta.abs();
            if abs_delta < 1e-4 {
                self.ltm_stability[i] = (self.ltm_stability[i] + 0.01).min(1.0);
            } else {
                self.ltm_stability[i] = (self.ltm_stability[i] * 0.99).max(0.0);
            }
        }
    }

    /// Decaimento das tags sinápticas
    pub fn decay_tags(&mut self) {
        for tag in &mut self.synaptic_tags {
            *tag *= 1.0 - self.tag_decay_rate;
            if *tag < 1e-3 {
                *tag = 0.0;
            }
        }
    }

    // ========================================================================
    // SYNAPTIC SCALING HOMEOSTÃTICO
    // ========================================================================

    /// Aplica synaptic scaling homeostático
    ///
    /// CORREÇÃO: Respeita piso mínimo de peso para evitar morte sináptica
    pub fn apply_synaptic_scaling(&mut self, rate_error: f64, eta: f64) {
        let scale = 1.0 - eta * rate_error;
        let scale = scale.clamp(0.9, 1.1);

        // Piso mínimo consistente com outras funções
        let min_weight = 0.02;

        for w in &mut self.weights {
            *w *= scale;
            *w = w.clamp(min_weight, self.weight_clamp);
        }

        let ltm_scale = 1.0 - (eta * 0.1) * rate_error;
        let ltm_scale = ltm_scale.clamp(0.95, 1.05);

        for w in &mut self.weights_ltm {
            *w *= ltm_scale;
            // LTM pode ir a zero (não consolidado ainda)
            *w = w.clamp(0.0, self.weight_clamp * 2.0);
        }
    }

    // ========================================================================
    // GETTERS E SETTERS
    // ========================================================================

    pub fn num_inputs(&self) -> usize {
        self.weights.len()
    }

    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    pub fn weight_norm(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    pub fn scale_learning_rate(&mut self, factor: f64) {
        self.learning_rate *= factor;
        self.stdp_a_plus *= factor;
        self.stdp_a_minus *= factor;
    }

    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.max(0.0);
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn reset_learning_params(&mut self) {
        self.learning_rate = 0.008;
        self.stdp_a_plus = 0.015;
        self.stdp_a_minus = 0.006;
    }

    pub fn set_weight_decay(&mut self, decay: f64) {
        self.weight_decay = decay.clamp(0.0, 1.0);
    }

    pub fn set_tag_decay_rate(&mut self, rate: f64) {
        self.tag_decay_rate = rate.clamp(0.0, 1.0);
    }

    pub fn set_plasticity_gain(&mut self, gain: f64) {
        self.plasticity_gain = gain.clamp(0.0, 2.0);
    }

    /// Retorna amplitudes STDP atuais
    pub fn get_stdp_amplitudes(&self) -> (f64, f64) {
        (self.stdp_a_plus, self.stdp_a_minus)
    }

    /// Define amplitudes STDP
    pub fn set_stdp_amplitudes(&mut self, a_plus: f64, a_minus: f64) {
        self.stdp_a_plus = a_plus.max(0.0);
        self.stdp_a_minus = a_minus.max(0.0);
    }

    fn effective_lr(&self) -> f64 {
        self.learning_rate * self.plasticity_gain
    }

    fn effective_stdp_params(&self) -> (f64, f64) {
        (
            self.stdp_a_plus * self.plasticity_gain,
            self.stdp_a_minus * self.plasticity_gain,
        )
    }

    /// Retorna estatísticas do estado atual (para debug)
    pub fn get_stats(&self) -> DendritomaStats {
        let weight_sum: f64 = self.weights.iter().sum();
        let weight_max = self.weights.iter().cloned().fold(0.0, f64::max);
        let weight_min = self.weights.iter().cloned().fold(f64::MAX, f64::min);
        let trace_sum: f64 = self.eligibility_trace.iter().sum();
        let resource_avg: f64 = self.synaptic_resources.iter().sum::<f64>()
                               / self.synaptic_resources.len() as f64;

        DendritomaStats {
            weight_sum,
            weight_max,
            weight_min,
            trace_sum,
            resource_avg,
            num_synapses: self.weights.len(),
        }
    }
}

/// Estatísticas do Dendritoma para diagnóstico
#[derive(Debug, Clone)]
pub struct DendritomaStats {
    pub weight_sum: f64,
    pub weight_max: f64,
    pub weight_min: f64,
    pub trace_sum: f64,
    pub resource_avg: f64,
    pub num_synapses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendritoma_initialization() {
        let d = Dendritoma::new(10);

        assert_eq!(d.weights.len(), 10);
        assert_eq!(d.eligibility_trace.len(), 10);
        assert_eq!(d.synaptic_resources.len(), 10);

        // Traces começam zerados
        assert_eq!(d.total_eligibility(), 0.0);

        // Recursos começam cheios
        for r in &d.synaptic_resources {
            assert_eq!(*r, 1.0);
        }
    }

    #[test]
    fn test_eligibility_trace_update() {
        let mut d = Dendritoma::new(5);

        // Inputs ativos
        let inputs = vec![1.0, 0.0, 1.0, 0.0, 0.5];

        // Atualiza com post_fired = true
        d.update_eligibility_traces(&inputs, true);

        // Traces devem aumentar para inputs ativos
        assert!(d.eligibility_trace[0] > 0.0);
        assert!(d.eligibility_trace[2] > 0.0);

        // Trace para input inativo deve ser ~0
        assert!(d.eligibility_trace[1] < 0.01);
    }

    #[test]
    fn test_stp_depression() {
        let mut d = Dendritoma::new(3);

        // Input repetido várias vezes
        for _ in 0..10 {
            let inputs = vec![1.0, 0.0, 0.0];
            d.integrate(&inputs);
        }

        // Recursos da sinapse 0 devem estar baixos
        assert!(d.synaptic_resources[0] < 0.5);

        // Recursos das outras devem estar cheios
        assert!(d.synaptic_resources[1] > 0.9);
        assert!(d.synaptic_resources[2] > 0.9);
    }

    #[test]
    fn test_stp_recovery() {
        let mut d = Dendritoma::new(3);

        // Depleta recursos
        for _ in 0..10 {
            let inputs = vec![1.0, 0.0, 0.0];
            d.integrate(&inputs);
        }

        let depleted = d.synaptic_resources[0];

        // Recupera por vários timesteps
        for _ in 0..500 {
            d.update_stp();
        }

        // Deve ter recuperado
        assert!(d.synaptic_resources[0] > depleted);
        assert!(d.synaptic_resources[0] > 0.8);
    }

    #[test]
    fn test_competitive_normalization() {
        let mut d = Dendritoma::new(5);
        d.target_weight_sum = 0.5;

        // Aumenta alguns pesos
        d.weights = vec![0.3, 0.3, 0.1, 0.1, 0.1]; // soma = 0.9

        d.apply_competitive_normalization();

        // Soma deve se aproximar do alvo
        let new_sum: f64 = d.weights.iter().sum();
        // Com suavização de 30%, não chega exatamente no alvo
        assert!(new_sum < 0.9);
    }

    #[test]
    fn test_stdp_asymmetric_window() {
        let d = Dendritoma::new(5);

        // tau_plus deve ser maior que tau_minus
        assert!(d.stdp_tau_plus > d.stdp_tau_minus);

        // Janela deve ser grande o suficiente
        assert!(d.stdp_window >= 40);
    }

    #[test]
    fn test_reward_modulated_learning() {
        let mut d = Dendritoma::new(3);

        // Configura traces manualmente
        d.eligibility_trace = vec![0.5, 0.0, 0.3];

        let initial_w0 = d.weights[0];
        let initial_w1 = d.weights[1];

        // Aplica reward positivo
        d.apply_reward_modulated_learning(1.0, 1.0);

        // Peso 0 deve aumentar (trace alto, reward positivo)
        assert!(d.weights[0] > initial_w0);

        // Peso 1 não deve mudar (trace zero)
        assert!((d.weights[1] - initial_w1).abs() < 1e-6);
    }
}
