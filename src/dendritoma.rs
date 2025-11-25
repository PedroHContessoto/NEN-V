/// Módulo responsável pela integração de sinais de entrada e aprendizado sináptico
///
/// O Dendritoma recebe e pondera os sinais de entrada, aplicando aprendizado
/// Hebbiano com normalização L2 para estabilidade.

#[derive(Debug, Clone)]
pub struct Dendritoma {
    /// Pesos sinápticos de curto prazo (STM - Short-Term Memory)
    pub weights: Vec<f64>,

    /// Pesos sinápticos de longo prazo (LTM - Long-Term Memory)
    pub weights_ltm: Vec<f64>,

    /// Fator de plasticidade para cada peso (modula a taxa de aprendizado)
    pub plasticity: Vec<f64>,

    // Parâmetros de aprendizado Hebbiano
    learning_rate: f64,

    // Parâmetros de STDP (Spike-Timing-Dependent Plasticity)
    /// Amplitude de potenciação (LTP) quando pre→post
    stdp_a_plus: f64,

    /// Amplitude de depressão (LTD) quando post→pre
    stdp_a_minus: f64,

    /// Constante de tempo para janela de potenciação (ms)
    stdp_tau_plus: f64,

    /// Constante de tempo para janela de depressão (ms)
    stdp_tau_minus: f64,

    /// Janela temporal para STDP (ms)
    stdp_window: i64,

    /// Parâmetros de iSTDP (Inhibitory STDP) - Vogels et al. 2011
    /// Taxa de aprendizado para sinapses inibitórias
    pub istdp_learning_rate: f64,

    /// Target firing rate para E/I balance (neurônio pós-sináptico alvo)
    pub istdp_target_rate: f64,

    /// Taxa de decaimento dos pesos STM (0.0 = sem decaimento, 1.0 = decaimento total)
    weight_decay: f64,

    /// Ganho de plasticidade global (modulado por energia, BCM, etc.)
    /// Valor padrão: 1.0. Valores < 1.0 reduzem aprendizado, > 1.0 aumentam.
    plasticity_gain: f64,

    /// Estabilidade de cada peso LTM [0.0, 1.0]
    /// 0.0 = recente/instável, 1.0 = consolidado/estável
    /// Memórias estáveis resistem mais a mudanças (anti-regressão)
    pub ltm_stability: Vec<f64>,

    /// Limite máximo para pesos (clamp superior)
    /// Valor padrão: 2.5. Pode ser ajustado para prevenir saturação.
    pub weight_clamp: f64,

    /// SYNAPTIC TAGGING AND CAPTURE (STC) - O Terceiro Pilar da Memória
    /// Buffer de Elegibilidade: indica a "relevância" da mudança recente em cada sinapse.
    /// Valores [0.0, 2.0] funcionam como um porteiro para a LTM.
    /// Tags fortes (> threshold) permitem consolidação, tags fracas bloqueiam ruído.
    pub synaptic_tags: Vec<f64>,

    /// Taxa de decaimento das etiquetas sinápticas (ex: 0.01 = 1% por passo)
    /// Tags são temporárias: se não reforçadas, decaem naturalmente
    tag_decay_rate: f64,

    /// Limiar de captura: só consolida se tag > threshold
    /// Valores típicos: 0.2 (moderado) a 0.5 (rigoroso)
    capture_threshold: f64,

    /// Sensibilidade à dopamina/reward para amplificação de tags
    /// Valores típicos: 5.0 (eventos emocionais geram tags 5x mais fortes)
    dopamine_sensitivity: f64,
}

impl Dendritoma {
    /// Cria um novo Dendritoma com pesos aleatórios iniciais
    ///
    /// # Argumentos
    /// * `num_inputs` - Número de conexões de entrada
    pub fn new(num_inputs: usize) -> Self {
        // Nota: Network::new() vai sobrescrever pesos com valores aleatórios 0.04-0.06
        // Inicializamos com 0.05 por simplicidade (será sobrescrito)
        let weights: Vec<f64> = vec![0.05; num_inputs];
        let plasticity = vec![1.0; num_inputs];
        let weights_ltm = vec![0.0; num_inputs];
        let ltm_stability = vec![0.0; num_inputs];
        // STC: Inicializa tags zeradas (sem relevância inicial)
        let synaptic_tags = vec![0.0; num_inputs];

        Self {
            weights,
            weights_ltm,
            plasticity,
            // CORREÇÃO: Aprendizado mais lento (0.005) para compensar maior atividade
            learning_rate: 0.005,
            // STDP balanceado: LTP > LTD (ratio 2:1) para permitir aprendizado
            // Combinado com weight_decay e punição de parede, cria contraste sem matar tudo
            stdp_a_plus: 0.012,          // LTP (fortalecimento) ligeiramente maior
            stdp_a_minus: 0.006,         // LTD (enfraquecimento)
            stdp_tau_plus: 20.0,
            stdp_tau_minus: 20.0,
            stdp_window: 20,
            // OTIMIZADO: iSTDP params (Vogels et al. 2011)
            istdp_learning_rate: 0.001,  // Taxa otimizada
            istdp_target_rate: 0.15,     // Target otimizado (15% FR)
            weight_decay: 0.0001,        // OTIMIZADO: Decay muito baixo
            plasticity_gain: 1.0,
            ltm_stability,
            weight_clamp: 2.5,           // Limite padrão para pesos
            // STC: Parâmetros de Synaptic Tagging and Capture
            synaptic_tags,
            tag_decay_rate: 0.01,        // 1% de decaimento por passo (~100 passos de vida)
            capture_threshold: 0.2,      // Limiar moderado (20% de certeza mínima)
            dopamine_sensitivity: 5.0,   // Reward amplifica tags 5x
        }
    }

    /// Cria um Dendritoma com parâmetros personalizados
    pub fn with_params(num_inputs: usize, learning_rate: f64) -> Self {
        let mut dendritoma = Self::new(num_inputs);
        dendritoma.learning_rate = learning_rate;
        dendritoma
    }

    /// Configura parâmetros STDP personalizados
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

    /// Integra os sinais de entrada através de uma soma ponderada
    ///
    /// # Argumentos
    /// * `inputs` - Vetor de sinais de entrada (pode conter valores positivos e negativos)
    ///
    /// # Retorna
    /// O potencial integrado (soma ponderada dos inputs)
    pub fn integrate(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Número de inputs deve ser igual ao número de pesos"
        );

        // DUAL-MEMORY: Usa peso efetivo (STM + LTM)
        inputs
            .iter()
            .zip(self.weights.iter().zip(self.weights_ltm.iter()))
            .map(|(input, (weight_stm, weight_ltm))| input * (weight_stm + weight_ltm))
            .sum()
    }

    /// Aplica aprendizado Hebbiano com normalização L2 (v2)
    ///
    /// Regra: "Neurónios que disparam juntos, conectam-se"
    /// - Fortalece pesos de inputs ativos quando o neurónio dispara
    /// - Normaliza o vetor de pesos usando norma L2 para estabilidade
    /// - Modulado por plasticity_gain (energia, BCM, etc.)
    ///
    /// # Argumentos
    /// * `inputs` - Vetor de sinais de entrada que estavam presentes durante o disparo
    pub fn apply_learning(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.weights.len());
        let lr = self.effective_lr();

        // Fase 1: Atualização Hebbiana
        for i in 0..self.weights.len() {
            let input_magnitude = inputs[i].abs();

            if input_magnitude > 0.0 {
                let hebbian_update = lr * self.plasticity[i] * input_magnitude;
                self.weights[i] += hebbian_update;
            }
        }

        // Fase 2: Normalização e CLAMPING (A Correção Crítica)
        for weight in &mut self.weights {
            *weight *= 1.0 - self.weight_decay;

            // Mantém os pesos numa faixa útil (0.0 a 2.5)
            *weight = weight.clamp(0.0, self.weight_clamp);
        }

        // Fase 3: Proteção de memória LTM
        self.apply_ltm_protection();
    }

    /// Aplica manutenção periódica: decaimento natural e limites de segurança
    /// Deve ser chamado a cada passo de tempo, independente do aprendizado.
    ///
    /// # Argumentos
    /// * `recent_activity` - Taxa de atividade recente [0.0, 1.0] para modular decay
    ///                       Neurônios ativos têm decay reduzido (proteção de memórias ativas)
    pub fn apply_weight_maintenance(&mut self, recent_activity: f64) {
        // Decay adaptativo: neurônios ativos têm decay reduzido
        // activity=0.0 → decay=100% (neurônio inativo, esquecimento pleno)
        // activity=1.0 → decay=10% (neurônio muito ativo, memórias protegidas)
        let activity_protection = 1.0 - (recent_activity * 0.9);
        let effective_decay = self.weight_decay * activity_protection;

        for weight in &mut self.weights {
            // 1. Decaimento Natural Adaptativo (Esquecimento com proteção de atividade)
            *weight *= 1.0 - effective_decay;

            // 2. Trava de Segurança (Clamp)
            // Impede matematicamente que o peso passe de 2.5
            *weight = weight.clamp(0.0, self.weight_clamp);
        }

        // 3. Proteção de Memória de Longo Prazo
        self.apply_ltm_protection();
    }

    /// Aplica proteção anti-regressão para memórias consolidadas
    ///
    /// Sinapses com LTM forte e alta estabilidade puxam STM de volta,
    /// evitando que novos padrões destruam representações antigas importantes.
    /// Isso permite lifelong learning sem catastrofe.
    fn apply_ltm_protection(&mut self) {
        for i in 0..self.weights.len() {
            let stability = self.ltm_stability[i];

            // Só protege se estabilidade for alta (> 0.8)
            if stability > 0.8 {
                let ltm = self.weights_ltm[i];

                // Se LTM é significativo e STM está divergindo, aplica correção
                if ltm.abs() > 0.1 {
                    // Força de atração proporcional à estabilidade
                    let attraction = 0.5 * stability;

                    // Puxa STM em direção a LTM (média ponderada)
                    self.weights[i] = self.weights[i] * (1.0 - attraction)
                        + ltm * attraction;
                }
            }
        }
    }

    /// Aplica STDP (Spike-Timing-Dependent Plasticity) com normalização L2
    ///
    /// Regra temporal: Fortalece/enfraquece conexões baseado na diferença temporal
    /// entre spikes pré-sinápticos e pós-sinápticos.
    ///
    /// - Δt > 0 (pre antes de post): Potenciação (LTP) com exponencial A+ * exp(-Δt/τ+)
    /// - Δt < 0 (post antes de pre): Depressão (LTD) com exponencial A- * exp(Δt/τ-)
    ///
    /// # Argumentos
    /// * `pre_spike_times` - Vetor com tempo do último spike de cada neurônio pré-sináptico
    /// * `post_spike_time` - Tempo do spike do neurônio pós-sináptico (este neurônio)
    ///
    /// # Retorna
    /// O número de pesos modificados
    pub fn apply_stdp_learning(
        &mut self,
        pre_spike_times: &[Option<i64>],
        post_spike_time: i64,
    ) -> usize {
        assert_eq!(
            pre_spike_times.len(),
            self.weights.len(),
            "Número de pre_spike_times deve ser igual ao número de pesos"
        );

        let mut modified_count = 0;

        // Fase 1: Atualização STDP baseada em timing
        for i in 0..self.weights.len() {
            if let Some(pre_time) = pre_spike_times[i] {
                let weight_before = self.weights[i];

                // Calcula diferença temporal: Δt = t_post - t_pre
                let delta_t = post_spike_time - pre_time;

                // Aplica janela STDP
                let weight_change = if delta_t > 0 {
                    // LTP: pre→post (causal)
                    // Potenciação decai exponencialmente com a distância temporal
                    self.stdp_a_plus
                        * self.plasticity[i]
                        * (-delta_t as f64 / self.stdp_tau_plus).exp()
                } else if delta_t < 0 {
                    // LTD: post→pre (anti-causal)
                    // Depressão decai exponencialmente
                    -self.stdp_a_minus
                        * self.plasticity[i]
                        * (delta_t as f64 / self.stdp_tau_minus).exp()
                } else {
                    // Δt = 0: spikes simultâneos (auto-conexão ou coincidência)
                    // CORREÇÃO: Ignora auto-conexões para evitar diluição do STDP
                    0.0
                };

                self.weights[i] += weight_change;

                // DEBUG: Log detalhado (apenas para primeiros 2 neurônios e primeiros 30 passos)
                if i < 2 && post_spike_time < 30 {
                    eprintln!("[STDP] t={} i={} pre_t={} post_t={} Δt={} w_before={:.6} Δw={:.6} w_after={:.6}",
                              post_spike_time, i, pre_time, post_spike_time, delta_t,
                              weight_before, weight_change, self.weights[i]);
                }

                modified_count += 1;
            }
        }

        // Fase 2: Normalização aditiva (preserva diferenças relativas do STDP)
        // Em vez de normalização L2 "hard", usa decaimento suave
        for weight in &mut self.weights {
            *weight *= 1.0 - self.weight_decay;
        }

        // Clipping para evitar explosão ou valores negativos
        for weight in &mut self.weights {
            *weight = weight.clamp(0.0, self.weight_clamp);
        }

        modified_count
    }

    /// Retorna o número de conexões de entrada
    pub fn num_inputs(&self) -> usize {
        self.weights.len()
    }

    /// Retorna a soma total dos pesos (útil para debugging)
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Retorna a norma L2 atual dos pesos
    pub fn weight_norm(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// Multiplica a taxa de aprendizado por um fator (para sono/vigília)
    pub fn scale_learning_rate(&mut self, factor: f64) {
        self.learning_rate *= factor;
        self.stdp_a_plus *= factor;
        self.stdp_a_minus *= factor;
    }

    /// Define a taxa de aprendizado diretamente
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.max(0.0);
    }

    /// Restaura parâmetros de aprendizado para valores padrão
    pub fn reset_learning_params(&mut self) {
        self.learning_rate = 0.01;
        self.stdp_a_plus = 0.15;
        self.stdp_a_minus = 0.005;
    }

    /// Retorna a taxa de aprendizado atual
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Método para aplicar STDP baseado em um par de spikes
    ///
    /// Este método substitui apply_stdp_learning() e implementa STDP corretamente
    /// baseado em pares (pré, pós) de spikes.
    /// Modulado por plasticity_gain (energia, BCM, etc.) e reward (dopamina).
    ///
    /// ATUALIZAÇÃO STC: Agora também cria Etiquetas Sinápticas (Synaptic Tags)
    /// baseadas na magnitude da mudança e no sinal de recompensa.
    ///
    /// # Argumentos
    /// * `pre_neuron_id` - ID do neurônio pré-sináptico que disparou
    /// * `delta_t` - Diferença temporal: t_post - t_pre
    /// * `reward` - Sinal de recompensa global [-1.0, +1.0], default=0.0 (neutro)
    ///
    /// # Retorna
    /// true se o peso foi modificado, false caso contrário
    pub fn apply_stdp_pair(&mut self, pre_neuron_id: usize, delta_t: i64, reward: f64) -> bool {
        if pre_neuron_id >= self.weights.len() || delta_t == 0 {
            return false;
        }

        // ENERGY-GATED: usa parâmetros STDP modulados por ganho de plasticidade
        let (a_plus_eff, a_minus_eff) = self.effective_stdp_params();

        // CORREÇÃO: Se o ganho for muito baixo, usa um mínimo para garantir aprendizado
        // Isso evita que a rede fique "presa" em estados de baixa energia sem aprender nada.
        let min_gain = 0.1;
        let final_a_plus = a_plus_eff.max(self.stdp_a_plus * min_gain);
        let final_a_minus = a_minus_eff.max(self.stdp_a_minus * min_gain);

        // REWARD MODULATION com inversão para reversão de aprendizado
        // reward > 0: amplifica LTP (aprendizado forte)
        // reward = 0: LTP/LTD normal (baseline STDP)
        // reward < 0: INVERTE para LTD (reversão/punição)

        let weight_change = if delta_t > 0 {
            // Causal (pré antes de pós): normalmente LTP
            if reward >= 0.0 {
                // Reward positivo ou neutro: LTP normal ou amplificado
                let reward_modulation = 1.0 + reward;
                final_a_plus
                    * self.plasticity[pre_neuron_id]
                    * (-delta_t as f64 / self.stdp_tau_plus).exp()
                    * reward_modulation
            } else {
                // Reward NEGATIVO: INVERTE para LTD (punição)
                // reward=-1.0 → LTD com força=100% de A+
                // reward=-0.5 → LTD com força=50% de A+
                let punishment_strength = -reward; // reward=-1.0 → strength=1.0
                -final_a_plus  // Nota: negativo para LTD
                    * self.plasticity[pre_neuron_id]
                    * (-delta_t as f64 / self.stdp_tau_plus).exp()
                    * punishment_strength
            }
        } else {
            // Anti-causal (pós antes de pré): normalmente LTD
            if reward >= 0.0 {
                // Reward positivo/neutro: LTD normal ou amplificado
                let reward_modulation = 1.0 + reward;
                -final_a_minus
                    * self.plasticity[pre_neuron_id]
                    * (delta_t.abs() as f64 / self.stdp_tau_minus).exp()
                    * reward_modulation
            } else {
                // Reward NEGATIVO: INVERTE para LTP (isso raramente acontece na prática)
                // Mantém LTD mas reduzida
                let reward_modulation = 1.0 + reward; // 0.0 a 1.0
                -final_a_minus
                    * self.plasticity[pre_neuron_id]
                    * (delta_t.abs() as f64 / self.stdp_tau_minus).exp()
                    * reward_modulation.max(0.0)
            }
        };

        // Aplica mudança
        self.weights[pre_neuron_id] += weight_change;

        // SYNAPTIC TAGGING: Cria/Atualiza Etiqueta Sináptica
        // A etiqueta indica a "relevância" desta mudança para consolidação futura

        // 1. Magnitude: O quanto a sinapse foi fisicamente alterada
        let magnitude = weight_change.abs();

        // 2. Fator de Relevância (Neuromodulação Dopaminérgica):
        // Reward > 0 (Prazer) ou Reward < 0 (Dor) multiplicam a importância.
        // Reward == 0 (Neutro) mantém importância base (1.0).
        let relevance_factor = 1.0 + (reward.abs() * self.dopamine_sensitivity);

        // 3. Atualiza a Tag: Acumula a importância (tags podem crescer com repetição)
        if magnitude > 1e-4 {
            // Multiplica por 10.0 para acelerar acumulação de tags
            // (sem isso, tags demorariam muito para atingir o threshold)
            self.synaptic_tags[pre_neuron_id] += magnitude * relevance_factor * 10.0;

            // Clamp para evitar explosão (teto de "certeza" = 2.0)
            self.synaptic_tags[pre_neuron_id] = self.synaptic_tags[pre_neuron_id].min(2.0);
        }

        // Decay proporcional ao peso (estabilização natural)
        // Pesos maiores decaem mais rápido, criando equilíbrio
        let proportional_decay = self.weights[pre_neuron_id] * 0.0001;
        self.weights[pre_neuron_id] -= proportional_decay;

        // Garante que o peso fique no intervalo [0.0, 2.5]
        self.weights[pre_neuron_id] = self.weights[pre_neuron_id].clamp(0.0, self.weight_clamp);

        true
    }

    /// Aplica iSTDP (Inhibitory STDP) - Vogels et al. 2011
    ///
    /// Regra: Sinapses inibitórias se fortalecem quando o neurônio pós-sináptico
    /// dispara acima do target rate, balanceando excitação/inibição automaticamente.
    ///
    /// # Argumentos
    /// * `pre_neuron_id` - ID do neurônio pré-sináptico (inibitório)
    /// * `post_firing_rate` - Taxa de disparo atual do neurônio pós-sináptico
    /// * `pre_fired` - Se o neurônio pré-sináptico disparou
    /// * `post_fired` - Se o neurônio pós-sináptico disparou
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

        // iSTDP: Ajusta peso inibitório baseado em desvio do target rate
        // Se post_rate > target → aumenta inibição (strengthens synapse)
        // Se post_rate < target → diminui inibição (weakens synapse)
        let rate_error = post_firing_rate - self.istdp_target_rate;

        // Apenas ajusta quando há correlação temporal (pre e post ativos próximos)
        if pre_fired && post_fired {
            // Vogels rule: ΔW = η * (rate - target)
            let weight_change = self.istdp_learning_rate * rate_error;
            self.weights[pre_neuron_id] += weight_change;

            // Pesos inibitórios permanecem positivos (magnitude da inibição)
            // A polaridade vem do neuron_type (Inhibitory → output = -1.0)
            self.weights[pre_neuron_id] = self.weights[pre_neuron_id].clamp(0.0, self.weight_clamp);
        }

        // Decay lento para evitar que inibição desapareça
        // (inibição baseline é necessária para estabilidade)
        self.weights[pre_neuron_id] *= 0.999;
    }

    /// Consolida memórias de curto prazo (STM) para longo prazo (LTM)
    /// Chamado durante o sono
    ///
    /// VERSÃO NÃO-ACUMULATIVA: LTM converge gradualmente para STM * factor
    /// em vez de crescer indefinidamente.
    ///
    /// ANTI-REGRESSÃO: Atualiza estabilidade de cada sinapse LTM baseado
    /// na magnitude das mudanças. Sinapses estáveis resistem a mudanças futuras.
    ///
    /// # Argumentos
    /// * `consolidation_rate` - Taxa de convergência (0.0-1.0, ex: 0.1 = 10% por passo)
    /// * `consolidation_factor` - Multiplicador do STM para LTM (ex: 2.0 = LTM será 2x STM)
    pub fn consolidate_memory(&mut self, consolidation_rate: f64, consolidation_factor: f64) {
        for i in 0..self.weights.len() {
            let stm = self.weights[i];
            let target_ltm = stm * consolidation_factor;

            // Consolidação: LTM converge gradualmente para target_ltm
            // Exemplo: rate=0.1, LTM se move 10% em direção ao alvo a cada passo
            let delta = consolidation_rate * (target_ltm - self.weights_ltm[i]);
            self.weights_ltm[i] += delta;

            // Clamp LTM para evitar explosão (permite até 2× o limite de STM)
            self.weights_ltm[i] = self.weights_ltm[i].clamp(0.0, self.weight_clamp * 2.0);

            // ANTI-REGRESSÃO: Atualiza estabilidade baseado em mudanças
            let abs_delta = delta.abs();

            if abs_delta < 1e-4 {
                // LTM quase não mudou → aumenta estabilidade
                self.ltm_stability[i] = (self.ltm_stability[i] + 0.01).min(1.0);
            } else {
                // LTM ainda está mudando → estabilidade diminui levemente
                self.ltm_stability[i] = (self.ltm_stability[i] * 0.99).max(0.0);
            }
        }

        // NOTA: NÃO normalizamos LTM!
        // LTM é acumulativo e representa memória consolidada
        // Normalizar destruiria a força da memória
    }

    /// Define a taxa de decaimento dos pesos STM
    ///
    /// # Argumentos
    /// * `decay` - Taxa de decaimento (0.0 = sem decaimento, 1.0 = decaimento total)
    pub fn set_weight_decay(&mut self, decay: f64) {
        self.weight_decay = decay.clamp(0.0, 1.0);
    }

    /// Define o ganho de plasticidade global
    ///
    /// Este ganho modula toda a plasticidade sináptica, permitindo controle
    /// dinâmico baseado em energia, BCM, ou outros mecanismos homeostáticos.
    ///
    /// # Argumentos
    /// * `gain` - Ganho de plasticidade (0.0 = sem aprendizado, 1.0 = aprendizado pleno)
    pub fn set_plasticity_gain(&mut self, gain: f64) {
        self.plasticity_gain = gain.clamp(0.0, 2.0);
    }

    /// Retorna a taxa de aprendizado efetiva (modulada pelo ganho de plasticidade)
    fn effective_lr(&self) -> f64 {
        self.learning_rate * self.plasticity_gain
    }

    /// Retorna os parâmetros STDP efetivos (modulados pelo ganho de plasticidade)
    fn effective_stdp_params(&self) -> (f64, f64) {
        (
            self.stdp_a_plus * self.plasticity_gain,
            self.stdp_a_minus * self.plasticity_gain,
        )
    }

    /// Aplica synaptic scaling homeostático
    ///
    /// Escala pesos sinápticos multiplicativamente baseado no erro de firing rate.
    /// Implementa homeostase de atividade: neurônios hiperativos têm pesos reduzidos,
    /// neurônios hipoativos têm pesos aumentados.
    ///
    /// Combinado com energy-gated learning, isso conecta:
    /// - Firing rate alvo
    /// - Consumo energético
    /// - Capacidade de aprendizado a longo prazo
    ///
    /// # Argumentos
    /// * `rate_error` - Diferença entre firing rate atual e alvo (positivo = hiperativo)
    /// * `eta` - Taxa de ajuste homeostático (pequena, ex: 0.01)
    pub fn apply_synaptic_scaling(&mut self, rate_error: f64, eta: f64) {
        // Escala multiplicativa: se hiperativo (error > 0), reduz pesos
        let scale = 1.0 - eta * rate_error;
        let scale = scale.clamp(0.9, 1.1); // Limita mudanças bruscas

        // Aplica scaling a STM
        for w in &mut self.weights {
            *w *= scale;
        }

        // LTM ajusta MUITO menos para preservar memórias consolidadas
        let ltm_scale = 1.0 - (eta * 0.1) * rate_error;
        let ltm_scale = ltm_scale.clamp(0.95, 1.05);

        for w in &mut self.weights_ltm {
            *w *= ltm_scale;
        }
    }

    /// SYNAPTIC TAGGING: Decaimento das Etiquetas Sinápticas
    ///
    /// Faz as etiquetas desaparecerem com o tempo (Esquecimento da Relevância).
    /// Tags são temporárias: se o evento não for repetido ou consolidado logo,
    /// a "memória química" se dissipa.
    ///
    /// Deve ser chamado a cada passo de tempo (update loop).
    ///
    /// Este mecanismo implementa a propriedade temporal das tags:
    /// - Tags recentes e não reforçadas decaem → não consolidam
    /// - Tags repetidas ou reforçadas acumulam → superam threshold → consolidam
    pub fn decay_tags(&mut self) {
        for tag in &mut self.synaptic_tags {
            // Decaimento exponencial: tag *= (1 - rate)
            *tag *= 1.0 - self.tag_decay_rate;

            // Limpeza de ruído residual (evita valores minúsculos)
            if *tag < 1e-3 {
                *tag = 0.0;
            }
        }
    }

    /// SYNAPTIC TAGGING: Consolidação "Tagged & Capture" - O Porteiro Rigoroso
    ///
    /// Transfere dados da STM para a LTM, usando a Tag como filtro de relevância.
    /// Apenas sinapses com tags acima do limiar são consolidadas.
    ///
    /// Este método substitui consolidate_memory() durante o sono para implementar
    /// consolidação seletiva baseada em relevância.
    ///
    /// Chamado apenas durante o ciclo de sono.
    ///
    /// # Argumentos
    /// * `base_consolidation_rate` - Taxa base de consolidação (ex: 0.01 = 1% por passo)
    ///
    /// # Funcionamento
    /// - Tag > threshold: consolida proporcionalmente à força da tag
    /// - Tag < threshold: LTM ignora a STM (ruído filtrado)
    /// - Consolidação consome parcialmente a tag (simula consumo de recursos)
    pub fn consolidate_memory_tagged(&mut self, base_consolidation_rate: f64) {
        for i in 0..self.weights.len() {
            let tag = self.synaptic_tags[i];

            // SÓ ENTRA SE TIVER CERTEZA (tag > threshold)
            if tag > self.capture_threshold {
                let stm = self.weights[i];
                let ltm = self.weights_ltm[i];

                // Velocidade dinâmica: Tags fortes consolidam mais rápido
                // tag=0.2 → rate=base, tag=1.0 → rate=base*5, tag=2.0 → rate=base*10
                let dynamic_rate = base_consolidation_rate * tag;

                // A LTM se move em direção à STM ("Capture")
                // LTM += rate * (STM - LTM)
                self.weights_ltm[i] += dynamic_rate * (stm - ltm);

                // Segurança: clamp LTM
                self.weights_ltm[i] = self.weights_ltm[i].clamp(0.0, self.weight_clamp * 2.0);

                // Consumo da Etiqueta:
                // A consolidação consome os recursos químicos (reset parcial)
                // Isso evita que a mesma tag consolide indefinidamente
                self.synaptic_tags[i] *= 0.5;

                // ANTI-REGRESSÃO: Atualiza estabilidade
                // Sinapses consolidadas por tagging são mais estáveis
                let abs_delta = (dynamic_rate * (stm - ltm)).abs();
                if abs_delta < 1e-4 {
                    // Consolidação estável → aumenta estabilidade
                    self.ltm_stability[i] = (self.ltm_stability[i] + 0.02).min(1.0);
                } else {
                    // Ainda mudando → estabilidade diminui levemente
                    self.ltm_stability[i] = (self.ltm_stability[i] * 0.98).max(0.0);
                }
            }
            // Se Tag < Threshold, a LTM ignora a STM (ruído filtrado).
        }
    }
}