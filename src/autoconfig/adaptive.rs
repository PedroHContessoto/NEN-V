//! Mecanismos adaptativos runtime para AutoConfig
//!
//! Este módulo implementa ajustes dinâmicos de parâmetros durante a execução,
//! permitindo que a rede se auto-regule baseado em métricas observadas.
//!
//! ## Hierarquia de Escalas Temporais (Biologically-Inspired)
//!
//! O sistema adaptativo opera em três escalas temporais inspiradas em sistemas
//! biológicos, cada uma com diferentes mecanismos e objetivos:
//!
//! ### 1. Escala Rápida (milissegundos–segundos de simulação)
//!
//! **Mecanismos:**
//! - STDP sináptico (Spike-Timing-Dependent Plasticity)
//! - Glia consumindo energia a cada spike
//! - Threshold adaptativo local (dentro de `decide_to_fire`)
//! - Balanço excitação/inibição imediato
//!
//! **Objetivo:** Responder rapidamente a padrões de atividade, aprender correlações
//! temporais entre spikes, e gerenciar recursos energéticos localmente.
//!
//! **Localização no código:**
//! - `NENV::decide_to_fire()` - threshold adaptativo
//! - `Dendritoma::apply_stdp()` - plasticidade sináptica
//! - `Glia::consume_energy()` - metabolismo local
//!
//! ### 2. Escala Média (segundos–minutos de simulação)
//!
//! **Mecanismos:**
//! - Synaptic scaling homeostático (`apply_homeostatic_plasticity`)
//! - Metaplasticidade via BCM (ajuste de meta_threshold)
//! - Modulação energética de plasticidade
//!
//! **Objetivo:** Manter firing rate de cada neurônio próximo ao seu `target_firing_rate`
//! individual, prevenir instabilidade, e balancear aprendizado com recursos.
//!
//! **Localização no código:**
//! - `NENV::apply_homeostatic_plasticity()` - synaptic scaling local
//! - `NENV::compute_bcm_gain()` - metaplasticidade
//! - `NENV::modulate_learning_by_energy()` - acoplamento energia/plasticidade
//!
//! ### 3. Escala Lenta (minutos–horas / múltiplos episódios)
//!
//! **Mecanismos:**
//! - Controlador PI global sobre firing rate médio (`AdaptiveState::compute_pi_control`)
//! - Ajuste de parâmetros metabólicos globais (energy_cost_fire, energy_recovery_rate)
//! - Gatilho e parâmetros de sono (frequência, duração, replay_noise)
//! - Integração com reward para modular intervenções (3-factor learning)
//!
//! **Objetivo:** Garantir homeostase global da rede, adaptar a dinâmica metabólica
//! ao contexto da tarefa, e consolidar memórias via sono guiado por performance.
//!
//! **Localização no código:**
//! - `AdaptiveState::compute_pi_control()` - controle global de FR
//! - `AdaptiveState::evaluate_sleep_outcome()` - otimização de sono via reward
//! - `monitor_and_adapt()` - orquestração das intervenções globais
//!
//! ## Separação de Preocupações (Design Pattern)
//!
//! Esta hierarquia segue o princípio de **separação de escalas temporais**:
//! - Escalas rápidas não devem ser influenciadas diretamente por controle lento
//!   (apenas indiretamente via parâmetros globais como threshold, learning rate)
//! - Controle lento observa médias temporais e espaciais (não responde a flutuações locais)
//! - Cada escala tem seu próprio "budget" de intervenções (cooldowns, thresholds)
//!
//! ## Analogia Biológica
//!
//! - **Escala rápida**: Canais iônicos, liberação de neurotransmissores
//! - **Escala média**: Expressão gênica local, síntese de receptores, regulação homeostática
//! - **Escala lenta**: Modulação neuroendócrina, ciclos circadianos, consolidação durante sono

use super::*;
use crate::network::Network;

/// Problemas detectáveis durante execução
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkIssue {
    /// Firing rate muito abaixo do target
    UnderFiring,
    /// Firing rate muito acima do target
    OverFiring,
    /// Energia média muito baixa (< 30%)
    EnergyDepletionRisk,
    /// Firing rate oscilando excessivamente
    Instability,
    /// Rede "morta" (nenhum neurônio disparando)
    DeadNetwork,
    /// Runaway excitation (todos disparando)
    RunawayExcitation,
}

impl NetworkIssue {
    pub fn description(&self) -> &str {
        match self {
            Self::UnderFiring => "Taxa de disparo abaixo do alvo",
            Self::OverFiring => "Taxa de disparo acima do alvo",
            Self::EnergyDepletionRisk => "Risco de esgotamento energético",
            Self::Instability => "Oscilações excessivas na taxa de disparo",
            Self::DeadNetwork => "Rede inativa (sem disparos)",
            Self::RunawayExcitation => "Excitação descontrolada",
        }
    }

    pub fn severity(&self) -> u8 {
        match self {
            Self::DeadNetwork => 10,
            Self::RunawayExcitation => 10,
            Self::EnergyDepletionRisk => 8,
            Self::Instability => 6,
            Self::UnderFiring => 4,
            Self::OverFiring => 4,
        }
    }
}

/// Resultado da avaliação do sono
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepOutcome {
    Improved,  // Performance melhorou após sono
    Worsened,  // Performance piorou após sono
    Neutral,   // Sem mudança significativa
    NoData,    // Sem dados para comparar
}

/// Ação corretiva a ser aplicada
#[derive(Debug, Clone)]
pub enum CorrectiveAction {
    /// Ajustar threshold de todos os neurônios
    AdjustThreshold { delta: f64 },
    /// Ajustar taxa de aprendizado
    AdjustLearningRate { factor: f64 },
    /// Ajustar energia recovery rate
    AdjustEnergyRecovery { factor: f64 },
    /// Ajustar homeostase (eta)
    AdjustHomeostasis { new_eta: f64 },
    /// Forçar ciclo de sono (consolidação)
    ForceSleepCycle { duration: usize },
}

/// Estado adaptativo da rede com controlador PI científico
#[derive(Debug, Clone)]
pub struct AdaptiveState {
    /// Histórico recente de firing rates (últimos N steps)
    firing_rate_history: Vec<f64>,
    /// Histórico de energia média
    energy_history: Vec<f64>,
    /// Último step em que adaptação foi aplicada
    last_adaptation_step: i64,
    /// Intervalo mínimo entre adaptações (evita oscilações)
    pub adaptation_cooldown: i64,
    /// Contador de adaptações aplicadas
    pub adaptation_count: usize,
    /// Tamanho da janela de histórico
    history_window: usize,

    // ========== CONTROLADOR PI ==========
    /// Target firing rate global
    target_firing_rate: f64,
    /// Erro integral acumulado (componente I)
    fr_error_integral: f64,
    /// Clamp do integrador para evitar wind-up
    fr_integral_clamp: f64,
    /// Ganho proporcional
    kp_fr: f64,
    /// Ganho integral
    ki_fr: f64,

    // ========== HISTERESE E ESTABILIDADE ==========
    /// Contador de steps estáveis (FR dentro da faixa)
    pub stable_steps: i64,
    /// Limiar para considerar estável (fração do target)
    stability_threshold: f64,

    // ========== REWARD TRACKING ==========
    /// Recompensa média (EMA)
    pub avg_reward: f64,
    /// Alpha para EMA de reward
    reward_alpha: f64,

    // ========== SONO E CONSOLIDAÇÃO ==========
    /// Snapshot de FR antes do último sono
    pre_sleep_fr: Option<f64>,
    /// Snapshot de reward antes do último sono
    pre_sleep_reward: Option<f64>,
    /// Número de ciclos de sono executados
    pub sleep_cycles: usize,
}

impl AdaptiveState {
    /// Cria um novo estado adaptativo com controlador PI
    pub fn new() -> Self {
        Self::with_target_fr(0.15) // Default fallback
    }

    /// Cria com target FR específico (recomendado: usar o do AutoConfig)
    pub fn with_target_fr(target_fr: f64) -> Self {
        Self {
            firing_rate_history: Vec::new(),
            energy_history: Vec::new(),
            last_adaptation_step: 0,
            adaptation_cooldown: 100,
            adaptation_count: 0,
            history_window: 50,

            // Controlador PI (parâmetros retunados AGRESSIVAMENTE para eliminar steady-state error)
            target_firing_rate: target_fr,
            fr_error_integral: 0.0,
            fr_integral_clamp: 5.0,      // Anti-windup bem maior (antes 2.0 → 5.0)
            kp_fr: 0.4,                  // Ganho proporcional mais suave (0.5 → 0.4)
            ki_fr: 0.05,                 // Ganho integral BEM mais agressivo (0.02 → 0.05)

            // Histerese
            stable_steps: 0,
            stability_threshold: 0.05, // 5% de tolerância

            // Reward
            avg_reward: 0.0,
            reward_alpha: 0.01,

            // Sono
            pre_sleep_fr: None,
            pre_sleep_reward: None,
            sleep_cycles: 0,
        }
    }

    /// Captura snapshot antes do sono
    pub fn pre_sleep_snapshot(&mut self, current_fr: f64) {
        self.pre_sleep_fr = Some(current_fr);
        self.pre_sleep_reward = Some(self.avg_reward);
    }

    /// Avalia resultado do sono e decide se houve melhora
    pub fn evaluate_sleep_outcome(&mut self, post_sleep_fr: f64) -> SleepOutcome {
        self.sleep_cycles += 1;

        let pre_fr = match self.pre_sleep_fr {
            Some(fr) => fr,
            None => return SleepOutcome::NoData,
        };

        let pre_reward = self.pre_sleep_reward.unwrap_or(0.0);

        // Calcula mudanças
        let fr_change = (post_sleep_fr - pre_fr).abs();
        let reward_change = self.avg_reward - pre_reward;

        // Avalia resultado
        if reward_change > 0.01 && fr_change < 0.05 {
            SleepOutcome::Improved // Reward melhorou, FR estável
        } else if reward_change < -0.01 {
            SleepOutcome::Worsened // Reward piorou
        } else {
            SleepOutcome::Neutral // Sem mudança significativa
        }
    }

    /// Registra métricas do step atual
    ///
    /// **Estabilidade agora é baseada no erro de FR, não no sinal de controle PI**
    pub fn record_metrics(&mut self, firing_rate: f64, avg_energy: f64) {
        self.firing_rate_history.push(firing_rate);
        self.energy_history.push(avg_energy);

        // Limita tamanho do histórico
        if self.firing_rate_history.len() > self.history_window {
            self.firing_rate_history.remove(0);
            self.energy_history.remove(0);
        }

        // NOVO: Estabilidade baseada em erro relativo de FR
        let err_frac = ((firing_rate - self.target_firing_rate).abs()
            / self.target_firing_rate)
            .abs();

        if err_frac < self.stability_threshold {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }
    }

    /// Detecta problemas na rede
    pub fn detect_issues(&self, target_fr: f64, current_step: i64) -> Vec<NetworkIssue> {
        let mut issues = Vec::new();

        // Precisa de dados suficientes
        if self.firing_rate_history.len() < 10 {
            return issues;
        }

        let recent_fr: f64 = self.firing_rate_history.iter()
            .rev()
            .take(10)
            .sum::<f64>() / 10.0;

        let recent_energy: f64 = self.energy_history.iter()
            .rev()
            .take(10)
            .sum::<f64>() / 10.0;

        // Check 1: Rede morta
        if recent_fr < 0.001 && current_step > 50 {
            issues.push(NetworkIssue::DeadNetwork);
            return issues; // Problema crítico, retorna imediatamente
        }

        // Check 2: Runaway excitation
        if recent_fr > 0.95 {
            issues.push(NetworkIssue::RunawayExcitation);
            return issues; // Problema crítico
        }

        // Check 3: Depleção energética
        if recent_energy < 30.0 {
            issues.push(NetworkIssue::EnergyDepletionRisk);
        }

        // Check 4: Under/Over firing (agora baseado em erro RELATIVO)
        let fr_error_frac = (recent_fr - target_fr).abs() / target_fr;

        // Considera problema se erro relativo > 20%
        if fr_error_frac > 0.2 {
            if recent_fr < target_fr {
                issues.push(NetworkIssue::UnderFiring);
            } else {
                issues.push(NetworkIssue::OverFiring);
            }
        }

        // Check 5: Instabilidade (variância alta)
        if self.firing_rate_history.len() >= 20 {
            let recent_20: Vec<f64> = self.firing_rate_history.iter()
                .rev()
                .take(20)
                .copied()
                .collect();

            let mean: f64 = recent_20.iter().sum::<f64>() / recent_20.len() as f64;
            let variance: f64 = recent_20.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / recent_20.len() as f64;

            let std_dev = variance.sqrt();

            // Se desvio padrão > 20% da média, há instabilidade
            if std_dev > mean * 0.2 && mean > 0.01 {
                issues.push(NetworkIssue::Instability);
            }
        }

        issues
    }

    /// Calcula ação usando controlador PI científico
    ///
    /// ## Teoria de Controle Aplicada:
    ///
    /// Controlador PI (Proporcional-Integral) para firing rate:
    ///
    /// ```text
    /// e(t) = FR_target - FR_atual
    /// u(t) = Kp * e(t) + Ki * ∫e(t)dt
    /// ```
    ///
    /// - **Termo P**: Responde imediatamente ao erro atual
    /// - **Termo I**: Elimina erro de estado estacionário
    /// - **Anti-windup**: Limita integral para evitar saturação
    /// - **Deadzone**: Histerese para reduzir "thrash"
    ///
    /// ## Mapeamento para Biologia:
    ///
    /// `u(t) → Δthreshold`:
    /// - u > 0: FR baixo → **reduz** threshold → neurônios disparam mais
    /// - u < 0: FR alto → **aumenta** threshold → neurônios disparam menos
    ///
    /// ## Debug Info (opcional)
    ///
    /// Retorna tupla (ação, debug_info) onde debug_info contém:
    /// - error: erro atual
    /// - integral: valor do integrador
    /// - u: sinal de controle
    /// - delta: ajuste aplicado (ou None se em deadzone)
    ///
    pub fn compute_pi_control(&mut self, current_fr: f64) -> Option<CorrectiveAction> {
        // Erro: target - atual (positivo = precisa aumentar FR)
        let error = self.target_firing_rate - current_fr;

        // Atualiza integral com anti-windup (evita saturação)
        self.fr_error_integral += error;
        let integral_before_clamp = self.fr_error_integral;
        self.fr_error_integral = self.fr_error_integral
            .clamp(-self.fr_integral_clamp, self.fr_integral_clamp);

        // Sinal de controle PI
        let u = self.kp_fr * error + self.ki_fr * self.fr_error_integral;

        // DEBUG: Armazena info para diagnóstico (se necessário)
        let _debug_saturated = (integral_before_clamp - self.fr_error_integral).abs() > 0.001;

        // Deadzone: se sinal é pequeno, não aplica correção (reduz thrash)
        // IMPORTANTE: Não modifica stable_steps aqui, pois estabilidade
        // é agora definida pelo erro de FR em record_metrics()
        if u.abs() < 0.01 {
            return None;
        }

        // Mapeia sinal de controle → ajuste de threshold
        // Ganho reduzido para -0.4 (resposta mais conservadora)
        let delta_threshold = -0.4 * u; // sinal invertido

        Some(CorrectiveAction::AdjustThreshold {
            delta: delta_threshold.clamp(-0.1, 0.1) // REDUZIDO: -0.3→-0.1 (evita saturação de threshold)
        })
    }

    /// Retorna estado interno do PI para debug
    pub fn pi_state(&self) -> (f64, f64, f64) {
        (self.target_firing_rate, self.fr_error_integral, self.fr_integral_clamp)
    }

    /// Sugere ações corretivas (mantém casos críticos + PI controller)
    pub fn suggest_actions(&mut self, issues: &[NetworkIssue], current_fr: f64) -> Vec<CorrectiveAction> {
        let mut actions = Vec::new();

        // Casos críticos têm prioridade
        let has_critical = issues.iter().any(|i| matches!(
            i,
            NetworkIssue::DeadNetwork | NetworkIssue::RunawayExcitation
        ));

        if has_critical {
            for issue in issues {
                match issue {
                    NetworkIssue::DeadNetwork => {
                        actions.push(CorrectiveAction::AdjustThreshold { delta: -0.1 });
                        actions.push(CorrectiveAction::AdjustEnergyRecovery { factor: 1.5 });
                    }
                    NetworkIssue::RunawayExcitation => {
                        actions.push(CorrectiveAction::AdjustThreshold { delta: 0.15 });
                        actions.push(CorrectiveAction::AdjustHomeostasis { new_eta: 0.1 });
                    }
                    _ => {}
                }
            }
        } else {
            // Situação normal: usa controlador PI
            if let Some(pi_action) = self.compute_pi_control(current_fr) {
                actions.push(pi_action);
            }

            // Energia ainda é tratada separadamente
            if issues.contains(&NetworkIssue::EnergyDepletionRisk) {
                actions.push(CorrectiveAction::AdjustEnergyRecovery { factor: 1.3 });
            }

            // Instabilidade: reduz learning
            if issues.contains(&NetworkIssue::Instability) {
                actions.push(CorrectiveAction::AdjustLearningRate { factor: 0.8 });
            }
        }

        actions
    }

    /// Atualiza reward tracking
    pub fn update_reward(&mut self, episode_reward: f64) {
        self.avg_reward = (1.0 - self.reward_alpha) * self.avg_reward
                        + self.reward_alpha * episode_reward;
    }

    /// Verifica se deve fazer adaptação baseado em reward
    /// Retorna true se FR está fora do alvo E performance está ruim
    pub fn should_adapt_for_task(&self, firing_rate: f64) -> bool {
        let fr_error = (firing_rate - self.target_firing_rate).abs() / self.target_firing_rate;

        // Se FR está muito próximo do alvo E reward está OK, não mexe
        if fr_error < 0.1 && self.avg_reward > 0.1 {
            return false; // Performance boa, não mexer
        }

        // Se FR está fora E reward está ruim/estagnado, pode adaptar
        if fr_error > 0.2 && self.avg_reward < 0.05 {
            return true; // Performance ruim, precisa ajustar
        }

        // Caso intermediário: usa só o erro de FR
        fr_error > 0.15
    }

    /// Verifica se está estável há tempo suficiente
    pub fn is_stable(&self, min_stable_steps: i64) -> bool {
        self.stable_steps >= min_stable_steps
    }

    /// Ajusta cooldown dinamicamente baseado em estabilidade
    pub fn update_adaptive_cooldown(&mut self) {
        // Se estável há tempo, aumenta cooldown (menos intervenções)
        if self.stable_steps > 1000 {
            self.adaptation_cooldown = (self.adaptation_cooldown as f64 * 1.2)
                .min(10_000.0) as i64;
        }
        // Se instável, reduz cooldown (mais responsivo)
        else if self.stable_steps == 0 {
            self.adaptation_cooldown = (self.adaptation_cooldown as f64 * 0.9)
                .max(50.0) as i64;
        }
    }

    /// Aplica uma ação corretiva na rede
    pub fn apply_action(
        &mut self,
        action: &CorrectiveAction,
        network: &mut Network,
        current_step: i64,
    ) -> bool {
        // Verifica cooldown (agora adaptativo)
        if current_step - self.last_adaptation_step < self.adaptation_cooldown {
            return false; // Ainda em cooldown
        }

        // Atualiza cooldown dinamicamente
        self.update_adaptive_cooldown();

        match action {
            CorrectiveAction::AdjustThreshold { delta } => {
                for neuron in &mut network.neurons {
                    // Range ampliado: [0.001, 5.0] para dar mais espaço ao PI
                    // (antes era [0.001, 2.0])
                    neuron.threshold = (neuron.threshold + delta).max(0.001).min(5.0);
                }
                self.last_adaptation_step = current_step;
                self.adaptation_count += 1;
                true
            }
            CorrectiveAction::AdjustLearningRate { factor } => {
                for neuron in &mut network.neurons {
                    let current_lr = neuron.dendritoma.get_learning_rate();
                    neuron.dendritoma.set_learning_rate(current_lr * factor);
                }
                self.last_adaptation_step = current_step;
                self.adaptation_count += 1;
                true
            }
            CorrectiveAction::AdjustEnergyRecovery { factor } => {
                for neuron in &mut network.neurons {
                    neuron.glia.energy_recovery_rate *= factor;
                }
                self.last_adaptation_step = current_step;
                self.adaptation_count += 1;
                true
            }
            CorrectiveAction::AdjustHomeostasis { new_eta } => {
                for neuron in &mut network.neurons {
                    neuron.homeo_eta = *new_eta;
                }
                self.last_adaptation_step = current_step;
                self.adaptation_count += 1;
                true
            }
            CorrectiveAction::ForceSleepCycle { duration } => {
                // Força entrada no modo sono
                network.enter_sleep(0.05, *duration);
                self.last_adaptation_step = current_step;
                self.adaptation_count += 1;
                true
            }
        }
    }

    /// Retorna número de adaptações aplicadas
    pub fn adaptation_count(&self) -> usize {
        self.adaptation_count
    }
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: cria AdaptiveState alinhado com o AutoConfig
pub fn adaptive_state_from_config(target_fr: f64) -> AdaptiveState {
    AdaptiveState::with_target_fr(target_fr)
}

/// Helper: Monitora e adapta a rede automaticamente
pub fn monitor_and_adapt(
    network: &mut Network,
    adaptive_state: &mut AdaptiveState,
    target_fr: f64,
    current_step: i64,
    verbose: bool,
) -> bool {
    // Coleta métricas
    let firing_rate = network.num_firing() as f64 / network.num_neurons() as f64;
    let avg_energy = network.average_energy();

    adaptive_state.record_metrics(firing_rate, avg_energy);

    // NOVO: Se reward tracking está ativo E performance está OK, não mexe
    if !adaptive_state.should_adapt_for_task(firing_rate) {
        if verbose && current_step % 5000 == 0 {
            println!("\n🎯 Performance aceitável para a tarefa (FR={:.4}, reward={:.3}) – não adaptando",
                firing_rate, adaptive_state.avg_reward);
        }
        return false;
    }

    // Verifica se está estável há tempo suficiente
    if adaptive_state.is_stable(2000) {
        if verbose && current_step % 5000 == 0 {
            println!("\n✅ Rede ESTÁVEL por {} steps (FR={:.4}, target={:.4})",
                adaptive_state.stable_steps, firing_rate, target_fr);
        }
        return false; // Estável, não mexe
    }

    // Detecta problemas
    let issues = adaptive_state.detect_issues(target_fr, current_step);

    if issues.is_empty() && firing_rate.abs() > 0.001 {
        return false; // Tudo OK e rede não está morta
    }

    // Reporta problemas apenas se não for ajuste fino do PI
    let has_non_trivial_issue = issues.iter().any(|i| i.severity() >= 6);

    if verbose && has_non_trivial_issue {
        println!("\n⚠️  PROBLEMAS DETECTADOS (step {}):", current_step);
        for issue in &issues {
            println!("  • [{}] {}", issue.severity(), issue.description());
        }
        println!("  FR={:.4}, target={:.4}, erro={:.1}%",
            firing_rate, target_fr,
            ((firing_rate - target_fr).abs() / target_fr * 100.0));
        println!("  Cooldown atual: {} steps", adaptive_state.adaptation_cooldown);
    }

    // Sugere e aplica ações (passa current FR para o PI controller)
    let actions = adaptive_state.suggest_actions(&issues, firing_rate);

    if verbose && !actions.is_empty() {
        println!("\n🔧 AÇÕES CORRETIVAS:");
    }

    let mut applied_any = false;

    for action in &actions {
        let applied = adaptive_state.apply_action(action, network, current_step);
        if applied {
            applied_any = true;
            if verbose {
                println!("  ✅ {:?}", action);
            }
        }
    }

    applied_any
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_state_initialization() {
        let state = AdaptiveState::new();
        assert_eq!(state.adaptation_count(), 0);
        assert_eq!(state.firing_rate_history.len(), 0);
    }

    #[test]
    fn test_record_metrics() {
        let mut state = AdaptiveState::new();

        for i in 0..100 {
            state.record_metrics(0.1 + i as f64 * 0.001, 50.0);
        }

        // Deve manter apenas últimos 50
        assert_eq!(state.firing_rate_history.len(), 50);
    }

    #[test]
    fn test_detect_dead_network() {
        let mut state = AdaptiveState::new();

        // Simula rede morta (FR = 0.0 por 60 steps)
        for _ in 0..60 {
            state.record_metrics(0.0, 80.0);
        }

        let issues = state.detect_issues(0.15, 60);

        assert!(issues.contains(&NetworkIssue::DeadNetwork));
    }

    #[test]
    fn test_detect_runaway_excitation() {
        let mut state = AdaptiveState::new();

        // Simula runaway (FR = 0.98)
        for _ in 0..20 {
            state.record_metrics(0.98, 50.0);
        }

        let issues = state.detect_issues(0.15, 20);

        assert!(issues.contains(&NetworkIssue::RunawayExcitation));
    }

    #[test]
    fn test_detect_energy_depletion() {
        let mut state = AdaptiveState::new();

        // Simula depleção energética
        for _ in 0..20 {
            state.record_metrics(0.15, 25.0); // Energia muito baixa
        }

        let issues = state.detect_issues(0.15, 20);

        assert!(issues.contains(&NetworkIssue::EnergyDepletionRisk));
    }

    #[test]
    fn test_suggest_actions_for_dead_network() {
        let state = AdaptiveState::new();
        let issues = vec![NetworkIssue::DeadNetwork];

        let actions = state.suggest_actions(&issues);

        // Deve sugerir redução de threshold e aumento de recovery
        assert!(actions.iter().any(|a| matches!(
            a,
            CorrectiveAction::AdjustThreshold { delta } if *delta < 0.0
        )));

        assert!(actions.iter().any(|a| matches!(
            a,
            CorrectiveAction::AdjustEnergyRecovery { factor } if *factor > 1.0
        )));
    }
}
