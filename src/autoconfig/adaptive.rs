//! # Sistema Adaptativo Runtime
//!
//! Monitora a rede durante execução e aplica correções automáticas
//! quando problemas são detectados.

use std::collections::VecDeque;
use crate::network::Network;
use crate::neuromodulation::NeuromodulatorType;
use super::AutoConfig;

/// Problemas detectáveis durante execução
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkIssue {
    /// Taxa de disparo abaixo do alvo
    UnderFiring,
    /// Taxa de disparo acima do alvo
    OverFiring,
    /// Risco de esgotamento energético
    EnergyDepletionRisk,
    /// Oscilações excessivas
    Instability,
    /// Rede completamente inativa
    DeadNetwork,
    /// Excitação descontrolada (>90%)
    RunawayExcitation,
    /// Aprendizado estagnado
    LearningStagnation,
    /// Colapso de pesos
    WeightCollapse,
}

impl NetworkIssue {
    /// Retorna descrição do problema
    pub fn description(&self) -> &str {
        match self {
            Self::UnderFiring => "Taxa de disparo abaixo do alvo",
            Self::OverFiring => "Taxa de disparo acima do alvo",
            Self::EnergyDepletionRisk => "Risco de esgotamento energético",
            Self::Instability => "Oscilações excessivas na taxa de disparo",
            Self::DeadNetwork => "Rede inativa (sem disparos)",
            Self::RunawayExcitation => "Excitação descontrolada",
            Self::LearningStagnation => "Aprendizado estagnado",
            Self::WeightCollapse => "Colapso de pesos sinápticos",
        }
    }

    /// Retorna severidade (1-10)
    pub fn severity(&self) -> u8 {
        match self {
            Self::DeadNetwork | Self::RunawayExcitation => 10,
            Self::WeightCollapse => 9,
            Self::EnergyDepletionRisk => 8,
            Self::LearningStagnation => 7,
            Self::Instability => 6,
            Self::UnderFiring | Self::OverFiring => 4,
        }
    }

    /// Verifica se é crítico (requer ação imediata)
    pub fn is_critical(&self) -> bool {
        self.severity() >= 9
    }
}

/// Resultado da avaliação do sono
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepOutcome {
    /// Performance melhorou após sono
    Improved,
    /// Performance piorou após sono
    Worsened,
    /// Sem mudança significativa
    Neutral,
    /// Dados insuficientes
    NoData,
}

/// Ação corretiva a ser aplicada
#[derive(Debug, Clone)]
pub enum CorrectiveAction {
    /// Ajusta threshold de todos os neurônios
    AdjustThreshold { delta: f64 },
    /// Ajusta taxa de aprendizado
    AdjustLearningRate { factor: f64 },
    /// Ajusta taxa de recuperação de energia
    AdjustEnergyRecovery { factor: f64 },
    /// Ajusta parâmetros homeostáticos
    AdjustHomeostasis { new_eta: f64 },
    /// Força ciclo de sono
    ForceSleepCycle { duration: usize },
    /// Ajusta amplitudes STDP
    AdjustSTDP { a_plus_factor: f64, a_minus_factor: f64 },
    /// Reinicializa pesos
    ReinitializeWeights { min_weight: f64, max_weight: f64 },
    /// Boost de neuromoduladores
    BoostNeuromodulation { dopamine: f64, norepinephrine: f64 },
}

impl CorrectiveAction {
    /// Retorna nome da ação
    pub fn name(&self) -> &str {
        match self {
            Self::AdjustThreshold { .. } => "AdjustThreshold",
            Self::AdjustLearningRate { .. } => "AdjustLearningRate",
            Self::AdjustEnergyRecovery { .. } => "AdjustEnergyRecovery",
            Self::AdjustHomeostasis { .. } => "AdjustHomeostasis",
            Self::ForceSleepCycle { .. } => "ForceSleepCycle",
            Self::AdjustSTDP { .. } => "AdjustSTDP",
            Self::ReinitializeWeights { .. } => "ReinitializeWeights",
            Self::BoostNeuromodulation { .. } => "BoostNeuromodulation",
        }
    }
}

/// Estado adaptativo runtime
#[derive(Debug)]
pub struct AdaptiveState {
    /// Referência à configuração original
    pub config: AutoConfig,

    /// Histórico de firing rate
    pub fr_history: VecDeque<f64>,

    /// Histórico de energia média
    pub energy_history: VecDeque<f64>,

    /// Histórico de reward
    pub reward_history: VecDeque<f64>,

    /// Histórico de gap de pesos
    pub weight_gap_history: VecDeque<f64>,

    /// Problemas detectados recentemente
    pub recent_issues: Vec<(i64, NetworkIssue)>,

    /// Ações corretivas aplicadas
    pub actions_taken: Vec<(i64, CorrectiveAction)>,

    /// Último timestep de intervenção
    pub last_intervention: i64,

    /// Cooldown entre intervenções
    pub intervention_cooldown: i64,

    /// Integradores PI
    pub pi_integral_fr: f64,
    pub pi_integral_energy: f64,

    /// Tamanho máximo do histórico
    history_size: usize,
}

impl AdaptiveState {
    /// Cria novo estado adaptativo
    pub fn new(config: AutoConfig) -> Self {
        Self {
            config,
            fr_history: VecDeque::with_capacity(1000),
            energy_history: VecDeque::with_capacity(1000),
            reward_history: VecDeque::with_capacity(1000),
            weight_gap_history: VecDeque::with_capacity(1000),
            recent_issues: Vec::new(),
            actions_taken: Vec::new(),
            last_intervention: -1000,
            intervention_cooldown: 100,
            pi_integral_fr: 0.0,
            pi_integral_energy: 0.0,
            history_size: 1000,
        }
    }

    /// Registra métricas do timestep atual
    pub fn record_metrics(&mut self, firing_rate: f64, avg_energy: f64, reward: Option<f64>) {
        self.fr_history.push_back(firing_rate);
        self.energy_history.push_back(avg_energy);

        if let Some(r) = reward {
            self.reward_history.push_back(r);
        }

        // Mantém tamanho máximo
        if self.fr_history.len() > self.history_size {
            self.fr_history.pop_front();
        }
        if self.energy_history.len() > self.history_size {
            self.energy_history.pop_front();
        }
        if self.reward_history.len() > self.history_size {
            self.reward_history.pop_front();
        }
    }

    /// Detecta problemas na rede
    pub fn detect_issues(&mut self, current_time: i64) -> Vec<NetworkIssue> {
        let mut issues = Vec::new();

        // Precisa de histórico mínimo
        if self.fr_history.len() < 50 {
            return issues;
        }

        let recent_fr: f64 = self.fr_history.iter().rev().take(20).sum::<f64>() / 20.0;
        let target_fr = self.config.params.target_firing_rate;

        // Verifica firing rate
        if recent_fr < 0.001 {
            issues.push(NetworkIssue::DeadNetwork);
        } else if recent_fr > 0.9 {
            issues.push(NetworkIssue::RunawayExcitation);
        } else if recent_fr < target_fr * 0.3 {
            issues.push(NetworkIssue::UnderFiring);
        } else if recent_fr > target_fr * 3.0 {
            issues.push(NetworkIssue::OverFiring);
        }

        // Verifica energia
        if !self.energy_history.is_empty() {
            let recent_energy: f64 = self.energy_history.iter().rev().take(20).sum::<f64>() / 20.0;
            if recent_energy < 30.0 {
                issues.push(NetworkIssue::EnergyDepletionRisk);
            }
        }

        // Verifica instabilidade (variância alta)
        if self.fr_history.len() >= 100 {
            let recent: Vec<f64> = self.fr_history.iter().rev().take(50).copied().collect();
            let mean: f64 = recent.iter().sum::<f64>() / 50.0;
            let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 50.0;

            if variance > mean * 2.0 {
                issues.push(NetworkIssue::Instability);
            }
        }

        // Registra problemas
        for issue in &issues {
            self.recent_issues.push((current_time, *issue));
        }

        // Limpa problemas antigos
        self.recent_issues.retain(|(t, _)| current_time - t < 1000);

        issues
    }

    /// Sugere ações corretivas para os problemas
    pub fn suggest_corrections(&self, issues: &[NetworkIssue]) -> Vec<CorrectiveAction> {
        let mut actions = Vec::new();

        for issue in issues {
            match issue {
                NetworkIssue::DeadNetwork => {
                    actions.push(CorrectiveAction::AdjustThreshold { delta: -0.05 });
                    actions.push(CorrectiveAction::AdjustLearningRate { factor: 1.5 });
                }
                NetworkIssue::RunawayExcitation => {
                    actions.push(CorrectiveAction::AdjustThreshold { delta: 0.1 });
                    actions.push(CorrectiveAction::AdjustHomeostasis { new_eta: 0.2 });
                }
                NetworkIssue::UnderFiring => {
                    actions.push(CorrectiveAction::AdjustThreshold { delta: -0.02 });
                }
                NetworkIssue::OverFiring => {
                    actions.push(CorrectiveAction::AdjustThreshold { delta: 0.02 });
                }
                NetworkIssue::EnergyDepletionRisk => {
                    actions.push(CorrectiveAction::AdjustEnergyRecovery { factor: 1.3 });
                }
                NetworkIssue::LearningStagnation => {
                    actions.push(CorrectiveAction::AdjustSTDP { a_plus_factor: 1.2, a_minus_factor: 0.8 });
                    actions.push(CorrectiveAction::BoostNeuromodulation { dopamine: 0.3, norepinephrine: 0.2 });
                }
                NetworkIssue::WeightCollapse => {
                    actions.push(CorrectiveAction::ReinitializeWeights { min_weight: 0.04, max_weight: 0.08 });
                }
                NetworkIssue::Instability => {
                    actions.push(CorrectiveAction::AdjustHomeostasis { new_eta: 0.15 });
                }
            }
        }

        actions
    }

    /// Aplica ação corretiva na rede
    pub fn apply_correction(&mut self, network: &mut Network, action: &CorrectiveAction, current_time: i64) {
        match action {
            CorrectiveAction::AdjustThreshold { delta } => {
                for neuron in &mut network.neurons {
                    neuron.threshold = (neuron.threshold + delta).clamp(0.05, 2.0);
                }
            }
            CorrectiveAction::AdjustLearningRate { factor } => {
                for neuron in &mut network.neurons {
                    neuron.dendritoma.set_learning_rate(
                        neuron.dendritoma.get_learning_rate() * factor
                    );
                }
            }
            CorrectiveAction::AdjustEnergyRecovery { factor } => {
                for neuron in &mut network.neurons {
                    neuron.glia.energy_recovery_rate *= factor;
                }
            }
            CorrectiveAction::AdjustHomeostasis { new_eta } => {
                for neuron in &mut network.neurons {
                    neuron.homeo_eta = *new_eta;
                }
            }
            CorrectiveAction::ForceSleepCycle { duration } => {
                network.enter_sleep(0.1, *duration);
            }
            CorrectiveAction::AdjustSTDP { a_plus_factor, a_minus_factor } => {
                for neuron in &mut network.neurons {
                    let (a_plus, a_minus) = neuron.dendritoma.get_stdp_amplitudes();
                    neuron.dendritoma.set_stdp_amplitudes(
                        a_plus * a_plus_factor,
                        a_minus * a_minus_factor,
                    );
                }
            }
            CorrectiveAction::ReinitializeWeights { min_weight, max_weight } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                for neuron in &mut network.neurons {
                    for w in &mut neuron.dendritoma.weights {
                        *w = rng.gen_range(*min_weight..*max_weight);
                    }
                }
            }
            CorrectiveAction::BoostNeuromodulation { dopamine, norepinephrine } => {
                network.neuromodulation.release(NeuromodulatorType::Dopamine, *dopamine);
                network.neuromodulation.release(NeuromodulatorType::Norepinephrine, *norepinephrine);
            }
        }

        self.actions_taken.push((current_time, action.clone()));
        self.last_intervention = current_time;
    }

    /// Monitora e adapta a rede automaticamente
    pub fn monitor_and_adapt(&mut self, network: &mut Network) {
        let current_time = network.current_time_step;

        // Respeita cooldown
        if current_time - self.last_intervention < self.intervention_cooldown {
            return;
        }

        // Coleta métricas
        let fr = network.num_firing() as f64 / network.num_neurons() as f64;
        let energy = network.average_energy();
        self.record_metrics(fr, energy, None);

        // Detecta problemas
        let issues = self.detect_issues(current_time);

        // Filtra para problemas críticos
        let critical_issues: Vec<_> = issues.iter()
            .filter(|i| i.is_critical())
            .collect();

        if !critical_issues.is_empty() {
            let corrections = self.suggest_corrections(&issues);
            // Aplica no máximo 2 correções por vez
            for action in corrections.iter().take(2) {
                self.apply_correction(network, action, current_time);
            }
        }
    }

    /// Retorna estatísticas do estado adaptativo
    pub fn get_stats(&self) -> AdaptiveStats {
        let avg_fr = if self.fr_history.is_empty() {
            0.0
        } else {
            self.fr_history.iter().sum::<f64>() / self.fr_history.len() as f64
        };

        let avg_energy = if self.energy_history.is_empty() {
            100.0
        } else {
            self.energy_history.iter().sum::<f64>() / self.energy_history.len() as f64
        };

        AdaptiveStats {
            avg_firing_rate: avg_fr,
            avg_energy,
            issues_detected: self.recent_issues.len(),
            actions_taken: self.actions_taken.len(),
        }
    }

    /// Limpa históricos
    pub fn clear_history(&mut self) {
        self.fr_history.clear();
        self.energy_history.clear();
        self.reward_history.clear();
        self.weight_gap_history.clear();
        self.recent_issues.clear();
        self.actions_taken.clear();
    }
}

/// Estatísticas do estado adaptativo
#[derive(Debug, Clone)]
pub struct AdaptiveStats {
    pub avg_firing_rate: f64,
    pub avg_energy: f64,
    pub issues_detected: usize,
    pub actions_taken: usize,
}

impl AdaptiveStats {
    /// Imprime relatório
    pub fn print_report(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│         ADAPTIVE STATE STATS            │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Avg Firing Rate:    {:>18.4} │", self.avg_firing_rate);
        println!("│ Avg Energy:         {:>18.2} │", self.avg_energy);
        println!("│ Issues Detected:    {:>18} │", self.issues_detected);
        println!("│ Actions Taken:      {:>18} │", self.actions_taken);
        println!("└─────────────────────────────────────────┘");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoconfig::{TaskSpec, TaskType, RewardDensity};

    fn create_test_config() -> AutoConfig {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };
        AutoConfig::from_task(task)
    }

    #[test]
    fn test_adaptive_state_creation() {
        let config = create_test_config();
        let state = AdaptiveState::new(config);

        assert!(state.fr_history.is_empty());
        assert!(state.recent_issues.is_empty());
    }

    #[test]
    fn test_record_metrics() {
        let config = create_test_config();
        let mut state = AdaptiveState::new(config);

        for i in 0..100 {
            state.record_metrics(0.1 + (i as f64 * 0.001), 80.0, Some(0.5));
        }

        assert_eq!(state.fr_history.len(), 100);
        assert_eq!(state.energy_history.len(), 100);
    }

    #[test]
    fn test_issue_severity() {
        assert!(NetworkIssue::DeadNetwork.is_critical());
        assert!(NetworkIssue::WeightCollapse.is_critical());
        assert!(!NetworkIssue::UnderFiring.is_critical());
    }

    #[test]
    fn test_suggestions() {
        let config = create_test_config();
        let state = AdaptiveState::new(config);

        let issues = vec![NetworkIssue::DeadNetwork];
        let actions = state.suggest_corrections(&issues);

        assert!(!actions.is_empty());
    }
}
