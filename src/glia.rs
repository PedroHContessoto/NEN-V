//! Módulo responsável pela modulação metabólica do neurônio
//!
//! A Glia modula a atividade do neurônio com base no seu estado metabólico,
//! implementando dinÃ¢micas homeostáticas através da gestão de energia.
//!
//! ## Novidades v2.0
//!
//! - **Reserva Energética**: Pool de emergÃªncia para bursts de atividade
//! - **Modulação de Plasticidade**: Energia afeta aprendizado de forma não-linear
//! - **Metabolismo Adaptativo**: Taxa de recuperação ajusta com demanda

#[derive(Debug, Clone)]
pub struct Glia {
    /// Energia atual do neurônio
    pub energy: f64,

    /// Prioridade do neurônio (mecanismo de atenção)
    pub priority: f64,

    /// Nível de alerta (estados globais da rede)
    pub alert_level: f64,

    // ========== CONSTANTES METABÃ“LICAS ==========
    pub max_energy: f64,
    pub energy_cost_fire: f64,
    pub energy_cost_maintenance: f64,
    pub energy_recovery_rate: f64,

    // ========== RESERVA ENERGÃ‰TICA (NOVO v2.0) ==========
    /// Reserva de energia para emergÃªncias
    pub energy_reserve: f64,

    /// Capacidade máxima da reserva
    pub max_reserve: f64,

    /// Taxa de transferÃªncia para reserva quando energia cheia
    pub reserve_fill_rate: f64,

    /// Threshold de energia para usar reserva
    pub reserve_use_threshold: f64,

    // ========== METABOLISMO ADAPTATIVO (NOVO v2.0) ==========
    /// Taxa de recuperação base (referÃªncia)
    base_recovery_rate: f64,

    /// Fator de adaptação do metabolismo (1.0 = normal)
    pub metabolic_adaptation: f64,

    /// Demanda metabólica recente (EMA)
    pub recent_demand: f64,

    /// Alpha para EMA da demanda
    demand_alpha: f64,

    // ========== MODULAÃ‡ÃƒO DE PLASTICIDADE ==========
    /// Threshold de energia para plasticidade plena
    pub plasticity_energy_threshold: f64,

    /// Custo energético por unidade de plasticidade
    pub plasticity_cost_factor: f64,
}

impl Glia {
    /// Cria uma nova instÃ¢ncia de Glia com parÃ¢metros padrão
    pub fn new() -> Self {
        Self {
            energy: 100.0,
            priority: 1.0,
            alert_level: 0.0,

            // ParÃ¢metros metabólicos principais
            max_energy: 100.0,
            energy_cost_fire: 10.0,
            energy_cost_maintenance: 0.1,
            energy_recovery_rate: 10.0,

            // Reserva energética v2.0
            energy_reserve: 20.0,      // Começa com alguma reserva
            max_reserve: 50.0,
            reserve_fill_rate: 0.5,
            reserve_use_threshold: 20.0, // Usa reserva se energia < 20

            // Metabolismo adaptativo v2.0
            base_recovery_rate: 10.0,
            metabolic_adaptation: 1.0,
            recent_demand: 0.0,
            demand_alpha: 0.01,

            // Modulação de plasticidade
            plasticity_energy_threshold: 50.0,
            plasticity_cost_factor: 0.1,
        }
    }

    /// Cria uma Glia com parÃ¢metros personalizados
    pub fn with_params(
        max_energy: f64,
        energy_cost_fire: f64,
        energy_cost_maintenance: f64,
        energy_recovery_rate: f64,
    ) -> Self {
        Self {
            energy: max_energy,
            priority: 1.0,
            alert_level: 0.0,
            max_energy,
            energy_cost_fire,
            energy_cost_maintenance,
            energy_recovery_rate,

            // Reserva proporcional
            energy_reserve: max_energy * 0.2,
            max_reserve: max_energy * 0.5,
            reserve_fill_rate: 0.5,
            reserve_use_threshold: max_energy * 0.2,

            // Metabolismo adaptativo
            base_recovery_rate: energy_recovery_rate,
            metabolic_adaptation: 1.0,
            recent_demand: 0.0,
            demand_alpha: 0.01,

            // Plasticidade
            plasticity_energy_threshold: max_energy * 0.5,
            plasticity_cost_factor: 0.1,
        }
    }

    /// Modula o potencial integrado baseado na energia disponível e priority
    ///
    /// Fórmula v2: potencial_modulado = potencial_integrado * energy_factor * priority
    pub fn modulate(&self, integrated_potential: f64) -> f64 {
        let energy_factor = (self.energy / self.max_energy).max(0.0);
        integrated_potential * energy_factor * self.priority
    }

    /// Calcula o fator de plasticidade baseado na energia
    ///
    /// Retorna um valor [0.0, 1.0] que modula o aprendizado
    /// Função sigmoide suave centrada em plasticity_energy_threshold
    pub fn compute_plasticity_factor(&self) -> f64 {
        let e = self.energy / self.max_energy;
        let threshold = self.plasticity_energy_threshold / self.max_energy;

        // Sigmoide suave
        // e = 0.2: factor â‰ˆ 0.05
        // e = 0.5: factor â‰ˆ 0.5
        // e = 0.8: factor â‰ˆ 0.95
        let steepness = 10.0;
        1.0 / (1.0 + (-steepness * (e - threshold)).exp())
    }

    /// Atualiza o estado metabólico da Glia após um passo de simulação
    ///
    /// Inclui:
    /// - Consumo/recuperação de energia
    /// - Gestão de reserva
    /// - Adaptação metabólica
    pub fn update_state(&mut self, did_fire: bool) {
        // Atualiza demanda recente
        let current_demand = if did_fire { 1.0 } else { 0.0 };
        self.recent_demand = (1.0 - self.demand_alpha) * self.recent_demand
                           + self.demand_alpha * current_demand;

        // Adapta metabolismo baseado na demanda
        // Alta demanda â†’ aumenta recuperação (até 50%)
        // Baixa demanda â†’ normaliza
        let target_adaptation = 1.0 + self.recent_demand * 0.5;
        self.metabolic_adaptation += 0.01 * (target_adaptation - self.metabolic_adaptation);
        self.metabolic_adaptation = self.metabolic_adaptation.clamp(0.8, 1.5);

        if did_fire {
            // Consome energia ao disparar
            self.energy -= self.energy_cost_fire;

            // Se energia baixa, usa reserva
            if self.energy < self.reserve_use_threshold && self.energy_reserve > 0.0 {
                let needed = (self.reserve_use_threshold - self.energy).min(self.energy_reserve);
                let transfer = needed.min(5.0); // Max 5 por step
                self.energy += transfer;
                self.energy_reserve -= transfer;
            }
        } else {
            // Recupera energia em repouso
            let deficit_fraction = 1.0 - self.energy / self.max_energy;
            let base_recovery = self.energy_recovery_rate * deficit_fraction * self.metabolic_adaptation;

            // Alert level aumenta recuperação
            let alert_boost = base_recovery * self.alert_level;

            self.energy += base_recovery + alert_boost;

            // Se energia cheia, preenche reserva
            if self.energy >= self.max_energy * 0.95 && self.energy_reserve < self.max_reserve {
                let overflow = (self.energy - self.max_energy * 0.95).min(self.reserve_fill_rate);
                self.energy_reserve += overflow;
                self.energy_reserve = self.energy_reserve.min(self.max_reserve);
            }
        }

        // Custo de manutenção constante
        self.energy -= self.energy_cost_maintenance;

        // Clamp
        self.energy = self.energy.clamp(0.0, self.max_energy);
    }

    /// Retorna a fração de energia atual (0.0 a 1.0)
    pub fn energy_fraction(&self) -> f64 {
        self.energy / self.max_energy
    }

    /// Retorna a fração de energia total (incluindo reserva)
    pub fn total_energy_fraction(&self) -> f64 {
        (self.energy + self.energy_reserve) / (self.max_energy + self.max_reserve)
    }

    /// Aplica custo energético de plasticidade sináptica
    pub fn consume_plasticity_energy(&mut self, plasticity_cost: f64) {
        let energy_cost = self.plasticity_cost_factor * plasticity_cost;
        self.energy -= energy_cost;
        self.energy = self.energy.clamp(0.0, self.max_energy);
    }

    /// Verifica se há energia suficiente para plasticidade
    pub fn can_learn(&self) -> bool {
        self.energy > self.plasticity_energy_threshold * 0.3
    }

    /// Ajusta parÃ¢metros metabólicos para modo sono
    pub fn enter_sleep_mode(&mut self) {
        self.energy_recovery_rate *= 1.5;
        self.energy_cost_fire *= 0.5;
        self.energy_cost_maintenance *= 0.5;
    }

    /// Restaura parÃ¢metros metabólicos para modo vigília
    pub fn exit_sleep_mode(&mut self) {
        self.energy_recovery_rate /= 1.5;
        self.energy_cost_fire /= 0.5;
        self.energy_cost_maintenance /= 0.5;
    }

    /// Força recarga completa de energia (útil para testes)
    pub fn full_recharge(&mut self) {
        self.energy = self.max_energy;
        self.energy_reserve = self.max_reserve;
    }

    /// Retorna estatísticas do estado metabólico
    pub fn get_stats(&self) -> GliaStats {
        GliaStats {
            energy: self.energy,
            energy_fraction: self.energy_fraction(),
            reserve: self.energy_reserve,
            total_fraction: self.total_energy_fraction(),
            plasticity_factor: self.compute_plasticity_factor(),
            metabolic_adaptation: self.metabolic_adaptation,
            recent_demand: self.recent_demand,
        }
    }
}

impl Default for Glia {
    fn default() -> Self {
        Self::new()
    }
}

/// Estatísticas do estado metabólico
#[derive(Debug, Clone)]
pub struct GliaStats {
    pub energy: f64,
    pub energy_fraction: f64,
    pub reserve: f64,
    pub total_fraction: f64,
    pub plasticity_factor: f64,
    pub metabolic_adaptation: f64,
    pub recent_demand: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_glia_initialization() {
        let glia = Glia::new();
        assert_eq!(glia.energy, 100.0);
        assert_eq!(glia.priority, 1.0);
        assert_eq!(glia.alert_level, 0.0);
        assert!(glia.energy_reserve > 0.0);
    }

    #[test]
    fn test_modulation_full_energy() {
        let glia = Glia::new();
        let modulated = glia.modulate(50.0);
        assert_relative_eq!(modulated, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_modulation_half_energy() {
        let mut glia = Glia::new();
        glia.energy = 50.0;
        let modulated = glia.modulate(50.0);
        assert_relative_eq!(modulated, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_consumption_on_fire() {
        let mut glia = Glia::new();
        let initial_energy = glia.energy;
        glia.update_state(true);

        assert!(glia.energy < initial_energy);
    }

    #[test]
    fn test_energy_recovery_at_rest() {
        let mut glia = Glia::new();
        glia.energy = 50.0;
        let initial_energy = glia.energy;

        glia.update_state(false);

        assert!(glia.energy > initial_energy);
    }

    #[test]
    fn test_reserve_usage() {
        let mut glia = Glia::new();
        glia.energy = 15.0; // Abaixo do threshold
        glia.energy_reserve = 30.0;

        let initial_reserve = glia.energy_reserve;

        // Dispara (consome energia)
        glia.update_state(true);

        // Reserva deve ter sido usada
        assert!(glia.energy_reserve < initial_reserve);
    }

    #[test]
    fn test_plasticity_factor() {
        let mut glia = Glia::new();

        // Energia plena â†’ fator alto
        glia.energy = 100.0;
        assert!(glia.compute_plasticity_factor() > 0.9);

        // Energia baixa â†’ fator baixo
        glia.energy = 10.0;
        assert!(glia.compute_plasticity_factor() < 0.1);

        // Energia no threshold â†’ fator ~0.5
        glia.energy = glia.plasticity_energy_threshold;
        let factor = glia.compute_plasticity_factor();
        assert!(factor > 0.4 && factor < 0.6);
    }

    #[test]
    fn test_metabolic_adaptation() {
        let mut glia = Glia::new();

        // Simula alta demanda (disparos frequentes)
        for _ in 0..1000 {
            glia.update_state(true);
            glia.energy = 80.0; // Mantém energia para poder disparar
        }

        // Adaptação deve ter aumentado
        assert!(glia.metabolic_adaptation > 1.0);
    }

    #[test]
    fn test_alert_level_accelerates_recovery() {
        let mut glia_normal = Glia::new();
        let mut glia_alert = Glia::new();

        glia_normal.energy = 50.0;
        glia_alert.energy = 50.0;
        glia_alert.alert_level = 1.0;

        glia_normal.update_state(false);
        glia_alert.update_state(false);

        assert!(glia_alert.energy > glia_normal.energy);
    }

    #[test]
    fn test_sleep_mode() {
        let mut glia = Glia::new();
        let original_recovery = glia.energy_recovery_rate;
        let original_cost = glia.energy_cost_fire;

        glia.enter_sleep_mode();

        assert!(glia.energy_recovery_rate > original_recovery);
        assert!(glia.energy_cost_fire < original_cost);

        glia.exit_sleep_mode();

        assert_relative_eq!(glia.energy_recovery_rate, original_recovery, epsilon = 1e-10);
        assert_relative_eq!(glia.energy_cost_fire, original_cost, epsilon = 1e-10);
    }
}