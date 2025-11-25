/// Módulo responsável pela modulação metabólica do neurónio
///
/// A Glia modula a atividade do neurónio com base no seu estado metabólico,
/// implementando dinâmicas homeostáticas através da gestão de energia.

#[derive(Debug, Clone)]
pub struct Glia {
    /// Energia atual do neurónio
    pub energy: f64,

    /// Prioridade do neurónio (para uso futuro em mecanismos de atenção)
    pub priority: f64,

    /// Nível de alerta (para uso futuro em estados globais da rede)
    pub alert_level: f64,

    // Constantes metabólicas
    pub max_energy: f64,
    pub energy_cost_fire: f64,
    pub energy_cost_maintenance: f64,
    pub energy_recovery_rate: f64,
}

impl Glia {
    /// Cria uma nova instância de Glia com parâmetros padrão
    pub fn new() -> Self {
        Self {
            energy: 100.0,
            priority: 1.0,
            alert_level: 0.0,
            max_energy: 100.0,
            energy_cost_fire: 10.0, // MANTIDO: Custo realista (caro)
            energy_cost_maintenance: 0.1,
            // AJUSTE HEMODINÂMICO: Aumentar "aporte de glicose" (de 2.0 para 10.0)
            // Isso permite recuperar 1 disparo (custo 10) em 1 passo de descanso.
            // Configuração ótima encontrada via busca metabólica.
            energy_recovery_rate: 10.0,
        }
    }

    /// Cria uma Glia com parâmetros personalizados
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
        }
    }

    /// Modula o potencial integrado baseado na energia disponível e priority
    ///
    /// Fórmula v2: potencial_modulado = potencial_integrado * energy_factor * priority
    ///
    /// - energy_factor: [0.0, 1.0] - Reduz potencial quando energia está baixa
    /// - priority: [1.0, 3.0] - Aumenta sensibilidade para inputs novos/importantes
    ///
    /// # Argumentos
    /// * `integrated_potential` - Potencial calculado pelo Dendritoma
    ///
    /// # Retorna
    /// Potencial modulado pronto para decisão de disparo
    pub fn modulate(&self, integrated_potential: f64) -> f64 {
        let energy_factor = (self.energy / self.max_energy).max(0.0);

        // Priority modula a sensibilidade do neurónio
        // Priority > 1.0 aumenta o potencial (neurónio mais reativo)
        // Priority = 1.0 não altera (comportamento padrão)
        integrated_potential * energy_factor * self.priority
    }

    /// Atualiza o estado metabólico da Glia após um passo de simulação
    ///
    /// - Se o neurónio disparou: consome energia
    /// - Se está em repouso: recupera energia (afetada por alert_level)
    /// - Sempre: aplica custo de manutenção
    ///
    /// O alert_level global aumenta a taxa de recuperação quando a rede
    /// está em estado de alerta, permitindo respostas mais rápidas.
    pub fn update_state(&mut self, did_fire: bool) {
        if did_fire {
            // Consome energia ao disparar
            self.energy -= self.energy_cost_fire;
        } else {
            // Recupera energia em repouso (taxa proporcional ao déficit)
            let base_recovery = self.energy_recovery_rate * (1.0 - self.energy / self.max_energy);

            // Alert_level aumenta a recuperação (1.0 = +100% de recuperação)
            let alert_boost = base_recovery * self.alert_level;

            self.energy += base_recovery + alert_boost;
        }

        // Custo de manutenção constante
        self.energy -= self.energy_cost_maintenance;

        // Garante que a energia permaneça dentro dos limites
        self.energy = self.energy.clamp(0.0, self.max_energy);
    }

    /// Retorna a fração de energia atual (0.0 a 1.0)
    pub fn energy_fraction(&self) -> f64 {
        self.energy / self.max_energy
    }

    /// Aplica custo energético de plasticidade sináptica
    ///
    /// Aprendizado consome energia proporcional à magnitude das mudanças
    /// sinápticas. Isso fecha o ciclo entre energia e aprendizado:
    /// - Baixa energia → menos aprendizado → menos custo → recuperação possível
    /// - Alta energia → mais aprendizado → mais custo → regulação natural
    ///
    /// # Argumentos
    /// * `plasticity_cost` - Magnitude total das mudanças de pesos
    pub fn consume_plasticity_energy(&mut self, plasticity_cost: f64) {
        // Fator de conversão: ajuste fino do custo energético de aprendizado
        // 0.1 = custo relativamente baixo; 1.0 = custo alto
        let cost_factor = 0.1;
        let energy_cost = cost_factor * plasticity_cost;

        self.energy -= energy_cost;
        self.energy = self.energy.clamp(0.0, self.max_energy);
    }

    /// Ajusta parâmetros metabólicos para modo sono
    ///
    /// Durante o sono, a rede opera em modo de baixo consumo energético
    /// e alta recuperação, facilitando consolidação de memória.
    pub fn enter_sleep_mode(&mut self) {
        self.energy_recovery_rate *= 1.5; // Recuperação 50% mais rápida
        self.energy_cost_fire *= 0.5;     // Spikes de replay custam menos
    }

    /// Restaura parâmetros metabólicos para modo vigília
    pub fn exit_sleep_mode(&mut self) {
        self.energy_recovery_rate /= 1.5; // Volta ao normal
        self.energy_cost_fire /= 0.5;     // Volta ao normal
    }
}

impl Default for Glia {
    fn default() -> Self {
        Self::new()
    }
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

        // Energia deve ter diminuído pelo custo de disparo + manutenção
        assert!(glia.energy < initial_energy);
        assert_relative_eq!(
            glia.energy,
            initial_energy - glia.energy_cost_fire - glia.energy_cost_maintenance,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_energy_recovery_at_rest() {
        let mut glia = Glia::new();
        glia.energy = 50.0;
        let initial_energy = glia.energy;

        glia.update_state(false);

        // Energia deve ter aumentado (recuperação > manutenção neste caso)
        assert!(glia.energy > initial_energy);
    }

    #[test]
    fn test_energy_bounds() {
        let mut glia = Glia::new();

        // Testa limite superior
        glia.energy = glia.max_energy;
        for _ in 0..10 {
            glia.update_state(false);
        }
        assert!(glia.energy <= glia.max_energy);

        // Testa limite inferior
        glia.energy = 0.0;
        glia.update_state(true);
        assert!(glia.energy >= 0.0);
    }

    // === Testes v0.2.0: Alert Level ===

    #[test]
    fn test_alert_level_accelerates_recovery() {
        let mut glia_normal = Glia::new();
        let mut glia_alert = Glia::new();

        // Ambos começam com 50% de energia
        glia_normal.energy = 50.0;
        glia_alert.energy = 50.0;

        // Glia alerta tem alert_level máximo
        glia_alert.alert_level = 1.0;

        // Ambos em repouso por 1 passo
        glia_normal.update_state(false);
        glia_alert.update_state(false);

        // Glia com alerta deve ter recuperado mais energia
        assert!(glia_alert.energy > glia_normal.energy);

        // base_recovery = 2.0 * (1.0 - 0.5) = 1.0
        // alert_boost = 1.0 * 1.0 = 1.0
        // diferença = 1.0
        let expected_diff = 1.0; // alert_boost
        assert_relative_eq!(
            glia_alert.energy - glia_normal.energy,
            expected_diff,
            epsilon = 0.05
        );
    }

    #[test]
    fn test_alert_level_zero_no_effect() {
        let mut glia_zero = Glia::new();
        let mut glia_normal = Glia::new();

        glia_zero.energy = 50.0;
        glia_normal.energy = 50.0;

        glia_zero.alert_level = 0.0;
        glia_normal.alert_level = 0.0;

        glia_zero.update_state(false);
        glia_normal.update_state(false);

        // Devem recuperar exatamente a mesma energia
        assert_relative_eq!(glia_zero.energy, glia_normal.energy, epsilon = 1e-10);
    }

    #[test]
    fn test_priority_modulation() {
        let mut glia = Glia::new();
        glia.priority = 2.0;
        glia.energy = 100.0; // Energia máxima

        let potential = 10.0;
        let modulated = glia.modulate(potential);

        // Com priority=2.0, potencial deve dobrar
        assert_relative_eq!(modulated, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_priority_and_energy_interaction() {
        let mut glia = Glia::new();
        glia.priority = 2.0;
        glia.energy = 50.0; // 50% de energia

        let potential = 10.0;
        let modulated = glia.modulate(potential);

        // energy_factor = 0.5, priority = 2.0
        // modulated = 10.0 * 0.5 * 2.0 = 10.0
        assert_relative_eq!(modulated, 10.0, epsilon = 1e-10);
    }
}
