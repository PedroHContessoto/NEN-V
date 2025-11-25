//! Validação e relatórios da configuração

use super::*;

impl AutoConfig {
    /// Valida configuração antes de criar a rede
    ///
    /// Detecta configurações absurdas que causariam falhas
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check 1: Balanço Energético
        self.validate_energy_balance(&mut errors);

        // Check 2: iSTDP Alinhamento
        self.validate_istdp_alignment(&mut errors);

        // Check 3: STDP Ratio
        self.validate_stdp_ratio(&mut errors);

        // Check 4: Valores Físicos
        self.validate_physical_values(&mut errors);

        // Check 5: Arquitetura
        self.validate_architecture(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_energy_balance(&self, errors: &mut Vec<String>) {
        let energy = &self.params.energy;
        let fr = self.params.target_firing_rate;

        let avg_cost = energy.energy_cost_fire * fr;
        let avg_gain = energy.energy_recovery_rate * (1.0 - fr);
        let balance = avg_gain - avg_cost;

        if balance <= 0.0 {
            errors.push(format!(
                "ERRO CRÍTICO: Metabolismo insustentável\n\
                 • Gasto médio: {:.3}\n\
                 • Ganho médio: {:.3}\n\
                 • Saldo: {:.3} (NEGATIVO!)",
                avg_cost, avg_gain, balance
            ));
        } else if balance < avg_cost * 0.1 {
            errors.push(format!(
                "AVISO: Margem energética baixa (<10%)\n\
                 • Saldo: {:.3}\n\
                 • Recomendado: >{:.3}",
                balance, avg_cost * 0.1
            ));
        }
    }

    fn validate_istdp_alignment(&self, errors: &mut Vec<String>) {
        let target_fr = self.params.target_firing_rate;
        let istdp_target = self.params.istdp.target_rate;

        let error = (istdp_target - target_fr).abs();
        if error > 0.01 {
            errors.push(format!(
                "ERRO: iSTDP desalinhado com Target FR\n\
                 • Target FR: {:.3}\n\
                 • iSTDP Target: {:.3}\n\
                 • Diferença: {:.3} (>{:.3})",
                target_fr, istdp_target, error, 0.01
            ));
        }
    }

    fn validate_stdp_ratio(&self, errors: &mut Vec<String>) {
        let stdp = &self.params.stdp;
        let ratio = stdp.a_plus / stdp.a_minus;

        if ratio < 1.0 {
            errors.push(format!(
                "ERRO: LTP menor que LTD (ratio={:.2})\n\
                 • LTP (A+): {:.4}\n\
                 • LTD (A-): {:.4}\n\
                 • Esperado: ratio > 1.0",
                ratio, stdp.a_plus, stdp.a_minus
            ));
        } else if ratio > 5.0 {
            errors.push(format!(
                "AVISO: LTP/LTD ratio muito alto ({:.2})\n\
                 • Risco de runaway potentiation\n\
                 • Recomendado: 1.5-3.0",
                ratio
            ));
        }
    }

    fn validate_physical_values(&self, errors: &mut Vec<String>) {
        let fr = self.params.target_firing_rate;

        if fr < 0.0 || fr > 1.0 {
            errors.push(format!(
                "ERRO: Target FR impossível ({:.3})\n\
                 • Deve estar em [0.0, 1.0]",
                fr
            ));
        }

        if self.params.energy.energy_recovery_rate <= 0.0 {
            errors.push(format!(
                "ERRO: Recovery rate não-positivo ({:.3})",
                self.params.energy.energy_recovery_rate
            ));
        }
    }

    fn validate_architecture(&self, errors: &mut Vec<String>) {
        let arch = &self.architecture;

        if arch.total_neurons < 3 {
            errors.push(format!(
                "ERRO: Rede muito pequena ({} neurônios)\n\
                 • Mínimo: 3 neurônios",
                arch.total_neurons
            ));
        }

        if arch.inhibitory_ratio < 0.0 || arch.inhibitory_ratio > 1.0 {
            errors.push(format!(
                "ERRO: Razão inibitória inválida ({:.2})",
                arch.inhibitory_ratio
            ));
        }
    }

    // ========================================================================
    // RELATÓRIOS
    // ========================================================================

    pub(crate) fn print_task_spec(&self) {
        println!("📥 ESPECIFICAÇÃO DA TAREFA:");
        println!("  • Sensores: {}", self.task_spec.num_sensors);
        println!("  • Atuadores: {}", self.task_spec.num_actuators);

        match &self.task_spec.task_type {
            TaskType::ReinforcementLearning { reward_density, .. } => {
                println!("  • Tipo: Reinforcement Learning");
                println!("  • Reward Density: {:?}", reward_density);
            }
            TaskType::SupervisedClassification { num_classes } => {
                println!("  • Tipo: Classificação Supervisionada");
                println!("  • Classes: {}", num_classes);
            }
            TaskType::AssociativeMemory { pattern_capacity } => {
                println!("  • Tipo: Memória Associativa");
                println!("  • Capacidade: {} padrões", pattern_capacity);
            }
        }
        println!();
    }

    pub(crate) fn print_architecture(&self) {
        let arch = &self.architecture;

        println!("🧮 ARQUITETURA (Derivada Automaticamente):");
        println!("  • Total de Neurônios: {}", arch.total_neurons);
        println!("    - Sensores: {} (índices {:?})",
                 arch.sensor_indices.len(), arch.sensor_indices);
        println!("    - Hidden: {} (índices {:?})",
                 arch.hidden_indices.len(), arch.hidden_indices);
        println!("    - Atuadores: {} (índices {:?})",
                 arch.actuator_indices.len(), arch.actuator_indices);
        println!("  • Topologia: {:?}", arch.connectivity);
        println!("  • Razão E/I: {:.1}% inibitórios", arch.inhibitory_ratio * 100.0);
        println!("  • Threshold Inicial: {:.3}", arch.initial_threshold);
        println!();
    }

    pub(crate) fn print_parameters(&self) {
        let p = &self.params;

        println!("📊 PARÂMETROS (Calculados - 60+ valores):");

        // Estruturais
        println!("  Estruturais:");
        println!("    • Target FR: {:.3} ({:.1}%)",
                 p.target_firing_rate, p.target_firing_rate * 100.0);
        println!("    • Learning Rate: {:.4}", p.learning_rate);
        println!("    • Avg Connections: {}", p.avg_connections);
        println!("    • Peso Excitatory: {:.3}", p.initial_excitatory_weight);
        println!("    • Peso Inhibitory: {:.3}", p.initial_inhibitory_weight);

        // Metabólicos
        println!("  Metabólicos:");
        println!("    • Max Energy: {:.1}", p.energy.max_energy);
        println!("    • Energy Cost (fire): {:.2}", p.energy.energy_cost_fire);
        println!("    • Energy Recovery: {:.2}/step", p.energy.energy_recovery_rate);

        // Plasticidade
        println!("  Plasticidade:");
        println!("    • STDP A+: {:.4}", p.stdp.a_plus);
        println!("    • STDP A-: {:.4}", p.stdp.a_minus);
        println!("    • Ratio LTP/LTD: {:.2}", p.stdp.a_plus / p.stdp.a_minus);
        println!("    • STDP Window: {} steps", p.stdp.window);
        println!("    • iSTDP LR: {:.4}", p.istdp.learning_rate);
        println!("    • iSTDP Target: {:.3}", p.istdp.target_rate);

        // Homeostase
        println!("  Homeostase:");
        println!("    • Refractory Period: {} ms", p.homeostatic.refractory_period);
        println!("    • Homeo Interval: {} steps", p.homeostatic.homeo_interval);
        println!("    • Homeo Eta: {:.3}", p.homeostatic.homeo_eta);

        // Memória
        println!("  Memória:");
        println!("    • Weight Decay: {:.5}", p.memory.weight_decay);
        println!("    • Tag Decay: {:.3}", p.memory.tag_decay_rate);
        println!("    • Capture Threshold: {:.3}", p.memory.capture_threshold);
        println!("    • Consolidation Rate: {:.4}", p.memory.consolidation_base_rate);

        // Sono
        println!("  Sono:");
        println!("    • Sleep Interval: {} steps", p.sleep.sleep_interval);
        println!("    • Sleep Duration: {} steps", p.sleep.sleep_duration);
        println!("    • Replay Noise: {:.2}", p.sleep.sleep_replay_noise);

        // RL (se aplicável)
        if let Some(rl) = &p.rl {
            println!("  Reinforcement Learning:");
            println!("    • Exploration Inicial: {:.2}", rl.initial_exploration_rate);
            println!("    • Exploration Decay: {:.4}", rl.exploration_decay_rate);
            println!("    • Eligibility Window: {} steps", rl.eligibility_trace_window);
        }

        println!();
    }

    pub(crate) fn print_verification(&self) {
        println!("✅ VERIFICAÇÕES:");

        // Balanço energético
        let energy = &self.params.energy;
        let fr = self.params.target_firing_rate;

        let avg_cost = energy.energy_cost_fire * fr;
        let avg_gain = energy.energy_recovery_rate * (1.0 - fr);
        let balance = avg_gain - avg_cost;
        let margin_pct = (balance / avg_cost) * 100.0;

        println!("  Balanço Energético:");
        println!("    • Gasto médio: {:.3}/step", avg_cost);
        println!("    • Ganho médio: {:.3}/step", avg_gain);
        println!("    • Saldo: {:.3}/step ({:+.1}% margem) {}",
                 balance,
                 margin_pct,
                 if balance > 0.0 { "✅ SUSTENTÁVEL" } else { "❌ INSUSTENTÁVEL" });

        // iSTDP
        let istdp_aligned = (self.params.istdp.target_rate - fr).abs() < 1e-6;
        println!("  iSTDP:");
        println!("    • Target: {:.3} {}",
                 self.params.istdp.target_rate,
                 if istdp_aligned { "✅ ALINHADO" } else { "❌ DESALINHADO" });

        // STDP Ratio
        let stdp_ratio = self.params.stdp.a_plus / self.params.stdp.a_minus;
        let ratio_ok = stdp_ratio >= 1.5 && stdp_ratio <= 3.0;
        println!("  STDP:");
        println!("    • LTP/LTD Ratio: {:.2} {}",
                 stdp_ratio,
                 if ratio_ok { "✅" } else { "⚠️  (esperado: 1.5-3.0)" });

        println!("\n════════════════════════════════════════\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_passes_for_valid_config() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let config = AutoConfig::from_task(task);
        let result = config.validate();

        assert!(result.is_ok(), "Validação falhou: {:?}", result.err());
    }
}
