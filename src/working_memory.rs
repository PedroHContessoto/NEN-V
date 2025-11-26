//! Módulo de Working Memory (Memória de Trabalho)
//!
//! Implementa memória de trabalho com dinÃ¢mica de atrator, permitindo
//! manutenção ativa de informação sem consolidação permanente.
//!
//! ## Características
//!
//! - **Capacidade Limitada**: ~7±2 slots (biologicamente plausível)
//! - **Manutenção Ativa**: Auto-excitação mantém padrões
//! - **Competição**: Inibição lateral limita capacidade
//! - **Decaimento Controlado**: Esquecimento gradual sem uso
//!
//! ## Uso
//!
//! ```rust,ignore
//! let mut wm = WorkingMemoryPool::new(7, 64);
//!
//! // Codifica padrão
//! let slot = wm.encode(pattern, current_time).unwrap();
//!
//! // Mantém a cada timestep
//! wm.sustain();
//!
//! // Recupera
//! let retrieved = wm.retrieve(slot);
//! ```

use std::collections::VecDeque;

/// Slot individual de memória de trabalho
#[derive(Debug, Clone)]
pub struct WMSlot {
    /// Padrão armazenado (vetor de ativações)
    pub pattern: Vec<f64>,

    /// Força de manutenção atual [0.0, 1.0]
    pub strength: f64,

    /// Timestamp de codificação
    pub encoded_at: i64,

    /// RelevÃ¢ncia acumulada (aumenta com uso)
    pub relevance: f64,

    /// Número de acessos
    pub access_count: usize,

    /// Tags associativas (para binding)
    pub tags: Vec<f64>,
}

impl WMSlot {
    /// Cria novo slot com padrão
    pub fn new(pattern: Vec<f64>, timestamp: i64) -> Self {
        let size = pattern.len();
        Self {
            pattern,
            strength: 1.0,
            encoded_at: timestamp,
            relevance: 1.0,
            access_count: 0,
            tags: vec![0.0; size.min(16)], // Tags limitadas
        }
    }

    /// Reforça o slot (quando acessado)
    pub fn reinforce(&mut self, amount: f64) {
        self.strength = (self.strength + amount).min(1.0);
        self.relevance += 0.1;
        self.access_count += 1;
    }

    /// Aplica decaimento
    pub fn decay(&mut self, rate: f64) {
        self.strength *= 1.0 - rate;
    }

    /// Verifica se slot está ativo (acima do threshold)
    pub fn is_active(&self, threshold: f64) -> bool {
        self.strength >= threshold
    }

    /// Similaridade com outro padrão (cosseno)
    pub fn similarity(&self, other: &[f64]) -> f64 {
        if self.pattern.len() != other.len() {
            return 0.0;
        }

        let dot: f64 = self.pattern.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self.pattern.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_other: f64 = other.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_self > 1e-6 && norm_other > 1e-6 {
            dot / (norm_self * norm_other)
        } else {
            0.0
        }
    }
}

/// Pool de Working Memory com dinÃ¢mica de atrator
#[derive(Debug)]
pub struct WorkingMemoryPool {
    /// Slots de memória
    slots: Vec<Option<WMSlot>>,

    /// Tamanho dos padrões
    pattern_size: usize,

    /// Capacidade máxima (número de slots)
    pub max_capacity: usize,

    /// Força da recorrÃªncia (auto-excitação)
    pub recurrent_strength: f64,

    /// Taxa de decaimento por timestep
    pub decay_rate: f64,

    /// Threshold para considerar slot ativo
    pub activation_threshold: f64,

    /// Força da inibição lateral
    pub lateral_inhibition: f64,

    /// Histórico de ativações (para análise)
    activation_history: VecDeque<Vec<f64>>,

    /// Tamanho máximo do histórico
    history_size: usize,

    /// Contador de operações
    pub operation_count: u64,
}

impl WorkingMemoryPool {
    /// Cria nova pool de working memory
    ///
    /// # Argumentos
    /// * `capacity` - Número máximo de slots (recomendado: 5-9)
    /// * `pattern_size` - Dimensão dos padrões a armazenar
    pub fn new(capacity: usize, pattern_size: usize) -> Self {
        Self {
            slots: vec![None; capacity],
            pattern_size,
            max_capacity: capacity,
            recurrent_strength: 0.85,
            decay_rate: 0.02,
            activation_threshold: 0.25,
            lateral_inhibition: 0.08,
            activation_history: VecDeque::with_capacity(100),
            history_size: 100,
            operation_count: 0,
        }
    }

    /// Cria com parÃ¢metros personalizados
    pub fn with_params(
        capacity: usize,
        pattern_size: usize,
        recurrent_strength: f64,
        decay_rate: f64,
        lateral_inhibition: f64,
    ) -> Self {
        let mut pool = Self::new(capacity, pattern_size);
        pool.recurrent_strength = recurrent_strength;
        pool.decay_rate = decay_rate;
        pool.lateral_inhibition = lateral_inhibition;
        pool
    }

    /// Tenta codificar padrão em slot disponível
    ///
    /// Retorna índice do slot usado, ou None se falhar
    pub fn encode(&mut self, pattern: Vec<f64>, current_time: i64) -> Option<usize> {
        if pattern.len() != self.pattern_size {
            return None;
        }

        self.operation_count += 1;

        // Primeiro verifica se padrão similar já existe
        if let Some((idx, similarity)) = self.find_similar(&pattern, 0.9) {
            // Reforça slot existente ao invés de criar novo
            if let Some(slot) = &mut self.slots[idx] {
                slot.reinforce(0.3);
                // Atualiza padrão com média ponderada
                for (i, val) in pattern.iter().enumerate() {
                    slot.pattern[i] = slot.pattern[i] * 0.7 + val * 0.3;
                }
            }
            return Some(idx);
        }

        // Procura slot disponível
        let slot_idx = self.find_available_slot()?;

        self.slots[slot_idx] = Some(WMSlot::new(pattern, current_time));

        Some(slot_idx)
    }

    /// Encontra slot disponível (vazio ou mais fraco)
    fn find_available_slot(&self) -> Option<usize> {
        // Primeiro procura slot vazio
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_none() {
                return Some(i);
            }
        }

        // Se todos ocupados, encontra o mais fraco
        let mut weakest_idx = 0;
        let mut weakest_strength = f64::MAX;

        for (i, slot_opt) in self.slots.iter().enumerate() {
            if let Some(slot) = slot_opt {
                // Considera força E relevÃ¢ncia
                let effective_strength = slot.strength * (1.0 + slot.relevance * 0.1);
                if effective_strength < weakest_strength {
                    weakest_strength = effective_strength;
                    weakest_idx = i;
                }
            }
        }

        Some(weakest_idx)
    }

    /// Encontra slot com padrão similar
    fn find_similar(&self, pattern: &[f64], threshold: f64) -> Option<(usize, f64)> {
        let mut best_idx = None;
        let mut best_sim = threshold;

        for (i, slot_opt) in self.slots.iter().enumerate() {
            if let Some(slot) = slot_opt {
                let sim = slot.similarity(pattern);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }
        }

        best_idx.map(|i| (i, best_sim))
    }

    /// Mantém padrões ativos (chamado a cada timestep)
    ///
    /// Implementa dinÃ¢mica de atrator com:
    /// - Auto-excitação (recorrÃªncia)
    /// - Inibição lateral (competição)
    /// - Decaimento natural
    pub fn sustain(&mut self) {
        // Coleta forças atuais para inibição lateral
        let strengths: Vec<f64> = self.slots.iter()
            .map(|s| s.as_ref().map(|slot| slot.strength).unwrap_or(0.0))
            .collect();

        let total_strength: f64 = strengths.iter().sum();
        let num_active = strengths.iter().filter(|&&s| s > self.activation_threshold).count();

        // Processa cada slot
        for (i, slot_opt) in self.slots.iter_mut().enumerate() {
            if let Some(slot) = slot_opt {
                // 1. Decaimento natural
                slot.decay(self.decay_rate);

                // 2. Auto-excitação (recorrÃªncia) - mantém memória ativa
                let self_excitation = self.recurrent_strength * slot.strength * 0.1;
                slot.strength += self_excitation;

                // 3. Inibição lateral - competição entre slots
                if total_strength > 0.0 && num_active > 1 {
                    let others_strength = total_strength - strengths[i];
                    let inhibition = self.lateral_inhibition * others_strength / (num_active as f64);
                    slot.strength -= inhibition;
                }

                // 4. Clamp
                slot.strength = slot.strength.clamp(0.0, 1.0);

                // 5. Remove se muito fraco
                if slot.strength < self.activation_threshold * 0.5 {
                    *slot_opt = None;
                }
            }
        }

        // Registra histórico
        let current_activations: Vec<f64> = self.slots.iter()
            .map(|s| s.as_ref().map(|slot| slot.strength).unwrap_or(0.0))
            .collect();

        self.activation_history.push_back(current_activations);
        if self.activation_history.len() > self.history_size {
            self.activation_history.pop_front();
        }
    }

    /// Recupera padrão de slot específico
    ///
    /// Reforça o slot acessado (uso fortalece memória)
    pub fn retrieve(&mut self, slot_idx: usize) -> Option<Vec<f64>> {
        self.operation_count += 1;

        if let Some(slot) = self.slots.get_mut(slot_idx)?.as_mut() {
            slot.reinforce(0.2);
            Some(slot.pattern.clone())
        } else {
            None
        }
    }

    /// Recupera padrão sem modificar estado (read-only)
    pub fn peek(&self, slot_idx: usize) -> Option<&Vec<f64>> {
        self.slots.get(slot_idx)?
            .as_ref()
            .map(|slot| &slot.pattern)
    }

    /// Busca slot mais similar ao query
    ///
    /// Retorna (índice, similaridade) se encontrar acima do threshold
    pub fn query(&mut self, query_pattern: &[f64], threshold: f64) -> Option<(usize, f64)> {
        self.operation_count += 1;

        let result = self.find_similar(query_pattern, threshold);

        // Reforça slot encontrado
        if let Some((idx, _)) = result {
            if let Some(slot) = &mut self.slots[idx] {
                slot.reinforce(0.1);
            }
        }

        result
    }

    /// Libera slot explicitamente
    pub fn release(&mut self, slot_idx: usize) {
        if slot_idx < self.slots.len() {
            self.slots[slot_idx] = None;
        }
    }

    /// Libera todos os slots
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }

    /// Retorna número de slots ativos
    pub fn active_count(&self) -> usize {
        self.slots.iter()
            .filter(|s| s.as_ref().map(|slot| slot.is_active(self.activation_threshold)).unwrap_or(false))
            .count()
    }

    /// Retorna capacidade restante
    pub fn available_capacity(&self) -> usize {
        self.max_capacity - self.active_count()
    }

    /// Verifica se está cheio
    pub fn is_full(&self) -> bool {
        self.active_count() >= self.max_capacity
    }

    /// Retorna vetor de forças de todos os slots
    pub fn get_strengths(&self) -> Vec<f64> {
        self.slots.iter()
            .map(|s| s.as_ref().map(|slot| slot.strength).unwrap_or(0.0))
            .collect()
    }

    /// Retorna estatísticas da working memory
    pub fn get_stats(&self) -> WMStats {
        let strengths = self.get_strengths();
        let active = self.active_count();

        let avg_strength = if active > 0 {
            strengths.iter().filter(|&&s| s > 0.0).sum::<f64>() / active as f64
        } else {
            0.0
        };

        let total_relevance: f64 = self.slots.iter()
            .filter_map(|s| s.as_ref())
            .map(|slot| slot.relevance)
            .sum();

        WMStats {
            active_slots: active,
            max_capacity: self.max_capacity,
            utilization: active as f64 / self.max_capacity as f64,
            avg_strength,
            total_relevance,
            operation_count: self.operation_count,
        }
    }

    /// Gera representação combinada de todos os padrões ativos
    ///
    /// Ãštil para criar contexto integrado
    pub fn get_combined_representation(&self) -> Vec<f64> {
        let mut combined = vec![0.0; self.pattern_size];
        let mut total_weight = 0.0;

        for slot_opt in &self.slots {
            if let Some(slot) = slot_opt {
                if slot.is_active(self.activation_threshold) {
                    let weight = slot.strength * slot.relevance;
                    for (i, &val) in slot.pattern.iter().enumerate() {
                        combined[i] += val * weight;
                    }
                    total_weight += weight;
                }
            }
        }

        if total_weight > 0.0 {
            for val in &mut combined {
                *val /= total_weight;
            }
        }

        combined
    }

    /// Aplica modulação de atenção aos slots
    ///
    /// Slots com padrões mais similares ao foco recebem boost
    pub fn apply_attention(&mut self, attention_focus: &[f64], boost: f64) {
        for slot_opt in &mut self.slots {
            if let Some(slot) = slot_opt {
                let similarity = slot.similarity(attention_focus);
                if similarity > 0.5 {
                    slot.strength = (slot.strength + boost * similarity).min(1.0);
                }
            }
        }
    }
}

/// Estatísticas da Working Memory
#[derive(Debug, Clone)]
pub struct WMStats {
    pub active_slots: usize,
    pub max_capacity: usize,
    pub utilization: f64,
    pub avg_strength: f64,
    pub total_relevance: f64,
    pub operation_count: u64,
}

impl WMStats {
    /// Imprime relatório formatado
    pub fn print_report(&self) {
        println!("â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”");
        println!("â”‚       WORKING MEMORY STATUS         â”‚");
        println!("â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤");
        println!("â”‚ Slots Ativos: {}/{:<20}â”‚", self.active_slots, self.max_capacity);
        println!("â”‚ Utilização:   {:.1}%{:<19}â”‚", self.utilization * 100.0, "");
        println!("â”‚ Força Média:  {:.3}{:<19}â”‚", self.avg_strength, "");
        println!("â”‚ RelevÃ¢ncia:   {:.2}{:<19}â”‚", self.total_relevance, "");
        println!("â”‚ Operações:    {:<21}â”‚", self.operation_count);
        println!("â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜");
    }
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wm_initialization() {
        let wm = WorkingMemoryPool::new(7, 10);

        assert_eq!(wm.max_capacity, 7);
        assert_eq!(wm.pattern_size, 10);
        assert_eq!(wm.active_count(), 0);
    }

    #[test]
    fn test_encode_and_retrieve() {
        let mut wm = WorkingMemoryPool::new(5, 4);

        let pattern = vec![1.0, 0.0, 1.0, 0.0];
        let slot = wm.encode(pattern.clone(), 0);

        assert!(slot.is_some());
        assert_eq!(wm.active_count(), 1);

        let retrieved = wm.retrieve(slot.unwrap());
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), pattern);
    }

    #[test]
    fn test_capacity_limit() {
        let mut wm = WorkingMemoryPool::new(3, 4);

        // Usa padrões bem distintos para evitar fusão por similaridade
        let patterns = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // Padrão 1
            vec![0.0, 1.0, 0.0, 0.0],  // Padrão 2
            vec![0.0, 0.0, 1.0, 0.0],  // Padrão 3
        ];

        // Preenche todos os slots
        for (i, pattern) in patterns.iter().enumerate() {
            let slot = wm.encode(pattern.clone(), i as i64);
            assert!(slot.is_some());
        }

        assert_eq!(wm.active_count(), 3);
        assert!(wm.is_full());

        // Ainda consegue codificar (substitui o mais fraco)
        let new_pattern = vec![0.0, 0.0, 0.0, 1.0];
        let slot = wm.encode(new_pattern, 10);
        assert!(slot.is_some());
    }

    #[test]
    fn test_sustain_decay() {
        let mut wm = WorkingMemoryPool::new(3, 4);
        wm.decay_rate = 0.1; // Decaimento rápido para teste

        let pattern = vec![1.0; 4];
        wm.encode(pattern, 0);

        let initial_strength = wm.get_strengths()[0];

        // Vários ciclos de sustain
        for _ in 0..50 {
            wm.sustain();
        }

        // Com recorrÃªncia, deve manter alguma atividade
        let final_strength = wm.get_strengths()[0];
        assert!(final_strength < initial_strength); // Decaiu
        assert!(final_strength > 0.0); // Mas não zerou (recorrÃªncia)
    }

    #[test]
    fn test_query_similarity() {
        let mut wm = WorkingMemoryPool::new(5, 4);

        let pattern1 = vec![1.0, 0.0, 0.0, 0.0];
        let pattern2 = vec![0.0, 1.0, 0.0, 0.0];

        wm.encode(pattern1.clone(), 0);
        wm.encode(pattern2.clone(), 1);

        // Query similar ao pattern1
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let result = wm.query(&query, 0.8);

        assert!(result.is_some());
        let (idx, sim) = result.unwrap();
        assert_eq!(idx, 0); // Deve encontrar pattern1
        assert!(sim > 0.9);
    }

    #[test]
    fn test_combined_representation() {
        let mut wm = WorkingMemoryPool::new(3, 4);

        wm.encode(vec![1.0, 0.0, 0.0, 0.0], 0);
        wm.encode(vec![0.0, 1.0, 0.0, 0.0], 1);

        let combined = wm.get_combined_representation();

        // Combinação deve ter elementos de ambos
        assert!(combined[0] > 0.0);
        assert!(combined[1] > 0.0);
    }

    #[test]
    fn test_lateral_inhibition() {
        let mut wm = WorkingMemoryPool::new(3, 4);
        wm.lateral_inhibition = 0.2; // Inibição forte

        // Codifica múltiplos padrões
        for i in 0..3 {
            wm.encode(vec![i as f64; 4], i as i64);
        }

        // Reforça um deles
        if let Some(slot) = &mut wm.slots[0] {
            slot.strength = 1.0;
        }

        // Sustain com inibição
        for _ in 0..20 {
            wm.sustain();
        }

        // O mais forte deve ter suprimido os outros parcialmente
        let strengths = wm.get_strengths();
        assert!(strengths[0] > strengths[1] || strengths[0] > strengths[2]);
    }
}
