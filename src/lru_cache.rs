//! # LRU Cache (Least Recently Used)
//!
//! Implementação simples de cache LRU para limitar uso de memória
//! em estruturas de dados que crescem indefinidamente.
//!
//! ## Uso Principal
//!
//! - Habituation map em CuriosityModule
//! - Caching de estados visitados
//! - Históricos com tamanho limitado

use std::collections::HashMap;
use std::hash::Hash;

/// Cache LRU com capacidade fixa
///
/// Quando a capacidade é excedida, o item menos recentemente usado é removido.
#[derive(Debug)]
pub struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Capacidade máxima
    capacity: usize,

    /// Mapa de valores
    map: HashMap<K, (V, u64)>,

    /// Contador de acesso (para determinar LRU)
    access_counter: u64,

    /// Estatísticas
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Cria novo cache LRU
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::with_capacity(capacity),
            access_counter: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Retorna capacidade máxima
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Retorna número atual de elementos
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Verifica se está vazio
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Verifica se está cheio
    pub fn is_full(&self) -> bool {
        self.map.len() >= self.capacity
    }

    /// Obtém valor (atualiza contador de acesso)
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.access_counter += 1;

        if let Some((value, access_time)) = self.map.get_mut(key) {
            *access_time = self.access_counter;
            self.hits += 1;
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Obtém valor mutável (atualiza contador de acesso)
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.access_counter += 1;

        if let Some((value, access_time)) = self.map.get_mut(key) {
            *access_time = self.access_counter;
            self.hits += 1;
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Verifica se contém chave (sem atualizar contador)
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Insere valor (pode causar evicção)
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.access_counter += 1;

        // Se já existe, apenas atualiza
        if let Some((old_value, access_time)) = self.map.get_mut(&key) {
            let old = std::mem::replace(old_value, value);
            *access_time = self.access_counter;
            return Some(old);
        }

        // Se cheio, remove o menos recentemente usado
        if self.is_full() {
            self.evict_lru();
        }

        self.map.insert(key, (value, self.access_counter));
        None
    }

    /// Remove e retorna valor
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|(value, _)| value)
    }

    /// Remove o item menos recentemente usado
    fn evict_lru(&mut self) {
        if self.map.is_empty() {
            return;
        }

        // Encontra a chave com menor tempo de acesso
        let lru_key = self.map
            .iter()
            .min_by_key(|(_, (_, access_time))| *access_time)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            self.map.remove(&key);
            self.evictions += 1;
        }
    }

    /// Limpa cache
    pub fn clear(&mut self) {
        self.map.clear();
        self.access_counter = 0;
    }

    /// Retorna estatísticas
    pub fn stats(&self) -> LruStats {
        LruStats {
            capacity: self.capacity,
            current_size: self.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Itera sobre valores
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values().map(|(v, _)| v)
    }

    /// Itera sobre pares (chave, valor)
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter().map(|(k, (v, _))| (k, v))
    }

    /// Remove entradas que satisfazem predicado
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.map.retain(|k, (v, _)| f(k, v));
    }

    /// Obtém ou insere com valor padrão
    pub fn get_or_insert(&mut self, key: K, default: V) -> &mut V
    where
        V: Clone,
    {
        self.access_counter += 1;

        if !self.map.contains_key(&key) {
            if self.is_full() {
                self.evict_lru();
            }
            self.map.insert(key.clone(), (default, self.access_counter));
            self.misses += 1;
        } else {
            self.hits += 1;
        }

        let (value, access_time) = self.map.get_mut(&key).unwrap();
        *access_time = self.access_counter;
        value
    }

    /// Obtém ou insere com função
    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> &mut V
    where
        F: FnOnce() -> V,
    {
        self.access_counter += 1;

        if !self.map.contains_key(&key) {
            if self.is_full() {
                self.evict_lru();
            }
            self.map.insert(key.clone(), (f(), self.access_counter));
            self.misses += 1;
        } else {
            self.hits += 1;
        }

        let (value, access_time) = self.map.get_mut(&key).unwrap();
        *access_time = self.access_counter;
        value
    }
}

/// Estatísticas do cache
#[derive(Debug, Clone)]
pub struct LruStats {
    pub capacity: usize,
    pub current_size: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

impl LruStats {
    /// Imprime relatório formatado
    pub fn print_report(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│          LRU CACHE STATS                │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Capacity:    {:>26} │", self.capacity);
        println!("│ Current:     {:>26} │", self.current_size);
        println!("│ Hits:        {:>26} │", self.hits);
        println!("│ Misses:      {:>26} │", self.misses);
        println!("│ Evictions:   {:>26} │", self.evictions);
        println!("│ Hit Rate:    {:>25.2}% │", self.hit_rate * 100.0);
        println!("└─────────────────────────────────────────┘");
    }
}

/// Cache LRU especializado para habituação (valores f64)
pub type HabituationCache = LruCache<u64, f64>;

impl HabituationCache {
    /// Obtém fator de habituação, retornando 1.0 se não encontrado
    pub fn get_habituation(&mut self, state_hash: u64) -> f64 {
        self.get(&state_hash).copied().unwrap_or(1.0)
    }

    /// Atualiza habituação com decay
    pub fn update_habituation(&mut self, state_hash: u64, decay_factor: f64) {
        let current = self.get_habituation(state_hash);
        self.insert(state_hash, current * decay_factor);
    }

    /// Remove entradas muito habituadas (threshold baixo)
    pub fn cleanup(&mut self, threshold: f64) {
        self.retain(|_, v| *v > threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut cache: LruCache<i32, String> = LruCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&"one".to_string()));
        assert_eq!(cache.get(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_eviction() {
        let mut cache: LruCache<i32, String> = LruCache::new(2);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        // Acessa 1 para torná-lo mais recente
        cache.get(&1);

        // Insere 3, deve evictar 2 (menos recente)
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_update() {
        let mut cache: LruCache<i32, String> = LruCache::new(2);

        cache.insert(1, "one".to_string());
        let old = cache.insert(1, "ONE".to_string());

        assert_eq!(old, Some("one".to_string()));
        assert_eq!(cache.get(&1), Some(&"ONE".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.insert(1, 10);
        cache.get(&1);  // hit
        cache.get(&2);  // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_habituation_cache() {
        let mut cache = HabituationCache::new(100);

        // Primeira vez: habituação = 1.0
        assert!((cache.get_habituation(42) - 1.0).abs() < 0.01);

        // Atualiza com decay
        cache.update_habituation(42, 0.9);
        assert!((cache.get_habituation(42) - 0.9).abs() < 0.01);

        // Decay acumulativo
        cache.update_habituation(42, 0.9);
        assert!((cache.get_habituation(42) - 0.81).abs() < 0.01);
    }

    #[test]
    fn test_retain() {
        let mut cache: LruCache<i32, f64> = LruCache::new(10);

        cache.insert(1, 0.9);
        cache.insert(2, 0.1);
        cache.insert(3, 0.5);

        // Remove valores < 0.3
        cache.retain(|_, v| *v >= 0.3);

        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
    }
}
