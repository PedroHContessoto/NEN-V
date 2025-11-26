//! # Sparse Matrix para Conectividade de Rede Neural
//!
//! Implementação eficiente de matriz esparsa para representar
//! conectividade em redes neurais.
//!
//! ## Motivação
//!
//! Para redes com N neurônios:
//! - Matriz densa: O(N²) memória
//! - Matriz esparsa: O(E) memória, onde E é número de conexões
//!
//! Para Grid2D com 8 vizinhos: E = 8N << N²
//!
//! ## Formato CSR (Compressed Sparse Row)
//!
//! - row_ptr[i]: índice em col_indices onde começam conexões do neurônio i
//! - col_indices: índices dos neurônios conectados
//! - values: pesos/valores das conexões (opcional para binário)

use std::collections::HashMap;

/// Matriz esparsa em formato CSR (Compressed Sparse Row)
#[derive(Debug, Clone)]
pub struct SparseConnectivity {
    /// Número de neurônios (linhas/colunas)
    size: usize,

    /// Ponteiros para início de cada linha
    /// row_ptr[i] é o índice em col_indices onde começam as conexões do neurônio i
    /// row_ptr[size] = número total de conexões
    row_ptr: Vec<usize>,

    /// Índices das colunas (neurônios conectados)
    col_indices: Vec<usize>,

    /// Número total de conexões
    num_connections: usize,
}

impl SparseConnectivity {
    /// Cria matriz esparsa vazia
    pub fn new(size: usize) -> Self {
        Self {
            size,
            row_ptr: vec![0; size + 1],
            col_indices: Vec::new(),
            num_connections: 0,
        }
    }

    /// Cria a partir de lista de adjacências
    pub fn from_adjacency_list(size: usize, adjacencies: &[Vec<usize>]) -> Self {
        assert_eq!(adjacencies.len(), size);

        let mut row_ptr = vec![0; size + 1];
        let mut col_indices = Vec::new();

        for (i, neighbors) in adjacencies.iter().enumerate() {
            row_ptr[i + 1] = row_ptr[i] + neighbors.len();
            col_indices.extend(neighbors.iter().copied());
        }

        let num_connections = col_indices.len();

        Self {
            size,
            row_ptr,
            col_indices,
            num_connections,
        }
    }

    /// Cria a partir de matriz densa
    pub fn from_dense(dense: &[Vec<u8>]) -> Self {
        let size = dense.len();
        let mut adjacencies: Vec<Vec<usize>> = vec![Vec::new(); size];

        for (i, row) in dense.iter().enumerate() {
            for (j, &connected) in row.iter().enumerate() {
                if connected == 1 {
                    adjacencies[i].push(j);
                }
            }
        }

        Self::from_adjacency_list(size, &adjacencies)
    }

    /// Cria topologia FullyConnected
    pub fn fully_connected(size: usize) -> Self {
        let mut adjacencies: Vec<Vec<usize>> = vec![Vec::new(); size];

        for i in 0..size {
            for j in 0..size {
                adjacencies[i].push(j);
            }
        }

        Self::from_adjacency_list(size, &adjacencies)
    }

    /// Cria topologia Grid2D (vizinhança de Moore - 8 vizinhos)
    pub fn grid_2d(width: usize, height: usize) -> Self {
        let size = width * height;
        let mut adjacencies: Vec<Vec<usize>> = vec![Vec::new(); size];

        for i in 0..size {
            let row = i / width;
            let col = i % width;

            // 8 vizinhos (Moore neighborhood)
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }

                    let new_row = row as i32 + dr;
                    let new_col = col as i32 + dc;

                    if new_row >= 0
                        && new_row < height as i32
                        && new_col >= 0
                        && new_col < width as i32
                    {
                        let j = (new_row as usize) * width + (new_col as usize);
                        if j < size {
                            adjacencies[i].push(j);
                        }
                    }
                }
            }
        }

        Self::from_adjacency_list(size, &adjacencies)
    }

    /// Cria topologia isolada (sem conexões)
    pub fn isolated(size: usize) -> Self {
        Self::from_adjacency_list(size, &vec![Vec::new(); size])
    }

    /// Retorna tamanho da matriz
    pub fn size(&self) -> usize {
        self.size
    }

    /// Retorna número total de conexões
    pub fn num_connections(&self) -> usize {
        self.num_connections
    }

    /// Verifica se neurônio i está conectado a neurônio j
    pub fn is_connected(&self, from: usize, to: usize) -> bool {
        if from >= self.size {
            return false;
        }

        let start = self.row_ptr[from];
        let end = self.row_ptr[from + 1];

        // Busca binária se lista ordenada, linear caso contrário
        self.col_indices[start..end].contains(&to)
    }

    /// Retorna vizinhos de um neurônio (neurônios que recebem input dele)
    pub fn neighbors(&self, neuron: usize) -> &[usize] {
        if neuron >= self.size {
            return &[];
        }

        let start = self.row_ptr[neuron];
        let end = self.row_ptr[neuron + 1];

        &self.col_indices[start..end]
    }

    /// Retorna número de conexões de saída de um neurônio
    pub fn out_degree(&self, neuron: usize) -> usize {
        if neuron >= self.size {
            return 0;
        }
        self.row_ptr[neuron + 1] - self.row_ptr[neuron]
    }

    /// Retorna grau médio de conexões
    pub fn average_degree(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        self.num_connections as f64 / self.size as f64
    }

    /// Converte para matriz densa (para compatibilidade)
    pub fn to_dense(&self) -> Vec<Vec<u8>> {
        let mut dense = vec![vec![0u8; self.size]; self.size];

        for i in 0..self.size {
            for &j in self.neighbors(i) {
                dense[i][j] = 1;
            }
        }

        dense
    }

    /// Retorna memória usada em bytes (aproximado)
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
    }

    /// Compara uso de memória com matriz densa
    pub fn memory_savings(&self) -> f64 {
        let dense_size = self.size * self.size;
        let sparse_size = self.row_ptr.len() + self.col_indices.len();

        if dense_size == 0 {
            return 0.0;
        }

        1.0 - (sparse_size as f64 / dense_size as f64)
    }

    /// Itera sobre todas as conexões (from, to)
    pub fn iter_connections(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.size).flat_map(move |from| {
            self.neighbors(from).iter().map(move |&to| (from, to))
        })
    }

    /// Aplica função a todos os vizinhos de um neurônio
    pub fn for_each_neighbor<F>(&self, neuron: usize, mut f: F)
    where
        F: FnMut(usize),
    {
        for &neighbor in self.neighbors(neuron) {
            f(neighbor);
        }
    }

    /// Coleta entradas para um neurônio destino
    ///
    /// Retorna vetor com valores de `outputs` apenas para neurônios conectados
    pub fn gather_inputs(&self, target: usize, outputs: &[f64]) -> Vec<(usize, f64)> {
        self.neighbors(target)
            .iter()
            .filter_map(|&source| {
                outputs.get(source).map(|&val| (source, val))
            })
            .collect()
    }
}

/// Matriz de conectividade com pesos
#[derive(Debug, Clone)]
pub struct WeightedSparseConnectivity {
    /// Estrutura de conectividade
    connectivity: SparseConnectivity,

    /// Pesos das conexões (mesmo tamanho que col_indices)
    weights: Vec<f64>,
}

impl WeightedSparseConnectivity {
    /// Cria a partir de conectividade não-ponderada
    pub fn from_connectivity(connectivity: SparseConnectivity, default_weight: f64) -> Self {
        let weights = vec![default_weight; connectivity.num_connections];
        Self { connectivity, weights }
    }

    /// Cria a partir de matriz densa de pesos
    pub fn from_dense_weights(dense: &[Vec<f64>]) -> Self {
        let size = dense.len();
        let mut adjacencies: Vec<Vec<usize>> = vec![Vec::new(); size];
        let mut weights = Vec::new();

        for (i, row) in dense.iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                if weight.abs() > 1e-10 {
                    adjacencies[i].push(j);
                    weights.push(weight);
                }
            }
        }

        let connectivity = SparseConnectivity::from_adjacency_list(size, &adjacencies);
        Self { connectivity, weights }
    }

    /// Retorna peso de uma conexão
    pub fn get_weight(&self, from: usize, to: usize) -> Option<f64> {
        if from >= self.connectivity.size {
            return None;
        }

        let start = self.connectivity.row_ptr[from];
        let end = self.connectivity.row_ptr[from + 1];

        for (idx, &neighbor) in self.connectivity.col_indices[start..end].iter().enumerate() {
            if neighbor == to {
                return Some(self.weights[start + idx]);
            }
        }

        None
    }

    /// Define peso de uma conexão existente
    pub fn set_weight(&mut self, from: usize, to: usize, weight: f64) -> bool {
        if from >= self.connectivity.size {
            return false;
        }

        let start = self.connectivity.row_ptr[from];
        let end = self.connectivity.row_ptr[from + 1];

        for (idx, &neighbor) in self.connectivity.col_indices[start..end].iter().enumerate() {
            if neighbor == to {
                self.weights[start + idx] = weight;
                return true;
            }
        }

        false
    }

    /// Retorna referência à conectividade
    pub fn connectivity(&self) -> &SparseConnectivity {
        &self.connectivity
    }

    /// Retorna referência aos pesos
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Retorna referência mutável aos pesos
    pub fn weights_mut(&mut self) -> &mut [f64] {
        &mut self.weights
    }

    /// Integra inputs ponderados para um neurônio
    pub fn integrate(&self, target: usize, outputs: &[f64]) -> f64 {
        let start = self.connectivity.row_ptr[target];
        let end = self.connectivity.row_ptr[target + 1];

        let mut sum = 0.0;
        for (idx, &source) in self.connectivity.col_indices[start..end].iter().enumerate() {
            if let Some(&output) = outputs.get(source) {
                sum += output * self.weights[start + idx];
            }
        }

        sum
    }
}

/// Builder para construção incremental de conectividade
#[derive(Debug)]
pub struct ConnectivityBuilder {
    size: usize,
    connections: HashMap<(usize, usize), ()>,
}

impl ConnectivityBuilder {
    /// Cria novo builder
    pub fn new(size: usize) -> Self {
        Self {
            size,
            connections: HashMap::new(),
        }
    }

    /// Adiciona conexão
    pub fn add_connection(&mut self, from: usize, to: usize) -> &mut Self {
        if from < self.size && to < self.size {
            self.connections.insert((from, to), ());
        }
        self
    }

    /// Remove conexão
    pub fn remove_connection(&mut self, from: usize, to: usize) -> &mut Self {
        self.connections.remove(&(from, to));
        self
    }

    /// Constrói matriz esparsa
    pub fn build(self) -> SparseConnectivity {
        let mut adjacencies: Vec<Vec<usize>> = vec![Vec::new(); self.size];

        for (from, to) in self.connections.keys() {
            adjacencies[*from].push(*to);
        }

        // Ordena para busca binária
        for neighbors in &mut adjacencies {
            neighbors.sort();
        }

        SparseConnectivity::from_adjacency_list(self.size, &adjacencies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_creation() {
        let sparse = SparseConnectivity::fully_connected(5);

        assert_eq!(sparse.size(), 5);
        assert_eq!(sparse.num_connections(), 25);  // 5 * 5

        // Todas as conexões devem existir
        for i in 0..5 {
            for j in 0..5 {
                assert!(sparse.is_connected(i, j));
            }
        }
    }

    #[test]
    fn test_grid_2d() {
        let sparse = SparseConnectivity::grid_2d(3, 3);

        assert_eq!(sparse.size(), 9);

        // Canto tem 3 vizinhos
        assert_eq!(sparse.out_degree(0), 3);

        // Borda tem 5 vizinhos
        assert_eq!(sparse.out_degree(1), 5);

        // Centro tem 8 vizinhos
        assert_eq!(sparse.out_degree(4), 8);
    }

    #[test]
    fn test_memory_savings() {
        let grid = SparseConnectivity::grid_2d(10, 10);  // 100 neurônios

        // Grid 2D deve economizar memória vs fully connected
        let savings = grid.memory_savings();
        assert!(savings > 0.5, "Grid should save > 50% memory vs dense");
    }

    #[test]
    fn test_from_dense() {
        let dense = vec![
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![0, 1, 0],
        ];

        let sparse = SparseConnectivity::from_dense(&dense);

        assert!(sparse.is_connected(0, 1));
        assert!(sparse.is_connected(0, 2));
        assert!(sparse.is_connected(1, 0));
        assert!(!sparse.is_connected(1, 2));
        assert!(sparse.is_connected(2, 1));
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let original = SparseConnectivity::grid_2d(4, 4);
        let dense = original.to_dense();
        let reconstructed = SparseConnectivity::from_dense(&dense);

        assert_eq!(original.num_connections(), reconstructed.num_connections());
    }

    #[test]
    fn test_weighted_connectivity() {
        let connectivity = SparseConnectivity::fully_connected(3);
        let mut weighted = WeightedSparseConnectivity::from_connectivity(connectivity, 0.5);

        assert_eq!(weighted.get_weight(0, 1), Some(0.5));

        weighted.set_weight(0, 1, 1.0);
        assert_eq!(weighted.get_weight(0, 1), Some(1.0));
    }

    #[test]
    fn test_builder() {
        let mut builder = ConnectivityBuilder::new(4);
        builder.add_connection(0, 1);
        builder.add_connection(0, 2);
        builder.add_connection(1, 3);
        let connectivity = builder.build();

        assert!(connectivity.is_connected(0, 1));
        assert!(connectivity.is_connected(0, 2));
        assert!(connectivity.is_connected(1, 3));
        assert!(!connectivity.is_connected(2, 3));
    }
}
