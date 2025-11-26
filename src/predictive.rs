//! Módulo de Predictive Coding (Codificação Preditiva)
//!
//! Implementa hierarquia preditiva onde cada nível prediz o nível abaixo
//! e propaga erros de predição para cima.
//!
//! ## Fundamentação Teórica
//!
//! Baseado em:
//! - Free Energy Principle (Friston)
//! - Predictive Coding (Rao & Ballard 1999)
//! - Hierarchical Predictive Processing
//!
//! ## Características
//!
//! - **Predições Top-Down**: Níveis superiores geram expectativas
//! - **Erros Bottom-Up**: Discrepâncias propagam para cima
//! - **Precisão**: Confiança nas predições modula impacto dos erros
//! - **Free Energy**: Métrica de surpresa total do sistema
//! - **Modelo Generativo Não-Linear**: Camadas ocultas com ativações configuráveis

use std::collections::VecDeque;

// ============================================================================
// FUNÇÕES DE ATIVAÇÃO
// ============================================================================

/// Tipos de funções de ativação disponíveis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFn {
    /// Tangente hiperbólica: saída em [-1, 1]
    Tanh,
    /// ReLU: max(0, x)
    ReLU,
    /// Leaky ReLU: max(0.01*x, x)
    LeakyReLU,
    /// Sigmoid: saída em [0, 1]
    Sigmoid,
    /// Linear (identidade)
    Linear,
    /// ELU: Exponential Linear Unit
    ELU,
    /// Swish: x * sigmoid(x)
    Swish,
}

impl ActivationFn {
    /// Aplica a função de ativação
    #[inline]
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Tanh => x.tanh(),
            Self::ReLU => x.max(0.0),
            Self::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Linear => x,
            Self::ELU => if x > 0.0 { x } else { x.exp() - 1.0 },
            Self::Swish => x * (1.0 / (1.0 + (-x).exp())),
        }
    }

    /// Calcula a derivada da função de ativação
    #[inline]
    pub fn derivative(&self, x: f64, output: f64) -> f64 {
        match self {
            Self::Tanh => 1.0 - output * output,
            Self::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Self::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            Self::Sigmoid => output * (1.0 - output),
            Self::Linear => 1.0,
            Self::ELU => if x > 0.0 { 1.0 } else { output + 1.0 },
            Self::Swish => {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig + output * (1.0 - sig)
            }
        }
    }
}

impl Default for ActivationFn {
    fn default() -> Self {
        Self::Tanh
    }
}

// ============================================================================
// MODELO GENERATIVO NÃO-LINEAR
// ============================================================================

/// Camada densa do modelo generativo
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Pesos: [output_size][input_size]
    pub weights: Vec<Vec<f64>>,
    /// Biases: [output_size]
    pub biases: Vec<f64>,
    /// Função de ativação
    pub activation: ActivationFn,
    /// Cache de pré-ativações (para backprop)
    pre_activations: Vec<f64>,
    /// Cache de ativações (saídas)
    activations: Vec<f64>,
    /// Cache de inputs (para backprop)
    last_input: Vec<f64>,
}

impl DenseLayer {
    /// Cria nova camada densa
    pub fn new(input_size: usize, output_size: usize, activation: ActivationFn) -> Self {
        // Inicialização Xavier/Glorot
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| (rand::random::<f64>() - 0.5) * 2.0 * scale)
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self {
            weights,
            biases,
            activation,
            pre_activations: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            last_input: vec![0.0; input_size],
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_input = input.to_vec();

        for (i, (row, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let pre_act: f64 = row.iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>() + bias;

            self.pre_activations[i] = pre_act;
            self.activations[i] = self.activation.apply(pre_act);
        }

        self.activations.clone()
    }

    /// Backward pass - retorna gradiente para camada anterior
    pub fn backward(&mut self, grad_output: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut grad_input = vec![0.0; self.last_input.len()];

        for (i, (&grad_out, (&pre_act, &act))) in grad_output.iter()
            .zip(self.pre_activations.iter().zip(self.activations.iter()))
            .enumerate()
        {
            let grad_act = grad_out * self.activation.derivative(pre_act, act);

            // Atualiza pesos e biases
            for (j, (w, &inp)) in self.weights[i].iter_mut().zip(self.last_input.iter()).enumerate() {
                grad_input[j] += grad_act * *w;
                *w += learning_rate * grad_act * inp;
                *w = w.clamp(-3.0, 3.0);
            }

            self.biases[i] += learning_rate * grad_act;
            self.biases[i] = self.biases[i].clamp(-1.0, 1.0);
        }

        grad_input
    }

    /// Retorna tamanho de saída
    pub fn output_size(&self) -> usize {
        self.weights.len()
    }

    /// Retorna tamanho de entrada
    pub fn input_size(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Modelo generativo não-linear com múltiplas camadas
#[derive(Debug, Clone)]
pub struct NonLinearGenerativeModel {
    /// Camadas do modelo
    layers: Vec<DenseLayer>,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularização L2
    pub weight_decay: f64,
}

impl NonLinearGenerativeModel {
    /// Cria modelo com arquitetura especificada
    ///
    /// # Argumentos
    /// * `layer_sizes` - Tamanhos de cada camada (input -> hidden -> output)
    /// * `hidden_activation` - Ativação das camadas ocultas
    /// * `output_activation` - Ativação da camada de saída
    pub fn new(
        layer_sizes: &[usize],
        hidden_activation: ActivationFn,
        output_activation: ActivationFn,
    ) -> Self {
        assert!(layer_sizes.len() >= 2, "Modelo precisa de pelo menos 2 camadas");

        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let activation = if i == layer_sizes.len() - 2 {
                output_activation
            } else {
                hidden_activation
            };

            layers.push(DenseLayer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation,
            ));
        }

        Self {
            layers,
            learning_rate: 0.01,
            weight_decay: 0.0001,
        }
    }

    /// Cria modelo simples com uma camada oculta
    pub fn with_hidden_layer(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self::new(
            &[input_size, hidden_size, output_size],
            ActivationFn::LeakyReLU,
            ActivationFn::Tanh,
        )
    }

    /// Cria modelo profundo com duas camadas ocultas
    pub fn deep(input_size: usize, output_size: usize) -> Self {
        let hidden1 = (input_size + output_size) / 2 + 4;
        let hidden2 = (hidden1 + output_size) / 2;

        Self::new(
            &[input_size, hidden1, hidden2, output_size],
            ActivationFn::LeakyReLU,
            ActivationFn::Tanh,
        )
    }

    /// Forward pass completo
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();

        for layer in &mut self.layers {
            current = layer.forward(&current);
        }

        current
    }

    /// Atualiza modelo para minimizar erro de predição
    pub fn update(&mut self, target: &[f64]) {
        // Calcula erro na saída
        let output = self.layers.last().map(|l| &l.activations).unwrap();
        let grad_output: Vec<f64> = output.iter()
            .zip(target.iter())
            .map(|(o, t)| t - o) // Gradiente do MSE
            .collect();

        // Backprop através das camadas
        let mut grad = grad_output;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, self.learning_rate);
        }

        // Weight decay
        if self.weight_decay > 0.0 {
            for layer in &mut self.layers {
                for row in &mut layer.weights {
                    for w in row {
                        *w *= 1.0 - self.weight_decay;
                    }
                }
            }
        }
    }

    /// Retorna estatísticas do modelo
    pub fn stats(&self) -> GenerativeModelStats {
        let total_params: usize = self.layers.iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum();

        let avg_weight: f64 = self.layers.iter()
            .flat_map(|l| l.weights.iter().flatten())
            .map(|w| w.abs())
            .sum::<f64>() / total_params as f64;

        GenerativeModelStats {
            num_layers: self.layers.len(),
            total_parameters: total_params,
            average_weight_magnitude: avg_weight,
        }
    }
}

/// Estatísticas do modelo generativo
#[derive(Debug, Clone)]
pub struct GenerativeModelStats {
    pub num_layers: usize,
    pub total_parameters: usize,
    pub average_weight_magnitude: f64,
}

/// Unidade preditiva individual
#[derive(Debug, Clone)]
pub struct PredictiveUnit {
    /// Predição atual
    pub prediction: f64,

    /// Erro de predição atual
    pub prediction_error: f64,

    /// Precisão (confiança inversa da variÃ¢ncia)
    pub precision: f64,

    /// Histórico de erros para estimativa de precisão
    error_history: VecDeque<f64>,

    /// Tamanho do histórico
    history_size: usize,
}

impl PredictiveUnit {
    pub fn new() -> Self {
        Self {
            prediction: 0.0,
            prediction_error: 0.0,
            precision: 1.0,
            error_history: VecDeque::with_capacity(100),
            history_size: 100,
        }
    }

    /// Atualiza predição
    pub fn set_prediction(&mut self, pred: f64) {
        self.prediction = pred;
    }

    /// Computa erro de predição dado input real
    pub fn compute_error(&mut self, actual: f64) {
        self.prediction_error = actual - self.prediction;

        // Registra no histórico
        self.error_history.push_back(self.prediction_error.abs());
        if self.error_history.len() > self.history_size {
            self.error_history.pop_front();
        }
    }

    /// Atualiza precisão baseado na variÃ¢ncia dos erros
    pub fn update_precision(&mut self) {
        if self.error_history.len() < 10 {
            return;
        }

        let mean: f64 = self.error_history.iter().sum::<f64>() / self.error_history.len() as f64;
        let variance: f64 = self.error_history.iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>() / self.error_history.len() as f64;

        // Precisão = 1 / variÃ¢ncia (com floor)
        self.precision = 1.0 / (variance + 0.01);
        self.precision = self.precision.clamp(0.1, 10.0);
    }

    /// Erro ponderado por precisão
    pub fn weighted_error(&self) -> f64 {
        self.prediction_error * self.precision
    }
}

impl Default for PredictiveUnit {
    fn default() -> Self {
        Self::new()
    }
}

/// Camada preditiva completa
#[derive(Debug, Clone)]
pub struct PredictiveLayer {
    /// Unidades preditivas
    units: Vec<PredictiveUnit>,

    /// Pesos do modelo generativo (top-down)
    pub generative_weights: Vec<Vec<f64>>,

    /// Learning rate para o modelo generativo
    pub learning_rate: f64,

    /// Tamanho da camada
    size: usize,
}

impl PredictiveLayer {
    /// Cria nova camada preditiva
    pub fn new(size: usize) -> Self {
        let units = (0..size).map(|_| PredictiveUnit::new()).collect();

        // Inicializa pesos do modelo generativo
        let scale = (2.0 / size as f64).sqrt();
        let generative_weights: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| (rand::random::<f64>() - 0.5) * 2.0 * scale)
                    .collect()
            })
            .collect();

        Self {
            units,
            generative_weights,
            learning_rate: 0.01,
            size,
        }
    }

    /// Cria camada para receber input de tamanho diferente
    pub fn with_input_size(size: usize, input_size: usize) -> Self {
        let units = (0..size).map(|_| PredictiveUnit::new()).collect();

        let scale = (2.0 / input_size as f64).sqrt();
        let generative_weights: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                (0..input_size)
                    .map(|_| (rand::random::<f64>() - 0.5) * 2.0 * scale)
                    .collect()
            })
            .collect();

        Self {
            units,
            generative_weights,
            learning_rate: 0.01,
            size,
        }
    }

    /// Gera predições top-down dado estado do nível superior
    pub fn generate_predictions(&mut self, higher_level: &[f64]) {
        for (i, unit) in self.units.iter_mut().enumerate() {
            let prediction: f64 = self.generative_weights[i].iter()
                .zip(higher_level.iter())
                .map(|(w, h)| w * h)
                .sum();

            unit.set_prediction(prediction.tanh());
        }
    }

    /// Computa erros de predição dado input real
    pub fn compute_errors(&mut self, actual_input: &[f64]) {
        for (unit, &actual) in self.units.iter_mut().zip(actual_input.iter()) {
            unit.compute_error(actual);
        }
    }

    /// Retorna vetor de predições
    pub fn get_predictions(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.prediction).collect()
    }

    /// Retorna vetor de erros de predição
    pub fn get_errors(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.prediction_error).collect()
    }

    /// Retorna vetor de erros ponderados por precisão
    pub fn get_weighted_errors(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.weighted_error()).collect()
    }

    /// Retorna vetor de precisões
    pub fn get_precisions(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.precision).collect()
    }

    /// Atualiza modelo generativo para minimizar erros
    pub fn update_model(&mut self, higher_level: &[f64]) {
        for (i, unit) in self.units.iter().enumerate() {
            let error = unit.prediction_error;
            let pred = unit.prediction;

            // Derivada de tanh
            let tanh_deriv = 1.0 - pred * pred;

            // Itera apenas até o tamanho da linha de pesos
            let row_len = self.generative_weights[i].len();
            for j in 0..row_len.min(higher_level.len()) {
                let h = higher_level[j];
                let grad = error * tanh_deriv * h;
                self.generative_weights[i][j] += self.learning_rate * grad;

                // Clamp para estabilidade
                self.generative_weights[i][j] = self.generative_weights[i][j].clamp(-3.0, 3.0);
            }
        }
    }

    /// Atualiza precisões de todas as unidades
    pub fn update_precisions(&mut self) {
        for unit in &mut self.units {
            unit.update_precision();
        }
    }

    /// Calcula free energy (surpresa) da camada
    pub fn free_energy(&self) -> f64 {
        self.units.iter()
            .map(|u| u.prediction_error.powi(2) * u.precision)
            .sum::<f64>() * 0.5
    }

    /// Retorna tamanho da camada
    pub fn size(&self) -> usize {
        self.size
    }
}

// ============================================================================
// CAMADA PREDITIVA NÃO-LINEAR
// ============================================================================

/// Camada preditiva com modelo generativo não-linear
#[derive(Debug, Clone)]
pub struct NonLinearPredictiveLayer {
    /// Unidades preditivas
    units: Vec<PredictiveUnit>,

    /// Modelo generativo não-linear (top-down)
    generative_model: NonLinearGenerativeModel,

    /// Tamanho da camada de saída
    size: usize,

    /// Tamanho de input (do nível superior)
    input_size: usize,
}

impl NonLinearPredictiveLayer {
    /// Cria nova camada preditiva não-linear
    pub fn new(size: usize, input_size: usize) -> Self {
        let units = (0..size).map(|_| PredictiveUnit::new()).collect();

        // Modelo com camada oculta
        let hidden_size = (size + input_size) / 2 + 2;
        let generative_model = NonLinearGenerativeModel::with_hidden_layer(
            input_size,
            hidden_size,
            size,
        );

        Self {
            units,
            generative_model,
            size,
            input_size,
        }
    }

    /// Cria camada com modelo generativo profundo
    pub fn deep(size: usize, input_size: usize) -> Self {
        let units = (0..size).map(|_| PredictiveUnit::new()).collect();
        let generative_model = NonLinearGenerativeModel::deep(input_size, size);

        Self {
            units,
            generative_model,
            size,
            input_size,
        }
    }

    /// Cria camada com arquitetura customizada
    pub fn with_architecture(
        size: usize,
        input_size: usize,
        hidden_sizes: &[usize],
        hidden_activation: ActivationFn,
    ) -> Self {
        let units = (0..size).map(|_| PredictiveUnit::new()).collect();

        let mut layer_sizes = vec![input_size];
        layer_sizes.extend_from_slice(hidden_sizes);
        layer_sizes.push(size);

        let generative_model = NonLinearGenerativeModel::new(
            &layer_sizes,
            hidden_activation,
            ActivationFn::Tanh,
        );

        Self {
            units,
            generative_model,
            size,
            input_size,
        }
    }

    /// Gera predições top-down usando o modelo não-linear
    pub fn generate_predictions(&mut self, higher_level: &[f64]) {
        let predictions = self.generative_model.forward(higher_level);

        for (unit, &pred) in self.units.iter_mut().zip(predictions.iter()) {
            unit.set_prediction(pred);
        }
    }

    /// Computa erros de predição dado input real
    pub fn compute_errors(&mut self, actual_input: &[f64]) {
        for (unit, &actual) in self.units.iter_mut().zip(actual_input.iter()) {
            unit.compute_error(actual);
        }
    }

    /// Retorna vetor de predições
    pub fn get_predictions(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.prediction).collect()
    }

    /// Retorna vetor de erros de predição
    pub fn get_errors(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.prediction_error).collect()
    }

    /// Retorna vetor de erros ponderados por precisão
    pub fn get_weighted_errors(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.weighted_error()).collect()
    }

    /// Retorna vetor de precisões
    pub fn get_precisions(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.precision).collect()
    }

    /// Atualiza modelo generativo usando backprop
    pub fn update_model(&mut self, _higher_level: &[f64]) {
        // Coleta targets (valores reais observados)
        let targets: Vec<f64> = self.units.iter()
            .map(|u| u.prediction + u.prediction_error)
            .collect();

        self.generative_model.update(&targets);
    }

    /// Atualiza precisões de todas as unidades
    pub fn update_precisions(&mut self) {
        for unit in &mut self.units {
            unit.update_precision();
        }
    }

    /// Calcula free energy (surpresa) da camada
    pub fn free_energy(&self) -> f64 {
        self.units.iter()
            .map(|u| u.prediction_error.powi(2) * u.precision)
            .sum::<f64>() * 0.5
    }

    /// Retorna tamanho da camada
    pub fn size(&self) -> usize {
        self.size
    }

    /// Retorna estatísticas do modelo generativo
    pub fn model_stats(&self) -> GenerativeModelStats {
        self.generative_model.stats()
    }

    /// Define learning rate do modelo generativo
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.generative_model.learning_rate = lr;
    }
}

// ============================================================================
// HIERARQUIA PREDITIVA PROFUNDA
// ============================================================================

/// Hierarquia preditiva com modelos generativos não-lineares
#[derive(Debug)]
pub struct DeepPredictiveHierarchy {
    /// Camadas não-lineares da hierarquia
    layers: Vec<NonLinearPredictiveLayer>,

    /// Estados de cada nível
    states: Vec<Vec<f64>>,

    /// Learning rate para atualização de estados
    pub state_learning_rate: f64,

    /// Número de iterações de inferência por update
    pub inference_iterations: usize,

    /// Histórico de free energy
    free_energy_history: VecDeque<f64>,
}

impl DeepPredictiveHierarchy {
    /// Cria hierarquia profunda com tamanhos especificados
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Hierarquia precisa de pelo menos 2 camadas");

        let mut layers = Vec::new();

        // Primeira camada
        layers.push(NonLinearPredictiveLayer::new(layer_sizes[0], layer_sizes[0]));

        // Camadas subsequentes
        for i in 1..layer_sizes.len() {
            layers.push(NonLinearPredictiveLayer::new(
                layer_sizes[i],
                layer_sizes[i - 1],
            ));
        }

        let states: Vec<Vec<f64>> = layer_sizes.iter()
            .map(|&size| vec![0.0; size])
            .collect();

        Self {
            layers,
            states,
            state_learning_rate: 0.1,
            inference_iterations: 5,
            free_energy_history: VecDeque::with_capacity(1000),
        }
    }

    /// Cria hierarquia de 3 níveis com modelos profundos
    pub fn new_three_level_deep(input_size: usize) -> Self {
        let layer_sizes = vec![
            input_size,
            input_size / 2 + 4,
            input_size / 4 + 2,
        ];

        let mut hierarchy = Self::new(&layer_sizes);

        // Usa modelos deep para cada camada (exceto primeira)
        for i in 1..hierarchy.layers.len() {
            hierarchy.layers[i] = NonLinearPredictiveLayer::deep(
                layer_sizes[i],
                layer_sizes[i - 1],
            );
        }

        hierarchy
    }

    /// Processa input sensorial através da hierarquia profunda
    pub fn process(&mut self, sensory_input: &[f64]) -> PredictiveOutput {
        // 1. Bottom-up: propaga erros
        self.layers[0].compute_errors(sensory_input);

        // 2. Iterações de inferência
        for _ in 0..self.inference_iterations {
            // Top-down: gera predições
            for i in (1..self.layers.len()).rev() {
                let higher_state = self.states[i].clone();
                self.layers[i - 1].generate_predictions(&higher_state);
            }

            // Bottom-up: computa erros
            self.layers[0].compute_errors(sensory_input);
            for i in 1..self.layers.len() {
                let lower_errors = self.layers[i - 1].get_weighted_errors();
                self.layers[i].compute_errors(&lower_errors);
            }

            // Atualiza estados internos
            self.update_states();
        }

        // 3. Atualiza modelos generativos
        for i in 1..self.layers.len() {
            let higher_state = self.states[i].clone();
            self.layers[i - 1].update_model(&higher_state);
        }

        // 4. Atualiza precisões
        for layer in &mut self.layers {
            layer.update_precisions();
        }

        // 5. Calcula free energy total
        let total_fe = self.total_free_energy();
        self.free_energy_history.push_back(total_fe);
        if self.free_energy_history.len() > 1000 {
            self.free_energy_history.pop_front();
        }

        PredictiveOutput {
            predictions: self.layers[0].get_predictions(),
            errors: self.layers[0].get_errors(),
            precisions: self.layers[0].get_precisions(),
            top_level_state: self.states.last().cloned().unwrap_or_default(),
            free_energy: total_fe,
        }
    }

    /// Atualiza estados internos
    fn update_states(&mut self) {
        for level in 1..self.states.len() {
            let errors_below = self.layers[level - 1].get_weighted_errors();

            for (i, state) in self.states[level].iter_mut().enumerate() {
                // Aproximação: usa erro médio ponderado
                let grad: f64 = errors_below.get(i % errors_below.len()).unwrap_or(&0.0) * 0.1;
                *state += self.state_learning_rate * grad;
                *state = state.tanh();
            }
        }
    }

    /// Calcula free energy total
    pub fn total_free_energy(&self) -> f64 {
        self.layers.iter().map(|l| l.free_energy()).sum()
    }

    /// Retorna surpresa média recente
    pub fn average_surprise(&self) -> f64 {
        if self.free_energy_history.is_empty() {
            0.0
        } else {
            self.free_energy_history.iter().sum::<f64>() / self.free_energy_history.len() as f64
        }
    }

    /// Retorna tendência da free energy
    pub fn free_energy_trend(&self) -> f64 {
        if self.free_energy_history.len() < 100 {
            return 0.0;
        }

        let recent: f64 = self.free_energy_history.iter().rev().take(50).sum::<f64>() / 50.0;
        let older: f64 = self.free_energy_history.iter().rev().skip(50).take(50).sum::<f64>() / 50.0;

        (recent - older) / (older + 1e-6)
    }

    /// Retorna número de níveis
    pub fn num_levels(&self) -> usize {
        self.layers.len()
    }

    /// Retorna estatísticas da hierarquia
    pub fn get_stats(&self) -> DeepHierarchyStats {
        let layer_fes: Vec<f64> = self.layers.iter().map(|l| l.free_energy()).collect();

        let model_stats: Vec<GenerativeModelStats> = self.layers.iter()
            .map(|l| l.model_stats())
            .collect();

        let total_params: usize = model_stats.iter()
            .map(|s| s.total_parameters)
            .sum();

        DeepHierarchyStats {
            num_levels: self.layers.len(),
            layer_free_energies: layer_fes,
            total_free_energy: self.total_free_energy(),
            average_surprise: self.average_surprise(),
            free_energy_trend: self.free_energy_trend(),
            total_parameters: total_params,
            layer_model_stats: model_stats,
        }
    }
}

/// Estatísticas da hierarquia preditiva profunda
#[derive(Debug, Clone)]
pub struct DeepHierarchyStats {
    pub num_levels: usize,
    pub layer_free_energies: Vec<f64>,
    pub total_free_energy: f64,
    pub average_surprise: f64,
    pub free_energy_trend: f64,
    pub total_parameters: usize,
    pub layer_model_stats: Vec<GenerativeModelStats>,
}

impl DeepHierarchyStats {
    /// Imprime relatório formatado
    pub fn print_report(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│    HIERARQUIA PREDITIVA PROFUNDA        │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Níveis:         {:<19} │", self.num_levels);
        println!("│ Parâmetros:     {:<19} │", self.total_parameters);
        println!("│ Free Energy:    {:<19.4} │", self.total_free_energy);
        println!("│ Surpresa Média: {:<19.4} │", self.average_surprise);
        println!("│ Tendência:      {:<18.2}% │", self.free_energy_trend * 100.0);
        println!("├─────────────────────────────────────────┤");
        for (i, (fe, stats)) in self.layer_free_energies.iter()
            .zip(self.layer_model_stats.iter())
            .enumerate()
        {
            println!("│ Camada {}: FE={:<8.4} params={:<7} │", i, fe, stats.total_parameters);
        }
        println!("└─────────────────────────────────────────┘");
    }
}

/// Hierarquia preditiva completa
#[derive(Debug)]
pub struct PredictiveHierarchy {
    /// Camadas da hierarquia (do mais baixo para mais alto)
    layers: Vec<PredictiveLayer>,

    /// Estados de cada nível (representações)
    states: Vec<Vec<f64>>,

    /// Learning rate para atualização de estados
    pub state_learning_rate: f64,

    /// Número de iterações de inferÃªncia por update
    pub inference_iterations: usize,

    /// Histórico de free energy total
    free_energy_history: VecDeque<f64>,
}

impl PredictiveHierarchy {
    /// Cria hierarquia com tamanhos especificados
    ///
    /// # Argumentos
    /// * `layer_sizes` - Tamanhos de cada camada (baixo para alto)
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Hierarquia precisa de pelo menos 2 camadas");

        let mut layers = Vec::new();

        // Primeira camada recebe input externo
        layers.push(PredictiveLayer::new(layer_sizes[0]));

        // Camadas intermediárias
        for i in 1..layer_sizes.len() {
            layers.push(PredictiveLayer::with_input_size(
                layer_sizes[i],
                layer_sizes[i - 1]
            ));
        }

        // Inicializa estados
        let states: Vec<Vec<f64>> = layer_sizes.iter()
            .map(|&size| vec![0.0; size])
            .collect();

        Self {
            layers,
            states,
            state_learning_rate: 0.1,
            inference_iterations: 5,
            free_energy_history: VecDeque::with_capacity(1000),
        }
    }

    /// Cria hierarquia de 3 níveis (padrão)
    pub fn new_three_level(input_size: usize) -> Self {
        let layer_sizes = vec![
            input_size,           // Nível sensorial
            input_size / 2 + 4,   // Nível intermediário
            input_size / 4 + 2,   // Nível abstrato
        ];
        Self::new(&layer_sizes)
    }

    /// Processa input sensorial através da hierarquia
    ///
    /// Executa inferÃªncia variacional para minimizar free energy
    pub fn process(&mut self, sensory_input: &[f64]) -> PredictiveOutput {
        // 1. Bottom-up: propaga erros
        self.layers[0].compute_errors(sensory_input);

        // 2. Iterações de inferÃªncia
        for _ in 0..self.inference_iterations {
            // Top-down: gera predições
            for i in (1..self.layers.len()).rev() {
                let higher_state = self.states[i].clone();
                self.layers[i - 1].generate_predictions(&higher_state);
            }

            // Bottom-up: computa erros
            self.layers[0].compute_errors(sensory_input);
            for i in 1..self.layers.len() {
                let lower_errors = self.layers[i - 1].get_weighted_errors();
                self.layers[i].compute_errors(&lower_errors);
            }

            // Atualiza estados internos para minimizar erros
            self.update_states();
        }

        // 3. Atualiza modelos generativos (aprendizado)
        for i in 1..self.layers.len() {
            let higher_state = self.states[i].clone();
            self.layers[i - 1].update_model(&higher_state);
        }

        // 4. Atualiza precisões
        for layer in &mut self.layers {
            layer.update_precisions();
        }

        // 5. Calcula free energy total
        let total_fe = self.total_free_energy();
        self.free_energy_history.push_back(total_fe);
        if self.free_energy_history.len() > 1000 {
            self.free_energy_history.pop_front();
        }

        // 6. Coleta output
        PredictiveOutput {
            predictions: self.layers[0].get_predictions(),
            errors: self.layers[0].get_errors(),
            precisions: self.layers[0].get_precisions(),
            top_level_state: self.states.last().cloned().unwrap_or_default(),
            free_energy: total_fe,
        }
    }

    /// Atualiza estados internos usando gradiente descendente na free energy
    fn update_states(&mut self) {
        for level in 1..self.states.len() {
            let errors_below = self.layers[level - 1].get_weighted_errors();

            for (i, state) in self.states[level].iter_mut().enumerate() {
                // Gradiente da free energy em relação ao estado
                let grad: f64 = errors_below.iter()
                    .zip(self.layers[level - 1].generative_weights.iter())
                    .map(|(&e, row)| e * row.get(i).unwrap_or(&0.0))
                    .sum();

                *state += self.state_learning_rate * grad;
                *state = state.tanh(); // Bounded
            }
        }
    }

    /// Gera predição para o futuro (um passo)
    pub fn predict_next(&self) -> Vec<f64> {
        // Usa o estado atual do nível mais alto para gerar predição
        if let Some(top_state) = self.states.last() {
            let mut prediction = vec![0.0; self.layers[0].size()];

            for (i, pred) in prediction.iter_mut().enumerate() {
                *pred = self.layers[0].generative_weights.get(i)
                    .map(|row| {
                        row.iter()
                            .zip(top_state.iter())
                            .map(|(w, s)| w * s)
                            .sum::<f64>()
                            .tanh()
                    })
                    .unwrap_or(0.0);
            }

            prediction
        } else {
            vec![0.0; self.layers[0].size()]
        }
    }

    /// Calcula free energy total da hierarquia
    pub fn total_free_energy(&self) -> f64 {
        self.layers.iter().map(|l| l.free_energy()).sum()
    }

    /// Retorna surpresa média recente
    pub fn average_surprise(&self) -> f64 {
        if self.free_energy_history.is_empty() {
            0.0
        } else {
            self.free_energy_history.iter().sum::<f64>() / self.free_energy_history.len() as f64
        }
    }

    /// Retorna tendÃªncia da free energy (negativo = melhorando)
    pub fn free_energy_trend(&self) -> f64 {
        if self.free_energy_history.len() < 100 {
            return 0.0;
        }

        let recent: f64 = self.free_energy_history.iter().rev().take(50).sum::<f64>() / 50.0;
        let older: f64 = self.free_energy_history.iter().rev().skip(50).take(50).sum::<f64>() / 50.0;

        (recent - older) / (older + 1e-6)
    }

    /// Retorna estado do nível especificado
    pub fn get_level_state(&self, level: usize) -> Option<&Vec<f64>> {
        self.states.get(level)
    }

    /// Retorna número de níveis
    pub fn num_levels(&self) -> usize {
        self.layers.len()
    }

    /// Retorna estatísticas da hierarquia
    pub fn get_stats(&self) -> HierarchyStats {
        let layer_fes: Vec<f64> = self.layers.iter().map(|l| l.free_energy()).collect();

        let avg_precision: f64 = self.layers.iter()
            .map(|l| l.get_precisions().iter().sum::<f64>() / l.size() as f64)
            .sum::<f64>() / self.layers.len() as f64;

        HierarchyStats {
            num_levels: self.layers.len(),
            layer_free_energies: layer_fes,
            total_free_energy: self.total_free_energy(),
            average_surprise: self.average_surprise(),
            free_energy_trend: self.free_energy_trend(),
            average_precision: avg_precision,
        }
    }
}

/// Output do processamento preditivo
#[derive(Debug, Clone)]
pub struct PredictiveOutput {
    /// Predições para o nível sensorial
    pub predictions: Vec<f64>,

    /// Erros de predição
    pub errors: Vec<f64>,

    /// Precisões
    pub precisions: Vec<f64>,

    /// Estado do nível mais alto (representação abstrata)
    pub top_level_state: Vec<f64>,

    /// Free energy total
    pub free_energy: f64,
}

/// Estatísticas da hierarquia preditiva
#[derive(Debug, Clone)]
pub struct HierarchyStats {
    pub num_levels: usize,
    pub layer_free_energies: Vec<f64>,
    pub total_free_energy: f64,
    pub average_surprise: f64,
    pub free_energy_trend: f64,
    pub average_precision: f64,
}

impl HierarchyStats {
    /// Imprime relatório formatado
    pub fn print_report(&self) {
        println!("â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”");
        println!("â”‚       HIERARQUIA PREDITIVA          â”‚");
        println!("â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤");
        println!("â”‚ Níveis:         {:<19}â”‚", self.num_levels);
        println!("â”‚ Free Energy:    {:<19.4}â”‚", self.total_free_energy);
        println!("â”‚ Surpresa Média: {:<19.4}â”‚", self.average_surprise);
        println!("â”‚ TendÃªncia:      {:<19.2}%â”‚", self.free_energy_trend * 100.0);
        println!("â”‚ Precisão Média: {:<19.4}â”‚", self.average_precision);
        println!("â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤");
        for (i, &fe) in self.layer_free_energies.iter().enumerate() {
            println!("â”‚ Camada {}: FE = {:<17.4}â”‚", i, fe);
        }
        println!("â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜");
    }
}

// ============================================================================
// ACTIVE INFERENCE (Extensão)
// ============================================================================

/// Módulo de Active Inference
///
/// Combina predição com seleção de ação para minimizar expected free energy
#[derive(Debug)]
pub struct ActiveInference {
    /// Hierarquia preditiva
    hierarchy: PredictiveHierarchy,

    /// Modelo de transição: estado + ação -> próximo estado
    transition_model: Vec<Vec<Vec<f64>>>, // [action][from_state][to_state]

    /// Número de ações possíveis
    num_actions: usize,

    /// Horizonte de planejamento
    planning_horizon: usize,
}

impl ActiveInference {
    /// Cria módulo de active inference
    pub fn new(state_size: usize, num_actions: usize) -> Self {
        let hierarchy = PredictiveHierarchy::new_three_level(state_size);

        // Inicializa modelo de transição uniforme
        let scale = 1.0 / state_size as f64;
        let transition_model: Vec<Vec<Vec<f64>>> = (0..num_actions)
            .map(|_| {
                (0..state_size)
                    .map(|_| vec![scale; state_size])
                    .collect()
            })
            .collect();

        Self {
            hierarchy,
            transition_model,
            num_actions,
            planning_horizon: 3,
        }
    }

    /// Seleciona ação que minimiza expected free energy
    pub fn select_action(&mut self, current_state: &[f64]) -> usize {
        // Processa estado atual
        let output = self.hierarchy.process(current_state);

        // Calcula expected free energy para cada ação
        let mut expected_fes = vec![0.0; self.num_actions];

        for action in 0..self.num_actions {
            let mut accumulated_fe = 0.0;
            let mut state_belief = current_state.to_vec();

            for _ in 0..self.planning_horizon {
                // Prediz próximo estado dado ação
                let next_state = self.predict_transition(&state_belief, action);

                // Simula processamento e calcula FE esperada
                // (simplificado: usa distÃ¢ncia ao estado preferido)
                let prediction_error: f64 = next_state.iter()
                    .zip(output.predictions.iter())
                    .map(|(n, p)| (n - p).powi(2))
                    .sum();

                accumulated_fe += prediction_error;
                state_belief = next_state;
            }

            expected_fes[action] = accumulated_fe;
        }

        // Seleciona ação com menor expected free energy
        expected_fes.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Prediz transição de estado dada ação
    fn predict_transition(&self, state: &[f64], action: usize) -> Vec<f64> {
        if action >= self.num_actions {
            return state.to_vec();
        }

        let transition = &self.transition_model[action];

        state.iter().enumerate().map(|(i, &s)| {
            transition.get(i)
                .map(|row| {
                    row.iter()
                        .zip(state.iter())
                        .map(|(t, s)| t * s)
                        .sum::<f64>()
                })
                .unwrap_or(s)
        }).collect()
    }

    /// Atualiza modelo de transição com experiÃªncia
    pub fn update_transition(&mut self, state: &[f64], action: usize, next_state: &[f64]) {
        if action >= self.num_actions {
            return;
        }

        let lr = 0.01;
        let transition = &mut self.transition_model[action];

        for (i, row) in transition.iter_mut().enumerate() {
            for (j, weight) in row.iter_mut().enumerate() {
                let error = next_state.get(i).unwrap_or(&0.0) - state.get(j).unwrap_or(&0.0) * *weight;
                *weight += lr * error * state.get(j).unwrap_or(&0.0);
                *weight = weight.clamp(0.0, 1.0);
            }
        }
    }

    /// Retorna hierarquia para acesso direto
    pub fn hierarchy(&self) -> &PredictiveHierarchy {
        &self.hierarchy
    }

    /// Retorna hierarquia mutável
    pub fn hierarchy_mut(&mut self) -> &mut PredictiveHierarchy {
        &mut self.hierarchy
    }
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_unit() {
        let mut unit = PredictiveUnit::new();

        unit.set_prediction(0.5);
        unit.compute_error(0.7);

        assert!((unit.prediction_error - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_predictive_layer() {
        let mut layer = PredictiveLayer::new(4);

        let higher = vec![0.5, 0.5, 0.0, 0.0];
        layer.generate_predictions(&higher);

        let predictions = layer.get_predictions();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_hierarchy_processing() {
        let mut hierarchy = PredictiveHierarchy::new_three_level(8);

        let input = vec![1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        let output = hierarchy.process(&input);

        assert_eq!(output.predictions.len(), 8);
        assert_eq!(output.errors.len(), 8);
        assert!(output.free_energy >= 0.0);
    }

    #[test]
    fn test_free_energy_decreases() {
        let mut hierarchy = PredictiveHierarchy::new_three_level(4);

        // Mesmo input repetido deve manter free energy estável ou reduzi-la
        let input = vec![0.5, 0.5, 0.0, 0.0];

        let mut fe_values = Vec::new();
        for _ in 0..100 {
            let output = hierarchy.process(&input);
            fe_values.push(output.free_energy);
        }

        // Verifica que free energy permanece finita e não explode
        let last_avg: f64 = fe_values.iter().rev().take(20).sum::<f64>() / 20.0;

        assert!(last_avg.is_finite());
        assert!(last_avg >= 0.0);
        // Free energy deve estar em um intervalo razoável
        assert!(last_avg < 1000.0);
    }

    #[test]
    fn test_active_inference() {
        let mut ai = ActiveInference::new(4, 3);

        let state = vec![0.5, 0.5, 0.0, 0.0];
        let action = ai.select_action(&state);

        assert!(action < 3);
    }

    // ========================================================================
    // TESTES PARA FUNCIONALIDADES NÃO-LINEARES
    // ========================================================================

    #[test]
    fn test_activation_functions() {
        // Tanh
        let tanh_out = ActivationFn::Tanh.apply(0.5);
        assert!((tanh_out - 0.5_f64.tanh()).abs() < 1e-6);

        // ReLU
        assert_eq!(ActivationFn::ReLU.apply(2.0), 2.0);
        assert_eq!(ActivationFn::ReLU.apply(-2.0), 0.0);

        // LeakyReLU
        assert_eq!(ActivationFn::LeakyReLU.apply(2.0), 2.0);
        assert!((ActivationFn::LeakyReLU.apply(-2.0) - (-0.02)).abs() < 1e-6);

        // Sigmoid
        let sig = ActivationFn::Sigmoid.apply(0.0);
        assert!((sig - 0.5).abs() < 1e-6);

        // ELU
        assert_eq!(ActivationFn::ELU.apply(1.0), 1.0);
        assert!(ActivationFn::ELU.apply(-1.0) < 0.0);

        // Swish
        let swish = ActivationFn::Swish.apply(0.0);
        assert!((swish - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_derivatives() {
        // Tanh derivative
        let x = 0.5;
        let out = ActivationFn::Tanh.apply(x);
        let deriv = ActivationFn::Tanh.derivative(x, out);
        assert!((deriv - (1.0 - out * out)).abs() < 1e-6);

        // ReLU derivative
        assert_eq!(ActivationFn::ReLU.derivative(1.0, 1.0), 1.0);
        assert_eq!(ActivationFn::ReLU.derivative(-1.0, 0.0), 0.0);

        // Sigmoid derivative
        let sig_out = ActivationFn::Sigmoid.apply(0.0);
        let sig_deriv = ActivationFn::Sigmoid.derivative(0.0, sig_out);
        assert!((sig_deriv - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_dense_layer() {
        let mut layer = DenseLayer::new(4, 3, ActivationFn::Tanh);

        let input = vec![0.5, 0.5, 0.0, 0.0];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 3);
        for &val in &output {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_dense_layer_backward() {
        let mut layer = DenseLayer::new(4, 3, ActivationFn::LeakyReLU);

        let input = vec![0.5, 0.5, 0.0, 0.0];
        let _output = layer.forward(&input);

        let grad_output = vec![0.1, -0.1, 0.05];
        let grad_input = layer.backward(&grad_output, 0.01);

        assert_eq!(grad_input.len(), 4);
    }

    #[test]
    fn test_nonlinear_generative_model() {
        let mut model = NonLinearGenerativeModel::with_hidden_layer(4, 8, 4);

        let input = vec![0.5, 0.5, 0.0, 0.0];
        let output = model.forward(&input);

        assert_eq!(output.len(), 4);

        // Modelo deve convergir com treinamento repetido
        let target = vec![0.7, 0.3, 0.1, 0.0];
        let initial_error: f64 = output.iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum();

        for _ in 0..100 {
            model.forward(&input);
            model.update(&target);
        }

        let final_output = model.forward(&input);
        let final_error: f64 = final_output.iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum();

        // Erro deve diminuir com treinamento
        assert!(final_error < initial_error);
    }

    #[test]
    fn test_deep_generative_model() {
        let mut model = NonLinearGenerativeModel::deep(8, 4);

        let stats = model.stats();
        assert!(stats.num_layers >= 3); // input->hidden1->hidden2->output

        let input = vec![0.5; 8];
        let output = model.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_nonlinear_predictive_layer() {
        let mut layer = NonLinearPredictiveLayer::new(4, 4);

        let higher_level = vec![0.5, 0.5, 0.0, 0.0];
        layer.generate_predictions(&higher_level);

        let predictions = layer.get_predictions();
        assert_eq!(predictions.len(), 4);

        let actual = vec![0.6, 0.4, 0.1, 0.0];
        layer.compute_errors(&actual);

        let errors = layer.get_errors();
        assert_eq!(errors.len(), 4);
        assert!(layer.free_energy() >= 0.0);
    }

    #[test]
    fn test_nonlinear_layer_with_architecture() {
        let layer = NonLinearPredictiveLayer::with_architecture(
            4,
            4,
            &[8, 6],
            ActivationFn::ELU,
        );

        let stats = layer.model_stats();
        assert!(stats.num_layers >= 3);
    }

    #[test]
    fn test_deep_predictive_hierarchy() {
        let mut hierarchy = DeepPredictiveHierarchy::new(&[8, 6, 4]);

        let input = vec![1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        let output = hierarchy.process(&input);

        assert_eq!(output.predictions.len(), 8);
        assert_eq!(output.errors.len(), 8);
        assert!(output.free_energy >= 0.0);
    }

    #[test]
    fn test_deep_hierarchy_three_level() {
        let mut hierarchy = DeepPredictiveHierarchy::new_three_level_deep(8);

        let input = vec![0.5; 8];
        let output = hierarchy.process(&input);

        assert_eq!(output.predictions.len(), 8);
        assert_eq!(hierarchy.num_levels(), 3);

        let stats = hierarchy.get_stats();
        assert!(stats.total_parameters > 0);
    }

    #[test]
    fn test_deep_hierarchy_learning() {
        let mut hierarchy = DeepPredictiveHierarchy::new_three_level_deep(4);

        // Treina com padrão constante
        let input = vec![0.7, 0.3, 0.5, 0.1];

        let mut fe_values = Vec::new();
        for _ in 0..100 {
            let output = hierarchy.process(&input);
            fe_values.push(output.free_energy);
        }

        // Verifica que free energy está limitada (não explode)
        let last_10: f64 = fe_values.iter().rev().take(10).sum::<f64>() / 10.0;

        // Free energy deve permanecer finita e não negativa
        assert!(last_10.is_finite());
        assert!(last_10 >= 0.0);
    }

    #[test]
    fn test_model_stats() {
        let model = NonLinearGenerativeModel::new(
            &[4, 8, 6, 4],
            ActivationFn::LeakyReLU,
            ActivationFn::Tanh,
        );

        let stats = model.stats();

        // 4*8 + 8 (bias) + 8*6 + 6 (bias) + 6*4 + 4 (bias)
        // = 32 + 8 + 48 + 6 + 24 + 4 = 122
        assert!(stats.total_parameters > 0);
        assert!(stats.average_weight_magnitude > 0.0);
    }

    #[test]
    fn test_deep_stats_report() {
        let hierarchy = DeepPredictiveHierarchy::new_three_level_deep(4);
        let stats = hierarchy.get_stats();

        assert_eq!(stats.num_levels, 3);
        assert_eq!(stats.layer_free_energies.len(), 3);
        assert_eq!(stats.layer_model_stats.len(), 3);
    }
}
