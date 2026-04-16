from config import *

class MemoryLayer:
    
    def __init__(self, n_neurons: int, time_window: int, device: str = 'cpu'):
        self.n_neurons = n_neurons
        self.time_window = time_window
        self.device = device
        
        print(f"  Memory Layer:")
        print(f"    Memory neurons: {n_neurons}")
        print(f"    Time window T: {time_window}")
        print(f"    Information decay: vt = (1 - t/T)")
        
        self.memory_values = None
        self.has_received_spike = None
        self.first_spike_time = None
        
        self.spike_reception_count = 0
        self.total_information = 0.0
        self.avg_spike_time = 0.0
    
    def reset(self, batch_size: int = 1):
        self.memory_values = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.has_received_spike = torch.zeros(batch_size, self.n_neurons, 
                                               dtype=torch.bool, device=self.device)
        self.first_spike_time = torch.full((batch_size, self.n_neurons), -1.0, 
                                            device=self.device)
        self.spike_reception_count = 0
        self.total_information = 0.0
        self.avg_spike_time = 0.0
    
    def compute_information_value(self, time_step: int) -> float:
        if time_step < 0 or time_step >= self.time_window:
            return 0.0
        return 1.0 - (time_step / self.time_window)
    
    def receive_spikes(self, spikes: torch.Tensor, time_step: int):
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)
        
        batch_size = spikes.shape[0]
        
        if self.memory_values is None or self.memory_values.shape[0] != batch_size:
            self.reset(batch_size)
        
        information_value = self.compute_information_value(time_step)
        spike_mask = spikes > 0
        new_spike_mask = spike_mask & (~self.has_received_spike)
        
        self.memory_values[new_spike_mask] = information_value
        self.first_spike_time[new_spike_mask] = float(time_step)
        self.has_received_spike = self.has_received_spike | spike_mask
    
    def get_memory_vector(self) -> torch.Tensor:
        if self.memory_values is None:
            return torch.zeros(1, self.n_neurons, device=self.device)
        return self.memory_values
    
    def get_memory_statistics(self) -> Dict:
        if self.memory_values is None:
            return {
                'spike_reception_rate': 0.0,
                'total_information': 0.0,
                'avg_information_per_neuron': 0.0,
                'avg_spike_time': 0.0,
                'active_neurons_count': 0
            }
        
        active_mask = self.has_received_spike
        active_count = active_mask.sum().item()
        total_neurons = self.n_neurons * self.memory_values.shape[0]
        spike_reception_rate = active_count / total_neurons if total_neurons > 0 else 0.0
        total_information = self.memory_values.sum().item()
        avg_information = total_information / total_neurons if total_neurons > 0 else 0.0
        
        valid_times = self.first_spike_time[active_mask]
        avg_spike_time = valid_times.mean().item() if valid_times.numel() > 0 else 0.0
        
        return {
            'spike_reception_rate': spike_reception_rate,
            'total_information': total_information,
            'avg_information_per_neuron': avg_information,
            'avg_spike_time': avg_spike_time,
            'active_neurons_count': int(active_count)
        }


class RandomProjector:
    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        scale = 1.0 / np.sqrt(output_dim)
        self.proj_matrix = torch.randn(output_dim, input_dim, device=device) * scale
        
        print(f"  Random Projector:")
        print(f"    Input dimension: {input_dim}")
        print(f"    Output dimension: {output_dim}")
        print(f"    Johnson-Lindenstrauss preserving distance property")
        
    def project(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mm(features, self.proj_matrix.T)


class SpikingSOMLayer:
    
    def __init__(self, n_input: int, n_competitive: int = 400, device: str = 'cpu'):
        self.n_input = n_input
        self.n_competitive = n_competitive
        self.device = device
        
        print(f"  Spiking-SOM Layer (LIF neuron competition + Memory Layer + Confidence Correction):")
        print(f"    Input: {n_input}, Competitive neurons: {n_competitive}")
        
        self.ttfs_time_window = ***  # Hidden hyperparameter
        self.memory_layer = MemoryLayer(
            n_neurons=n_input,
            time_window=self.ttfs_time_window,
            device=device
        )
        
        self.competitive_weights = torch.rand(n_competitive, n_input, device=device) * 0.5 + 0.3
        self.competitive_weights = F.normalize(self.competitive_weights, p=2, dim=1)
        
        self.tau_m = ***      # Hidden hyperparameter
        self.v_thresh = ***   # Hidden hyperparameter
        self.v_reset = ***    # Hidden hyperparameter
        self.v_rest = ***     # Hidden hyperparameter
        
        self.v_neurons = None
        self.has_spiked = None
        self.first_spike_time = None
        
        self.learning_rate = ***          # Hidden hyperparameter
        self.sigma_init = ***             # Hidden hyperparameter
        self.sigma_decay = ***            # Hidden hyperparameter
        self.current_sigma = self.sigma_init
        
        self.neighborhood_range = torch.arange(n_competitive, device=device).float()
        
        self.bmu_history = []
        self.confidence_scores = []
        self.memory_stats_history = []
        
        self.neuron_class_activations = {i: {} for i in range(n_competitive)}
        self.neuron_confidence_scores = {}
        self.boundary_neurons = set()
        self.non_boundary_neurons = set()
    
    def reset_neurons(self, batch_size: int):
        self.v_neurons = torch.full((batch_size, self.n_competitive), self.v_rest, device=self.device)
        self.has_spiked = torch.zeros(batch_size, self.n_competitive, dtype=torch.bool, device=self.device)
        self.first_spike_time = torch.full((batch_size, self.n_competitive), -1.0, device=self.device)
        self.memory_layer.reset(batch_size)
    
    def _calculate_current_from_memory(self, memory_vector: torch.Tensor, time_step: int) -> torch.Tensor:
        batch_size = memory_vector.shape[0]
        
        ad_distance = torch.sqrt(torch.sum(
            (self.competitive_weights.unsqueeze(0) - memory_vector.unsqueeze(1)) ** 2, dim=2
        ))
        
        weight_ranks = torch.argsort(torch.argsort(self.competitive_weights, dim=1), dim=1)
        memory_ranks = torch.argsort(torch.argsort(memory_vector, dim=1), dim=1)
        
        rd_distance = torch.zeros(batch_size, self.n_competitive, device=self.device)
        for b in range(batch_size):
            valid_mask = memory_vector[b] != 0
            if valid_mask.sum() > 0:
                rd_distance[b] = torch.sqrt(torch.sum(
                    (weight_ranks[:, valid_mask] - memory_ranks[b, valid_mask].unsqueeze(0)) ** 2, dim=1
                ))
        
        a, b_param = ***, ***  # Hidden hyperparameters
        total_distance = a * ad_distance + b_param * rd_distance
        
        max_distance = total_distance.max(dim=1, keepdim=True).values
        normalized_distance = total_distance / (max_distance + 1e-8)
        input_current = (1.0 - normalized_distance) * 2.0
        
        return input_current
    
    def _update_neurons(self, input_current: torch.Tensor, time_step: int):
        dv = ((self.v_rest - self.v_neurons) + input_current) / self.tau_m
        self.v_neurons += dv
        
        spike_mask = (self.v_neurons >= self.v_thresh) & (~self.has_spiked)
        self.first_spike_time[spike_mask] = float(time_step)
        self.has_spiked = self.has_spiked | spike_mask
        self.v_neurons[spike_mask] = self.v_reset
        
        return spike_mask
    
    def _update_weights(self, bmu_indices: torch.Tensor, info_vector: torch.Tensor):
        batch_size = bmu_indices.shape[0]
        
        for b in range(batch_size):
            bmu_idx = bmu_indices[b].item()
            bmu_pos = self.neighborhood_range[bmu_idx]
            distances = torch.abs(self.neighborhood_range - bmu_pos)
            
            neighborhood = torch.exp(-distances**2 / (2 * self.current_sigma**2))
            
            weight_update = self.learning_rate * neighborhood.unsqueeze(1) * (info_vector[b] - self.competitive_weights)
            self.competitive_weights += weight_update
        
        self.competitive_weights = F.normalize(self.competitive_weights, p=2, dim=1)
        self.current_sigma *= self.sigma_decay
        self.current_sigma = max(self.current_sigma, 1.0)
    
    def process_ttfs_spikes(self, input_spikes: torch.Tensor, training: bool = True, 
                           label: int = None) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        batch_size = 1
        time_steps = self.ttfs_time_window
        
        self.reset_neurons(batch_size)
        
        bmu_indices = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        confidence_scores = torch.zeros(batch_size, device=self.device)
        
        ttfs_spikes = input_spikes[:time_steps, :].unsqueeze(0)
        
        for t in range(time_steps):
            current_spikes = ttfs_spikes[:, t, :]
            self.memory_layer.receive_spikes(current_spikes, t)
            
            if self.memory_layer.has_received_spike.all():
                break
        
        memory_vector = self.memory_layer.get_memory_vector()
        
        input_current = self._calculate_current_from_memory(memory_vector, 0)
        max_competition_steps = min(10, time_steps)
        
        for t in range(max_competition_steps):
            spike_mask_neurons = self._update_neurons(input_current, t)
            
            if bmu_indices[0] == -1 and spike_mask_neurons[0].any():
                first_spike_idx = torch.where(spike_mask_neurons[0])[0]
                if len(first_spike_idx) > 0:
                    bmu_indices[0] = first_spike_idx[0]
                    break
        
        if bmu_indices[0] == -1:
            bmu_indices[0] = torch.argmax(self.v_neurons[0])
        
        bmu_idx = bmu_indices[0].item()
        
        if label is not None and training:
            if label not in self.neuron_class_activations[bmu_idx]:
                self.neuron_class_activations[bmu_idx][label] = 0
            self.neuron_class_activations[bmu_idx][label] += 1
        
        if label is not None and self.neuron_class_activations[bmu_idx]:
            activations = self.neuron_class_activations[bmu_idx]
            total_activations = sum(activations.values())
            
            if total_activations > 0:
                max_activation_class = max(activations.items(), key=lambda x: x[1])[0]
                max_activations = activations[max_activation_class]
                
                confidence = max_activations / total_activations
                confidence_scores[0] = confidence
                
                self.neuron_confidence_scores[bmu_idx] = confidence
            else:
                confidence_scores[0] = 0.5
        else:
            bmu_time = self.first_spike_time[0, bmu_indices[0]]
            if bmu_time != -1:
                confidence = 1.0 - (bmu_time / max_competition_steps)
                confidence_scores[0] = max(0.0, confidence)
            else:
                max_v = self.v_neurons[0].max()
                min_v = self.v_neurons[0].min()
                if max_v > min_v:
                    confidence = (self.v_neurons[0, bmu_indices[0]] - min_v) / (max_v - min_v)
                    confidence_scores[0] = confidence
                else:
                    confidence_scores[0] = 0.5
        
        if training:
            self._update_weights(bmu_indices, memory_vector)
        
        memory_stats = self.memory_layer.get_memory_statistics()
        self.memory_stats_history.append(memory_stats)
        
        som_output = torch.zeros(self.n_competitive, device=self.device)
        som_output[bmu_idx] = 1.0
        
        confidence = confidence_scores[0].item()
        
        self.bmu_history.append(bmu_idx)
        self.confidence_scores.append(confidence)
        
        return bmu_idx, confidence, som_output, memory_vector.squeeze(0)
    
    def get_memory_layer_stats(self) -> Dict:
        if len(self.memory_stats_history) == 0:
            return {
                'avg_spike_reception_rate': 0.0,
                'avg_total_information': 0.0,
                'avg_spike_time': 0.0,
                'total_samples_processed': 0
            }
        
        avg_reception_rate = np.mean([s['spike_reception_rate'] for s in self.memory_stats_history])
        avg_total_info = np.mean([s['total_information'] for s in self.memory_stats_history])
        avg_spike_time = np.mean([s['avg_spike_time'] for s in self.memory_stats_history])
        
        return {
            'avg_spike_reception_rate': avg_reception_rate,
            'avg_total_information': avg_total_info,
            'avg_spike_time': avg_spike_time,
            'total_samples_processed': len(self.memory_stats_history)
        }
    
    def identify_boundary_neurons(self, confidence_threshold: float = 0.6):
        print(f"\n  Identifying boundary neurons (threshold={confidence_threshold})...")
        
        self.boundary_neurons = set()
        self.non_boundary_neurons = set()
        
        for neuron_idx in range(self.n_competitive):
            if neuron_idx in self.neuron_confidence_scores:
                confidence = self.neuron_confidence_scores[neuron_idx]
                
                if confidence < confidence_threshold:
                    self.boundary_neurons.add(neuron_idx)
                else:
                    self.non_boundary_neurons.add(neuron_idx)
            else:
                self.boundary_neurons.add(neuron_idx)
        
        n_boundary = len(self.boundary_neurons)
        n_non_boundary = len(self.non_boundary_neurons)
        total = n_boundary + n_non_boundary
        
        print(f"    Boundary neurons: {n_boundary} ({n_boundary/total*100:.1f}%)")
        print(f"    Non-boundary neurons: {n_non_boundary} ({n_non_boundary/total*100:.1f}%)")
        print(f"    Total neurons: {total}")
        
        if self.boundary_neurons:
            boundary_confidences = [self.neuron_confidence_scores.get(idx, 0.0) 
                                   for idx in self.boundary_neurons]
            if boundary_confidences:
                print(f"    Boundary neurons confidence range: [{min(boundary_confidences):.3f}, {max(boundary_confidences):.3f}]")
        
        if self.non_boundary_neurons:
            non_boundary_confidences = [self.neuron_confidence_scores.get(idx, 0.0) 
                                       for idx in self.non_boundary_neurons]
            if non_boundary_confidences:
                print(f"    Non-boundary neurons confidence range: [{min(non_boundary_confidences):.3f}, {max(non_boundary_confidences):.3f}]")
        
        return self.boundary_neurons, self.non_boundary_neurons


class EnhancedHybridSpikeEncoder:
    
    def __init__(self, time_steps: int = 50, device: str = 'cuda'):
        self.time_steps = time_steps
        self.device = device
        self.ttfs_window = (***, ***)      # Hidden hyperparameter
        self.poisson_window = (***, ***)   # Hidden hyperparameter
        
        self.high_intensity_threshold = ***  # Hidden hyperparameter
        self.edge_threshold = ***            # Hidden hyperparameter
    
    def detect_edges(self, img: torch.Tensor, threshold: float = None) -> torch.Tensor:
        img = img.to(self.device)
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            img = img.unsqueeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        if gradient_magnitude.dim() == 0:
            gradient_magnitude = gradient_magnitude.unsqueeze(0).unsqueeze(0)
        elif gradient_magnitude.dim() == 1:
            gradient_magnitude = gradient_magnitude.unsqueeze(0)
        
        return gradient_magnitude > (threshold if threshold else self.edge_threshold)
    
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        images = images.to(self.device)
        batch_size = images.shape[0]
        n_pixels = 784
        spike_train = torch.zeros(batch_size, self.time_steps, n_pixels, device=self.device)
        
        for b in range(batch_size):
            img = images[b, 0]
            img_flat = img.view(-1)
            edge_mask = self.detect_edges(img)
            edge_flat = edge_mask.view(-1)
            
            for idx in range(n_pixels):
                intensity = img_flat[idx].item()
                is_edge = edge_flat[idx].item() if idx < edge_flat.numel() else False
                
                if intensity < 0.03:
                    continue
                
                if intensity > self.high_intensity_threshold or is_edge:
                    ttfs_time = self.ttfs_window[0] + int((1.0 - intensity) * 
                                (self.ttfs_window[1] - self.ttfs_window[0]))
                    ttfs_time = max(self.ttfs_window[0], 
                                   min(self.ttfs_window[1] - 1, ttfs_time))
                    spike_train[b, ttfs_time, idx] = 1.0
                else:
                    ttfs_time = self.ttfs_window[0] + int((1.0 - intensity * 0.8) * 
                                (self.ttfs_window[1] - self.ttfs_window[0]))
                    ttfs_time = max(self.ttfs_window[0], 
                                   min(self.ttfs_window[1] - 1, ttfs_time))
                    spike_train[b, ttfs_time, idx] = 1.0
                
                if intensity > 0.7:
                    n_spikes = 5 + int(intensity * 2)
                elif intensity > 0.4:
                    n_spikes = 3 + int(intensity * 2)
                else:
                    n_spikes = 1 + int(intensity * 2)
                
                poisson_window_size = self.poisson_window[1] - self.poisson_window[0]
                interval = max(2, poisson_window_size // (n_spikes + 1))
                
                for spike_idx in range(n_spikes):
                    poisson_time = self.poisson_window[0] + (spike_idx + 1) * interval
                    jitter = int(torch.randn(1).item() * 1.5)
                    poisson_time = poisson_time + jitter
                    
                    if self.poisson_window[0] <= poisson_time < self.poisson_window[1]:
                        spike_train[b, poisson_time, idx] = 1.0
        
        total_spikes = spike_train.sum().item()
        ttfs_spikes = spike_train[:, self.ttfs_window[0]:self.ttfs_window[1], :].sum().item()
        poisson_spikes = spike_train[:, self.poisson_window[0]:self.poisson_window[1], :].sum().item()
        
        info = {
            'total_sparsity': total_spikes / spike_train.numel(),
            'ttfs_spikes': ttfs_spikes,
            'poisson_spikes': poisson_spikes,
            'total_spikes': total_spikes,
            'ttfs_ratio': ttfs_spikes / total_spikes if total_spikes > 0 else 0,
            'poisson_ratio': poisson_spikes / total_spikes if total_spikes > 0 else 0,
        }
        
        return spike_train, info


class FullLIF:
    
    def __init__(self, n_neurons: int, device: str = 'cpu'):
        self.n_neurons = n_neurons
        self.device = device
        self.tau_m = ***      # Hidden hyperparameter
        self.v_thresh = ***   # Hidden hyperparameter
        self.v_reset = ***    # Hidden hyperparameter
        self.v_rest = ***     # Hidden hyperparameter
        self.R = ***          # Hidden hyperparameter
        self.refrac_period = ***  # Hidden hyperparameter
        
        self.v = None
        self.spike = None
        self.refrac_count = None
    
    def reset(self, batch_size: int):
        self.v = torch.full((batch_size, self.n_neurons), self.v_rest, device=self.device)
        self.spike = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.refrac_count = torch.zeros(batch_size, self.n_neurons, device=self.device)
    
    @torch.no_grad()
    def update(self, I_ext: torch.Tensor, dt: float = 1.0):
        in_refrac = self.refrac_count > 0
        dv = ((self.v_rest - self.v + self.R * I_ext) / self.tau_m) * dt
        self.v = torch.where(in_refrac, self.v, self.v + dv)
        
        self.spike = ((self.v >= self.v_thresh) & (~in_refrac)).float()
        
        self.v = torch.where(self.spike.bool(), torch.full_like(self.v, self.v_reset), self.v)
        self.refrac_count = torch.where(self.spike.bool(),
                                       torch.full_like(self.refrac_count, float(self.refrac_period)),
                                       self.refrac_count)
        self.refrac_count = torch.clamp(self.refrac_count - 1, min=0)
        
        return self.spike, self.v


class EnhancedExcitatoryInhibitorySTDP:
    
    def __init__(self, n_neurons: int, device: str = 'cpu', excitatory_ratio: float = 0.8):
        self.n_neurons = n_neurons
        self.device = device
        
        self.n_excitatory = int(n_neurons * excitatory_ratio)
        self.n_inhibitory = n_neurons - self.n_excitatory
        
        self.is_excitatory = torch.zeros(n_neurons, dtype=torch.bool, device=device)
        self.is_excitatory[:self.n_excitatory] = True
        
        # Hidden STDP hyperparameters
        self.a_plus_exc_fast = ***   # Hidden
        self.a_minus_exc_fast = ***  # Hidden
        self.tau_plus_exc_fast = *** # Hidden
        self.tau_minus_exc_fast = *** # Hidden
        
        self.a_plus_exc_slow = ***   # Hidden
        self.a_minus_exc_slow = ***  # Hidden
        self.tau_plus_exc_slow = *** # Hidden
        self.tau_minus_exc_slow = *** # Hidden
        
        self.a_plus_inh = ***   # Hidden
        self.a_minus_inh = ***  # Hidden
        self.tau_plus_inh = *** # Hidden
        self.tau_minus_inh = *** # Hidden
        
        self.trace_pre_fast = None
        self.trace_post_fast = None
        self.trace_pre_slow = None
        self.trace_post_slow = None
        
        self.learning_step = 0
        self.weight_update_history = []
    
    def reset(self, batch_size: int):
        self.trace_pre_fast = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.trace_post_fast = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.trace_pre_slow = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.trace_post_slow = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.learning_step = 0
    
    @torch.no_grad()
    def update_trace(self, spike_pre: torch.Tensor, spike_post: torch.Tensor, dt: float = 1.0):
        decay_pre_fast = torch.exp(torch.tensor(-dt / self.tau_plus_exc_fast, device=self.device))
        decay_post_fast = torch.exp(torch.tensor(-dt / self.tau_minus_exc_fast, device=self.device))
        
        decay_pre_slow = torch.exp(torch.tensor(-dt / self.tau_plus_exc_slow, device=self.device))
        decay_post_slow = torch.exp(torch.tensor(-dt / self.tau_minus_exc_slow, device=self.device))
        
        decay_pre_inh = torch.exp(torch.tensor(-dt / self.tau_plus_inh, device=self.device))
        decay_post_inh = torch.exp(torch.tensor(-dt / self.tau_minus_inh, device=self.device))
        
        decay_pre_f = torch.where(self.is_excitatory, decay_pre_fast, decay_pre_inh)
        decay_post_f = torch.where(self.is_excitatory, decay_post_fast, decay_post_inh)
        self.trace_pre_fast = self.trace_pre_fast * decay_pre_f + spike_pre
        self.trace_post_fast = self.trace_post_fast * decay_post_f + spike_post
        
        if self.is_excitatory.any():
            exc_mask = self.is_excitatory.unsqueeze(0).expand_as(spike_pre)
            self.trace_pre_slow = torch.where(exc_mask,
                                              self.trace_pre_slow * decay_pre_slow + spike_pre,
                                              self.trace_pre_slow)
            self.trace_post_slow = torch.where(exc_mask,
                                               self.trace_post_slow * decay_post_slow + spike_post,
                                               self.trace_post_slow)
    
    @torch.no_grad()
    def compute_weight_update(self, W: torch.Tensor, spike_pre: torch.Tensor,
                             spike_post: torch.Tensor, lr_scale: float = 1.0) -> torch.Tensor:
        ltp_fast = torch.matmul(spike_post.unsqueeze(2), self.trace_pre_fast.unsqueeze(1)).mean(dim=0)
        ltd_fast = torch.matmul(self.trace_post_fast.unsqueeze(2), spike_pre.unsqueeze(1)).mean(dim=0)
        
        ltp_slow = torch.matmul(spike_post.unsqueeze(2), self.trace_pre_slow.unsqueeze(1)).mean(dim=0)
        ltd_slow = torch.matmul(self.trace_post_slow.unsqueeze(2), spike_pre.unsqueeze(1)).mean(dim=0)
        
        self.learning_step += 1
        adaptive_factor = 1.0 / (1.0 + self.learning_step / 1000.0)
        
        exc_mask = self.is_excitatory
        delta_w_exc = (
            (self.a_plus_exc_fast * ltp_fast - self.a_minus_exc_fast * ltd_fast) * 0.7 +
            (self.a_plus_exc_slow * ltp_slow - self.a_minus_exc_slow * ltd_slow) * 0.3
        )
        
        delta_w_inh = (
            self.a_plus_inh * ltp_fast - self.a_minus_inh * ltd_fast
        )
        
        delta_w = torch.where(exc_mask.unsqueeze(1), delta_w_exc, delta_w_inh)
        delta_w = delta_w * lr_scale * adaptive_factor
        
        W_new = W + delta_w
        
        W_clipped = torch.zeros_like(W_new)
        W_clipped[exc_mask] = torch.clamp(W_new[exc_mask], 0.0, 1.0)
        W_clipped[~exc_mask] = torch.clamp(W_new[~exc_mask], -1.0, 0.0)
        
        self.weight_update_history.append(delta_w.abs().mean().item())
        if len(self.weight_update_history) > 100:
            self.weight_update_history.pop(0)
        
        return W_clipped


class OptimizedTemporalLiquidLayer:
    
    def __init__(self, n_input: int, n_neurons: int = 450, device: str = 'cpu',
                 time_steps: int = 50, n_time_windows: int = 5):
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.device = device
        self.time_steps = time_steps
        self.n_time_windows = n_time_windows
        self.window_size = time_steps // n_time_windows
        
        print(f"  LSM-STDP Liquid Layer (Deep Feature Extraction Layer):")
        print(f"    Input: {n_input}, Neurons: {n_neurons}")
        print(f"    Time windows: {n_time_windows} x {self.window_size} steps")
        
        self.neurons = FullLIF(n_neurons, device)
        self.stdp = EnhancedExcitatoryInhibitorySTDP(n_neurons, device, excitatory_ratio=***)
        
        self.w_input = torch.randn(n_input, n_neurons, device=device) * ***  # Hidden
        mask = torch.rand_like(self.w_input) < ***  # Hidden
        self.w_input = self.w_input * mask.float()
        
        self.w_recurrent = torch.randn(n_neurons, n_neurons, device=device) * ***  # Hidden
        mask = torch.rand_like(self.w_recurrent) < ***  # Hidden
        self.w_recurrent = self.w_recurrent * mask.float()
        self.w_recurrent.fill_diagonal_(0)
        
        self._apply_ei_constraints()
        
        temporal_dim = n_neurons * 2
        spatial_dim = n_neurons * 2
        total_dim = temporal_dim + spatial_dim
        
        print(f"    Signal 1 - Temporal (1st window): {temporal_dim} dims")
        print(f"    Signal 2 - Spatial (global): {spatial_dim} dims")
        print(f"    Total output features: {total_dim} dims")
    
    def _apply_ei_constraints(self):
        exc_mask = self.stdp.is_excitatory
        self.w_recurrent[:, exc_mask] = torch.abs(self.w_recurrent[:, exc_mask])
        self.w_recurrent[:, ~exc_mask] = -torch.abs(self.w_recurrent[:, ~exc_mask])
    
    @torch.no_grad()
    def process(self, input_spikes: torch.Tensor, time_steps: int,
                apply_stdp: bool = True, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        batch_size = input_spikes.shape[0]
        
        self.neurons.reset(batch_size)
        self.stdp.reset(batch_size)
        
        spike_history = torch.zeros(batch_size, time_steps, self.n_neurons, device=self.device)
        voltage_history = torch.zeros(batch_size, time_steps, self.n_neurons, device=self.device)
        
        total_spike_count = torch.zeros(batch_size, self.n_neurons, device=self.device)
        first_spike_time = torch.full((batch_size, self.n_neurons), time_steps,
                                     device=self.device, dtype=torch.float32)
        
        for t in range(time_steps):
            current_input = input_spikes[:, t, :]
            I_ext = torch.matmul(current_input, self.w_input)
            I_rec = torch.matmul(self.neurons.spike, self.w_recurrent)
            I_total = I_ext + I_rec
            
            spike, v = self.neurons.update(I_total, dt=1.0)
            
            spike_history[:, t, :] = spike
            voltage_history[:, t, :] = v
            
            total_spike_count += spike
            
            spike_mask = spike > 0
            first_spike_time = torch.where(
                spike_mask & (first_spike_time == time_steps),
                torch.full_like(first_spike_time, float(t)),
                first_spike_time
            )
            
            self.stdp.update_trace(spike, spike, dt=1.0)
            
            if apply_stdp and training and t % *** == 0 and t > ***:  # Hidden
                self.w_recurrent = self.stdp.compute_weight_update(
                    self.w_recurrent, spike, spike, lr_scale=***  # Hidden
                )
                self._apply_ei_constraints()
        
        temporal_features = self._extract_first_window_temporal(
            spike_history, first_spike_time, batch_size, time_steps
        )
        
        spatial_features = self._extract_spatial_signal(
            total_spike_count, voltage_history, time_steps
        )
        
        features = torch.cat([temporal_features, spatial_features], dim=1)
        
        diagnostics = {
            'avg_spike_freq': (total_spike_count / time_steps).mean().item(),
            'feature_dim': features.shape[1],
            'temporal_dim': temporal_features.shape[1],
            'spatial_dim': spatial_features.shape[1],
            'spike_history': spike_history.clone(),
            'voltage_history': voltage_history.clone(),
            'total_spike_count': total_spike_count.clone(),
        }
        
        return features, diagnostics
    
    def _extract_first_window_temporal(self, spike_history: torch.Tensor,
                                       first_spike_time: torch.Tensor,
                                       batch_size: int, time_steps: int) -> torch.Tensor:
        feature_list = []
        
        start_t = 0
        end_t = self.window_size
        
        window_spikes = spike_history[:, start_t:end_t, :].sum(dim=1) / self.window_size
        feature_list.append(window_spikes)
        
        window_first_spike = torch.full((batch_size, self.n_neurons),
                                       float(self.window_size), device=self.device)
        for t_local in range(self.window_size):
            t_global = start_t + t_local
            if t_global < time_steps:
                spikes = spike_history[:, t_global, :]
                window_first_spike = torch.where(
                    (spikes > 0) & (window_first_spike == self.window_size),
                    torch.full_like(window_first_spike, float(t_local)),
                    window_first_spike
                )
        feature_list.append(window_first_spike / self.window_size)
        
        temporal_features = torch.cat(feature_list, dim=1)
        return temporal_features
    
    def _extract_spatial_signal(self, total_spike_count: torch.Tensor,
                                 voltage_history: torch.Tensor,
                                 time_steps: int) -> torch.Tensor:
        global_spike_freq = total_spike_count / time_steps
        global_voltage = voltage_history.mean(dim=1)
        
        spatial_features = torch.cat([global_spike_freq, global_voltage], dim=1)
        return spatial_features


class ImprovedStableSOM:
    
    def __init__(self, n_neurons: int, input_dim: int, n_classes: int = 10,
                 device: str = 'cpu', normalize_features: bool = True):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.device = device
        self.normalize_features = normalize_features
        
        print(f"  Traditional SOM (Final Clustering Layer - Feature Normalization):")
        print(f"    Neurons: {n_neurons}, Input dim: {input_dim}")
        print(f"    Feature normalization: {normalize_features}")
        
        self.grid_size = int(np.ceil(np.sqrt(n_neurons / 0.866)))
        self.positions = self._init_hexagonal_positions()
        
        self.weights = None
        self.neuron_class_map = {}
        self.neuron_class_confidence = {}
        self.class_to_neurons = None
        
        self.best_weights = None
        self.best_error = float('inf')
        self.patience = ***  # Hidden hyperparameter
        self.patience_counter = 0
        
        self.train_accuracy_history = []
        self.train_features_cache = None
        self.train_labels_cache = None
    
    def _init_hexagonal_positions(self) -> torch.Tensor:
        positions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if len(positions) >= self.n_neurons:
                    break
                x = col * 1.5
                y = row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2
                positions.append([x, y])
        
        positions = positions[:self.n_neurons]
        return torch.tensor(positions, dtype=torch.float32, device=self.device)
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, p=2, dim=1)
    
    @torch.no_grad()
    def _find_bmu_batch(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_features:
            features_norm = F.normalize(features, p=2, dim=1)
            weights_norm = F.normalize(self.weights, p=2, dim=1)
        else:
            features_norm = features
            weights_norm = self.weights
        distances = torch.cdist(features_norm, weights_norm, p=2)
        bmu_distances, bmu_indices = torch.min(distances, dim=1)
        return bmu_indices, bmu_distances
    
    def _pca_initialization(self, features: torch.Tensor):
        print("    t-SNE initialization...")
        from sklearn.preprocessing import StandardScaler
        
        features_cpu = features.cpu().numpy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_cpu)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=***, n_iter=***, verbose=0)  # Hidden
        features_2d = tsne.fit_transform(features_scaled)
        
        features_2d_norm = (features_2d - features_2d.min(axis=0)) / \
                          (features_2d.max(axis=0) - features_2d.min(axis=0) + 1e-8)
        
        self.weights = torch.zeros(self.n_neurons, self.input_dim,
                                   device=self.device, dtype=torch.float32)
        
        pos_np = self.positions.cpu().numpy()
        pos_norm = (pos_np - pos_np.min(axis=0)) / (pos_np.max(axis=0) - pos_np.min(axis=0) + 1e-7)
        
        for i in range(self.n_neurons):
            distances = np.sqrt((features_2d_norm[:, 0] - pos_norm[i, 0])**2 +
                              (features_2d_norm[:, 1] - pos_norm[i, 1])**2)
            nearest_idx = np.argmin(distances)
            self.weights[i] = features[nearest_idx].clone()
    
    @torch.no_grad()
    def train_unsupervised(self, features: torch.Tensor, n_epochs: int = 250,
                          learning_rate: float = None, batch_size: int = None,
                          visualizer=None, train_labels: torch.Tensor = None):
        n_samples = features.shape[0]
        
        if self.normalize_features:
            print("    Applying L2 feature normalization...")
            features = self._normalize_features(features)
        
        self._pca_initialization(features)
        
        self.train_features_cache = features.clone()
        if train_labels is not None:
            self.train_labels_cache = train_labels.clone()
        
        if train_labels is not None:
            print("\n  Building initial label mapping for accuracy tracking...")
            self._build_initial_label_mapping(features, train_labels)
        
        print(f"\n  Traditional SOM Training: {n_epochs} epochs")
        
        pbar = tqdm(range(n_epochs), desc="    [Training]")
        
        for epoch in pbar:
            indices = torch.randperm(n_samples, device=self.device)
            progress = epoch / n_epochs
            
            current_lr = *** * np.exp(-*** * progress)  # Hidden
            
            sigma_init = ***   # Hidden
            sigma_final = ***  # Hidden
            current_sigma = sigma_final + (sigma_init - sigma_final) * np.exp(-*** * progress)  # Hidden
            
            epoch_errors = []
            
            n_batches = (n_samples + batch_size - 1) // batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_features = features[batch_indices]
                
                bmu_indices, errors = self._find_bmu_batch(batch_features)
                epoch_errors.extend(errors.cpu().tolist())
                
                for i, idx in enumerate(batch_indices):
                    x = features[idx]
                    bmu_idx = bmu_indices[i].item()
                    bmu_pos = self.positions[bmu_idx]
                    distances = torch.norm(self.positions - bmu_pos, dim=1)
                    
                    exp_input = -distances**2 / (2 * current_sigma**2)
                    exp_input = torch.clamp(exp_input, min=-50, max=0)
                    h = torch.exp(exp_input)
                    
                    delta = (x - self.weights) * h.unsqueeze(1) * current_lr
                    delta = torch.clamp(delta, min=-2.5, max=2.5)
                    self.weights += delta
            
            avg_error = np.mean(epoch_errors)
            
            if avg_error < self.best_error:
                self.best_error = avg_error
                self.best_weights = self.weights.clone()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\n    Early stopping at epoch {epoch}")
                self.weights = self.best_weights
                break
            
            if (epoch + 1) % 2 == 0 and self.train_labels_cache is not None:
                self._update_label_mapping(features, train_labels)
                train_pred = self.predict(self.train_features_cache)
                train_acc = (train_pred == self.train_labels_cache.cpu().numpy()).mean()
                self.train_accuracy_history.append(train_acc)
            
            if visualizer is not None:
                sample_indices = torch.randperm(n_samples, device=self.device)[:500]
                sample_features = features[sample_indices]
                bmu_test, _ = self._find_bmu_batch(sample_features)
    
    def _build_initial_label_mapping(self, features: torch.Tensor, labels: torch.Tensor):
        bmu_indices, _ = self._find_bmu_batch(features)
        
        for i in range(self.n_neurons):
            self.neuron_class_map[i] = {}
        
        for i, bmu_idx in enumerate(bmu_indices):
            label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
            if label not in self.neuron_class_map[bmu_idx.item()]:
                self.neuron_class_map[bmu_idx.item()][label] = 0
            self.neuron_class_map[bmu_idx.item()][label] += 1
    
    def _update_label_mapping(self, features: torch.Tensor, labels: torch.Tensor):
        bmu_indices, _ = self._find_bmu_batch(features)
        
        for i, bmu_idx in enumerate(bmu_indices):
            label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
            if label not in self.neuron_class_map[bmu_idx.item()]:
                self.neuron_class_map[bmu_idx.item()][label] = 0
            self.neuron_class_map[bmu_idx.item()][label] += 1
    
    def assign_labels(self, features: torch.Tensor, labels: torch.Tensor):
        print("\n  Assigning class labels to neurons...")
        
        bmu_indices, _ = self._find_bmu_batch(features)
        
        neuron_class_count = {i: {} for i in range(self.n_neurons)}
        
        for i, bmu_idx in enumerate(bmu_indices):
            label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
            if label not in neuron_class_count[bmu_idx.item()]:
                neuron_class_count[bmu_idx.item()][label] = 0
            neuron_class_count[bmu_idx.item()][label] += 1
        
        for neuron_idx in range(self.n_neurons):
            if neuron_class_count[neuron_idx]:
                max_class = max(neuron_class_count[neuron_idx].items(), key=lambda x: x[1])
                self.neuron_class_map[neuron_idx] = max_class[0]
                
                total_count = sum(neuron_class_count[neuron_idx].values())
                self.neuron_class_confidence[neuron_idx] = max_class[1] / total_count
            else:
                self.neuron_class_map[neuron_idx] = 0
                self.neuron_class_confidence[neuron_idx] = 0.0
        
        assigned = len([n for n in range(self.n_neurons) if neuron_class_count[n]])
        print(f"    Neurons with assigned labels: {assigned}/{self.n_neurons}")
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=1)
        
        bmu_indices, _ = self._find_bmu_batch(features)
        
        predictions = np.array([self.neuron_class_map.get(idx.item(), -1) 
                               for idx in bmu_indices])
        
        return predictions


class DualSOMSystem:
    
    def __init__(self, n_spiking_som: int, n_lsm: int, n_traditional_som: int,
                 device: str = 'cpu', time_steps: int = 50,
                 confidence_threshold: float = None, normalize_features: bool = True):
        self.device = device
        self.time_steps = time_steps
        self.confidence_threshold = confidence_threshold
        self.normalize_features = normalize_features
        
        self.encoder = EnhancedHybridSpikeEncoder(time_steps=time_steps, device=device)
        
        self.spiking_som = SpikingSOMLayer(n_input=784, n_competitive=n_spiking_som, device=device)
        
        self.liquid_layer = OptimizedTemporalLiquidLayer(
            n_input=784, n_neurons=n_lsm, device=device, time_steps=time_steps
        )
        
        self.random_projector = RandomProjector(
            input_dim=784, output_dim=n_lsm * 4, device=device
        )
        
        self.traditional_som = None
        self.n_traditional_som = n_traditional_som
        
        self.processing_stats = {
            'total_samples': 0,
            'easy_samples': 0,
            'hard_samples': 0,
            'confidence_scores': [],
            'easy_ratio': 0.0,
            'hard_ratio': 0.0,
            'avg_confidence': 0.0
        }
        
        print(f"  Dual SOM System initialized:")
        print(f"    Spiking SOM neurons: {n_spiking_som}")
        print(f"    LSM neurons: {n_lsm}")
        print(f"    Traditional SOM neurons: {n_traditional_som}")
        print(f"    Confidence threshold: {confidence_threshold}")
    
    def process_batch(self, images: torch.Tensor, training: bool = True,
                     labels: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = images.shape[0]
        images = images.to(self.device)
        
        spike_trains, spike_info = self.encoder.encode(images)
        
        features_list = []
        confidence_scores = []
        memory_vectors = []
        easy_count = 0
        hard_count = 0
        
        for i in range(batch_size):
            spike_train = spike_trains[i]
            
            bmu_idx, confidence, som_output, memory_vector = self.spiking_som.process_ttfs_spikes(
                spike_train, training=training,
                label=labels[i].item() if labels is not None else None
            )
            
            confidence_scores.append(confidence)
            memory_vectors.append(memory_vector)
            
            if confidence >= self.confidence_threshold:
                easy_count += 1
                mem_features = self.random_projector.project(memory_vector.unsqueeze(0))
                if self.normalize_features:
                    mem_features = F.normalize(mem_features, p=2, dim=1)
                features_list.append(mem_features.squeeze(0))
            else:
                hard_count += 1
                lsm_features, lsm_diag = self.liquid_layer.process(
                    spike_train.unsqueeze(0), self.time_steps,
                    apply_stdp=training, training=training
                )
                if self.normalize_features:
                    lsm_features = F.normalize(lsm_features, p=2, dim=1)
                features_list.append(lsm_features.squeeze(0))
        
        features = torch.stack(features_list)
        
        self.processing_stats['total_samples'] += batch_size
        self.processing_stats['easy_samples'] += easy_count
        self.processing_stats['hard_samples'] += hard_count
        self.processing_stats['confidence_scores'].extend(confidence_scores)
        
        total = self.processing_stats['total_samples']
        self.processing_stats['easy_ratio'] = self.processing_stats['easy_samples'] / total
        self.processing_stats['hard_ratio'] = self.processing_stats['hard_samples'] / total
        self.processing_stats['avg_confidence'] = np.mean(self.processing_stats['confidence_scores'])
        
        batch_info = {
            'confidence_scores': confidence_scores,
            'memory_vectors': memory_vectors,
            'spike_train': spike_trains,
            'easy_count': easy_count,
            'hard_count': hard_count,
        }
        
        return features, batch_info
    
    def get_processing_stats(self) -> Dict:
        return self.processing_stats.copy()
