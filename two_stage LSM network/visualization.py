from config import *
from core import SpikingSOMLayer, OptimizedTemporalLiquidLayer

class EnhancedVisualizationAndMonitoring:
    
    def __init__(self, save_dir='./dual_som_optimized_visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        setup_matplotlib_fonts()
        
        self.training_history = {
            'epoch': [],
            'quantization_error': [],
            'neuron_activation': [],
            'feature_variance': [],
            'learning_rate': [],
            'timestamp': []
        }
        
        self.performance_metrics = {
            'accuracy': [],
            'f1_score': [],
            'training_time': [],
            'memory_usage': []
        }
        
        self.dual_som_stats = {
            'spiking_som_confidence': [],
            'easy_samples_ratio': [],
            'hard_samples_ratio': [],
            'lsm_processed_ratio': []
        }
        
        print(f"  Enhanced Visualization & Monitoring initialized: {save_dir}")
    
    def record_training_step(self, epoch, quantization_error, neuron_activation=None,
                            feature_variance=None, learning_rate=None):
        self.training_history['epoch'].append(epoch)
        self.training_history['quantization_error'].append(quantization_error)
        self.training_history['neuron_activation'].append(
            neuron_activation if neuron_activation is not None else 0
        )
        self.training_history['feature_variance'].append(
            feature_variance if feature_variance is not None else 0
        )
        self.training_history['learning_rate'].append(
            learning_rate if learning_rate is not None else 0
        )
        self.training_history['timestamp'].append(time.time())
    
    def record_dual_som_stats(self, spiking_confidences: List[float], 
                             easy_ratio: float, hard_ratio: float, lsm_ratio: float):
        if len(spiking_confidences) > 0:
            confidences_cpu = []
            for conf in spiking_confidences:
                if isinstance(conf, torch.Tensor):
                    confidences_cpu.append(conf.cpu().item())
                else:
                    confidences_cpu.append(float(conf))
            self.dual_som_stats['spiking_som_confidence'].append(np.mean(confidences_cpu))
        self.dual_som_stats['easy_samples_ratio'].append(float(easy_ratio))
        self.dual_som_stats['hard_samples_ratio'].append(float(hard_ratio))
        self.dual_som_stats['lsm_processed_ratio'].append(float(lsm_ratio))
    
    def record_performance(self, accuracy, f1_score, training_time, memory_usage=0):
        self.performance_metrics['accuracy'].append(accuracy)
        self.performance_metrics['f1_score'].append(f1_score)
        self.performance_metrics['training_time'].append(training_time)
        self.performance_metrics['memory_usage'].append(memory_usage)
    
    def plot_memory_vectors_standalone(self, images: torch.Tensor, memory_vectors: List[torch.Tensor],
                                       labels: torch.Tensor, sample_indices: List[int],
                                       confidence_scores: List[float] = None,
                                       title_prefix="Memory_Vector_Standalone"):
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import gaussian_kde
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib import cm
        
        print(f"\n  Plotting standalone memory vector visualizations...")
        
        standalone_dir = os.path.join(self.save_dir, "memory_vectors_standalone")
        os.makedirs(standalone_dir, exist_ok=True)
        
        memory_colors = ['#0d0887', '#3b049a', '#7201a8', '#a82296', '#cb4679',
                        '#e56b5d', '#f89441', '#fdc328', '#f0f921']
        memory_cmap = LinearSegmentedColormap.from_list('memory_pro', memory_colors, N=256)
        
        for i, idx in enumerate(sample_indices):
            fig = plt.figure(figsize=(20, 16))
            fig.patch.set_facecolor('white')
            
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
            
            if isinstance(images, torch.Tensor):
                img = images[idx, 0].cpu().numpy()
            else:
                img = images[idx, 0]
            
            mem_vec = memory_vectors[idx]
            if isinstance(mem_vec, torch.Tensor):
                mem_vec = mem_vec.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                label = labels[idx].cpu().item()
            else:
                label = labels[idx]
            conf = confidence_scores[idx] if confidence_scores and idx < len(confidence_scores) else 0.0
            
            mem_img = mem_vec.reshape(28, 28)
            
            mem_min, mem_max = mem_img.min(), mem_img.max()
            if mem_max > mem_min:
                mem_img_norm = (mem_img - mem_min) / (mem_max - mem_min)
            else:
                mem_img_norm = mem_img
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img, cmap='gray', aspect='equal')
            ax1.set_title(f'Original Image\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)})', fontsize=12, fontweight='bold', pad=10)
            ax1.axis('off')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(mem_img_norm, cmap=memory_cmap, aspect='equal', 
                            vmin=0, vmax=1, interpolation='bilinear')
            ax2.set_title('Memory Vector Heatmap\n(Information Value Encoding)', 
                         fontsize=12, fontweight='bold', pad=10)
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, 
                                label='Information Value')
            cbar2.ax.tick_params(labelsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            ax3 = fig.add_subplot(gs[0, 2], projection='3d')
            X = np.arange(28)
            Y = np.arange(28)
            X, Y = np.meshgrid(X, Y)
            
            surf = ax3.plot_surface(X, Y, mem_img_norm, cmap=memory_cmap,
                                   linewidth=0, antialiased=True, alpha=0.9)
            ax3.set_xlabel('X Position', fontsize=9, labelpad=5)
            ax3.set_ylabel('Y Position', fontsize=9, labelpad=5)
            ax3.set_zlabel('Info Value', fontsize=9, labelpad=5)
            ax3.set_title('3D Information Landscape', fontsize=12, fontweight='bold', pad=10)
            ax3.view_init(elev=30, azim=45)
            ax3.tick_params(labelsize=8)
            fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, label='Info Value')
            
            ax4 = fig.add_subplot(gs[1, 0])
            row_profiles = mem_img_norm.mean(axis=1)
            x_pos = np.arange(28)
            ax4.fill_between(x_pos, row_profiles, alpha=0.6, color='#3498db')
            ax4.plot(x_pos, row_profiles, 'b-', linewidth=2, color='#2980b9')
            ax4.set_xlabel('Row Index', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Average Information Value', fontsize=10, fontweight='bold')
            ax4.set_title('Row-wise Profile', fontsize=12, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.set_xlim(0, 27)
            ax4.set_ylim(0, 1)
            for spine in ax4.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax5 = fig.add_subplot(gs[1, 1])
            col_profiles = mem_img_norm.mean(axis=0)
            ax5.fill_between(x_pos, col_profiles, alpha=0.6, color='#e74c3c')
            ax5.plot(x_pos, col_profiles, 'r-', linewidth=2, color='#c0392b')
            ax5.set_xlabel('Column Index', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Average Information Value', fontsize=10, fontweight='bold')
            ax5.set_title('Column-wise Profile', fontsize=12, fontweight='bold', pad=10)
            ax5.grid(True, alpha=0.3, linestyle='--')
            ax5.set_xlim(0, 27)
            ax5.set_ylim(0, 1)
            for spine in ax5.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax6 = fig.add_subplot(gs[1, 2])
            non_zero_mask = mem_vec > 0
            non_zero_values = mem_vec[non_zero_mask]
            
            if len(non_zero_values) > 0:
                n, bins, patches = ax6.hist(non_zero_values, bins=40, density=True, 
                                           alpha=0.6, color='#9b59b6', edgecolor='white')
                
                if len(non_zero_values) > 5:
                    kde_x = np.linspace(non_zero_values.min(), non_zero_values.max(), 200)
                    kde = gaussian_kde(non_zero_values)
                    ax6.plot(kde_x, kde(kde_x), 'r-', linewidth=2.5, label='KDE')
                
                mean_val = non_zero_values.mean()
                median_val = np.median(non_zero_values)
                ax6.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_val:.3f}')
                ax6.axvline(median_val, color='#27ae60', linestyle=':', linewidth=2,
                           label=f'Median: {median_val:.3f}')
            
            ax6.set_xlabel('Information Value', fontsize=10, fontweight='bold')
            ax6.set_ylabel('Density', fontsize=10, fontweight='bold')
            ax6.set_title('Distribution Analysis', fontsize=12, fontweight='bold', pad=10)
            ax6.legend(fontsize=9, loc='upper right')
            ax6.grid(True, alpha=0.3, linestyle='--')
            for spine in ax6.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax7 = fig.add_subplot(gs[2, 0])
            binary_activation = (mem_img > 0).astype(float)
            ax7.imshow(binary_activation, cmap='Greys', aspect='equal')
            ax7.set_title(f'Activation Map\nActive: {np.sum(binary_activation):.0f}/784 neurons', 
                         fontsize=12, fontweight='bold', pad=10)
            ax7.axis('off')
            for spine in ax7.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            ax8 = fig.add_subplot(gs[2, 1])
            levels = np.linspace(0, 1, 21)
            contour = ax8.contourf(X, Y, mem_img_norm, levels=levels, cmap=memory_cmap)
            ax8.contour(X, Y, mem_img_norm, levels=10, colors='white', linewidths=0.5, alpha=0.5)
            ax8.set_title('Information Intensity Contour', fontsize=12, fontweight='bold', pad=10)
            ax8.set_xlabel('X Position', fontsize=10, fontweight='bold')
            ax8.set_ylabel('Y Position', fontsize=10, fontweight='bold')
            plt.colorbar(contour, ax=ax8, fraction=0.046, pad=0.04)
            ax8.set_aspect('equal')
            for spine in ax8.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.axis('off')
            
            active_neurons = np.sum(mem_vec > 0)
            total_info = mem_vec.sum()
            mean_info = non_zero_values.mean() if len(non_zero_values) > 0 else 0
            std_info = non_zero_values.std() if len(non_zero_values) > 0 else 0
            max_info = mem_vec.max()
            sparsity = 1.0 - active_neurons / len(mem_vec)
            entropy = -np.sum(mem_vec[mem_vec > 0] * np.log(mem_vec[mem_vec > 0] + 1e-10))
            
            stats_text = f"""
MEMORY VECTOR STATISTICS
============================
  Sample Index:        {idx:>15}
  Class Label:         {label:>15}
  Confidence:          {conf:>14.4f}
----------------------------
  ACTIVE NEURON STATISTICS
  Active Neurons:      {active_neurons:>10}/{len(mem_vec)}
  Activation Rate:     {active_neurons/len(mem_vec)*100:>14.2f}%
  Sparsity:            {sparsity*100:>14.2f}%
----------------------------
  INFORMATION VALUE STATISTICS
  Total Information:   {total_info:>14.4f}
  Mean (active):       {mean_info:>14.4f}
  Std Dev (active):    {std_info:>14.4f}
  Max Value:           {max_info:>14.4f}
  Information Entropy: {entropy:>14.4f}
----------------------------
  TEMPORAL ENCODING INFO
  Time Window T:       {15:>15}
  Decay Function:      1 - t/T
============================
            """
            
            ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                             edgecolor='#333333', linewidth=2))
            
            fig.suptitle(f'Spiking SOM Memory Vector - Standalone Visualization\n'
                        f'Sample {idx} | Class {label} ({FASHION_MNIST_CLASSES.get(label, label)}) | Confidence: {conf:.4f}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            save_path = os.path.join(standalone_dir, f'{title_prefix}_{idx:02d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"    Saved {len(sample_indices)} standalone memory vector images to: {standalone_dir}")
    
    def plot_lsm_feature_vectors_standalone(self, images: torch.Tensor, lsm_features: torch.Tensor,
                                            spike_trains: torch.Tensor, labels: torch.Tensor,
                                            sample_indices: List[int],
                                            confidence_scores: List[float] = None,
                                            title_prefix="LSM_Feature_Standalone"):
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import gaussian_kde
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib import cm
        
        print(f"\n  Plotting standalone LSM feature vector visualizations...")
        
        standalone_dir = os.path.join(self.save_dir, "lsm_features_standalone")
        os.makedirs(standalone_dir, exist_ok=True)
        
        lsm_colors = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560',
                     '#f38181', '#fce38a', '#eaffd0', '#95e1d3']
        lsm_cmap = LinearSegmentedColormap.from_list('lsm_pro', lsm_colors, N=256)
        
        feature_dim = lsm_features.shape[1]
        temporal_dim = feature_dim // 2
        
        for i, idx in enumerate(sample_indices):
            fig = plt.figure(figsize=(24, 18))
            fig.patch.set_facecolor('white')
            
            gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
            
            if isinstance(images, torch.Tensor):
                img = images[idx, 0].cpu().numpy()
            else:
                img = images[idx, 0]
            
            feat = lsm_features[idx]
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            
            spikes = spike_trains[idx]
            if isinstance(spikes, torch.Tensor):
                spikes = spikes.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                label = labels[idx].cpu().item()
            else:
                label = labels[idx]
            conf = confidence_scores[i] if confidence_scores and i < len(confidence_scores) else 0.0
            
            temporal_feat = feat[:temporal_dim]
            spatial_feat = feat[temporal_dim:]
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img, cmap='gray', aspect='equal')
            ax1.set_title(f'Original Image\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)})', fontsize=12, fontweight='bold', pad=10)
            ax1.axis('off')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            ax2 = fig.add_subplot(gs[0, 1])
            feat_2d = feat.reshape(1, -1)
            im2 = ax2.imshow(feat_2d, cmap=lsm_cmap, aspect='auto', interpolation='nearest')
            ax2.axvline(temporal_dim, color='white', linestyle='--', linewidth=2, label='Temporal|Spatial')
            ax2.set_title(f'LSM Feature Vector ({feature_dim} dims)\nTemporal ({temporal_dim}) | Spatial ({temporal_dim})', 
                         fontsize=12, fontweight='bold', pad=10)
            ax2.set_xlabel('Feature Dimension', fontsize=10, fontweight='bold')
            ax2.set_yticks([])
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Feature Value')
            cbar2.ax.tick_params(labelsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax3 = fig.add_subplot(gs[0, 2])
            active_neurons = np.where(spikes.sum(axis=0) > 0)[0]
            if len(active_neurons) > 0:
                if len(active_neurons) > 300:
                    active_neurons = active_neurons[:300]
                
                spike_times = []
                neuron_ids = []
                for t in range(spikes.shape[0]):
                    for n in active_neurons:
                        if spikes[t, n] > 0:
                            spike_times.append(t)
                            neuron_ids.append(n)
                
                if len(spike_times) > 0:
                    ax3.scatter(spike_times, neuron_ids, s=2, c='#e74c3c', alpha=0.7, marker='|')
            
            ax3.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Neuron ID', fontsize=10, fontweight='bold')
            ax3.set_title('Spike Raster Plot', fontsize=12, fontweight='bold', pad=10)
            ax3.grid(True, alpha=0.3, linestyle='--')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax4 = fig.add_subplot(gs[1, 0])
            x_temporal = np.arange(len(temporal_feat))
            colors_temporal = ['#3498db' if v >= 0 else '#e74c3c' for v in temporal_feat]
            ax4.bar(x_temporal, temporal_feat, color=colors_temporal, alpha=0.7, width=1.0)
            ax4.axhline(0, color='black', linewidth=0.8)
            ax4.axhline(temporal_feat.mean(), color='#27ae60', linestyle='--', linewidth=2, 
                       label=f'Mean: {temporal_feat.mean():.4f}')
            ax4.set_xlabel('Feature Index', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Feature Value', fontsize=10, fontweight='bold')
            ax4.set_title(f'Temporal Features ({temporal_dim} dims)', fontsize=12, fontweight='bold', pad=10)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, linestyle='--')
            for spine in ax4.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax5 = fig.add_subplot(gs[1, 1])
            x_spatial = np.arange(len(spatial_feat))
            colors_spatial = ['#9b59b6' if v >= 0 else '#e67e22' for v in spatial_feat]
            ax5.bar(x_spatial, spatial_feat, color=colors_spatial, alpha=0.7, width=1.0)
            ax5.axhline(0, color='black', linewidth=0.8)
            ax5.axhline(spatial_feat.mean(), color='#27ae60', linestyle='--', linewidth=2,
                       label=f'Mean: {spatial_feat.mean():.4f}')
            ax5.set_xlabel('Feature Index', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Feature Value', fontsize=10, fontweight='bold')
            ax5.set_title(f'Spatial Features ({temporal_dim} dims)', fontsize=12, fontweight='bold', pad=10)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, linestyle='--')
            for spine in ax5.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.hist(temporal_feat, bins=50, alpha=0.6, color='#3498db', label='Temporal', density=True)
            ax6.hist(spatial_feat, bins=50, alpha=0.6, color='#9b59b6', label='Spatial', density=True)
            ax6.axvline(temporal_feat.mean(), color='#2980b9', linestyle='--', linewidth=2)
            ax6.axvline(spatial_feat.mean(), color='#8e44ad', linestyle='--', linewidth=2)
            ax6.set_xlabel('Feature Value', fontsize=10, fontweight='bold')
            ax6.set_ylabel('Density', fontsize=10, fontweight='bold')
            ax6.set_title('Feature Distribution Comparison', fontsize=12, fontweight='bold', pad=10)
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3, linestyle='--')
            for spine in ax6.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax7 = fig.add_subplot(gs[2, 0])
            cumulative_energy = np.cumsum(feat**2) / np.sum(feat**2)
            ax7.plot(np.arange(len(cumulative_energy)), cumulative_energy, 'g-', linewidth=2)
            ax7.axhline(0.9, color='red', linestyle='--', linewidth=1.5, label='90% Energy')
            ax7.axvline(np.argmax(cumulative_energy >= 0.9), color='orange', linestyle=':', linewidth=1.5)
            ax7.fill_between(np.arange(len(cumulative_energy)), cumulative_energy, alpha=0.3, color='green')
            ax7.set_xlabel('Feature Index', fontsize=10, fontweight='bold')
            ax7.set_ylabel('Cumulative Energy Ratio', fontsize=10, fontweight='bold')
            ax7.set_title('Cumulative Feature Energy', fontsize=12, fontweight='bold', pad=10)
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3, linestyle='--')
            ax7.set_ylim(0, 1.05)
            for spine in ax7.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax8 = fig.add_subplot(gs[2, 1])
            n_cols = int(np.sqrt(len(feat)))
            n_rows = int(np.ceil(len(feat) / n_cols))
            feat_padded = np.zeros(n_rows * n_cols)
            feat_padded[:len(feat)] = feat
            feat_2d_grid = feat_padded.reshape(n_rows, n_cols)
            
            im8 = ax8.imshow(feat_2d_grid, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
            ax8.set_title('Feature 2D Grid Visualization', fontsize=12, fontweight='bold', pad=10)
            ax8.set_xlabel('Grid Column', fontsize=10, fontweight='bold')
            ax8.set_ylabel('Grid Row', fontsize=10, fontweight='bold')
            plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            for spine in ax8.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax9 = fig.add_subplot(gs[2, 2])
            spike_activity = spikes.sum(axis=0).reshape(28, 28)
            im9 = ax9.imshow(spike_activity, cmap='hot', aspect='equal', interpolation='bilinear')
            ax9.set_title(f'Spike Activity Heatmap\nTotal Spikes: {spikes.sum():.0f}', 
                         fontsize=12, fontweight='bold', pad=10)
            ax9.axis('off')
            plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04, label='Spike Count')
            for spine in ax9.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            ax10 = fig.add_subplot(gs[3, :])
            ax10.axis('off')
            
            feat_mean = feat.mean()
            feat_std = feat.std()
            feat_min = feat.min()
            feat_max = feat.max()
            feat_norm = np.linalg.norm(feat)
            feat_sparsity = 1.0 - np.sum(np.abs(feat) > 1e-6) / len(feat)
            
            temp_mean = temporal_feat.mean()
            temp_std = temporal_feat.std()
            spatial_mean = spatial_feat.mean()
            spatial_std = spatial_feat.std()
            
            n_active_neurons = np.sum(spikes.sum(axis=0) > 0)
            avg_spike_rate = spikes.sum() / spikes.shape[0]
            
            stats_text = f"""
LSM FEATURE VECTOR - STANDALONE VISUALIZATION STATISTICS
============================================================
  Sample Index: {idx:>8}  |  Class Label: {label:>6}  |  Confidence: {conf:>8.4f}  |  Feature Dimension: {feature_dim:>6}
------------------------------------------------------------
  OVERALL FEATURE STATISTICS                           |  TEMPORAL FEATURES                    |  SPATIAL FEATURES
------------------------------------------------------------
  Mean:              {feat_mean:>12.6f}                   |  Mean:          {temp_mean:>12.6f}           |  Mean:          {spatial_mean:>12.6f}
  Std Dev:           {feat_std:>12.6f}                   |  Std Dev:       {temp_std:>12.6f}           |  Std Dev:       {spatial_std:>12.6f}
  Min:               {feat_min:>12.6f}                   |  Min:           {temporal_feat.min():>12.6f}           |  Min:           {spatial_feat.min():>12.6f}
  Max:               {feat_max:>12.6f}                   |  Max:           {temporal_feat.max():>12.6f}           |  Max:           {spatial_feat.max():>12.6f}
  L2 Norm:           {feat_norm:>12.6f}                   |  L2 Norm:       {np.linalg.norm(temporal_feat):>12.6f}           |  L2 Norm:       {np.linalg.norm(spatial_feat):>12.6f}
  Sparsity:          {feat_sparsity*100:>11.2f}%                   |  Range:         [{temporal_feat.min():>6.3f}, {temporal_feat.max():>6.3f}]        |  Range:         [{spatial_feat.min():>6.3f}, {spatial_feat.max():>6.3f}]
------------------------------------------------------------
  SPIKE ACTIVITY STATISTICS
------------------------------------------------------------
  Total Spikes:      {spikes.sum():>8.0f}  |  Active Neurons: {n_active_neurons:>6}/{spikes.shape[1]}  |  Avg Spike Rate: {avg_spike_rate:>8.4f}  |  Time Steps: {spikes.shape[0]:>4}
============================================================
            """
            
            ax10.text(0.02, 0.95, stats_text, transform=ax10.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                              edgecolor='#333333', linewidth=2))
            
            fig.suptitle(f'LSM-STDP Feature Vector - Standalone Visualization (Hard Sample)\n'
                        f'Sample {idx} | Class {label} ({FASHION_MNIST_CLASSES.get(label, label)}) | Confidence: {conf:.4f}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            save_path = os.path.join(standalone_dir, f'{title_prefix}_{idx:02d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"    Saved {len(sample_indices)} standalone LSM feature images to: {standalone_dir}")

    def plot_memory_vectors_individual(self, images: torch.Tensor, memory_vectors: List[torch.Tensor], 
                                       labels: torch.Tensor, sample_indices: List[int],
                                       confidence_scores: List[float] = None,
                                       title_prefix="Memory_Vector_Sample"):
        print(f"\n  Plotting memory vector visualizations (individual images)...")
        
        individual_dir = os.path.join(self.save_dir, "memory_vectors_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, idx in enumerate(sample_indices):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.patch.set_facecolor('white')
            
            if isinstance(images, torch.Tensor):
                img = images[idx, 0].cpu().numpy()
            else:
                img = images[idx, 0]
            
            mem_vec = memory_vectors[idx]
            if isinstance(mem_vec, torch.Tensor):
                mem_vec = mem_vec.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                label = labels[idx].cpu().item()
            else:
                label = labels[idx]
            
            conf = confidence_scores[idx] if confidence_scores and idx < len(confidence_scores) else 0.0
            
            mem_img = mem_vec.reshape(28, 28)
            
            mem_min, mem_max = mem_img.min(), mem_img.max()
            if mem_max > mem_min:
                mem_img_norm = (mem_img - mem_min) / (mem_max - mem_min)
            else:
                mem_img_norm = mem_img
            
            ax1 = axes[0]
            im1 = ax1.imshow(img, cmap='gray', aspect='auto')
            ax1.set_title(f'Original Image\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)})', fontsize=12, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            ax2 = axes[1]
            im2 = ax2.imshow(mem_img_norm, cmap='hot', aspect='auto', vmin=0, vmax=1)
            ax2.set_title(f'Memory Vector\n(Information Value Encoding)', fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            ax3 = axes[2]
            overlay = np.zeros((28, 28, 3))
            overlay[:, :, 0] = img
            overlay[:, :, 1] = mem_img_norm
            overlay = np.clip(overlay, 0, 1)
            ax3.imshow(overlay)
            ax3.set_title('Overlay\n(Red: Original, Green: Memory)', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            ax4 = axes[3]
            non_zero_mask = mem_vec > 0
            non_zero_values = mem_vec[non_zero_mask]
            if len(non_zero_values) > 0:
                ax4.hist(non_zero_values, bins=30, color='coral', alpha=0.7, edgecolor='black')
                ax4.axvline(non_zero_values.mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {non_zero_values.mean():.3f}')
            ax4.set_xlabel('Information Value', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax4.set_title(f'Memory Vector Distribution\nActive Neurons: {np.sum(non_zero_mask)}/{len(mem_vec)}', 
                         fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            fig.suptitle(f'Spiking SOM Memory Vector - Sample {idx}\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)}), Confidence: {conf:.3f}', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            save_path = os.path.join(individual_dir, f'{title_prefix}_{idx:02d}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
        print(f"    Saved {len(sample_indices)} individual memory vector images to: {individual_dir}")
    
    def plot_lsm_feature_vectors_individual(self, images: torch.Tensor, lsm_features: torch.Tensor,
                                            spike_trains: torch.Tensor, labels: torch.Tensor, 
                                            sample_indices: List[int],
                                            confidence_scores: List[float] = None,
                                            title_prefix="LSM_Feature_Sample"):
        print(f"\n  Plotting LSM feature vector visualizations (individual images)...")
        
        individual_dir = os.path.join(self.save_dir, "lsm_features_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        feature_dim = lsm_features.shape[1]
        
        for i, idx in enumerate(sample_indices):
            fig = plt.figure(figsize=(20, 10))
            fig.patch.set_facecolor('white')
            
            if isinstance(images, torch.Tensor):
                img = images[idx, 0].cpu().numpy()
            else:
                img = images[idx, 0]
            
            feat = lsm_features[idx]
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            
            spikes = spike_trains[idx]
            if isinstance(spikes, torch.Tensor):
                spikes = spikes.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                label = labels[idx].cpu().item()
            else:
                label = labels[idx]
            
            conf = confidence_scores[idx] if confidence_scores and idx < len(confidence_scores) else 0.0
            
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img, cmap='gray')
            ax1.set_title(f'Original Image\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)})', fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            active_neurons = np.where(spikes.sum(axis=0) > 0)[0]
            if len(active_neurons) > 0:
                if len(active_neurons) > 200:
                    active_neurons = active_neurons[:200]
                
                spike_times = []
                neuron_ids = []
                for t in range(spikes.shape[0]):
                    for n in active_neurons:
                        if spikes[t, n] > 0:
                            spike_times.append(t)
                            neuron_ids.append(n)
                
                if len(spike_times) > 0:
                    ax2.scatter(spike_times, neuron_ids, s=1, c='black', alpha=0.5)
            ax2.set_xlabel('Time Step', fontsize=9)
            ax2.set_ylabel('Neuron ID', fontsize=9)
            ax2.set_title('Spike Raster Plot', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(gs[0, 2])
            feat_2d = feat.reshape(-1, 1).T
            im3 = ax3.imshow(feat_2d, cmap='RdBu_r', aspect='auto', interpolation='nearest')
            ax3.set_xlabel('Feature Dimension', fontsize=9)
            ax3.set_ylabel('Sample', fontsize=9)
            ax3.set_title(f'LSM Feature Vector\nDimension: {feature_dim}', fontsize=11, fontweight='bold')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.hist(feat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.axvline(feat.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {feat.mean():.4f}')
            ax4.axvline(0, color='green', linestyle=':', linewidth=1)
            ax4.set_xlabel('Feature Value', fontsize=9)
            ax4.set_ylabel('Frequency', fontsize=9)
            ax4.set_title('Feature Distribution', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(gs[1, 1])
            temporal_dim = feature_dim // 2
            temporal_feat = feat[:temporal_dim]
            spatial_feat = feat[temporal_dim:]
            
            x_temporal = np.arange(len(temporal_feat))
            x_spatial = np.arange(len(spatial_feat)) + len(temporal_feat)
            
            ax5.bar(x_temporal, temporal_feat, alpha=0.7, label='Temporal Features', 
                   color='coral', width=1.0)
            ax5.bar(x_spatial, spatial_feat, alpha=0.7, label='Spatial Features', 
                   color='steelblue', width=1.0)
            ax5.axhline(0, color='black', linewidth=0.5)
            ax5.axvline(temporal_dim, color='red', linestyle='--', linewidth=1, 
                       label=f'Temporal/Spatial Boundary ({temporal_dim})')
            ax5.set_xlabel('Feature Dimension Index', fontsize=9)
            ax5.set_ylabel('Feature Value', fontsize=9)
            ax5.set_title('LSM Feature Vector Segmentation', fontsize=11, fontweight='bold')
            ax5.legend(fontsize=8, loc='upper right')
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[1, 2])
            stats_text = f"""
            Sample #{idx} Statistics:
            -------------------------
            Label: {label}
            Confidence: {conf:.4f}
            Feature Dimension: {feature_dim}
            -------------------------
            Temporal Part:
              - Mean: {temporal_feat.mean():.4f}
              - Std: {temporal_feat.std():.4f}
              - Range: [{temporal_feat.min():.4f}, {temporal_feat.max():.4f}]
            -------------------------
            Spatial Part:
              - Mean: {spatial_feat.mean():.4f}
              - Std: {spatial_feat.std():.4f}
              - Range: [{spatial_feat.min():.4f}, {spatial_feat.max():.4f}]
            -------------------------
            Spike Activity:
              - Total spikes: {spikes.sum():.0f}
              - Active neurons: {np.sum(spikes.sum(axis=0) > 0)}
            """
            ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax6.axis('off')
            ax6.set_title('Statistics', fontsize=11, fontweight='bold')
            
            fig.suptitle(f'LSM Feature Vector - Sample {idx}\n(Hard Sample - Low Confidence)', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = os.path.join(individual_dir, f'{title_prefix}_{idx:02d}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
        print(f"    Saved {len(sample_indices)} individual LSM feature images to: {individual_dir}")
    
    def plot_memory_vs_lsm_comparison_individual(self, images: torch.Tensor,
                                                  memory_vectors: List[torch.Tensor],
                                                  lsm_features: torch.Tensor,
                                                  labels: torch.Tensor,
                                                  sample_indices: List[int],
                                                  confidence_scores: List[float],
                                                  title_prefix="Memory_vs_LSM_Sample"):
        print(f"\n  Plotting memory vs LSM comparison visualizations (individual images)...")
        
        individual_dir = os.path.join(self.save_dir, "memory_vs_lsm_comparison_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, idx in enumerate(sample_indices):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.patch.set_facecolor('white')
            
            if isinstance(images, torch.Tensor):
                img = images[idx, 0].cpu().numpy()
            else:
                img = images[idx, 0]
            
            mem_vec = memory_vectors[idx]
            if isinstance(mem_vec, torch.Tensor):
                mem_vec = mem_vec.cpu().numpy()
            
            lsm_feat = lsm_features[idx]
            if isinstance(lsm_feat, torch.Tensor):
                lsm_feat = lsm_feat.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                label = labels[idx].cpu().item()
            else:
                label = labels[idx]
            
            conf = confidence_scores[idx] if idx < len(confidence_scores) else 0.0
            
            ax1 = axes[0, 0]
            ax1.imshow(img, cmap='gray')
            ax1.set_title(f'Original Image\nClass: {label} ({FASHION_MNIST_CLASSES.get(label, label)}), Confidence: {conf:.3f}', 
                         fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            ax2 = axes[0, 1]
            mem_img = mem_vec.reshape(28, 28)
            mem_norm = (mem_img - mem_img.min()) / (mem_img.max() - mem_img.min() + 1e-8)
            im2 = ax2.imshow(mem_norm, cmap='hot', aspect='auto')
            ax2.set_title(f'Memory Vector (784 dims)\nActive: {np.sum(mem_vec > 0)}', 
                         fontsize=11, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            ax3 = axes[0, 2]
            non_zero = mem_vec[mem_vec > 0]
            if len(non_zero) > 0:
                ax3.hist(non_zero, bins=30, color='coral', alpha=0.7, edgecolor='black')
                ax3.axvline(non_zero.mean(), color='red', linestyle='--', linewidth=2)
            ax3.set_xlabel('Information Value', fontsize=9)
            ax3.set_ylabel('Frequency', fontsize=9)
            ax3.set_title('Memory Vector Distribution', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 0]
            feat_2d = lsm_feat.reshape(-1, 1).T
            im4 = ax4.imshow(feat_2d, cmap='RdBu_r', aspect='auto')
            ax4.set_title(f'LSM Feature ({len(lsm_feat)} dims)', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Dimension', fontsize=9)
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            
            ax5 = axes[1, 1]
            ax5.hist(lsm_feat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax5.axvline(lsm_feat.mean(), color='red', linestyle='--', linewidth=2)
            ax5.set_xlabel('Feature Value', fontsize=9)
            ax5.set_ylabel('Frequency', fontsize=9)
            ax5.set_title('LSM Feature Distribution', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            ax6 = axes[1, 2]
            mem_norm_val = np.linalg.norm(mem_vec)
            lsm_norm_val = np.linalg.norm(lsm_feat)
            mem_sparsity = 1.0 - np.sum(mem_vec > 0) / len(mem_vec)
            lsm_sparsity = 1.0 - np.sum(np.abs(lsm_feat) > 1e-6) / len(lsm_feat)
            
            x_pos = [0, 1]
            norms = [mem_norm_val, lsm_norm_val]
            sparsities = [mem_sparsity, lsm_sparsity]
            
            width = 0.35
            ax6_twin = ax6.twinx()
            
            bars1 = ax6.bar([x - width/2 for x in x_pos], norms, width, 
                           label='L2 Norm', color='steelblue', alpha=0.8)
            bars2 = ax6_twin.bar([x + width/2 for x in x_pos], sparsities, width, 
                                 label='Sparsity', color='coral', alpha=0.8)
            
            ax6.set_ylabel('L2 Norm', fontsize=9, color='steelblue')
            ax6_twin.set_ylabel('Sparsity', fontsize=9, color='coral')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(['Memory Vector', 'LSM Feature'])
            ax6.set_title('Feature Comparison', fontsize=11, fontweight='bold')
            
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            fig.suptitle(f'Memory Vector vs LSM Feature Comparison - Sample {idx}', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            save_path = os.path.join(individual_dir, f'{title_prefix}_{idx:02d}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
        print(f"    Saved {len(sample_indices)} individual comparison images to: {individual_dir}")
    
    def plot_dual_som_statistics(self, title="Dual_SOM_Statistics"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        if len(self.dual_som_stats['spiking_som_confidence']) > 0:
            axes[0, 0].plot(self.dual_som_stats['spiking_som_confidence'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Batch', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Avg Confidence', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Spiking SOM Confidence', fontsize=13, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        if len(self.dual_som_stats['easy_samples_ratio']) > 0:
            axes[0, 1].plot(self.dual_som_stats['easy_samples_ratio'], 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Batch', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Easy Samples Ratio', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Easy Samples (High Confidence)', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        if len(self.dual_som_stats['hard_samples_ratio']) > 0:
            axes[1, 0].plot(self.dual_som_stats['hard_samples_ratio'], 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Batch', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Hard Samples Ratio', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Hard Samples (Low Confidence)', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        if len(self.dual_som_stats['lsm_processed_ratio']) > 0:
            axes[1, 1].plot(self.dual_som_stats['lsm_processed_ratio'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Batch', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('LSM Processed Ratio', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('LSM-STDP Processed', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def plot_training_curves(self, title="Training_Curves"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        axes[0, 0].plot(self.training_history['epoch'], 
                       self.training_history['quantization_error'], 
                       'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Quantization Error', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Quantization Error', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.training_history['epoch'], 
                       self.training_history['neuron_activation'], 
                       'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Activation Rate', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Neuron Activation', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.training_history['epoch'], 
                       self.training_history['feature_variance'], 
                       'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Feature Variance', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Feature Variance', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.training_history['epoch'], 
                       self.training_history['learning_rate'], 
                       'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def plot_confusion_matrix(self, cm, title="Confusion_Matrix"):
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        fig, ax = plt.subplots(figsize=(18, 15))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_accuracy = cm / (row_sums + 1e-8)
        
        from matplotlib.colors import LinearSegmentedColormap
        colors_gradient = [
            '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
            '#4292c6', '#2171b5', '#08519c', '#08306b'
        ]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('blue_accuracy', colors_gradient, N=n_bins)
        
        im = ax.imshow(cm_accuracy, cmap=cmap, aspect='auto', interpolation='bilinear', 
                      vmin=0, vmax=1)
        
        for i in range(10):
            for j in range(10):
                value = cm[i, j]
                accuracy = cm_accuracy[i, j]
                
                if accuracy > 0.5:
                    text_color = "white"
                    fontweight = 'bold'
                else:
                    text_color = "black"
                    fontweight = 'normal'
                
                ax.text(j, i, int(value), ha="center", va="center",
                       color=text_color, fontsize=12, fontweight=fontweight,
                       family='monospace')
        
        ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_title('Confusion Matrix - Test Set Performance\n(White=Low Accuracy, Deep Blue=High Accuracy)', 
                    fontsize=19, fontweight='bold', pad=25, color='#1a1a1a')
        
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(class_names, fontsize=12, rotation=45, ha='right', fontweight='bold')
        ax.set_yticklabels(class_names, fontsize=12, fontweight='bold')
        
        ax.grid(False)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(2.5)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        cbar.set_label('Classification Accuracy', fontsize=14, fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=12)
        cbar.outline.set_edgecolor('#333333')
        cbar.outline.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def _detect_and_remove_outliers(self, features_2d: np.ndarray, labels: np.ndarray,
                                    contamination: float = 0.05, 
                                    min_neighbors_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        n_samples = features_2d.shape[0]
        outlier_mask = np.zeros(n_samples, dtype=bool)
        
        try:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            lof_outliers = lof.fit_predict(features_2d) == -1
            outlier_mask |= lof_outliers
        except Exception as e:
            print(f"    Warning: LOF detection failed: {e}")
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_features = features_2d[label_mask]
            
            if len(label_features) < 5:
                continue
            
            center = np.mean(label_features, axis=0)
            distances = np.sqrt(np.sum((label_features - center) ** 2, axis=1))
            
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            
            label_outliers = distances > threshold
            original_indices = np.where(label_mask)[0]
            outlier_mask[original_indices[label_outliers]] = True
        
        keep_mask = ~outlier_mask
        cleaned_features = features_2d[keep_mask]
        cleaned_labels = labels[keep_mask]
        
        outlier_info = {
            'total_samples': n_samples,
            'outliers_detected': np.sum(outlier_mask),
            'outlier_ratio': np.sum(outlier_mask) / n_samples,
            'remaining_samples': len(cleaned_features),
            'outliers_per_class': {}
        }
        
        for label in unique_labels:
            label_mask = labels == label
            label_outliers = np.sum(outlier_mask & label_mask)
            outlier_info['outliers_per_class'][int(label)] = {
                'total': np.sum(label_mask),
                'outliers': label_outliers,
                'ratio': label_outliers / np.sum(label_mask) if np.sum(label_mask) > 0 else 0
            }
        
        return cleaned_features, cleaned_labels, outlier_info
    
    def plot_optimized_tsne(self, features: torch.Tensor, labels: torch.Tensor, 
                           title="Optimized_tSNE",
                           remove_outliers: bool = True,
                           contamination: float = 0.03):
        print(f"\n  Plotting optimized t-SNE visualization...")
        
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
            
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        
        from sklearn.preprocessing import StandardScaler
        
        print("    Note: Using StandardScaler instead of L2 normalization to preserve inter-class separability")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_np)
        
        n_samples = features_scaled.shape[0]
        perplexity = min(50, max(5, n_samples // 100))
        
        print(f"    t-SNE parameters: perplexity={perplexity}, n_iter=2000")
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=2000,
            early_exaggeration=15,
            learning_rate='auto',
            init='pca',
            random_state=42,
            metric='euclidean',
            n_jobs=-1
        )
        
        features_2d = tsne.fit_transform(features_scaled)
        
        if remove_outliers:
            print(f"    Detecting and removing outliers (expected ratio: {contamination*100:.1f}%)...")
            features_2d_cleaned, labels_cleaned, outlier_info = self._detect_and_remove_outliers(
                features_2d, labels_np, contamination=contamination
            )
            
            print(f"    Outlier statistics:")
            print(f"      Original samples: {outlier_info['total_samples']}")
            print(f"      Detected outliers: {outlier_info['outliers_detected']} ({outlier_info['outlier_ratio']*100:.2f}%)")
            print(f"      Remaining samples: {outlier_info['remaining_samples']}")
            
            features_2d = features_2d_cleaned
            labels_np = labels_cleaned
        else:
            outlier_info = None
        
        fig, ax = plt.subplots(figsize=(16, 14))
        fig.patch.set_facecolor('white')
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for class_idx in range(10):
            mask = labels_np == class_idx
            if np.sum(mask) > 0:
                ax.scatter(
                    features_2d[mask, 0], 
                    features_2d[mask, 1],
                    c=[colors[class_idx]],
                    label=f'Class {class_idx}',
                    alpha=0.75,
                    s=60,
                    edgecolors='white',
                    linewidths=0.8
                )
                
                center_x = np.mean(features_2d[mask, 0])
                center_y = np.mean(features_2d[mask, 1])
                ax.scatter(
                    [center_x], [center_y],
                    c=[colors[class_idx]],
                    marker='*',
                    s=300,
                    edgecolors='black',
                    linewidths=1.5,
                    alpha=1.0,
                    zorder=10
                )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold', labelpad=12)
        
        title_suffix = "(Outliers Removed)" if remove_outliers else "(All Samples)"
        ax.set_title(f'Optimized t-SNE Visualization\n{title_suffix}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
        
        return features_2d, outlier_info
    
    def plot_spiking_som_weights(self, spiking_som_layer, title="Spiking_SOM_Weights"):
        print(f"\n  Plotting Spiking SOM weight visualization...")
        
        weights = spiking_som_layer.competitive_weights.cpu().detach().numpy()
        n_competitive = weights.shape[0]
        
        n_cols = int(np.ceil(np.sqrt(n_competitive)))
        n_rows = int(np.ceil(n_competitive / n_cols))
        
        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Spiking SOM Competitive Layer Weights\n' + 
                    f'Total: {n_competitive} neurons', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        for idx in range(n_competitive):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            weight_vector = weights[idx]
            weight_image = weight_vector.reshape(28, 28)
            
            weight_min = weight_image.min()
            weight_max = weight_image.max()
            if weight_max > weight_min:
                weight_image_normalized = (weight_image - weight_min) / (weight_max - weight_min)
            else:
                weight_image_normalized = weight_image
            
            ax.imshow(weight_image_normalized, cmap='gray', aspect='auto', 
                     interpolation='bilinear', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.99])
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def plot_lsm_weights(self, lsm_layer, title="LSM_Weights"):
        print(f"\n  Plotting LSM weight visualization...")
        
        w_input = lsm_layer.w_input.cpu().detach().numpy()
        w_recurrent = lsm_layer.w_recurrent.cpu().detach().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'LSM-STDP Liquid Layer Weights', fontsize=18, fontweight='bold', y=0.98)
        
        ax1 = axes[0]
        im1 = ax1.imshow(w_input.T, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        ax1.set_title(f'Input Weights\nShape: {w_input.shape}', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Input Neurons (784 pixels)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Liquid Neurons', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Weight Value', fontsize=11, fontweight='bold')
        
        ax2 = axes[1]
        im2 = ax2.imshow(w_recurrent, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        ax2.set_title(f'Recurrent Weights\nShape: {w_recurrent.shape}', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Pre-synaptic Neurons', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Post-synaptic Neurons', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Weight Value', fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def plot_weight_statistics(self, spiking_som_layer, lsm_layer, title="Weight_Statistics"):
        print(f"\n  Plotting weight statistics distribution...")
        
        spiking_weights = spiking_som_layer.competitive_weights.cpu().detach().numpy().flatten()
        lsm_input_weights = lsm_layer.w_input.cpu().detach().numpy().flatten()
        lsm_recurrent_weights = lsm_layer.w_recurrent.cpu().detach().numpy().flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('white')
        
        ax1 = axes[0, 0]
        ax1.hist(spiking_weights, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(spiking_weights.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {spiking_weights.mean():.4f}')
        ax1.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'Spiking SOM Weights Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.hist(lsm_input_weights, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.axvline(lsm_input_weights.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {lsm_input_weights.mean():.4f}')
        ax2.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'LSM Input Weights Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.hist(lsm_recurrent_weights, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.axvline(lsm_recurrent_weights.mean(), color='purple', linestyle='--', linewidth=2,
                   label=f'Mean: {lsm_recurrent_weights.mean():.4f}')
        ax3.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title(f'LSM Recurrent Weights Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_data = [
            ['Layer', 'Mean', 'Std', 'Min', 'Max', 'Sparsity'],
            ['Spiking SOM', f'{spiking_weights.mean():.4f}', f'{spiking_weights.std():.4f}',
             f'{spiking_weights.min():.4f}', f'{spiking_weights.max():.4f}',
             f'{np.sum(spiking_weights == 0) / len(spiking_weights) * 100:.1f}%'],
            ['LSM Input', f'{lsm_input_weights.mean():.4f}', f'{lsm_input_weights.std():.4f}',
             f'{lsm_input_weights.min():.4f}', f'{lsm_input_weights.max():.4f}',
             f'{np.sum(lsm_input_weights == 0) / len(lsm_input_weights) * 100:.1f}%'],
            ['LSM Recurrent', f'{lsm_recurrent_weights.mean():.4f}', f'{lsm_recurrent_weights.std():.4f}',
             f'{lsm_recurrent_weights.min():.4f}', f'{lsm_recurrent_weights.max():.4f}',
             f'{np.sum(lsm_recurrent_weights == 0) / len(lsm_recurrent_weights) * 100:.1f}%']
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(6):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
        
        for i in range(1, 4):
            for j in range(6):
                cell = table[(i, j)]
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')
                cell.set_edgecolor('#333333')
        
        ax4.set_title('Weight Statistics Summary', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def plot_train_accuracy_curve(self, train_accuracy_history: List[float], 
                                 title="Train_Accuracy_Curve"):
        if not train_accuracy_history:
            print(f"    Warning: No training accuracy history to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        
        epochs_list = [i * 2 + 2 for i in range(len(train_accuracy_history))]
        
        ax.plot(epochs_list, train_accuracy_history, 'o-', 
               color='#2E86AB', linewidth=2.5, markersize=6, 
               markerfacecolor='#A23B72', markeredgewidth=1.5, 
               markeredgecolor='#2E86AB', label='Training Accuracy')
        
        ax.fill_between(epochs_list, train_accuracy_history, alpha=0.15, color='#2E86AB')
        
        max_acc = max(train_accuracy_history)
        max_epoch = epochs_list[train_accuracy_history.index(max_acc)]
        ax.scatter([max_epoch], [max_acc], color='#F18F01', s=200, 
                  zorder=5, edgecolors='#C41E3A', linewidth=2)
        ax.annotate(f'Peak: {max_acc:.4f}', xy=(max_epoch, max_acc),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#F18F01', alpha=0.7),
                   fontsize=11, fontweight='bold', color='white',
                   arrowprops=dict(arrowstyle='->', color='#C41E3A', lw=2))
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold', labelpad=12)
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold', labelpad=12)
        ax.set_title('Training Accuracy Curve', fontsize=15, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        y_min = min(train_accuracy_history) - 0.05
        y_max = max(train_accuracy_history) + 0.05
        ax.set_ylim(max(0, y_min), min(1.0, y_max))
        
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {save_path}")
    
    def generate_report(self, final_metrics: Dict, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Dual SOM System Training Report (Feature Normalization Optimized + Memory/LSM Visualization)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("System Architecture:\n")
            f.write("-" * 40 + "\n")
            f.write("1. TTFS-Poisson Hybrid Encoding\n")
            f.write("2. Spiking SOM Coarse Clustering + Memory Layer\n")
            f.write("3. Confidence Evaluation and Routing (Based on Neuron Class Activation Counts):\n")
            f.write("   - High confidence samples -> Memory Vector -> Random Projection -> L2 Normalization -> Traditional SOM\n")
            f.write("   - Low confidence samples -> LSM-STDP -> L2 Normalization -> Traditional SOM\n")
            f.write("4. Feature Normalization: LSM output and Spiking SOM direction vectors L2 normalized\n")
            f.write("5. Random Projection unified feature dimension (Johnson-Lindenstrauss)\n")
            f.write("6. Traditional SOM Final Clustering\n")
            f.write("7. Memory Vector Visualization\n")
            f.write("8. LSM Feature Vector Visualization\n\n")
            
            f.write("Visualization Features:\n")
            f.write("-" * 40 + "\n")
            f.write("- Memory Vector Visualization: Shows Spiking SOM output memory vectors\n")
            f.write("- LSM Feature Vector Visualization: Shows LSM output deep features\n")
            f.write("- Comparison Visualization: Memory vector vs LSM feature analysis\n")
            f.write("- Individual Sample Images: Each sample saved as separate image file\n\n")
            
            f.write("Final Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key:30s}: {value:.4f}\n")
                else:
                    f.write(f"{key:30s}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"    Report saved: {save_path}")
