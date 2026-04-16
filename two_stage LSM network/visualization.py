from config import *
from core import SpikingSOMLayer, OptimizedTemporalLiquidLayer

class EnhancedVisualizationAndMonitoring:
    
    def __init__(self, save_dir=None):
        # Save directory removed for GitHub upload
        self.save_dir = '***'  # Path hidden
        # os.makedirs(save_dir, exist_ok=True)  # Removed
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
        
        print(f"  Enhanced Visualization & Monitoring initialized")
    
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
        
        # Save directory removed for GitHub upload
        # standalone_dir = os.path.join(self.save_dir, "memory_vectors_standalone")
        # os.makedirs(standalone_dir, exist_ok=True)
        
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
  Time Window T:       Hidden
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
            
            # Save path removed for GitHub upload
            # save_path = os.path.join(standalone_dir, f'{title_prefix}_{idx:02d}.png')
            # plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"    Processed {len(sample_indices)} standalone memory vector visualizations")
    
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
        
        # Save directory removed for GitHub upload
        # standalone_dir = os.path.join(self.save_dir, "lsm_features_standalone")
        # os.makedirs(standalone_dir, exist_ok=True)
        
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
            
            # ... (visualization code continues with same pattern)
            
            plt.close()
        
        print(f"    Processed {len(sample_indices)} standalone LSM feature visualizations")

    def plot_memory_vectors_individual(self, images: torch.Tensor, memory_vectors: List[torch.Tensor], 
                                       labels: torch.Tensor, sample_indices: List[int],
                                       confidence_scores: List[float] = None,
                                       title_prefix="Memory_Vector_Sample"):
        print(f"\n  Plotting memory vector visualizations (individual images)...")
        
        # Save directory removed for GitHub upload
        # individual_dir = os.path.join(self.save_dir, "memory_vectors_individual")
        # os.makedirs(individual_dir, exist_ok=True)
        
        for i, idx in enumerate(sample_indices):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.patch.set_facecolor('white')
            
            # ... (visualization code)
            
            plt.close()
            
        print(f"    Processed {len(sample_indices)} individual memory vector images")
    
    def plot_lsm_feature_vectors_individual(self, images: torch.Tensor, lsm_features: torch.Tensor,
                                            spike_trains: torch.Tensor, labels: torch.Tensor, 
                                            sample_indices: List[int],
                                            confidence_scores: List[float] = None,
                                            title_prefix="LSM_Feature_Sample"):
        print(f"\n  Plotting LSM feature vector visualizations (individual images)...")
        
        # Save directory removed for GitHub upload
        # individual_dir = os.path.join(self.save_dir, "lsm_features_individual")
        # os.makedirs(individual_dir, exist_ok=True)
        
        feature_dim = lsm_features.shape[1]
        
        for i, idx in enumerate(sample_indices):
            fig = plt.figure(figsize=(20, 10))
            fig.patch.set_facecolor('white')
            
            # ... (visualization code)
            
            plt.close()
            
        print(f"    Processed {len(sample_indices)} individual LSM feature images")
    
    def plot_memory_vs_lsm_comparison_individual(self, images: torch.Tensor,
                                                  memory_vectors: List[torch.Tensor],
                                                  lsm_features: torch.Tensor,
                                                  labels: torch.Tensor,
                                                  sample_indices: List[int],
                                                  confidence_scores: List[float],
                                                  title_prefix="Memory_vs_LSM_Sample"):
        print(f"\n  Plotting memory vs LSM comparison visualizations (individual images)...")
        
        # Save directory removed for GitHub upload
        # individual_dir = os.path.join(self.save_dir, "memory_vs_lsm_comparison_individual")
        # os.makedirs(individual_dir, exist_ok=True)
        
        for i, idx in enumerate(sample_indices):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.patch.set_facecolor('white')
            
            # ... (visualization code)
            
            plt.close()
            
        print(f"    Processed {len(sample_indices)} comparison images")

    def plot_training_curves(self, title_prefix="Training_Curves"):
        """Plot training curves - visualization only, no save"""
        print(f"\n  Plotting training curves...")
        
        if len(self.training_history['epoch']) == 0:
            print("    No training history to plot")
            return
        
        # ... (visualization code without save)
        print("    Training curves displayed")

    def plot_dual_som_statistics(self, title_prefix="Dual_SOM_Statistics"):
        """Plot dual SOM statistics - visualization only, no save"""
        print(f"\n  Plotting dual SOM statistics...")
        
        if len(self.dual_som_stats['spiking_som_confidence']) == 0:
            print("    No dual SOM statistics to plot")
            return
        
        # ... (visualization code without save)
        print("    Dual SOM statistics displayed")

    def plot_confusion_matrix(self, cm: np.ndarray, title_prefix="Confusion_Matrix"):
        """Plot confusion matrix - visualization only, no save"""
        print(f"\n  Plotting confusion matrix...")
        
        # ... (visualization code without save)
        print("    Confusion matrix displayed")

    def plot_optimized_tsne(self, features: torch.Tensor, labels: torch.Tensor,
                           title_prefix="tSNE_Visualization",
                           remove_outliers: bool = True,
                           contamination: float = 0.03):
        """Plot t-SNE visualization - visualization only, no save"""
        print(f"\n  Plotting optimized t-SNE visualization...")
        
        # ... (visualization code without save)
        print("    t-SNE visualization displayed")

    def plot_spiking_som_weights(self, spiking_som, title_prefix="Spiking_SOM_Weights"):
        """Plot spiking SOM weights - visualization only, no save"""
        print(f"\n  Plotting spiking SOM weights...")
        
        # ... (visualization code without save)
        print("    Spiking SOM weights displayed")

    def plot_lsm_weights(self, liquid_layer, title_prefix="LSM_Weights"):
        """Plot LSM weights - visualization only, no save"""
        print(f"\n  Plotting LSM weights...")
        
        # ... (visualization code without save)
        print("    LSM weights displayed")

    def plot_weight_statistics(self, spiking_som, liquid_layer, title_prefix="Weight_Statistics"):
        """Plot weight statistics - visualization only, no save"""
        print(f"\n  Plotting weight statistics...")
        
        # ... (visualization code without save)
        print("    Weight statistics displayed")

    def plot_train_accuracy_curve(self, accuracy_history: List[float], title_prefix="Train_Accuracy"):
        """Plot training accuracy curve - visualization only, no save"""
        print(f"\n  Plotting training accuracy curve...")
        
        # ... (visualization code without save)
        print("    Training accuracy curve displayed")

    def generate_report(self, metrics: Dict):
        """Generate and print final report"""
        print(f"\n  Generating training report...")
        
        print("\n" + "=" * 60)
        print("TRAINING REPORT")
        print("=" * 60)
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  Test F1-Macro: {metrics['test_f1']:.4f}")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Generalization Gap: {metrics['gap']:.2%}")
        print(f"  Easy Samples Ratio: {metrics['easy_samples_ratio']*100:.1f}%")
        print(f"  Hard Samples Ratio: {metrics['hard_samples_ratio']*100:.1f}%")
        print(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
        print(f"  Total Training Time: {metrics['total_time']:.2f}s")
        print("=" * 60)
