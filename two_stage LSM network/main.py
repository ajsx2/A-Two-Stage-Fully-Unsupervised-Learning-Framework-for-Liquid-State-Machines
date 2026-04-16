from config import *
from core import DualSOMSystem, ImprovedStableSOM
from visualization import EnhancedVisualizationAndMonitoring
from utils import load_fashion_mnist_with_augmentation, comprehensive_evaluation

def main():
    print("=" * 80)
    print("Dual SOM System (Feature Normalization Optimized + Memory/LSM Feature Visualization)")
    print("   Simple: Memory Vector + Random Projection + L2 Norm | Hard: LSM Deep Features + L2 Norm")
    print("   New: Memory Vector Visualization + LSM Feature Vector Visualization (Individual Images)")
    print("=" * 80)
    
    # Hidden hyperparameters for GitHub upload
    n_train = ***           # Hidden
    n_test = ***            # Hidden
    n_spiking_som = ***     # Hidden
    n_lsm = ***             # Hidden
    n_traditional_som = *** # Hidden
    time_steps = ***        # Hidden
    batch_size = ***        # Hidden
    confidence_threshold = ***  # Hidden
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_visualization_samples = ***  # Hidden
    
    print(f"\n  Configuration:")
    print(f"    Device: {device}")
    print(f"    Spiking SOM neurons: [Hidden]")
    print(f"    LSM neurons: [Hidden]")
    print(f"    Traditional SOM neurons: [Hidden]")
    print(f"    Confidence threshold: [Hidden]")
    print(f"    Visualization samples: [Hidden]")
    
    # Save directory removed for GitHub upload
    visualizer = EnhancedVisualizationAndMonitoring(save_dir=None)
    
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading")
    print("=" * 80)
    start_time = time.time()
    train_images, train_labels, test_images, test_labels = \
        load_fashion_mnist_with_augmentation(n_train, n_test)
    data_load_time = time.time() - start_time
    print(f"  Data loading time: {data_load_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("STEP 2: Dual SOM System Initialization (Feature Normalization Version)")
    print("=" * 80)
    
    dual_som = DualSOMSystem(
        n_spiking_som=n_spiking_som,
        n_lsm=n_lsm,
        n_traditional_som=n_traditional_som,
        device=device,
        time_steps=time_steps,
        confidence_threshold=confidence_threshold,
        normalize_features=True
    )
    
    print("\n" + "=" * 80)
    print(f"STEP 2.5: Memory Vector and LSM Feature Visualization ({n_visualization_samples} samples)")
    print("=" * 80)
    
    viz_indices = list(range(n_visualization_samples))
    viz_images = train_images[viz_indices]
    viz_labels = train_labels[viz_indices]
    
    all_memory_vectors = []
    all_lsm_features = []
    all_spike_trains = []
    all_confidence_scores = []
    easy_sample_indices = []
    hard_sample_indices = []
    
    print(f"\n  Processing {n_visualization_samples} samples for visualization...")
    for i in tqdm(range(n_visualization_samples), desc="    [Visualization]"):
        img = viz_images[i:i+1]
        label = viz_labels[i:i+1]
        
        features, info = dual_som.process_batch(img, training=False, labels=label)
        
        all_memory_vectors.append(info['memory_vectors'][0])
        all_confidence_scores.append(info['confidence_scores'][0])
        
        all_spike_trains.append(info['spike_train'][0])
        
        if info['confidence_scores'][0] >= confidence_threshold:
            easy_sample_indices.append(i)
        else:
            hard_sample_indices.append(i)
            all_lsm_features.append(features[0].cpu())
    
    print("\n  Plotting memory vector visualizations (individual images)...")
    visualizer.plot_memory_vectors_individual(
        images=viz_images,
        memory_vectors=all_memory_vectors,
        labels=viz_labels,
        sample_indices=list(range(n_visualization_samples)),
        confidence_scores=all_confidence_scores,
        title_prefix="Memory_Vector_Sample"
    )
    
    print("\n  Plotting memory vector STANDALONE visualizations (professional)...")
    visualizer.plot_memory_vectors_standalone(
        images=viz_images,
        memory_vectors=all_memory_vectors,
        labels=viz_labels,
        sample_indices=list(range(n_visualization_samples)),
        confidence_scores=all_confidence_scores,
        title_prefix="Memory_Vector_Standalone"
    )
    
    if len(hard_sample_indices) > 0:
        print(f"\n  Plotting LSM feature vector visualizations ({len(hard_sample_indices)} hard samples)...")
        
        hard_spike_trains = torch.stack([all_spike_trains[i] for i in hard_sample_indices])
        hard_lsm_features = torch.stack(all_lsm_features)
        hard_labels = viz_labels[hard_sample_indices]
        hard_confidences = [all_confidence_scores[i] for i in hard_sample_indices]
        hard_images = viz_images[hard_sample_indices]
        
        hard_sample_sequential_indices = list(range(len(hard_sample_indices)))
        
        visualizer.plot_lsm_feature_vectors_individual(
            images=hard_images,
            lsm_features=hard_lsm_features,
            spike_trains=hard_spike_trains,
            labels=hard_labels,
            sample_indices=hard_sample_sequential_indices,
            confidence_scores=hard_confidences,
            title_prefix="LSM_Feature_Sample"
        )
        
        print(f"\n  Plotting LSM feature vector STANDALONE visualizations ({len(hard_sample_indices)} hard samples)...")
        visualizer.plot_lsm_feature_vectors_standalone(
            images=hard_images,
            lsm_features=hard_lsm_features,
            spike_trains=hard_spike_trains,
            labels=hard_labels,
            sample_indices=hard_sample_sequential_indices,
            confidence_scores=hard_confidences,
            title_prefix="LSM_Feature_Standalone"
        )
    else:
        print(f"\n  No hard samples found, skipping LSM feature visualization")
    
    print("\n  Plotting memory vs LSM comparison visualizations (individual images)...")
    all_features_for_comparison = []
    for i in range(n_visualization_samples):
        img = viz_images[i:i+1]
        label = viz_labels[i:i+1]
        features, _ = dual_som.process_batch(img, training=False, labels=label)
        all_features_for_comparison.append(features[0].cpu())
    
    all_features_tensor = torch.stack(all_features_for_comparison)
    
    visualizer.plot_memory_vs_lsm_comparison_individual(
        images=viz_images,
        memory_vectors=all_memory_vectors,
        lsm_features=all_features_tensor,
        labels=viz_labels,
        sample_indices=list(range(n_visualization_samples)),
        confidence_scores=all_confidence_scores,
        title_prefix="Memory_vs_LSM_Sample"
    )
    
    print(f"\n  Visualization sample statistics:")
    print(f"    Total samples: {n_visualization_samples}")
    print(f"    Easy samples (high confidence): {len(easy_sample_indices)}")
    print(f"    Hard samples (low confidence): {len(hard_sample_indices)}")
    
    print("\n" + "=" * 80)
    print("STEP 3: Batch Feature Extraction")
    print("  Simple samples: Memory Vector -> Random Projection -> L2 Norm -> Unified Dimension")
    print("  Hard samples: LSM-STDP Deep Features -> L2 Norm")
    print("  Confidence calculation: Based on neuron class activation counts")
    print("=" * 80)
    
    start_time = time.time()
    n_batches = (n_train + batch_size - 1) // batch_size
    train_features_list = []
    
    print(f"\n  Processing {n_batches} training batches...")
    for batch_idx in tqdm(range(n_batches), desc="    [Training]"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_train)
        batch_images = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        
        batch_features, batch_info = dual_som.process_batch(
            batch_images, training=True, labels=batch_labels
        )
        train_features_list.append(batch_features.cpu())
        
        visualizer.record_dual_som_stats(
            spiking_confidences=batch_info['confidence_scores'],
            easy_ratio=batch_info['easy_count'] / len(batch_images),
            hard_ratio=batch_info['hard_count'] / len(batch_images),
            lsm_ratio=batch_info['hard_count'] / len(batch_images)
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    train_features = torch.cat(train_features_list, dim=0)
    train_processing_time = time.time() - start_time
    print(f"  Training set processing time: {train_processing_time:.2f}s")
    
    print(f"\n  Processing test set...")
    n_batches_test = (n_test + batch_size - 1) // batch_size
    test_features_list = []
    
    for batch_idx in tqdm(range(n_batches_test), desc="    [Testing]"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_test)
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        
        batch_features, _ = dual_som.process_batch(
            batch_images, training=False, labels=batch_labels
        )
        test_features_list.append(batch_features.cpu())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    test_features = torch.cat(test_features_list, dim=0)
    test_processing_time = time.time() - start_time - train_processing_time
    print(f"  Test set processing time: {test_processing_time:.2f}s")
    
    stats = dual_som.get_processing_stats()
    print(f"\n  Dual SOM processing statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Easy samples: {stats['easy_samples']} ({stats['easy_ratio']*100:.1f}%)")
    print(f"    Hard samples: {stats['hard_samples']} ({stats['hard_ratio']*100:.1f}%)")
    print(f"    Average confidence: {stats['avg_confidence']:.4f}")
    
    memory_stats = dual_som.spiking_som.get_memory_layer_stats()
    print(f"\n  Memory Layer Statistics:")
    print(f"    Average spike reception rate: {memory_stats['avg_spike_reception_rate']*100:.2f}%")
    print(f"    Average total information: {memory_stats['avg_total_information']:.4f}")
    print(f"    Average spike time: {memory_stats['avg_spike_time']:.4f}")
    
    if len(dual_som.processing_stats['confidence_scores']) > 0:
        confidences = np.array(dual_som.processing_stats['confidence_scores'])
        print(f"\n  Confidence distribution statistics:")
        print(f"    Minimum: {confidences.min():.4f}")
        print(f"    Maximum: {confidences.max():.4f}")
        print(f"    Median: {np.median(confidences):.4f}")
        print(f"    Std: {confidences.std():.4f}")
    
    print("\n  Plotting dual SOM statistics...")
    visualizer.plot_dual_som_statistics("Dual_SOM_Statistics")
    
    print("\n" + "=" * 80)
    print("STEP 3.5: Spiking SOM Layer Boundary Neuron Identification")
    print("=" * 80)
    
    spiking_boundary_threshold = ***  # Hidden
    dual_som.spiking_som.identify_boundary_neurons(confidence_threshold=spiking_boundary_threshold)
    
    n_spiking_boundary = len(dual_som.spiking_som.boundary_neurons)
    n_spiking_non_boundary = len(dual_som.spiking_som.non_boundary_neurons)
    total_spiking_neurons = n_spiking_boundary + n_spiking_non_boundary
    print(f"\n  Spiking SOM boundary neuron statistics:")
    print(f"    Boundary neurons: {n_spiking_boundary} ({n_spiking_boundary/total_spiking_neurons*100:.1f}%)")
    print(f"    Non-boundary neurons: {n_spiking_non_boundary} ({n_spiking_non_boundary/total_spiking_neurons*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("STEP 4: Traditional SOM Final Clustering (Feature Normalization)")
    print("=" * 80)
    
    dual_som.traditional_som = ImprovedStableSOM(
        n_neurons=n_traditional_som,
        input_dim=train_features.shape[1],
        n_classes=10,
        device=device,
        normalize_features=True
    )
    
    start_time = time.time()
    dual_som.traditional_som.train_unsupervised(
        train_features.to(device), n_epochs=***,  # Hidden
        learning_rate=***, batch_size=***,  # Hidden
        visualizer=visualizer, train_labels=train_labels.to(device)
    )
    som_train_time = time.time() - start_time
    print(f"  SOM training time: {som_train_time:.2f}s")
    
    print("\n  Plotting training curves...")
    visualizer.plot_training_curves("SOM_Training_Curves")
    
    if dual_som.traditional_som.train_accuracy_history:
        visualizer.plot_train_accuracy_curve(
            dual_som.traditional_som.train_accuracy_history,
            "Train_Accuracy_Curve"
        )
    
    dual_som.traditional_som.assign_labels(train_features.to(device), train_labels.to(device))
    
    print("\n" + "=" * 80)
    print("STEP 5: Prediction & Evaluation")
    print("=" * 80)
    
    start_time = time.time()
    train_pred = dual_som.traditional_som.predict(train_features.to(device))
    test_pred = dual_som.traditional_som.predict(test_features.to(device))
    prediction_time = time.time() - start_time
    print(f"  Prediction time: {prediction_time:.2f}s")
    
    train_metrics = comprehensive_evaluation(train_pred, train_labels.cpu().numpy())
    test_metrics = comprehensive_evaluation(test_pred, test_labels.cpu().numpy())
    
    total_time = data_load_time + train_processing_time + test_processing_time + som_train_time
    visualizer.record_performance(
        test_metrics['accuracy'], 
        test_metrics['f1_macro'],
        total_time
    )
    
    print("\n  Generating visualizations...")
    visualizer.plot_confusion_matrix(test_metrics['confusion_matrix'], "Test_Confusion_Matrix")
    
    print("\n  Plotting optimized t-SNE visualization...")
    visualizer.plot_optimized_tsne(
        train_features, train_labels,
        "Optimized_tSNE_Training_Outliers_Removed",
        remove_outliers=True,
        contamination=***  # Hidden
    )
    
    print("\n  Generating weight visualizations...")
    visualizer.plot_spiking_som_weights(dual_som.spiking_som, "Spiking_SOM_Weights")
    visualizer.plot_lsm_weights(dual_som.liquid_layer, "LSM_Weights")
    visualizer.plot_weight_statistics(dual_som.spiking_som, dual_som.liquid_layer, "Weight_Statistics")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\n  Training:")
    print(f"    Accuracy: {train_metrics['accuracy']:.2%}")
    print(f"    F1-Macro: {train_metrics['f1_macro']:.4f}")
    print(f"    Coverage: {train_metrics['coverage']:.2%}")
    
    print(f"\n  Testing:")
    print(f"    Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"    F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"    Coverage: {test_metrics['coverage']:.2%}")
    
    gap = train_metrics['accuracy'] - test_metrics['accuracy']
    print(f"\n  Train-Test Gap: {gap:.2%}")
    
    print(f"\n  Dual SOM System Statistics:")
    print(f"    Easy samples ratio: {stats['easy_ratio']*100:.1f}%")
    print(f"    Hard samples ratio: {stats['hard_ratio']*100:.1f}%")
    print(f"    Average confidence: {stats['avg_confidence']:.4f}")
    
    print(f"\n  Visualization Features:")
    print(f"    Memory vector visualization: Generated (individual images)")
    print(f"    LSM feature vector visualization: Generated (individual images)")
    print(f"    Comparison visualization: Generated (individual images)")
    
    final_metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_macro'],
        'train_accuracy': train_metrics['accuracy'],
        'gap': gap,
        'dual_som': True,
        'feature_normalization': True,
        'outlier_removal': True,
        'memory_vector_visualization': True,
        'lsm_feature_visualization': True,
        'easy_samples_ratio': stats['easy_ratio'],
        'hard_samples_ratio': stats['hard_ratio'],
        'avg_confidence': stats['avg_confidence'],
        'total_time': total_time,
        'visualization_samples': n_visualization_samples,
    }
    
    print("\n  Generating training report...")
    visualizer.generate_report(final_metrics)
    
    print("\n" + "=" * 80)
    print("Dual SOM System (Feature Normalization Optimized + Memory/LSM Feature Visualization) Training Complete")
    print("=" * 80)
    
    return final_metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    results = main()
    
    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"Test Accuracy:            {results['test_accuracy']:.2%}")
    print(f"Test F1-Macro:            {results['test_f1']:.4f}")
    print(f"Train Accuracy:           {results['train_accuracy']:.2%}")
    print(f"Generalization Gap:       {results['gap']:.2%}")
    print(f"Memory Vector Vis:        {results['memory_vector_visualization']}")
    print(f"LSM Feature Vis:          {results['lsm_feature_visualization']}")
    print(f"Visualization Samples:    {results['visualization_samples']}")
    print(f"Easy Samples Ratio:       {results['easy_samples_ratio']*100:.1f}%")
    print(f"Hard Samples Ratio:       {results['hard_samples_ratio']*100:.1f}%")
    print(f"Total Time:               {results['total_time']:.2f}s")
