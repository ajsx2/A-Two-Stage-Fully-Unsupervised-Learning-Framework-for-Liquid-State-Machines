from config import *

def load_fashion_mnist_with_augmentation(n_train=60000, n_test=10000):
    try:
        from torchvision import datasets, transforms
        
        print("  Loading Fashion-MNIST dataset...")
        
        # Data path removed for GitHub upload
        train_dataset = datasets.FashionMNIST(
            root='***', train=True, download=True,  # Path hidden
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.FashionMNIST(
            root='***', train=False, download=True,  # Path hidden
            transform=transforms.ToTensor()
        )
        
        train_indices = torch.randperm(len(train_dataset))[:n_train]
        test_indices = torch.randperm(len(test_dataset))[:n_test]
        
        train_data = torch.stack([train_dataset[i][0] for i in train_indices])
        train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
        test_data = torch.stack([test_dataset[i][0] for i in test_indices])
        test_labels = torch.tensor([test_dataset[i][1] for i in test_indices])
        
        print(f"\n  Dataset split:")
        print(f"    Train: {train_data.shape[0]}")
        print(f"    Test: {test_data.shape[0]}")
        
        return train_data, train_labels, test_data, test_labels
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")


def comprehensive_evaluation(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    valid_mask = predictions != -1
    if not valid_mask.any():
        return {'accuracy': 0.0, 'coverage': 0.0, 'f1_macro': 0.0,
                'confusion_matrix': np.zeros((10, 10))}
    
    pred_valid = predictions[valid_mask]
    label_valid = labels[valid_mask]
    
    accuracy = (pred_valid == label_valid).mean()
    coverage = valid_mask.mean()
    f1_macro = f1_score(label_valid, pred_valid, average='macro', zero_division=0)
    cm = confusion_matrix(label_valid, pred_valid, labels=range(10))
    
    return {'accuracy': accuracy, 'coverage': coverage,
            'f1_macro': f1_macro, 'confusion_matrix': cm}
