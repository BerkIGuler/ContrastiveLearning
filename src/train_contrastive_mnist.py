import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")
    umap = None
import random
from tqdm import tqdm
import warnings
import yaml
from pathlib import Path
import os
from datetime import datetime

warnings.filterwarnings('ignore')


def load_config(config_path='./configs/mnist_train_infonce_config.yml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found!")
        print("Creating default configuration...")

        # Create default config
        default_config = create_default_config()

        # Create configs directory if it doesn't exist
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save default config
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)

        print(f"✅ Default configuration saved to {config_path}")
        return default_config


def create_default_config():
    """Create default configuration dictionary"""
    return {
        'experiment': {
            'name': 'MNIST Contrastive Learning Temperature Study',
            'description': 'Studying the effect of temperature parameter on contrastive learning',
            'random_seed': 42
        },
        'model': {
            'embedding_dim': 32,
            'architecture': 'SimpleEncoder',
            'dropout_rate': 0.3
        },
        'training': {
            'temperatures': [0.05, 0.1, 0.3, 0.5, 0.7],
            'epochs': 10,
            'batch_size': 256,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'num_workers': 2
        },
        'data': {
            'dataset': 'MNIST',
            'data_root': './data',
            'max_samples_visualization': 1500,
            'normalization': {
                'mean': [0.1307],
                'std': [0.3081]
            }
        },
        'augmentation': {
            'rotation_degrees': 15,
            'translate_range': 0.1,
            'scale_range': [0.8, 1.0],
            'crop_size': 28
        },
        'visualization': {
            'methods': ['tsne', 'pca', 'umap'],
            'tsne_perplexity': 30,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'plot_height': 600,
            'plot_width': 800,
            'max_samples': 2000
        },
        'analysis': {
            'quality_metrics': ['silhouette_score', 'separation_ratio'],
            'num_classes': 10
        },
        'output': {
            'base_dir': './outputs',
            'save_models': True,
            'save_embeddings': True,
            'save_plots': True
        }
    }


def create_output_directory(config):
    """Create output directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = config['output']['base_dir']
    experiment_name = config['experiment']['name'].lower().replace(' ', '_')

    output_dir = Path(output_base) / f"{experiment_name}_{timestamp}"

    # Create subdirectories
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'embeddings').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (output_dir / 'configs').mkdir(parents=True, exist_ok=True)

    # Save the configuration used for this experiment
    config_save_path = output_dir / 'configs' / 'experiment_config.yml'
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)

    print(f"📁 Output directory created: {output_dir}")
    return output_dir


def save_model_checkpoint(model, tau, losses, config, output_dir):
    """Save model checkpoint immediately after training"""
    if not config['output']['save_models']:
        return

    model_dir = output_dir / 'models'
    model_filename = f"model_tau_{tau:.3f}.pth"
    model_path = model_dir / model_filename

    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': tau,
        'training_losses': losses,
        'config': config,
        'model_architecture': config['model']['architecture'],
        'embedding_dim': config['model']['embedding_dim'],
        'epochs_trained': len(losses),
        'final_loss': losses[-1] if losses else None,
        'timestamp': datetime.now().isoformat()
    }

    # Save the checkpoint
    torch.save(checkpoint, model_path)
    print(f"💾 Model saved: {model_path}")

    # Also save losses separately as numpy array for easy loading
    losses_path = model_dir / f"losses_tau_{tau:.3f}.npy"
    np.save(losses_path, np.array(losses))

    return model_path


def load_model_checkpoint(checkpoint_path, config):
    """Load a saved model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model with the same architecture
    model = SimpleEncoder(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint


# Set random seeds for reproducibility
def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ContrastiveDataset(Dataset):
    """Dataset that generates positive pairs through augmentation"""

    def __init__(self, dataset, transform_positive=None):
        self.dataset = dataset
        self.transform_positive = transform_positive

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Original image (anchor)
        anchor = image

        # Generate positive pair through augmentation
        if self.transform_positive:
            positive = self.transform_positive(image)
        else:
            positive = image

        return anchor, positive, label


class SimpleEncoder(nn.Module):
    """Simple CNN encoder for MNIST"""

    def __init__(self, config):
        super(SimpleEncoder, self).__init__()
        embedding_dim = config['model']['embedding_dim']
        dropout_rate = config['model']['dropout_rate']

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after convolutions
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))  # 7x7 -> 7x7
        x = self.pool(x)  # 7x7 -> 3x3

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # L2 normalize embeddings
        return F.normalize(x, p=2, dim=1)


class InfoNCELoss(nn.Module):
    """InfoNCE Loss implementation"""

    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives=None):
        # If negatives not provided, use in-batch negatives
        if negatives is None:
            # Create batch of negatives by shifting positive samples
            batch_size = anchor.size(0)
            negatives = []
            for i in range(batch_size):
                neg_indices = list(range(batch_size))
                neg_indices.remove(i)
                negatives.append(positive[neg_indices])
            negatives = torch.stack(negatives)

        batch_size = anchor.size(0)
        losses = []

        for i in range(batch_size):
            # Positive similarity
            pos_sim = torch.dot(anchor[i], positive[i]) / self.temperature

            # Negative similarities
            neg_sims = torch.matmul(negatives[i], anchor[i]) / self.temperature

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss = -F.log_softmax(logits, dim=0)[0]
            losses.append(loss)

        return torch.stack(losses).mean()


def get_augmentation_transforms(config):
    """Define augmentation transforms for positive pairs"""
    aug_config = config['augmentation']
    norm_config = config['data']['normalization']

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(aug_config['rotation_degrees']),
        transforms.RandomAffine(
            degrees=0,
            translate=(aug_config['translate_range'], aug_config['translate_range'])
        ),
        transforms.RandomResizedCrop(
            aug_config['crop_size'],
            scale=tuple(aug_config['scale_range'])
        ),
        transforms.ToTensor(),
        transforms.Normalize(tuple(norm_config['mean']), tuple(norm_config['std']))
    ])


def get_base_transforms(config):
    """Get base transforms for the dataset"""
    norm_config = config['data']['normalization']
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(norm_config['mean']), tuple(norm_config['std']))
    ])


def train_contrastive_model(temperature, config, output_dir):
    """Train contrastive model with given temperature and save immediately"""
    print(f"Training with temperature τ = {temperature}")

    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_workers = config['training']['num_workers']

    # Data loading
    base_transform = get_base_transforms(config)

    train_dataset = torchvision.datasets.MNIST(
        root=config['data']['data_root'],
        train=True,
        download=True,
        transform=base_transform
    )

    contrastive_dataset = ContrastiveDataset(
        train_dataset,
        transform_positive=get_augmentation_transforms(config)
    )

    train_loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEncoder(config).to(device)

    # Configure optimizer based on config
    optimizer_name = config['training']['optimizer']
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = InfoNCELoss(temperature=temperature)

    # Training loop
    model.train()
    losses = []

    for epoch in tqdm(range(epochs), desc=f"τ={temperature}"):
        epoch_losses = []
        for batch_idx, (anchor, positive, labels) in enumerate(train_loader):
            anchor, positive = anchor.to(device), positive.to(device)

            optimizer.zero_grad()

            # Get embeddings
            anchor_embed = model(anchor)
            positive_embed = model(positive)

            # Compute loss
            loss = criterion(anchor_embed, positive_embed)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

    # Save model immediately after training is completed
    model_path = save_model_checkpoint(model, temperature, losses, config, output_dir)
    print(f"✅ Training completed and model saved for τ = {temperature}")

    return model, losses, model_path


def save_embeddings(embeddings_dict, labels, output_dir, config):
    """Save embeddings to disk"""
    if not config['output']['save_embeddings']:
        return

    embeddings_dir = output_dir / 'embeddings'

    # Save each temperature's embeddings
    for tau, embeddings in embeddings_dict.items():
        embeddings_file = embeddings_dir / f"embeddings_tau_{tau:.3f}.npy"
        np.save(embeddings_file, embeddings)

    # Save labels (same for all temperatures)
    labels_file = embeddings_dir / "labels.npy"
    np.save(labels_file, labels)

    print(f"💾 Embeddings saved to {embeddings_dir}")


def extract_embeddings(model, config, dataset_type='test'):
    """Extract embeddings from trained model"""
    max_samples = config['visualization']['max_samples']

    base_transform = get_base_transforms(config)

    dataset = torchvision.datasets.MNIST(
        root=config['data']['data_root'],
        train=(dataset_type == 'train'),
        download=True,
        transform=base_transform
    )

    # Limit samples for visualization
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embed = model(images)
            embeddings.append(embed.cpu().numpy())
            labels.extend(lbls.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def create_3d_visualization(embeddings_dict, labels, config, method='tsne'):
    """Create interactive 3D visualization of embeddings"""
    vis_config = config['visualization']

    fig = make_subplots(
        rows=1, cols=len(embeddings_dict),
        subplot_titles=[f'τ = {tau}' for tau in embeddings_dict.keys()],
        specs=[[{'type': 'scatter3d'} for _ in range(len(embeddings_dict))]]
    )

    # Use multiple color palettes to ensure we have enough colors
    num_classes = config['analysis']['num_classes']
    colors = []

    # Combine multiple color palettes to get enough colors
    color_palettes = [
        px.colors.qualitative.Set1,
        px.colors.qualitative.Set2,
        px.colors.qualitative.Pastel1,
        px.colors.qualitative.Dark2
    ]

    for palette in color_palettes:
        colors.extend(palette)
        if len(colors) >= num_classes:
            break

    # Ensure we have at least num_classes colors
    colors = colors[:num_classes]

    for col_idx, (tau, embeddings) in enumerate(embeddings_dict.items()):
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=3,
                random_state=config['experiment']['random_seed'],
                perplexity=vis_config['tsne_perplexity']
            )
        else:  # PCA
            reducer = PCA(n_components=3, random_state=config['experiment']['random_seed'])

        embeddings_3d = reducer.fit_transform(embeddings)

        # Add traces for each digit class
        for digit in range(config['analysis']['num_classes']):
            mask = labels == digit
            if np.any(mask):
                fig.add_trace(
                    go.Scatter3d(
                        x=embeddings_3d[mask, 0],
                        y=embeddings_3d[mask, 1],
                        z=embeddings_3d[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=colors[digit],
                            opacity=0.7
                        ),
                        name=f'Digit {digit}',
                        legendgroup=f'digit_{digit}',
                        showlegend=(col_idx == 0)  # Show legend only for first subplot
                    ),
                    row=1, col=col_idx + 1
                )

    fig.update_layout(
        title=f'3D Embedding Visualization ({method.upper()}): Effect of Temperature τ',
        height=vis_config['plot_height'],
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )

    return fig


def analyze_embedding_quality(embeddings_dict, labels, config):
    """Analyze embedding quality metrics"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score

    quality_metrics = {}
    num_classes = config['analysis']['num_classes']

    for tau, embeddings in embeddings_dict.items():
        # Silhouette score (higher is better)
        sil_score = silhouette_score(embeddings, labels)

        # Intra-class vs inter-class distance ratio
        class_centroids = []
        intra_distances = []

        for digit in range(num_classes):
            mask = labels == digit
            if np.any(mask):
                class_embed = embeddings[mask]
                centroid = np.mean(class_embed, axis=0)
                class_centroids.append(centroid)

                # Average intra-class distance
                if len(class_embed) > 1:
                    distances = np.linalg.norm(class_embed - centroid, axis=1)
                    intra_distances.extend(distances)

        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0

        # Inter-class distances (between centroids)
        if len(class_centroids) > 1:
            centroids = np.array(class_centroids)
            inter_distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    inter_distances.append(dist)
            avg_inter_distance = np.mean(inter_distances)
        else:
            avg_inter_distance = 0

        quality_metrics[tau] = {
            'silhouette_score': sil_score,
            'intra_class_distance': avg_intra_distance,
            'inter_class_distance': avg_inter_distance,
            'separation_ratio': avg_inter_distance / avg_intra_distance if avg_intra_distance > 0 else 0
        }

    return quality_metrics


def plot_training_curves(losses_dict, config):
    """Plot training loss curves for different temperatures"""
    fig = go.Figure()

    for tau, losses in losses_dict.items():
        fig.add_trace(go.Scatter(
            x=list(range(1, len(losses) + 1)),
            y=losses,
            mode='lines+markers',
            name=f'τ = {tau}',
            line=dict(width=2)
        ))

    fig.update_layout(
        title='Training Loss Curves: Effect of Temperature Parameter',
        xaxis_title='Epoch',
        yaxis_title='InfoNCE Loss',
        hovermode='x unified',
        width=config['visualization']['plot_width'],
        height=config['visualization']['plot_height']
    )

    return fig


def save_plots(figures, output_dir, config):
    """Save all plots to disk"""
    if not config['output']['save_plots']:
        return

    plots_dir = output_dir / 'plots'

    for name, fig in figures.items():
        # Save as HTML (interactive) - always works
        html_path = plots_dir / f"{name}.html"
        fig.write_html(html_path)

        # Save as PNG (static) - requires kaleido
        try:
            png_path = plots_dir / f"{name}.png"
            fig.write_image(png_path, width=config['visualization']['plot_width'],
                            height=config['visualization']['plot_height'])
        except Exception as e:
            print(f"⚠️  Could not save PNG for {name}: {e}")
            print("   HTML version saved successfully. Install 'kaleido' for PNG export.")

    print(f"💾 Plots saved to {plots_dir}")


def main():
    """Main execution function"""
    print("🚀 Starting MNIST Contrastive Learning Experiments")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Create output directory
    output_dir = create_output_directory(config)

    # Set random seeds
    set_random_seeds(config['experiment']['random_seed'])

    # Print experiment info
    print(f"📋 Experiment: {config['experiment']['name']}")
    print(f"📝 Description: {config['experiment']['description']}")
    print(f"🌡️  Temperature values: {config['training']['temperatures']}")
    print(f"🏃 Epochs: {config['training']['epochs']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🧠 Embedding dimension: {config['model']['embedding_dim']}")

    # Different temperature values from config
    temperatures = config['training']['temperatures']

    # Storage for results
    models = {}
    losses_dict = {}
    embeddings_dict = {}
    model_paths = {}

    # Train models with different temperatures
    for tau in temperatures:
        print(f"\n🔥 Experiment with τ = {tau}")
        print("-" * 40)

        model, losses, model_path = train_contrastive_model(tau, config, output_dir)
        models[tau] = model
        losses_dict[tau] = losses
        model_paths[tau] = model_path

        # Extract embeddings
        print("📊 Extracting embeddings...")
        embeddings, labels = extract_embeddings(model, config)
        embeddings_dict[tau] = embeddings

    # Save embeddings
    save_embeddings(embeddings_dict, labels, output_dir, config)

    print("\n📈 Creating visualizations...")

    # Dictionary to store all figures for saving
    figures = {}

    # Plot training curves
    loss_fig = plot_training_curves(losses_dict, config)
    figures['training_curves'] = loss_fig
    loss_fig.show()

    # Create 3D embedding visualizations for each method in config
    for method in config['visualization']['methods']:
        print(f"Creating {method.upper()} visualization...")
        try:
            vis_fig = create_3d_visualization(embeddings_dict, labels, config, method=method)
            if vis_fig is not None:  # Check if visualization was created successfully
                figures[f'embeddings_3d_{method}'] = vis_fig
                vis_fig.show()
        except Exception as e:
            print(f"⚠️  Could not create {method.upper()} visualization: {e}")
            if method == 'umap':
                print("   Install UMAP with: pip install umap-learn")

    # Analyze embedding quality
    print("\n📊 Embedding Quality Analysis:")
    print("=" * 50)
    quality_metrics = analyze_embedding_quality(embeddings_dict, labels, config)

    for tau, metrics in quality_metrics.items():
        print(f"\nTemperature τ = {tau}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")

    # Create quality metrics visualization
    metrics_fig = go.Figure()

    tau_values = list(quality_metrics.keys())
    silhouette_scores = [quality_metrics[tau]['silhouette_score'] for tau in tau_values]
    separation_ratios = [quality_metrics[tau]['separation_ratio'] for tau in tau_values]

    metrics_fig.add_trace(go.Scatter(
        x=tau_values,
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        yaxis='y1'
    ))

    metrics_fig.add_trace(go.Scatter(
        x=tau_values,
        y=separation_ratios,
        mode='lines+markers',
        name='Separation Ratio',
        yaxis='y2'
    ))

    metrics_fig.update_layout(
        title='Embedding Quality vs Temperature Parameter',
        xaxis_title='Temperature (τ)',
        yaxis=dict(title='Silhouette Score', side='left'),
        yaxis2=dict(title='Separation Ratio', side='right', overlaying='y'),
        hovermode='x unified',
        width=config['visualization']['plot_width'],
        height=config['visualization']['plot_height']
    )

    figures['quality_metrics'] = metrics_fig
    metrics_fig.show()

    # Save all plots
    save_plots(figures, output_dir, config)

    # Save quality metrics as JSON
    import json

    # Convert numpy values to regular Python floats for JSON serialization
    json_metrics = {}
    for tau, metrics in quality_metrics.items():
        json_metrics[str(tau)] = {
            metric_name: float(value) for metric_name, value in metrics.items()
        }

    metrics_path = output_dir / 'quality_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)

    print("\n✅ Experiments completed!")
    print(f"📁 All outputs saved to: {output_dir}")
    print("\n📄 Saved files:")
    print(f"  • Models: {len(model_paths)} checkpoints in {output_dir / 'models'}")
    print(f"  • Embeddings: {len(embeddings_dict)} arrays in {output_dir / 'embeddings'}")
    print(f"  • Plots: {len(figures)} visualizations in {output_dir / 'plots'}")
    print(f"  • Config: experiment_config.yml in {output_dir / 'configs'}")
    print(f"  • Metrics: quality_metrics.json")

    print("\n🔍 Key Observations:")
    print("• Lower τ values create sharper, more confident predictions")
    print("• Higher τ values lead to softer, more exploratory learning")
    print("• Optimal τ balances between over-confidence and under-confidence")
    print("• 3D visualizations show clustering quality for different τ values")
    print(f"\n⚙️  Configuration used: {config['experiment']['name']}")

    # Print model loading example
    print(f"\n💡 To load a saved model:")
    print(f"model, checkpoint = load_model_checkpoint('{model_paths[temperatures[0]]}', config)")

    return output_dir, model_paths, quality_metrics


if __name__ == "__main__":
    main()