import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    def __init__(self, embedding_dim=128):
        super(SimpleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))             # 7x7 -> 7x7
        x = self.pool(x)                      # 7x7 -> 3x3
        
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

def get_augmentation_transforms():
    """Define augmentation transforms for positive pairs"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def train_contrastive_model(temperature, epochs=20, batch_size=128):
    """Train contrastive model with given temperature"""
    print(f"Training with temperature τ = {temperature}")
    
    # Data loading
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=base_transform
    )
    
    contrastive_dataset = ContrastiveDataset(
        train_dataset, 
        transform_positive=get_augmentation_transforms()
    )
    
    train_loader = DataLoader(
        contrastive_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEncoder(embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    return model, losses

def extract_embeddings(model, dataset_type='test', max_samples=2000):
    """Extract embeddings from trained model"""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=(dataset_type=='train'), download=True, transform=base_transform
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

def create_3d_visualization(embeddings_dict, labels, method='tsne'):
    """Create interactive 3D visualization of embeddings"""
    fig = make_subplots(
        rows=1, cols=len(embeddings_dict),
        subplot_titles=[f'τ = {tau}' for tau in embeddings_dict.keys()],
        specs=[[{'type': 'scatter3d'} for _ in range(len(embeddings_dict))]]
    )
    
    colors = px.colors.qualitative.Set1[:10]  # Colors for 10 digit classes
    
    for col_idx, (tau, embeddings) in enumerate(embeddings_dict.items()):
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        else:  # PCA
            reducer = PCA(n_components=3, random_state=42)
        
        embeddings_3d = reducer.fit_transform(embeddings)
        
        # Add traces for each digit class
        for digit in range(10):
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
        height=600,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    return fig

def analyze_embedding_quality(embeddings_dict, labels):
    """Analyze embedding quality metrics"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    
    quality_metrics = {}
    
    for tau, embeddings in embeddings_dict.items():
        # Silhouette score (higher is better)
        sil_score = silhouette_score(embeddings, labels)
        
        # Intra-class vs inter-class distance ratio
        class_centroids = []
        intra_distances = []
        
        for digit in range(10):
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
                for j in range(i+1, len(centroids)):
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

def plot_training_curves(losses_dict):
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
        width=800,
        height=500
    )
    
    return fig

def main():
    """Main execution function"""
    print("🚀 Starting MNIST Contrastive Learning Experiments")
    print("=" * 60)
    
    # Different temperature values to experiment with
    temperatures = [0.05, 0.1, 0.3, 0.5]
    
    # Storage for results
    models = {}
    losses_dict = {}
    embeddings_dict = {}
    
    # Train models with different temperatures
    for tau in temperatures:
        print(f"\n🔥 Experiment with τ = {tau}")
        print("-" * 40)
        
        model, losses = train_contrastive_model(tau, epochs=15)
        models[tau] = model
        losses_dict[tau] = losses
        
        # Extract embeddings
        print("📊 Extracting embeddings...")
        embeddings, labels = extract_embeddings(model, max_samples=1500)
        embeddings_dict[tau] = embeddings
    
    print("\n📈 Creating visualizations...")
    
    # Plot training curves
    loss_fig = plot_training_curves(losses_dict)
    loss_fig.show()
    
    # Create 3D embedding visualizations
    tsne_fig = create_3d_visualization(embeddings_dict, labels, method='tsne')
    tsne_fig.show()
    
    pca_fig = create_3d_visualization(embeddings_dict, labels, method='pca')
    pca_fig.show()
    
    # Analyze embedding quality
    print("\n📊 Embedding Quality Analysis:")
    print("=" * 50)
    quality_metrics = analyze_embedding_quality(embeddings_dict, labels)
    
    for tau, metrics in quality_metrics.items():
        print(f"\nTemperature τ = {tau}:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Intra-class Distance: {metrics['intra_class_distance']:.4f}")
        print(f"  Inter-class Distance: {metrics['inter_class_distance']:.4f}")
        print(f"  Separation Ratio: {metrics['separation_ratio']:.4f}")
    
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
        width=800,
        height=500
    )
    
    metrics_fig.show()
    
    print("\n✅ Experiments completed!")
    print("\n🔍 Key Observations:")
    print("• Lower τ values create sharper, more confident predictions")
    print("• Higher τ values lead to softer, more exploratory learning")
    print("• Optimal τ balances between over-confidence and under-confidence")
    print("• 3D visualizations show clustering quality for different τ values")

if __name__ == "__main__":
    main()
