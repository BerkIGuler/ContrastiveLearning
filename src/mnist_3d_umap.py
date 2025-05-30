import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
from plotly.offline import plot
import time

def load_mnist_sample(n_samples=5000):
    """
    Load a sample of MNIST data for faster processing
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Sample data for faster processing
    indices = np.random.choice(len(mnist.data), n_samples, replace=False)
    X_sample = mnist.data[indices]
    y_sample = mnist.target[indices].astype(int)
    
    print(f"Loaded {n_samples} samples from MNIST")
    return X_sample, y_sample

def preprocess_data(X):
    """
    Normalize the pixel values
    """
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_umap_3d(X, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply UMAP dimensionality reduction to 3D
    """
    print("Applying UMAP dimensionality reduction...")
    start_time = time.time()
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    
    embedding = reducer.fit_transform(X)
    
    end_time = time.time()
    print(f"UMAP completed in {end_time - start_time:.2f} seconds")
    
    return embedding

def create_interactive_plot(embedding, labels, output_file='mnist_3d_umap.html'):
    """
    Create interactive 3D scatter plot with Plotly
    """
    print("Creating interactive 3D plot...")
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'z': embedding[:, 2],
        'digit': labels.astype(str)
    })
    
    # Define colors for each digit class
    colors = [
        '#1f77b4',  # 0 - blue
        '#ff7f0e',  # 1 - orange
        '#2ca02c',  # 2 - green
        '#d62728',  # 3 - red
        '#9467bd',  # 4 - purple
        '#8c564b',  # 5 - brown
        '#e377c2',  # 6 - pink
        '#7f7f7f',  # 7 - gray
        '#bcbd22',  # 8 - olive
        '#17becf'   # 9 - cyan
    ]
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='digit',
        color_discrete_sequence=colors,
        title='MNIST Dataset - 3D UMAP Visualization',
        labels={
            'x': 'UMAP Dimension 1',
            'y': 'UMAP Dimension 2',
            'z': 'UMAP Dimension 3',
            'digit': 'Digit Class'
        },
        hover_data={'digit': True}
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': 'MNIST Dataset - 3D UMAP Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            title=dict(
                text='Digit Classes',
                font=dict(size=16, color='black')
            ),
            font=dict(size=14),
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            itemsizing='constant',
            itemwidth=30
        ),
        # Make plot responsive and full screen
        autosize=True,
        margin=dict(l=0, r=150, t=50, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Update traces for better visibility
    fig.update_traces(
        marker=dict(
            size=4,
            opacity=0.8,
            line=dict(width=0)
        ),
        textfont=dict(size=12)
    )
    
    # Save as HTML file with full screen configuration
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
        'responsive': True
    }
    
    plot(fig, filename=output_file, auto_open=False, config=config)
    print(f"Interactive plot saved as '{output_file}'")
    
    return fig

def main():
    """
    Main function to run the complete pipeline
    """
    print("Starting MNIST 3D UMAP Visualization Pipeline")
    print("=" * 50)
    
    try:
        # Load MNIST data (using a sample for faster processing)
        X, y = load_mnist_sample(n_samples=5000)  # Adjust sample size as needed
        
        # Preprocess the data
        X_processed = preprocess_data(X)
        
        # Apply UMAP for 3D visualization
        embedding_3d = apply_umap_3d(X_processed)
        
        # Create interactive plot
        _ = create_interactive_plot(embedding_3d, y, output_file="../outputs/temperature/mnist_3d_umap.html")
        
        print("\n" + "=" * 50)
        print("Visualization completed successfully!")
        print("Open 'mnist_3d_umap.html' in your web browser to view the interactive plot.")
        print("\nFeatures of the visualization:")
        print("- Rotate: Click and drag")
        print("- Zoom: Scroll wheel")
        print("- Pan: Shift + click and drag")
        print("- Toggle classes: Click on legend items")
        print("- Hover for details")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Make sure you have all required packages installed:")
        print("pip install numpy pandas scikit-learn umap-learn plotly")

if __name__ == "__main__":
    main()