import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import umap
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import base64
from io import BytesIO
from PIL import Image
import time

# Global variables to store data
embedding_data = None
labels_data = None
original_images_data = None


def load_mnist_sample(n_samples=2000):
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
    return X_sample, y_sample, indices


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


def image_to_base64(image_array, size=(128, 128)):
    """
    Convert MNIST image array to base64 string for display in Dash
    """
    # Reshape 784-dim vector to 28x28 image
    img = image_array.reshape(28, 28)

    # Normalize to 0-255 range
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # Create PIL image and resize for better visibility
    pil_img = Image.fromarray(img, mode='L')
    pil_img = pil_img.resize(size, Image.NEAREST)  # Keep pixelated look

    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def create_3d_plot(embedding, labels):
    """
    Create the 3D scatter plot
    """
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
        '#17becf'  # 9 - cyan
    ]

    fig = go.Figure()

    # Create a single trace with all points to maintain index mapping
    fig.add_trace(go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=[colors[label] for label in labels],
            opacity=0.8,
            line=dict(width=0)
        ),
        text=[f"Digit: {label}" for label in labels],  # Simple text for hover
        customdata=list(range(len(labels))),  # Store indices
        hovertemplate=(
                "<b>%{text}</b><br>" +
                "UMAP 1: %{x:.3f}<br>" +
                "UMAP 2: %{y:.3f}<br>" +
                "UMAP 3: %{z:.3f}<br>" +
                "Index: %{customdata}<br>" +
                "<extra></extra>"
        ),
        showlegend=False
    ))

    # Add invisible traces for legend
    for digit in range(10):
        mask = labels == digit
        if np.sum(mask) > 0:
            fig.add_trace(go.Scatter3d(
                x=[None],  # Invisible points
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[digit],
                    opacity=0.8
                ),
                name=f'Digit {digit}',
                showlegend=True,
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'MNIST Dataset - 3D UMAP Visualization (Hover for Images)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
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
                font=dict(size=14, color='black')
            ),
            font=dict(size=12),
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=1.02
        ),
        margin=dict(l=0, r=150, t=50, b=0),
        height=700
    )

    return fig


def prepare_data():
    """
    Load and process all data
    """
    global embedding_data, labels_data, original_images_data

    # Load MNIST data
    X, y, indices = load_mnist_sample(n_samples=2000)

    # Preprocess the data
    X_processed = preprocess_data(X)

    # Apply UMAP for 3D visualization
    embedding_3d = apply_umap_3d(X_processed)

    # Store in global variables
    embedding_data = embedding_3d
    labels_data = y
    original_images_data = X

    print("Data preparation completed!")
    return embedding_3d, y, X


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("MNIST 3D UMAP Visualization with Image Hover",
                style={'text-align': 'center', 'margin-bottom': '20px'}),

        html.Div([
            # Main plot container
            html.Div([
                dcc.Graph(
                    id='3d-scatter-plot',
                    style={'height': '700px'}
                )
            ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Image display container
            html.Div([
                html.Div(id='hover-info',
                         style={
                             'border': '2px solid #ddd',
                             'border-radius': '10px',
                             'padding': '20px',
                             'margin': '20px',
                             'background-color': '#f9f9f9',
                             'text-align': 'center',
                             'min-height': '300px'
                         })
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
        ])
    ]),

    html.Div([
        html.P("Instructions:", style={'font-weight': 'bold', 'margin-top': '20px'}),
        html.Ul([
            html.Li("Hover over any point in the 3D plot to see the corresponding MNIST digit image"),
            html.Li("Rotate: Click and drag"),
            html.Li("Zoom: Scroll wheel"),
            html.Li("Pan: Shift + click and drag"),
            html.Li("Toggle classes: Click on legend items")
        ])
    ], style={'margin': '20px', 'padding': '20px', 'background-color': '#f0f0f0', 'border-radius': '5px'})
])


# Callback for hover events
@app.callback(
    Output('hover-info', 'children'),
    Input('3d-scatter-plot', 'hoverData')
)
def display_hover_data(hoverData):
    if hoverData is None:
        return html.Div([
            html.H3("Hover over a point to see the MNIST image",
                    style={'color': '#666', 'text-align': 'center', 'margin-top': '100px'})
        ])

    # Extract point information
    point = hoverData['points'][0]

    # Get index from customdata, with fallback to pointIndex
    if 'customdata' in point and point['customdata'] is not None:
        index = point['customdata']
    elif 'pointIndex' in point:
        index = point['pointIndex']
    else:
        # Fallback: try to find the closest point
        x_coord = point['x']
        y_coord = point['y']
        z_coord = point['z']

        # Find closest point in embedding space
        distances = np.sqrt((embedding_data[:, 0] - x_coord) ** 2 +
                            (embedding_data[:, 1] - y_coord) ** 2 +
                            (embedding_data[:, 2] - z_coord) ** 2)
        index = np.argmin(distances)

    # Ensure index is valid
    if index >= len(labels_data):
        index = 0

    digit = labels_data[index]
    x_coord = point.get('x', 0)
    y_coord = point.get('y', 0)
    z_coord = point.get('z', 0)

    # Get the image
    image_base64 = image_to_base64(original_images_data[index])

    return html.Div([
        html.H3(f"Digit: {digit}", style={'color': '#333', 'margin-bottom': '15px'}),
        html.Img(src=image_base64,
                 style={'max-width': '200px', 'max-height': '200px', 'border': '2px solid #333'}),
        html.Div([
            html.P(f"UMAP Coordinates:", style={'font-weight': 'bold', 'margin': '15px 0 5px 0'}),
            html.P(f"X: {x_coord:.3f}", style={'margin': '2px 0'}),
            html.P(f"Y: {y_coord:.3f}", style={'margin': '2px 0'}),
            html.P(f"Z: {z_coord:.3f}", style={'margin': '2px 0'}),
            html.P(f"Index: {index}", style={'margin': '10px 0 2px 0', 'font-size': '12px', 'color': '#666'})
        ])
    ])


# Callback to initialize the plot
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Input('3d-scatter-plot', 'id')
)
def update_plot(plot_id):
    if embedding_data is None:
        # Return empty plot if data not ready
        return go.Figure().add_annotation(
            text="Loading data...",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font_size=20
        )

    return create_3d_plot(embedding_data, labels_data)


def main():
    """
    Main function to prepare data and run the Dash app
    """
    print("Starting MNIST 3D UMAP Dash App")
    print("=" * 50)

    try:
        # Prepare all data
        prepare_data()

        print("\n" + "=" * 50)
        print("Starting Dash server...")
        print("Open your browser and go to: http://127.0.0.1:8050")
        print("=" * 50)

        # Run the app
        app.run(debug=True, port=8050)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Make sure you have all required packages installed:")
        print("pip install numpy pandas scikit-learn umap-learn plotly dash pillow")


if __name__ == "__main__":
    main()