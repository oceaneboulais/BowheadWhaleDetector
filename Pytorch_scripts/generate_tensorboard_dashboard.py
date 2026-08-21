#!/usr/bin/env python3
"""
Generate TensorBoard Dashboard for Latent Space Embeddings

This script creates a comprehensive TensorBoard visualization dashboard including:
- Latent space embeddings (32D, UMAP 2D/3D, PaCMAP 2D/3D)
- Training metrics and loss curves
- Sample spectrograms and reconstructions
- Cluster visualizations

USAGE:
    python3 generate_tensorboard_dashboard.py --dir <model_directory>
    
EXAMPLE:
    python3 generate_tensorboard_dashboard.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
    
After running, start TensorBoard with:
    tensorboard --logdir=<model_directory>/tensorboard_logs
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)


def load_embeddings(directory):
    """Load all available embeddings from model directory."""
    embeddings = {}
    
    # Load latent embeddings
    latent_path = os.path.join(directory, 'MATLAB', 'latent_embeddings.mat')
    if os.path.exists(latent_path):
        print(f"Loading latent embeddings: {latent_path}")
        data = loadmat(latent_path)
        embeddings['latent'] = {
            'embeddings': data['latent_embeddings'],
            'clusters': data.get('clusters', None),
            'optimal_k': data.get('optimal_k', np.array([[0]]))[0, 0] if 'optimal_k' in data else 0,
            'dataset_label': data.get('dataset_label', 'Unknown')
        }
        # Flatten clusters if needed
        if embeddings['latent']['clusters'] is not None and embeddings['latent']['clusters'].ndim > 1:
            embeddings['latent']['clusters'] = embeddings['latent']['clusters'].flatten()
    
    # Load UMAP 2D embeddings
    umap_2d_path = os.path.join(directory, 'UMAP', 'umap_embeddings_2d.mat')
    if os.path.exists(umap_2d_path):
        print(f"Loading UMAP 2D embeddings: {umap_2d_path}")
        data = loadmat(umap_2d_path)
        embeddings['umap_2d'] = {
            'embeddings': data['umap_embeddings_2d'],
            'clusters': data.get('clusters', None)
        }
        if embeddings['umap_2d']['clusters'] is not None and embeddings['umap_2d']['clusters'].ndim > 1:
            embeddings['umap_2d']['clusters'] = embeddings['umap_2d']['clusters'].flatten()
    
    # Load UMAP 3D embeddings
    umap_3d_path = os.path.join(directory, 'UMAP', 'umap_embeddings_3d.mat')
    if os.path.exists(umap_3d_path):
        print(f"Loading UMAP 3D embeddings: {umap_3d_path}")
        data = loadmat(umap_3d_path)
        embeddings['umap_3d'] = {
            'embeddings': data['umap_embeddings_3d'],
            'clusters': data.get('clusters', None)
        }
        if embeddings['umap_3d']['clusters'] is not None and embeddings['umap_3d']['clusters'].ndim > 1:
            embeddings['umap_3d']['clusters'] = embeddings['umap_3d']['clusters'].flatten()
    
    # Load PaCMAP 2D embeddings
    pacmap_2d_path = os.path.join(directory, 'PaCMAP', 'pacmap_embeddings_2d.mat')
    if os.path.exists(pacmap_2d_path):
        print(f"Loading PaCMAP 2D embeddings: {pacmap_2d_path}")
        data = loadmat(pacmap_2d_path)
        embeddings['pacmap_2d'] = {
            'embeddings': data['pacmap_embeddings_2d'],
            'clusters': data.get('clusters', None)
        }
        if embeddings['pacmap_2d']['clusters'] is not None and embeddings['pacmap_2d']['clusters'].ndim > 1:
            embeddings['pacmap_2d']['clusters'] = embeddings['pacmap_2d']['clusters'].flatten()
    
    # Load PaCMAP 3D embeddings
    pacmap_3d_path = os.path.join(directory, 'PaCMAP', 'pacmap_embeddings_3d.mat')
    if os.path.exists(pacmap_3d_path):
        print(f"Loading PaCMAP 3D embeddings: {pacmap_3d_path}")
        data = loadmat(pacmap_3d_path)
        embeddings['pacmap_3d'] = {
            'embeddings': data['pacmap_embeddings_3d'],
            'clusters': data.get('clusters', None)
        }
        if embeddings['pacmap_3d']['clusters'] is not None and embeddings['pacmap_3d']['clusters'].ndim > 1:
            embeddings['pacmap_3d']['clusters'] = embeddings['pacmap_3d']['clusters'].flatten()
    
    # Load PaCMAP 5D embeddings
    pacmap_5d_path = os.path.join(directory, 'PaCMAP', 'pacmap_embeddings_5d.mat')
    if os.path.exists(pacmap_5d_path):
        print(f"Loading PaCMAP 5D embeddings: {pacmap_5d_path}")
        data = loadmat(pacmap_5d_path)
        embeddings['pacmap_5d'] = {
            'embeddings': data['pacmap_embeddings_5d'],
            'clusters': data.get('clusters', None)
        }
        if embeddings['pacmap_5d']['clusters'] is not None and embeddings['pacmap_5d']['clusters'].ndim > 1:
            embeddings['pacmap_5d']['clusters'] = embeddings['pacmap_5d']['clusters'].flatten()
    
    return embeddings


def load_reconstruction_data(directory):
    """Load reconstruction samples for visualization."""
    recon_path = os.path.join(directory, 'MATLAB', 'reconstruction_data.mat')
    if not os.path.exists(recon_path):
        return None
    
    print(f"Loading reconstruction data: {recon_path}")
    data = loadmat(recon_path)
    
    return {
        'original': data.get('originals', None),
        'reconstructed': data.get('reconstructions', None),
        'filenames': data.get('filenames', None)
    }


def create_embedding_figure(embeddings, clusters, title, figsize=(10, 8)):
    """Create matplotlib figure for 2D embeddings."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if clusters is not None:
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=clusters,
            cmap='tab10',
            s=1,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            s=1,
            alpha=0.6
        )
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return fig


def write_embeddings_to_tensorboard(writer, embeddings, max_samples=10000):
    """Write embeddings to TensorBoard projector."""
    print("\nWriting embeddings to TensorBoard...")
    
    for emb_name, emb_data in embeddings.items():
        emb_array = emb_data['embeddings']
        clusters = emb_data.get('clusters', None)
        
        # Subsample if too many points (for performance)
        if len(emb_array) > max_samples:
            print(f"  Subsampling {emb_name} from {len(emb_array)} to {max_samples} points")
            indices = np.random.choice(len(emb_array), max_samples, replace=False)
            emb_array = emb_array[indices]
            if clusters is not None:
                clusters = clusters[indices]
        
        # Convert to proper format
        emb_tensor = emb_array.astype(np.float32)
        
        # Create metadata (cluster labels)
        metadata = None
        if clusters is not None:
            metadata = [f"Cluster_{int(c)}" for c in clusters]
        
        # Write to TensorBoard
        writer.add_embedding(
            emb_tensor,
            metadata=metadata,
            tag=f'embeddings/{emb_name}',
            global_step=0
        )
        print(f"  ✓ Written {emb_name}: shape {emb_tensor.shape}")


def create_3d_projection_figure(embeddings, clusters, title, timestamp):
    """Create 3D projection figure with multiple views."""
    fig = plt.figure(figsize=(18, 6))
    
    # Add title with timestamp
    fig.suptitle(f'{title}\nGenerated: {timestamp}', fontsize=16, fontweight='bold')
    
    # View 1: XY
    ax1 = fig.add_subplot(131, projection='3d')
    if clusters is not None:
        scatter = ax1.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                             c=clusters, cmap='viridis', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Cluster')
    else:
        ax1.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   c='blue', s=1, alpha=0.6)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_zlabel('Dimension 3')
    ax1.set_title('View: XYZ')
    ax1.view_init(elev=20, azim=45)
    
    # View 2: Side view
    ax2 = fig.add_subplot(132, projection='3d')
    if clusters is not None:
        ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   c=clusters, cmap='viridis', s=1, alpha=0.6)
    else:
        ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   c='blue', s=1, alpha=0.6)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_zlabel('Dimension 3')
    ax2.set_title('View: Side')
    ax2.view_init(elev=0, azim=0)
    
    # View 3: Top view
    ax3 = fig.add_subplot(133, projection='3d')
    if clusters is not None:
        ax3.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   c=clusters, cmap='viridis', s=1, alpha=0.6)
    else:
        ax3.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   c='blue', s=1, alpha=0.6)
    ax3.set_xlabel('Dimension 1')
    ax3.set_ylabel('Dimension 2')
    ax3.set_zlabel('Dimension 3')
    ax3.set_title('View: Top')
    ax3.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    return fig


def create_5d_parallel_coords_figure(embeddings, clusters, title, timestamp):
    """Create parallel coordinates figure for 5D embeddings."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Subsample for visualization
    max_lines = 2000
    if len(embeddings) > max_lines:
        indices = np.random.choice(len(embeddings), max_lines, replace=False)
        embeddings_sub = embeddings[indices]
        clusters_sub = clusters[indices] if clusters is not None else None
    else:
        embeddings_sub = embeddings
        clusters_sub = clusters
    
    # Normalize each dimension to [0, 1] for parallel coordinates
    normalized = (embeddings_sub - embeddings_sub.min(axis=0)) / (embeddings_sub.max(axis=0) - embeddings_sub.min(axis=0) + 1e-8)
    
    # Plot lines
    x_positions = np.arange(5)
    if clusters_sub is not None:
        colors = plt.cm.viridis(clusters_sub / clusters_sub.max())
        for i in range(len(normalized)):
            ax.plot(x_positions, normalized[i], c=colors[i], alpha=0.3, linewidth=0.5)
    else:
        for i in range(len(normalized)):
            ax.plot(x_positions, normalized[i], c='blue', alpha=0.3, linewidth=0.5)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'Dim {i+1}' for i in range(5)])
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'{title} (Parallel Coordinates)\nGenerated: {timestamp}\nShowing {len(normalized):,} / {len(embeddings):,} samples',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_clusters_for_k(embeddings, k):
    """Re-run KMeans with k clusters on latent embeddings; return updated embeddings dict."""
    import copy
    if 'latent' not in embeddings:
        return embeddings

    latent_raw = embeddings['latent']['embeddings']
    print(f"  Running KMeans k={k} on latent embeddings ({latent_raw.shape[0]:,} samples)...")
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(latent_raw).astype(float)

    # Apply the same sample-order labels to every embedding type
    updated = copy.deepcopy(embeddings)
    for name in updated:
        updated[name]['clusters'] = labels
    return updated


def write_visualizations_to_tensorboard(writer, embeddings, directory, global_step=0):
    """Write embedding visualization figures to TensorBoard."""
    print(f"\nWriting visualization figures (k={global_step if global_step > 0 else 'original'})...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # PRIORITY 1: PaCMAP 3D (most important - shown first)
    if 'pacmap_3d' in embeddings:
        emb_data = embeddings['pacmap_3d']
        # Subsample for visualization
        max_viz = 5000
        if len(emb_data['embeddings']) > max_viz:
            indices = np.random.choice(len(emb_data['embeddings']), max_viz, replace=False)
            viz_embeddings = emb_data['embeddings'][indices]
            viz_clusters = emb_data.get('clusters')[indices] if emb_data.get('clusters') is not None else None
        else:
            viz_embeddings = emb_data['embeddings']
            viz_clusters = emb_data.get('clusters')
        
        fig = create_3d_projection_figure(
            viz_embeddings,
            viz_clusters,
            'PaCMAP 3D Projection - PRIMARY VIEW',
            timestamp
        )
        writer.add_figure('AAA_PRIMARY/pacmap_3d_multi_view', fig, global_step=global_step)
        plt.close(fig)
        print(f"  ✓ Saved PaCMAP 3D (PRIMARY - top of dashboard)")
    
    # PRIORITY 2: PaCMAP 5D (second most important)
    if 'pacmap_5d' in embeddings:
        emb_data = embeddings['pacmap_5d']
        
        # Create parallel coordinates view
        fig = create_5d_parallel_coords_figure(
            emb_data['embeddings'],
            emb_data.get('clusters'),
            'PaCMAP 5D Projection - SECONDARY VIEW',
            timestamp
        )
        writer.add_figure('AAA_PRIMARY/pacmap_5d_parallel_coords', fig, global_step=global_step)
        plt.close(fig)
        print(f"  ✓ Saved PaCMAP 5D parallel coordinates (PRIMARY - top of dashboard)")
        
        # Also create 3D projection of first 3 dimensions
        fig = create_3d_projection_figure(
            emb_data['embeddings'][:, :3],
            emb_data.get('clusters'),
            'PaCMAP 5D Projection (First 3 Dimensions)',
            timestamp
        )
        writer.add_figure('AAA_PRIMARY/pacmap_5d_first3dims', fig, global_step=global_step)
        plt.close(fig)
        print(f"  ✓ Saved PaCMAP 5D first 3 dims view")
    
    # 2D embeddings (lower priority)
    for emb_name in ['umap_2d', 'pacmap_2d']:
        if emb_name in embeddings:
            emb_data = embeddings[emb_name]
            fig = create_embedding_figure(
                emb_data['embeddings'],
                emb_data.get('clusters', None),
                f"{emb_name.upper().replace('_', ' ')} Projection\nGenerated: {timestamp}"
            )
            writer.add_figure(f'visualizations/{emb_name}', fig, global_step=global_step)
            plt.close(fig)
            print(f"  ✓ Saved {emb_name} figure")
    
    # Load and display static PaCMAP HTML plots info
    pacmap_dir = os.path.join(directory, 'PaCMAP')
    if os.path.exists(pacmap_dir):
        html_files = []
        for fname in ['pacmap_3d_interactive.html', 'pacmap_5d_parallel_coordinates.html', 'pacmap_5d_linked_3d_views.html']:
            fpath = os.path.join(pacmap_dir, fname)
            if os.path.exists(fpath):
                file_size = os.path.getsize(fpath) / 1024 / 1024
                html_files.append(f"- **{fname}** ({file_size:.1f} MB)")
        
        if html_files:
            html_info = "\n".join(html_files)
            writer.add_text('interactive_visualizations/available_html_files', 
                          f"# Interactive HTML Visualizations\n\nAvailable in `{pacmap_dir}:`\n\n{html_info}\n\n**Open these files in a web browser for fully interactive 3D rotation and exploration.**",
                          global_step=0)
            print(f"  ✓ Added HTML file information")


def write_reconstruction_samples(writer, recon_data, num_samples=16):
    """Write reconstruction comparison images to TensorBoard."""
    if recon_data is None:
        return
    
    print("\nWriting reconstruction samples...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    originals = recon_data['original']
    reconstructed = recon_data['reconstructed']
    
    if originals is None or reconstructed is None:
        print("  No reconstruction data available")
        return
    
    # Select random samples
    num_available = min(len(originals), num_samples)
    indices = np.random.choice(len(originals), num_available, replace=False)
    
    # Create comparison figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    fig.suptitle(f'Autoencoder Reconstruction Samples\nGenerated: {timestamp}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    for idx, sample_idx in enumerate(indices[:8]):
        # Original
        axes[idx * 2].imshow(originals[sample_idx], cmap='viridis', aspect='auto')
        axes[idx * 2].set_title(f'Original {sample_idx}', fontsize=10)
        axes[idx * 2].axis('off')
        
        # Reconstructed
        axes[idx * 2 + 1].imshow(reconstructed[sample_idx], cmap='viridis', aspect='auto')
        axes[idx * 2 + 1].set_title(f'Reconstructed {sample_idx}', fontsize=10)
        axes[idx * 2 + 1].axis('off')
    
    plt.tight_layout()
    writer.add_figure('reconstructions/samples', fig, global_step=0)
    plt.close(fig)
    print(f"  ✓ Saved {num_available} reconstruction samples")


def write_plot_descriptions(writer):
    """Write an explanatory paragraph for each plot type to TensorBoard's TEXT tab."""
    print("\nWriting plot descriptions...")

    descriptions = {
        'AAA_PRIMARY/pacmap_3d_multi_view': (
            "## PaCMAP 3D Multi-View\n\n"
            "This is the **primary visualization**. PaCMAP (Pairwise Controlled Manifold Approximation) "
            "compresses the 32-dimensional latent space into 3 dimensions while preserving both local "
            "neighbourhood structure and global inter-cluster relationships. Three camera angles are shown "
            "side-by-side (oblique, side, top-down) so you can inspect the cluster geometry from multiple "
            "perspectives. Each point is one spectrogram window; colour indicates the k-means cluster "
            "assignment. Tight, well-separated blobs suggest the autoencoder has learnt discriminative "
            "features for different call types."
        ),
        'AAA_PRIMARY/pacmap_5d_parallel_coords': (
            "## PaCMAP 5D — Parallel Coordinates\n\n"
            "A 5-dimensional PaCMAP embedding is shown as a **parallel coordinates plot**. Each vertical "
            "axis represents one latent dimension (normalised to [0, 1]) and each line traces one "
            "spectrogram sample across all five axes. Colour encodes cluster label. Bands of lines that "
            "stay close together across axes indicate strong cluster coherence; crossings between axes "
            "reveal correlations or anti-correlations between dimensions. This view captures more variance "
            "than the 3D plot while remaining human-readable."
        ),
        'AAA_PRIMARY/pacmap_5d_first3dims': (
            "## PaCMAP 5D — First 3 Dimensions (3D Projection)\n\n"
            "The first three dimensions of the 5D PaCMAP embedding are projected into 3D space using the "
            "same multi-view layout as the primary PaCMAP 3D panel. Because these three dimensions "
            "typically capture the most variance, this view is a quick sanity-check that the 5D structure "
            "is consistent with the dedicated 3D run. Differences between this panel and the primary 3D "
            "plot highlight information encoded in dimensions 4 and 5."
        ),
        'visualizations/umap_2d': (
            "## UMAP 2D Projection\n\n"
            "UMAP (Uniform Manifold Approximation and Projection) reduces the 32-dimensional latent "
            "vectors to 2 dimensions, optimising for both local and global structure. The 2D scatter plot "
            "is the most compact overview of the embedding landscape. Each dot is one spectrogram window; "
            "colours represent k-means cluster labels. UMAP tends to produce more visually separated "
            "clusters than t-SNE at the cost of less interpretable inter-cluster distances."
        ),
        'visualizations/pacmap_2d': (
            "## PaCMAP 2D Projection\n\n"
            "A 2-dimensional PaCMAP reduction of the latent space. Compared to the 3D and 5D views this "
            "discards more information, but is useful for quick qualitative comparisons and for exporting "
            "publication-ready flat scatter plots. Cluster colours match those used in all other panels. "
            "Overlapping clusters here that are separated in 3D indicate structure that requires the extra "
            "dimension to resolve."
        ),
        'reconstructions/samples': (
            "## Autoencoder Reconstruction Samples\n\n"
            "Side-by-side comparison of **original** (left) and **autoencoder-reconstructed** (right) "
            "log-mel spectrograms. Each pair shows a randomly selected bowhead whale call window. "
            "High visual similarity between originals and reconstructions indicates that the bottleneck "
            "latent code retains the acoustic detail needed to reproduce the spectrogram. Blurry or "
            "artefact-laden reconstructions suggest the model is under-fitting or that the latent "
            "dimension is too small. The colour map is viridis: dark blue = low energy, yellow = "
            "high energy."
        ),
    }

    for tag, text in descriptions.items():
        writer.add_text(f'plot_descriptions/{tag.replace("/", "_")}', text, global_step=0)

    print(f"  ✓ Written descriptions for {len(descriptions)} plot types")


def write_summary_text(writer, embeddings, directory):
    """Write summary statistics as text to TensorBoard."""
    print("\nWriting summary text...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_only = datetime.now().strftime('%B %d, %Y')
    time_only = datetime.now().strftime('%I:%M:%S %p')
    
    # Extract model name and date from directory
    model_name = os.path.basename(directory)
    if 'Date' in model_name:
        model_date = model_name.split('Date')[-1].replace('.dir', '')
    else:
        model_date = 'Unknown'
    
    summary_lines = [
        "# 🎯 Bowhead Whale Autoencoder Dashboard",
        "",
        "## 📅 Run Information",
        f"- **Dashboard Generated:** {date_only} at {time_only}",
        f"- **Model Directory:** `{model_name}`",
        f"- **Model Training Date:** {model_date}",
        f"- **Full Path:** `{directory}`",
        "",
        "---",
        "",
        "## 📊 Available Embeddings",
        ""
    ]
    
    # Sort embeddings to show PaCMAP first
    sorted_emb_names = sorted(embeddings.keys(), key=lambda x: (0 if 'pacmap' in x else 1, x))
    
    for emb_name in sorted_emb_names:
        emb_data = embeddings[emb_name]
        shape = emb_data['embeddings'].shape
        clusters = emb_data.get('clusters', None)
        n_clusters = len(np.unique(clusters)) if clusters is not None else 0
        
        # Add emoji based on type
        if 'pacmap_5d' in emb_name:
            emoji = "🌟"
            label = "PaCMAP 5D (PRIMARY - 5 Dimensions)"
        elif 'pacmap_3d' in emb_name:
            emoji = "⭐"
            label = "PaCMAP 3D (PRIMARY - 3 Dimensions)"
        elif 'pacmap_2d' in emb_name:
            emoji = "📌"
            label = "PaCMAP 2D"
        elif 'umap' in emb_name:
            emoji = "🗺️"
            label = emb_name.upper().replace('_', ' ')
        else:
            emoji = "🔹"
            label = emb_name.upper().replace('_', ' ')
        
        summary_lines.append(f"### {emoji} {label}")
        summary_lines.append(f"- **Shape:** {shape[0]:,} samples × {shape[1]} dimensions")
        if n_clusters > 0:
            summary_lines.append(f"- **Clusters:** {n_clusters}")
        summary_lines.append("")
    
    summary_lines.extend([
        "---",
        "",
        "## 🔬 Dataset Information",
        ""
    ])
    
    if 'latent' in embeddings:
        dataset_label = embeddings['latent'].get('dataset_label', 'Unknown')
        optimal_k = embeddings['latent'].get('optimal_k', 0)
        summary_lines.append(f"- **Dataset:** {dataset_label}")
        summary_lines.append(f"- **Optimal K:** {optimal_k}")
        summary_lines.append(f"- **Latent Dimensions:** {embeddings['latent']['embeddings'].shape[1]}")
    
    summary_lines.extend([
        "",
        "---",
        "",
        "## 📂 Navigation Guide",
        "",
        "### IMAGES Tab (👈 Start Here!)",
        "- **AAA_PRIMARY/** - PaCMAP 3D and 5D visualizations at the TOP",
        "- **visualizations/** - 2D embedding projections",
        "- **reconstructions/** - Autoencoder reconstruction samples",
        "",
        "### PROJECTOR Tab",
        "- Interactive 3D/2D exploration of embeddings",
        "- Select different embedding types from dropdown",
        "- Use PCA/t-SNE/UMAP for visualization",
        "",
        "### TEXT Tab",
        "- This summary and interactive HTML file locations",
        "",
        "---",
        "",
        f"**Last Updated:** {timestamp}",
    ])
    
    summary_text = "\n".join(summary_lines)
    writer.add_text('AAA_DASHBOARD_INFO/summary', summary_text, global_step=0)
    print("  ✓ Saved summary text (top of TEXT tab)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate TensorBoard dashboard for latent space embeddings'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum samples for embedding projector (default: 10000)'
    )
    parser.add_argument(
        '--no-info',
        action='store_true',
        default=False,
        help='Skip writing plot description paragraphs to the TEXT tab'
    )
    parser.add_argument(
        '--launch',
        action='store_true',
        default=False,
        help='Automatically launch TensorBoard after generation'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port for TensorBoard server (default: 6006)'
    )

    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("TensorBoard Dashboard Generation")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print(f"Max samples for projector: {args.max_samples}")
    print("=" * 70)
    
    # Create TensorBoard log directory
    log_dir = os.path.join(args.dir, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"\nTensorBoard logs: {log_dir}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load embeddings
    print("\n" + "=" * 70)
    embeddings = load_embeddings(args.dir)
    
    if not embeddings:
        print("ERROR: No embeddings found in directory")
        sys.exit(1)
    
    print(f"\nLoaded {len(embeddings)} embedding types")
    
    # Load reconstruction data
    print("\n" + "=" * 70)
    recon_data = load_reconstruction_data(args.dir)
    
    # Write to TensorBoard
    print("\n" + "=" * 70)
    write_embeddings_to_tensorboard(writer, embeddings, max_samples=args.max_samples)

    # Write figures for each k (2-5); TensorBoard step slider = k slider
    print("\n" + "=" * 70)
    print("Writing figures for k=2..5 (use the TensorBoard step slider to change k)")
    for k in range(2, 6):
        embeddings_k = compute_clusters_for_k(embeddings, k)
        write_visualizations_to_tensorboard(writer, embeddings_k, args.dir, global_step=k)

    write_reconstruction_samples(writer, recon_data)
    write_summary_text(writer, embeddings, args.dir)
    if not args.no_info:
        write_plot_descriptions(writer)
    
    # Close writer
    writer.close()
    
    print("\n" + "=" * 70)
    print("✓ TensorBoard dashboard generation complete!")
    print("=" * 70)
    print(f"\nTo view the dashboard, run:")
    print(f"  tensorboard --logdir={log_dir} --port={args.port}")
    print(f"\nThen open your browser to: http://localhost:{args.port}")
    print("\n")

    if args.launch:
        import subprocess
        print(f"Launching TensorBoard on port {args.port}...")
        subprocess.Popen(
            ['tensorboard', f'--logdir={log_dir}', f'--port={args.port}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time, webbrowser
        time.sleep(3)
        webbrowser.open(f'http://localhost:{args.port}')
        print(f"✓ TensorBoard launched — http://localhost:{args.port}")


if __name__ == '__main__':
    main()
