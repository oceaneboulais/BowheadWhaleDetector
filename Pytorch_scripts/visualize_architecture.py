#!/usr/bin/env python3
"""
Autoencoder Architecture Visualizer

Generates publication-quality block diagrams showing:
- Encoder architecture with layer dimensions
- Latent space representation
- Decoder architecture with layer dimensions
- Data flow and tensor shapes

Supports both 3-layer and 4-layer (extra_conv) configurations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import argparse


class AutoencoderVisualizer:
    """Creates publication-quality architecture diagrams for autoencoders."""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, base_channels=32, extra_conv=False):
        self.nrow = nrow
        self.ncol = ncol
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.extra_conv = extra_conv
        
        # Calculate dimensions at each layer
        self._calculate_dimensions()
        
        # Color scheme
        self.colors = {
            'input': '#2E86AB',      # Blue
            'conv': '#A23B72',       # Purple
            'pool': '#F18F01',       # Orange
            'latent': '#C73E1D',     # Red
            'linear': '#6A994E',     # Green
            'deconv': '#BC4B51',     # Dark pink
            'output': '#2E86AB',     # Blue
            'batch_norm': '#8B8C7A', # Gray
        }
    
    def _calculate_dimensions(self):
        """Calculate tensor dimensions at each layer."""
        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        
        if self.extra_conv:
            # 4 pooling layers: divide by 16
            self.layers = [
                ('Input', 1, self.nrow, self.ncol),
                ('Conv1+BN+ReLU', c1, self.nrow, self.ncol),
                ('Pool1', c1, self.nrow//2, self.ncol//2),
                ('Conv2+BN+ReLU', c2, self.nrow//2, self.ncol//2),
                ('Pool2', c2, self.nrow//4, self.ncol//4),
                ('Conv3+BN+ReLU', c3, self.nrow//4, self.ncol//4),
                ('Pool3', c3, self.nrow//8, self.ncol//8),
                ('Conv4+BN+ReLU', c4, self.nrow//8, self.ncol//8),
                ('Pool4', c4, self.nrow//16, self.ncol//16),
                ('Flatten', c4 * (self.nrow//16) * (self.ncol//16), 1, 1),
                ('Linear1', self.latent_dim * 2, 1, 1),
                ('Latent', self.latent_dim, 1, 1),
                ('Linear2', self.latent_dim * 2, 1, 1),
                ('Linear3', c4 * (self.nrow//16) * (self.ncol//16), 1, 1),
                ('Reshape', c4, self.nrow//16, self.ncol//16),
                ('Deconv1+BN+ReLU', c3, self.nrow//8, self.ncol//8),
                ('Deconv2+BN+ReLU', c2, self.nrow//4, self.ncol//4),
                ('Deconv3+BN+ReLU', c1, self.nrow//2, self.ncol//2),
                ('Deconv4', 1, self.nrow, self.ncol),
                ('Output', 1, self.nrow, self.ncol),
            ]
            self.split_idx = 11  # Index of latent layer
        else:
            # 3 pooling layers: divide by 8
            self.layers = [
                ('Input', 1, self.nrow, self.ncol),
                ('Conv1+BN+ReLU', c1, self.nrow, self.ncol),
                ('Pool1', c1, self.nrow//2, self.ncol//2),
                ('Conv2+BN+ReLU', c2, self.nrow//2, self.ncol//2),
                ('Pool2', c2, self.nrow//4, self.ncol//4),
                ('Conv3+BN+ReLU', c3, self.nrow//4, self.ncol//4),
                ('Pool3', c3, self.nrow//8, self.ncol//8),
                ('Flatten', c3 * (self.nrow//8) * (self.ncol//8), 1, 1),
                ('Linear1', self.latent_dim * 2, 1, 1),
                ('Latent', self.latent_dim, 1, 1),
                ('Linear2', self.latent_dim * 2, 1, 1),
                ('Linear3', c3 * (self.nrow//8) * (self.ncol//8), 1, 1),
                ('Reshape', c3, self.nrow//8, self.ncol//8),
                ('Deconv1+BN+ReLU', c2, self.nrow//4, self.ncol//4),
                ('Deconv2+BN+ReLU', c1, self.nrow//2, self.ncol//2),
                ('Deconv3', 1, self.nrow, self.ncol),
                ('Output', 1, self.nrow, self.ncol),
            ]
            self.split_idx = 9  # Index of latent layer
    
    def _get_layer_color(self, name):
        """Get color for layer type."""
        if 'Input' in name or 'Output' in name:
            return self.colors['input']
        elif 'Conv' in name and 'Deconv' not in name:
            return self.colors['conv']
        elif 'Pool' in name:
            return self.colors['pool']
        elif 'Latent' in name:
            return self.colors['latent']
        elif 'Linear' in name:
            return self.colors['linear']
        elif 'Deconv' in name:
            return self.colors['deconv']
        else:
            return self.colors['batch_norm']
    
    def create_detailed_diagram(self, save_path='autoencoder_architecture.png', dpi=300):
        """Create a detailed horizontal flow diagram."""
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        title = f'Autoencoder Architecture ({self.nrow}×{self.ncol} → {self.latent_dim}D)'
        subtitle = f'{len([l for l in self.layers if "Conv" in l[0] and "Deconv" not in l[0]])} Conv Layers, ' + \
                   f'{self.base_channels} Base Channels, ' + \
                   f'{"4-layer (extra_conv)" if self.extra_conv else "3-layer (standard)"}'
        
        ax.text(11, 11.5, title, ha='center', va='center', fontsize=18, fontweight='bold')
        ax.text(11, 10.8, subtitle, ha='center', va='center', fontsize=11, style='italic', alpha=0.7)
        
        # Draw layers
        num_layers = len(self.layers)
        x_spacing = 20 / (num_layers - 1)
        y_center = 6
        
        for i, (name, channels, h, w) in enumerate(self.layers):
            x = 1 + i * x_spacing
            
            # Determine box size based on tensor dimensions
            if 'Flatten' in name or 'Linear' in name or 'Reshape' in name:
                box_width = 0.6
                box_height = 0.8
            elif 'Latent' in name:
                box_width = 1.0
                box_height = 1.5
            else:
                # Scale box size based on spatial dimensions
                scale = min(h / self.nrow, w / self.ncol)
                box_width = 0.3 + scale * 0.5
                box_height = 0.5 + scale * 2.5
            
            # Draw box
            color = self._get_layer_color(name)
            box = FancyBboxPatch(
                (x - box_width/2, y_center - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=2 if 'Latent' in name else 1.5,
                alpha=0.9 if 'Latent' in name else 0.7
            )
            ax.add_patch(box)
            
            # Layer name
            ax.text(x, y_center + box_height/2 + 0.3, name,
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Dimensions
            if h == 1 and w == 1:
                dim_text = f'{channels}'
            else:
                dim_text = f'{channels}×{h}×{w}'
            
            ax.text(x, y_center, dim_text,
                   ha='center', va='center', fontsize=7,
                   color='white', fontweight='bold')
            
            # Draw arrows between layers
            if i < num_layers - 1:
                x_next = 1 + (i + 1) * x_spacing
                _, next_channels, next_h, next_w = self.layers[i+1]
                next_box_width = 1.0 if 'Latent' in self.layers[i+1][0] else 0.6 if any(k in self.layers[i+1][0] for k in ['Flatten', 'Linear', 'Reshape']) else 0.3 + min(next_h/self.nrow, next_w/self.ncol) * 0.5
                
                arrow = FancyArrowPatch(
                    (x + box_width/2, y_center),
                    (x_next - next_box_width/2 - 0.05, y_center),
                    arrowstyle='->,head_width=0.3,head_length=0.3',
                    color='black',
                    linewidth=2 if i == self.split_idx - 1 or i == self.split_idx else 1.5,
                    alpha=0.6
                )
                ax.add_patch(arrow)
        
        # Add labels for encoder, latent, decoder sections
        encoder_x = 1 + (self.split_idx - 1) * x_spacing / 2
        decoder_x = 1 + (self.split_idx + 1 + num_layers - 1) * x_spacing / 2
        
        ax.text(encoder_x, 1.5, 'ENCODER', ha='center', va='center',
               fontsize=14, fontweight='bold', color=self.colors['conv'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=self.colors['conv'], linewidth=2))
        
        ax.text(1 + self.split_idx * x_spacing, 1.5, 'LATENT', ha='center', va='center',
               fontsize=14, fontweight='bold', color=self.colors['latent'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=self.colors['latent'], linewidth=2))
        
        ax.text(decoder_x, 1.5, 'DECODER', ha='center', va='center',
               fontsize=14, fontweight='bold', color=self.colors['deconv'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=self.colors['deconv'], linewidth=2))
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['conv'], edgecolor='black', label='Convolution + BN + ReLU'),
            mpatches.Patch(facecolor=self.colors['pool'], edgecolor='black', label='MaxPool2d (2×2)'),
            mpatches.Patch(facecolor=self.colors['linear'], edgecolor='black', label='Linear Layer'),
            mpatches.Patch(facecolor=self.colors['latent'], edgecolor='black', label='Latent Space'),
            mpatches.Patch(facecolor=self.colors['deconv'], edgecolor='black', label='Transposed Conv + BN + ReLU'),
        ]
        ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
                 frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved detailed diagram: {save_path}")
    
    def create_vertical_diagram(self, save_path='autoencoder_vertical.png', dpi=300):
        """Create a vertical flow diagram (top to bottom)."""
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 18)
        ax.axis('off')
        
        # Title
        title = f'Autoencoder Architecture - Vertical View'
        subtitle = f'{self.nrow}×{self.ncol} → {self.latent_dim}D Latent Space'
        ax.text(7, 17, title, ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(7, 16.3, subtitle, ha='center', va='center', fontsize=10, style='italic', alpha=0.7)
        
        # Draw layers from top to bottom
        num_layers = len(self.layers)
        y_start = 15
        y_spacing = 14 / (num_layers - 1)
        
        for i, (name, channels, h, w) in enumerate(self.layers):
            y = y_start - i * y_spacing
            x = 7
            
            # Box dimensions
            if 'Latent' in name:
                box_width = 3
                box_height = 1.2
            elif 'Flatten' in name or 'Linear' in name or 'Reshape' in name:
                box_width = 2.5
                box_height = 0.6
            else:
                scale = min(h / self.nrow, w / self.ncol)
                box_width = 1.5 + scale * 3
                box_height = 0.4 + scale * 0.8
            
            # Draw box
            color = self._get_layer_color(name)
            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=2 if 'Latent' in name else 1.5,
                alpha=0.9 if 'Latent' in name else 0.7
            )
            ax.add_patch(box)
            
            # Layer name
            ax.text(x - box_width/2 - 0.3, y, name,
                   ha='right', va='center', fontsize=9, fontweight='bold')
            
            # Dimensions
            if h == 1 and w == 1:
                dim_text = f'{channels}'
            else:
                dim_text = f'{channels}×{h}×{w}'
            
            ax.text(x, y, dim_text,
                   ha='center', va='center', fontsize=8,
                   color='white', fontweight='bold')
            
            # Draw arrows
            if i < num_layers - 1:
                y_next = y_start - (i + 1) * y_spacing
                arrow = FancyArrowPatch(
                    (x, y - box_height/2),
                    (x, y_next + box_height/2 + 0.1),
                    arrowstyle='->,head_width=0.4,head_length=0.4',
                    color='black',
                    linewidth=2 if i == self.split_idx - 1 or i == self.split_idx else 1.5,
                    alpha=0.6
                )
                ax.add_patch(arrow)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved vertical diagram: {save_path}")
    
    def create_simplified_diagram(self, save_path='autoencoder_simple.png', dpi=300):
        """Create a simplified, publication-ready diagram."""
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        title = f'Autoencoder: {self.nrow}×{self.ncol} → {self.latent_dim}D'
        ax.text(9, 7.2, title, ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Simplified representation
        sections = [
            ('Input\nSpectrogram', 2, f'1×{self.nrow}×{self.ncol}', self.colors['input']),
            ('Encoder\n(CNN)', 5, f'{self.nrow//8}×{self.ncol//8}×{self.base_channels*4}', self.colors['conv']),
            ('Latent\nSpace', 9, f'{self.latent_dim}D', self.colors['latent']),
            ('Decoder\n(CNN)', 13, f'{self.nrow//8}×{self.ncol//8}×{self.base_channels*4}', self.colors['deconv']),
            ('Output\nReconstruction', 16, f'1×{self.nrow}×{self.ncol}', self.colors['output']),
        ]
        
        for name, x, dims, color in sections:
            # Draw box
            is_latent = 'Latent' in name
            box_width = 2.5 if is_latent else 2
            box_height = 3 if is_latent else 2.5
            
            box = FancyBboxPatch(
                (x - box_width/2, 4 - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=3 if is_latent else 2,
                alpha=0.85
            )
            ax.add_patch(box)
            
            # Text
            ax.text(x, 4.5, name, ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
            ax.text(x, 3.5, dims, ha='center', va='center',
                   fontsize=9, color='white', style='italic')
        
        # Arrows
        for i in range(len(sections) - 1):
            x1 = sections[i][1] + 1
            x2 = sections[i+1][1] - 1
            arrow = FancyArrowPatch(
                (x1, 4), (x2, 4),
                arrowstyle='->,head_width=0.5,head_length=0.5',
                color='black', linewidth=3, alpha=0.7
            )
            ax.add_patch(arrow)
        
        # Add compression ratio
        input_size = self.nrow * self.ncol
        compression = input_size / self.latent_dim
        ax.text(9, 1.5, f'Compression: {compression:.1f}× ({input_size} → {self.latent_dim})',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=1.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved simplified diagram: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate autoencoder architecture diagrams',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--nrow', type=int, default=121, help='Input height (default: 121)')
    parser.add_argument('--ncol', type=int, default=104, help='Input width (default: 104)')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension (default: 32)')
    parser.add_argument('--channels', type=int, default=32, help='Base channels (default: 32)')
    parser.add_argument('--extra-conv', action='store_true', help='Use 4-layer architecture')
    parser.add_argument('--output-dir', type=str, default='architecture_diagrams',
                       help='Output directory (default: architecture_diagrams)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI (default: 300)')
    parser.add_argument('--style', type=str, default='all', 
                       choices=['all', 'detailed', 'vertical', 'simple'],
                       help='Diagram style to generate (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizer
    viz = AutoencoderVisualizer(
        nrow=args.nrow,
        ncol=args.ncol,
        latent_dim=args.latent_dim,
        base_channels=args.channels,
        extra_conv=args.extra_conv
    )
    
    print(f"\n{'='*70}")
    print(f"AUTOENCODER ARCHITECTURE VISUALIZER")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Input:       {args.nrow}×{args.ncol}")
    print(f"  Latent:      {args.latent_dim}D")
    print(f"  Channels:    {args.channels} (base)")
    print(f"  Layers:      {'4-layer (extra_conv)' if args.extra_conv else '3-layer (standard)'}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Generate diagrams
    arch_tag = f"{args.latent_dim}LD_{args.channels}C_{'4L' if args.extra_conv else '3L'}"
    
    if args.style in ['all', 'detailed']:
        viz.create_detailed_diagram(
            os.path.join(args.output_dir, f'autoencoder_detailed_{arch_tag}.png'),
            dpi=args.dpi
        )
    
    if args.style in ['all', 'vertical']:
        viz.create_vertical_diagram(
            os.path.join(args.output_dir, f'autoencoder_vertical_{arch_tag}.png'),
            dpi=args.dpi
        )
    
    if args.style in ['all', 'simple']:
        viz.create_simplified_diagram(
            os.path.join(args.output_dir, f'autoencoder_simple_{arch_tag}.png'),
            dpi=args.dpi
        )
    
    print(f"\n✓ All diagrams generated successfully!")
    print(f"  Location: {args.output_dir}/")
    print(f"\nGenerated files:")
    for f in os.listdir(args.output_dir):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
            print(f"  - {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
