#!/usr/bin/env python3
"""
Autoencoder Architecture Visualizer - PlotNeuralNet Integration

Generates LaTeX code for publication-quality diagrams using PlotNeuralNet.
Requires: https://github.com/HarisIqbal88/PlotNeuralNet

SETUP INSTRUCTIONS:
1. Clone PlotNeuralNet:
   cd ~/Public/Bowhead_DL_Project
   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git

2. Install LaTeX (if not already installed):
   MacOS: brew install --cask mactex
   Ubuntu: sudo apt-get install texlive-full

3. Generate diagram:
   python3 visualize_architecture_plotneuralnet.py --latent-dim 32 --channels 32
   cd architecture_diagrams_latex
   pdflatex autoencoder_arch.tex

OUTPUT:
- autoencoder_arch.tex (LaTeX source)
- autoencoder_arch.pdf (rendered diagram)
"""

import os
import argparse
from pathlib import Path


class PlotNeuralNetGenerator:
    """Generates PlotNeuralNet LaTeX code for autoencoder architectures."""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, base_channels=32, 
                 extra_conv=False, in_channels=1):
        self.nrow = nrow
        self.ncol = ncol
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.extra_conv = extra_conv
        self.in_channels = in_channels
        
        # Calculate dimensions
        self._calculate_layers()
    
    def _calculate_layers(self):
        """Calculate layer configurations."""
        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        
        if self.extra_conv:
            self.encoder_layers = [
                ('conv1', c1, self.nrow, self.ncol, 3),
                ('pool1', c1, self.nrow//2, self.ncol//2, 2),
                ('conv2', c2, self.nrow//2, self.ncol//2, 3),
                ('pool2', c2, self.nrow//4, self.ncol//4, 2),
                ('conv3', c3, self.nrow//4, self.ncol//4, 3),
                ('pool3', c3, self.nrow//8, self.ncol//8, 2),
                ('conv4', c4, self.nrow//8, self.ncol//8, 3),
                ('pool4', c4, self.nrow//16, self.ncol//16, 2),
            ]
            self.decoder_layers = [
                ('deconv1', c3, self.nrow//8, self.ncol//8, 2),
                ('deconv2', c2, self.nrow//4, self.ncol//4, 2),
                ('deconv3', c1, self.nrow//2, self.ncol//2, 2),
                ('deconv4', 1, self.nrow, self.ncol, 2),
            ]
        else:
            self.encoder_layers = [
                ('conv1', c1, self.nrow, self.ncol, 3),
                ('pool1', c1, self.nrow//2, self.ncol//2, 2),
                ('conv2', c2, self.nrow//2, self.ncol//2, 3),
                ('pool2', c2, self.nrow//4, self.ncol//4, 2),
                ('conv3', c3, self.nrow//4, self.ncol//4, 3),
                ('pool3', c3, self.nrow//8, self.ncol//8, 2),
            ]
            self.decoder_layers = [
                ('deconv1', c2, self.nrow//4, self.ncol//4, 2),
                ('deconv2', c1, self.nrow//2, self.ncol//2, 2),
                ('deconv3', 1, self.nrow, self.ncol, 2),
            ]
    
    def generate_latex(self, output_path='autoencoder_arch.tex'):
        """Generate PlotNeuralNet LaTeX code."""
        
        latex_content = self._generate_header()
        latex_content += self._generate_encoder()
        latex_content += self._generate_latent()
        latex_content += self._generate_decoder()
        latex_content += self._generate_footer()
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        print(f"✓ Generated LaTeX: {output_path}")
        print(f"\nTo compile:")
        print(f"  cd {os.path.dirname(output_path)}")
        print(f"  pdflatex {os.path.basename(output_path)}")
        
        return output_path
    
    def _generate_header(self):
        """Generate LaTeX document header."""
        return f"""\\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage{{import}}
\\subimport{{../../PlotNeuralNet/layers/}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}}

\\def\\ConvColor{{rgb:yellow,5;red,2.5;white,5}}
\\def\\ConvReluColor{{rgb:yellow,5;red,5;white,5}}
\\def\\PoolColor{{rgb:red,1;black,0.3}}
\\def\\DcnvColor{{rgb:blue,5;green,2.5;white,5}}
\\def\\SoftmaxColor{{rgb:magenta,5;black,7}}
\\def\\LatentColor{{rgb:red,5;black,3}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=\\edgecolor,opacity=0.7]

% Title
\\node[canvas is zy plane at x=0] (title) at (-3,0,12) {{\\Huge\\bf Autoencoder Architecture}};
\\node[canvas is zy plane at x=0] (subtitle) at (-3,0,10.5) {{\\Large {self.nrow}$\\times${self.ncol} $\\rightarrow$ {self.latent_dim}D Latent ({"2-channel" if self.in_channels == 2 else "1-channel"} input)}};

% Input
\\pic[shift={{(0,0,0)}}] at (0,0,0) {{Box={{
    name=input,
    caption=Input,
    xlabel={{{self.in_channels}}},
    zlabel={self.nrow},
    fill=\\ConvColor,
    height={self.nrow/10},
    width={self.in_channels*2},
    depth={self.ncol/10}
}}}};

"""
    
    def _generate_encoder(self):
        """Generate encoder layers."""
        content = "% ENCODER\n"
        prev_name = "input"
        x_offset = 0
        
        for i, (name, channels, h, w, kernel) in enumerate(self.encoder_layers):
            x_offset += 3
            
            if 'pool' in name:
                content += f"""\\pic[shift={{({x_offset},0,0)}}] at ({prev_name}-east) {{{{Box={{
    name={name},
    caption={name.upper()},
    fill=\\PoolColor,
    opacity=0.5,
    height={h/10},
    width={channels/10},
    depth={w/10}
}}}}}};

"""
            else:
                content += f"""\\pic[shift={{({x_offset},0,0)}}] at ({prev_name}-east) {{{{RightBandedBox={{
    name={name},
    caption={name.upper()},
    xlabel={{{channels}}},
    zlabel={h},
    fill=\\ConvReluColor,
    bandfill=\\ConvColor,
    height={h/10},
    width={channels/10},
    depth={w/10}
}}}}}};

"""
            
            # Connection
            content += f"\\draw [connection]  ({prev_name}-east) -- node {{\\midarrow}} ({name}-west);\n\n"
            prev_name = name
        
        return content
    
    def _generate_latent(self):
        """Generate latent space representation."""
        last_encoder = self.encoder_layers[-1][0]
        
        return f"""% LATENT SPACE
\\pic[shift={{(4,0,0)}}] at ({last_encoder}-east) {{{{Ball={{
    name=latent,
    caption=LATENT,
    fill=\\LatentColor,
    opacity=0.8,
    radius=2.5,
    logo=\\Large {self.latent_dim}D
}}}}}};

\\draw [connection]  ({last_encoder}-east) -- node {{\\midarrow}} (latent-west);

"""
    
    def _generate_decoder(self):
        """Generate decoder layers."""
        content = "% DECODER\n"
        prev_name = "latent"
        x_offset = 0
        
        for i, (name, channels, h, w, kernel) in enumerate(self.decoder_layers):
            x_offset += 4
            
            content += f"""\\pic[shift={{({x_offset},0,0)}}] at (latent-east) {{{{RightBandedBox={{
    name={name},
    caption={name.upper()},
    xlabel={{{channels}}},
    zlabel={h},
    fill=\\DcnvColor,
    bandfill=\\ConvColor,
    height={h/10},
    width={channels/10},
    depth={w/10}
}}}}}};

\\draw [connection]  ({prev_name}-east) -- node {{\\midarrow}} ({name}-west);

"""
            prev_name = name
        
        # Output
        content += f"""% OUTPUT
\\pic[shift={{(4,0,0)}}] at ({prev_name}-east) {{{{Box={{
    name=output,
    caption=Output,
    xlabel={{1}},
    zlabel={self.nrow},
    fill=\\ConvColor,
    height={self.nrow/10},
    width=2,
    depth={self.ncol/10}
}}}}}};

\\draw [connection]  ({prev_name}-east) -- node {{\\midarrow}} (output-west);

"""
        return content
    
    def _generate_footer(self):
        """Generate LaTeX document footer."""
        return """\\end{tikzpicture}
\\end{document}
"""


def generate_readme(output_dir):
    """Generate README with setup instructions."""
    readme = """# PlotNeuralNet Autoencoder Diagrams

## Quick Start

### 1. Install PlotNeuralNet
```bash
cd /Users/oboulais/Public/Bowhead_DL_Project
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

### 2. Install LaTeX (if needed)
```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-full

# After install, refresh PATH
eval "$(/usr/libexec/path_helper)"
```

### 3. Generate Diagram
```bash
# Generate LaTeX code
python3 visualize_architecture_plotneuralnet.py --latent-dim 32 --channels 32

# Compile to PDF
cd architecture_diagrams_latex
pdflatex autoencoder_arch.tex
```

## Advantages of PlotNeuralNet

✅ **Publication-quality** 3D diagrams with proper LaTeX typography
✅ **Highly customizable** colors, sizes, and labels
✅ **Vector graphics** (PDF) - infinite zoom without quality loss
✅ **LaTeX integration** - use in papers, presentations, posters
✅ **Professional appearance** - commonly used in top-tier papers

## Comparison

| Feature | Matplotlib (current) | PlotNeuralNet |
|---------|---------------------|---------------|
| Output Format | PNG/JPG (raster) | PDF (vector) |
| Typography | Basic | LaTeX quality |
| 3D Rendering | 2D projections | True 3D with perspective |
| Customization | Python code | LaTeX styling |
| File Size (typical) | 200-500 KB | 50-150 KB |
| Zoom Quality | Pixelated at high zoom | Perfect at any zoom |
| Use in Papers | Acceptable | Preferred |

## Generated Files

- `autoencoder_arch.tex` - LaTeX source code
- `autoencoder_arch.pdf` - Compiled diagram (publication-ready)
- `autoencoder_arch.aux`, `.log` - LaTeX compilation artifacts (can delete)

## Customization

Edit the generated `.tex` file to customize:
- Colors: `\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}`
- Spacing: `shift={(X,Y,Z)}` coordinates
- Labels: `caption=`, `xlabel=`, `zlabel=`
- Sizes: `height=`, `width=`, `depth=`

## Troubleshooting

**Error: `! LaTeX Error: File 'init.tex' not found`**
- Solution: Make sure PlotNeuralNet is cloned in the correct location
- Check path: `/Users/oboulais/Public/Bowhead_DL_Project/PlotNeuralNet/`

**Error: `pdflatex: command not found`**
- Solution: Install MacTeX and refresh PATH
- Run: `eval "$(/usr/libexec/path_helper)"`

**Diagram looks wrong/misaligned**
- Solution: Adjust spacing in `shift={(x,y,z)}` values
- Increase x-offset between layers for more spacing

## Example Usage

```bash
# Default 121×104 → 32D
python3 visualize_architecture_plotneuralnet.py

# Custom configuration
python3 visualize_architecture_plotneuralnet.py \\
    --latent-dim 16 \\
    --channels 64 \\
    --extra-conv \\
    --in-channels 2

# Compile
cd architecture_diagrams_latex
pdflatex autoencoder_arch.tex
open autoencoder_arch.pdf  # macOS
```

## Integration with Papers

To include in LaTeX documents:
```latex
\\documentclass{article}
\\usepackage{graphicx}
\\begin{document}
\\begin{figure}[h]
    \\centering
    \\includegraphics[width=\\textwidth]{architecture_diagrams_latex/autoencoder_arch.pdf}
    \\caption{Autoencoder architecture with 32D latent space.}
    \\label{fig:autoencoder}
\\end{figure}
\\end{document}
```
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"✓ Generated README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PlotNeuralNet LaTeX diagrams for autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration
  python3 visualize_architecture_plotneuralnet.py
  
  # 2-channel input (SNR+NTV)
  python3 visualize_architecture_plotneuralnet.py --in-channels 2
  
  # 4-layer architecture
  python3 visualize_architecture_plotneuralnet.py --extra-conv
  
  # Full custom
  python3 visualize_architecture_plotneuralnet.py --latent-dim 16 --channels 64 --in-channels 2

After generation:
  cd architecture_diagrams_latex
  pdflatex autoencoder_arch.tex
  open autoencoder_arch.pdf
        """
    )
    
    parser.add_argument('--nrow', type=int, default=121, help='Input height')
    parser.add_argument('--ncol', type=int, default=104, help='Input width')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--channels', type=int, default=32, help='Base channels')
    parser.add_argument('--in-channels', type=int, default=1, 
                       help='Input channels (1=single gram, 2=SNR+NTV)')
    parser.add_argument('--extra-conv', action='store_true', help='Use 4-layer architecture')
    parser.add_argument('--output-dir', type=str, default='architecture_diagrams_latex',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PLOTNEURALNET AUTOENCODER DIAGRAM GENERATOR")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Input:       {args.in_channels}×{args.nrow}×{args.ncol} ({'SNR+NTV' if args.in_channels == 2 else 'single gram'})")
    print(f"  Latent:      {args.latent_dim}D")
    print(f"  Channels:    {args.channels} (base)")
    print(f"  Layers:      {'4-layer (extra_conv)' if args.extra_conv else '3-layer (standard)'}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Generate LaTeX
    generator = PlotNeuralNetGenerator(
        nrow=args.nrow,
        ncol=args.ncol,
        latent_dim=args.latent_dim,
        base_channels=args.channels,
        extra_conv=args.extra_conv,
        in_channels=args.in_channels
    )
    
    output_path = os.path.join(args.output_dir, 'autoencoder_arch.tex')
    generator.generate_latex(output_path)
    
    # Generate README
    generate_readme(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Ensure PlotNeuralNet is installed:")
    print(f"   cd /Users/oboulais/Public/Bowhead_DL_Project")
    print(f"   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")
    print(f"")
    print(f"2. Compile LaTeX to PDF:")
    print(f"   cd {args.output_dir}")
    print(f"   pdflatex autoencoder_arch.tex")
    print(f"")
    print(f"3. View result:")
    print(f"   open autoencoder_arch.pdf  # macOS")
    print(f"   xdg-open autoencoder_arch.pdf  # Linux")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
