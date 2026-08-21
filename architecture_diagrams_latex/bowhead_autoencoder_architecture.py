#!/usr/bin/env python3
"""
Generate architecture diagram for Bowhead Whale Autoencoder
Uses PlotNeuralNet to create a clear, professional visualization
"""

import sys
import os

# Add PlotNeuralNet to path
sys.path.append('../PlotNeuralNet')

from pycore.tikzeng import *
from pycore.blocks import *

# Create architecture with large, clear labels
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # INPUT LAYER
    to_input('../PlotNeuralNet/examples/fcn8s/cats.jpg', to='(-4,0,0)', width=10, height=10),
    
    to_Conv(name='input', s_filer='', n_filer='', offset="(0,0,0)", 
            to="(0,0,0)", height=60, depth=52, width=1, 
            caption="Input\\\\1×121×104\\\\Spectrogram", xlabel=True, fill='{rgb:blue,1;red,1;green,1;white,5}'),
    
    to_connection("input", "conv1"),
    
    # ENCODER SECTION
    to_Conv(name='conv1', s_filer='5×5', n_filer=32, offset="(2,0,0)", 
            to="(input-east)", height=60, depth=52, width=4, 
            caption="Conv1\\\\5×5, 32\\\\+BN+ReLU", xlabel=True, fill='{rgb:yellow,5;red,2.5;white,5}'),
    
    to_Pool(name="pool1", offset="(0,0,0)", to="(conv1-east)", 
            height=30, depth=26, width=1, opacity=0.5, 
            caption="MaxPool\\\\2×2", xlabel=True),
    
    to_connection("pool1", "conv2"),
    
    to_Conv(name='conv2', s_filer='3×3', n_filer=64, offset="(2,0,0)", 
            to="(pool1-east)", height=30, depth=26, width=6, 
            caption="Conv2\\\\3×3, 64\\\\+BN+ReLU", xlabel=True, fill='{rgb:yellow,5;red,2.5;white,4}'),
    
    to_Pool(name="pool2", offset="(0,0,0)", to="(conv2-east)", 
            height=15, depth=13, width=1, opacity=0.5, 
            caption="MaxPool\\\\2×2", xlabel=True),
    
    to_connection("pool2", "conv3"),
    
    to_Conv(name='conv3', s_filer='3×3', n_filer=128, offset="(2,0,0)", 
            to="(pool2-east)", height=15, depth=13, width=10, 
            caption="Conv3\\\\3×3, 128\\\\+BN+ReLU", xlabel=True, fill='{rgb:yellow,5;red,2.5;white,3}'),
    
    to_Pool(name="pool3", offset="(0,0,0)", to="(conv3-east)", 
            height=8, depth=7, width=1, opacity=0.5, 
            caption="MaxPool\\\\2×2", xlabel=True),
    
    to_connection("pool3", "flatten"),
    
    # LATENT SPACE
    to_SoftMax(name="flatten", s_filer='', offset="(2,0,0)", to="(pool3-east)", 
               width=1, height=8, depth=8, 
               caption="Flatten\\\\26,624", xlabel=True, fill='{rgb:green,1;white,3}'),
    
    to_connection("flatten", "fc1"),
    
    to_SoftMax(name="fc1", s_filer='', offset="(3,0,0)", to="(flatten-east)", 
               width=2, height=6, depth=6, 
               caption="Dense\\\\64", xlabel=True, fill='{rgb:green,1;white,2}'),
    
    to_connection("fc1", "latent"),
    
    to_SoftMax(name="latent", s_filer='', offset="(3,0,0)", to="(fc1-east)", 
               width=3, height=4, depth=4, 
               caption="\\textbf{LATENT}\\\\\\textbf{32-dim}", xlabel=True, fill='{rgb:red,5;white,1}'),
    
    to_connection("latent", "fc2"),
    
    # DECODER SECTION
    to_SoftMax(name="fc2", s_filer='', offset="(3,0,0)", to="(latent-east)", 
               width=2, height=6, depth=6, 
               caption="Dense\\\\64", xlabel=True, fill='{rgb:green,1;white,2}'),
    
    to_connection("fc2", "fc3"),
    
    to_SoftMax(name="fc3", s_filer='', offset="(3,0,0)", to="(fc2-east)", 
               width=1, height=8, depth=8, 
               caption="Dense\\\\26,624", xlabel=True, fill='{rgb:green,1;white,3}'),
    
    to_connection("fc3", "reshape"),
    
    to_Conv(name='reshape', s_filer='', n_filer='', offset="(2,0,0)", 
            to="(fc3-east)", height=8, depth=7, width=10, 
            caption="Reshape\\\\128×16×13", xlabel=True, fill='{rgb:blue,1;white,3}'),
    
    to_connection("reshape", "deconv1"),
    
    to_ConvConvRelu(name='deconv1', s_filer='2×2', n_filer='(64,64)', offset="(2,0,0)", 
                    to="(reshape-east)", height=15, depth=13, width=6, 
                    caption="TransConv1\\\\2×2, 64\\\\+BN+ReLU", xlabel=True),
    
    to_connection("deconv1", "deconv2"),
    
    to_ConvConvRelu(name='deconv2', s_filer='2×2', n_filer='(32,32)', offset="(2,0,0)", 
                    to="(deconv1-east)", height=30, depth=26, width=4, 
                    caption="TransConv2\\\\2×2, 32\\\\+BN+ReLU", xlabel=True),
    
    to_connection("deconv2", "output"),
    
    to_Conv(name='output', s_filer='2×2', n_filer=1, offset="(2,0,0)", 
            to="(deconv2-east)", height=60, depth=52, width=1, 
            caption="Output\\\\1×121×104\\\\Reconstruction", xlabel=True, fill='{rgb:blue,1;red,1;green,1;white,5}'),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"✓ Generated {namefile}.tex")
    print(f"  Run: cd architecture_diagrams_latex && pdflatex {namefile}.tex")
    print(f"  Or upload to Overleaf for compilation")

if __name__ == '__main__':
    main()
