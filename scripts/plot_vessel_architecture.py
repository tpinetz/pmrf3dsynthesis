import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_vessel_segmentation_architecture(save_path: str = "assets/vessel_network_architecture.png"):
    """
    Plots the 3D-to-2D U-Net architecture for OCT vessel segmentation.
    """
    fig, ax = plt.subplots(figsize=(16, 8.2), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.5, 9.8)
    ax.axis('off')

    # Title
    ax.text(8, 9.4, '3D-to-2D Vessel Segmentation Network Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
    ax.text(8, 9.0, 'Asymmetric 3D U-Net with Axial-Slicing Skip Connections and En-Face Projection Head', 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569')

    # Palette
    c_input = '#e0f2fe'     # Light blue
    c_enc = '#bfdbfe'       # Blue
    c_bottleneck = '#818cf8'# Indigo
    c_dec = '#fbcfe8'       # Pink
    c_proj = '#fde68a'      # Yellow
    c_out = '#bbf7d0'       # Mint green

    # Level 0 (Top): Input & conv_first
    rect_in = patches.FancyBboxPatch((0.5, 7.0), 2.2, 1.25, boxstyle='round,pad=0.1', facecolor=c_input, edgecolor='#0284c7', linewidth=1.5)
    ax.add_patch(rect_in)
    ax.text(1.6, 7.8, 'Input OCT Volume', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.6, 7.3, '1 x 128 x 256 x 256\n(B, C, W, H, D)', ha='center', va='center', fontsize=8, color='#334155')

    rect_e0 = patches.FancyBboxPatch((3.4, 7.0), 2.4, 1.25, boxstyle='round,pad=0.1', facecolor=c_enc, edgecolor='#2563eb', linewidth=1.5)
    ax.add_patch(rect_e0)
    ax.text(4.6, 7.8, 'conv_first + ResBlock', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.6, 7.3, 'Conv3d(1->4, k=3x1x3)\n1x ResBlock(4->4)\nShape: 4 x 128 x 256 x 256', ha='center', va='center', fontsize=7.5, color='#1e3a8a')

    # Encoder Level 1
    rect_e1 = patches.FancyBboxPatch((3.4, 5.0), 2.4, 1.25, boxstyle='round,pad=0.1', facecolor=c_enc, edgecolor='#2563eb', linewidth=1.5)
    ax.add_patch(rect_e1)
    ax.text(4.6, 5.8, 'Encoder Level 1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.6, 5.3, 'Downsample (1x2x2)\n1x ResBlock(8->8)\nShape: 8 x 128 x 128 x 128', ha='center', va='center', fontsize=7.5, color='#1e3a8a')

    # Encoder Level 2
    rect_e2 = patches.FancyBboxPatch((3.4, 3.0), 2.4, 1.25, boxstyle='round,pad=0.1', facecolor=c_enc, edgecolor='#2563eb', linewidth=1.5)
    ax.add_patch(rect_e2)
    ax.text(4.6, 3.8, 'Encoder Level 2', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.6, 3.3, 'Downsample (1x2x2)\n2x ResBlock(16->16)\nShape: 16 x 128 x 64 x 64', ha='center', va='center', fontsize=7.5, color='#1e3a8a')

    # Bottleneck Level 3
    rect_b = patches.FancyBboxPatch((6.8, 1.0), 2.6, 1.3, boxstyle='round,pad=0.1', facecolor=c_bottleneck, edgecolor='#4338ca', linewidth=1.5)
    ax.add_patch(rect_b)
    ax.text(8.1, 1.9, 'Bottleneck (Level 3)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(8.1, 1.4, 'Downsample (1x2x2)\n4x ResBlock(32->32)\nShape: 32 x 128 x 32 x 32', ha='center', va='center', fontsize=7.5, color='#e0e7ff')

    # Decoder Level 2
    rect_d2 = patches.FancyBboxPatch((10.4, 3.0), 2.5, 1.25, boxstyle='round,pad=0.1', facecolor=c_dec, edgecolor='#db2777', linewidth=1.5)
    ax.add_patch(rect_d2)
    ax.text(11.65, 3.8, 'Decoder Level 2', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.65, 3.3, 'Cat(Skip2, Up) -> Conv3d\n2x ResBlock(16->16)\nShape: 16 x 128 x 32 x 64', ha='center', va='center', fontsize=7.5, color='#831843')

    # Decoder Level 1
    rect_d1 = patches.FancyBboxPatch((10.4, 5.0), 2.5, 1.25, boxstyle='round,pad=0.1', facecolor=c_dec, edgecolor='#db2777', linewidth=1.5)
    ax.add_patch(rect_d1)
    ax.text(11.65, 5.8, 'Decoder Level 1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.65, 5.3, 'Cat(Skip1, Up) -> Conv3d\n1x ResBlock(8->8)\nShape: 8 x 128 x 32 x 128', ha='center', va='center', fontsize=7.5, color='#831843')

    # Decoder Level 0
    rect_d0 = patches.FancyBboxPatch((10.4, 7.0), 2.5, 1.25, boxstyle='round,pad=0.1', facecolor=c_dec, edgecolor='#db2777', linewidth=1.5)
    ax.add_patch(rect_d0)
    ax.text(11.65, 7.8, 'Decoder Level 0', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.65, 7.3, 'Cat(Skip0, Up) -> Conv3d\n1x ResBlock(4->4)\nShape: 4 x 128 x 32 x 256', ha='center', va='center', fontsize=7.5, color='#831843')

    # Projection Head
    rect_proj = patches.FancyBboxPatch((13.5, 7.0), 2.1, 1.25, boxstyle='round,pad=0.1', facecolor=c_proj, edgecolor='#d97706', linewidth=1.5)
    ax.add_patch(rect_proj)
    ax.text(14.55, 7.8, 'Axial Projection', ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(14.55, 7.3, 'Mean(dim=3: H->1)\nConv3d(4->1, k=3)\nSigmoid()', ha='center', va='center', fontsize=7.5, color='#78350f')

    # 2D Output
    rect_out = patches.FancyBboxPatch((13.5, 5.0), 2.1, 1.25, boxstyle='round,pad=0.1', facecolor=c_out, edgecolor='#16a34a', linewidth=1.5)
    ax.add_patch(rect_out)
    ax.text(14.55, 5.8, '2D En-Face Map', ha='center', va='center', fontsize=10, fontweight='bold', color='#14532d')
    ax.text(14.55, 5.3, 'Vessels in [0, 1]\nShape: 128 x 256\n(B-scans x A-scans)', ha='center', va='center', fontsize=7.5, color='#15803d')

    # Arrows
    def draw_arrow(x1, y1, x2, y2, text='', color='#475569', lw=1.5, ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, text, ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))

    draw_arrow(2.7, 7.6, 3.4, 7.6, color='#0284c7', lw=1.8)
    draw_arrow(4.6, 7.0, 4.6, 6.25, text='AvgPool(1x2x2) + 1x1 Conv', color='#2563eb')
    draw_arrow(4.6, 5.0, 4.6, 4.25, text='AvgPool(1x2x2) + 1x1 Conv', color='#2563eb')
    draw_arrow(4.6, 3.0, 6.8, 1.8, text='AvgPool(1x2x2) + 1x1 Conv', color='#2563eb')

    draw_arrow(9.4, 1.8, 11.65, 3.0, text='Trilinear(1,1,2) + 1x1 Conv', color='#db2777')
    draw_arrow(11.65, 4.25, 11.65, 5.0, text='Trilinear(1,1,2) + 1x1 Conv', color='#db2777')
    draw_arrow(11.65, 6.25, 11.65, 7.0, text='Trilinear(1,1,2) + 1x1 Conv', color='#db2777')

    draw_arrow(5.8, 7.6, 10.4, 7.6, text='Skip 0: AvgPool(1, 8, 1) -> (4x128x32x256)', color='#0284c7', ls='--', lw=1.6)
    draw_arrow(5.8, 5.6, 10.4, 5.6, text='Skip 1: AvgPool(1, 4, 1) -> (8x128x32x128)', color='#0284c7', ls='--', lw=1.6)
    draw_arrow(5.8, 3.6, 10.4, 3.6, text='Skip 2: AvgPool(1, 2, 1) -> (16x128x32x64)', color='#0284c7', ls='--', lw=1.6)

    draw_arrow(12.9, 7.6, 13.5, 7.6, color='#d97706', lw=1.8)
    draw_arrow(14.55, 7.0, 14.55, 6.25, text='squeeze', color='#16a34a', lw=1.8)

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight', dpi=200)
    print(f"Architecture plot saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot vessel segmentation architecture.")
    parser.add_argument("--save_path", type=str, default="assets/vessel_network_architecture.png", help="Output path for the plot.")
    args = parser.parse_args()
    plot_vessel_segmentation_architecture(args.save_path)


if __name__ == "__main__":
    main()
