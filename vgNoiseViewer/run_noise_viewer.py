#!/usr/bin/env python3
"""
Launch the Noise Viewer application.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import tkinter as tk
from noise_viewer.app import NoiseViewer


def main():
    """Main entry point."""
    root = tk.Tk()
    app = NoiseViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

