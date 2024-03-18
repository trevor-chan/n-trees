from matplotlib.colors import ListedColormap
import random

def desert_cmap():
    desert_colors = [
                    "#fddd9e","#e29c52","#d57b2c","#c75a05",
                    "#a5d2d0","#8cb6b5","#739a99","#416262",
                    "#aebdad","#98a388","#807f62","#555434",
                    "#e8e7d7","#b2b1aa","#7a7a7b","#55555c",
                    "#fee5c4","#d1b895","#a68b68","#7d613e",
                    "#d57c7c","#b44b4b","#872a2a","#5c0d0d",
                    ]
    random.shuffle(desert_colors)
    return ListedColormap(desert_colors)