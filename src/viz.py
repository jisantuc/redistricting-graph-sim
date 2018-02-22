import matplotlib.pyplot as plt
from pysal.contrib.viz import mapping as maps


def gr_to_patches(mggg_gr, ax, district_no):
    """Get a patch collection for a district number
    """

    pass


def plot_gr(mggg_gr, ax=None):
    """Plot the current state of an MGGG adjacency graph
    """

    pysal_geoms = mggg_gr.shape_df['geometry']
    bbox = mggg_gr.loaded_geodata.bbox
    base = maps.map_poly_shp(pysal_geoms, bbox=bbox)
    base.set_linewidth(0.75)
    base.set_edgecolor('0.8')
    maps.setup_ax([base], [bbox], ax=ax)
    plt.show()
