from copy import deepcopy

import pandas as pd
from numpy import float64, int64, object_

from pysal.cg.shapes import Polygon
from pysal.contrib.shapely_ext import envelope, union


def merge_rows(contained_row,
               container_row,
               dtypes,
               geometry_col='geometry',
               centroid_x_col='CENTROID_XCOORDINATES',
               centroid_y_col='CENTROID_YCOORDINATES'):
    """Merge attributes of two rows

    This function assumes that the container_row geometry contains the
    contained_row's geometry but _does not check to ensure that that's
    the case_.

    Args:
        contained_row (pd.Series): row of inner geometry attributes
        container_row (pd.Series): row of outer geometry attributes
        dtypes (pd.Series): series of types for the two rows

    Returns:
        pd.Series
    """

    # Ensure all the indices match
    assert contained_row.index.equals(container_row.index)
    assert container_row.index.equals(dtypes.index)

    # Make a new series with the same index and no values
    new_ser = pd.Series(index=contained_row.index)

    # get the geometry data we want
    (new_geom, (centroid_x, centroid_y)) = merge_geometry(
        contained_row, container_row)

    # Go through the datatypes and combine values accordingly
    for k, dtype in dtypes.items():
        if k == geometry_col:
            new_ser[k] = new_geom
        elif k == centroid_x_col:
            new_ser[k] = centroid_x
        elif k == centroid_y_col:
            new_ser[k] = centroid_y
        elif dtype.type in [int64, float64]:
            new_ser[k] = contained_row[k] + container_row[k]
        elif dtype.type == object_:
            new_ser[k] = container_row[k]
        else:
            raise NotImplementedError(
                'I don\'t know how to handle these: %s and %s', k, dtype)

    return new_ser


def merge_geometry(contained_row, container_row):
    """Merge geometry attributes of two rows

    Args:
        contained_row (pd.Series): row of inner geometry attributes
        container_row (pd.Series): row of outer geometry attributes

    Returns:
        (geometry, centroid)
    """

    new_geom = union(contained_row['geometry'], container_row['geometry'])
    return (new_geom, new_geom.centroid)


def find_interlopers(mggg_graph):
    interlopers = []
    visited = []
    tmp = mggg_graph.neighbors.copy()
    for x in tmp:
        neighbors = tmp[x]
        inner_geom = mggg_graph.shape_df.loc[x, 'geometry']
        # Candidates are those neighboring
        candidates = [
            y for y in neighbors if neighbors | tmp[y] == tmp[y] and y != x
        ]
        for n in candidates:
            if n in visited:
                continue
            neighbor_geom = mggg_graph.shape_df.loc[n, 'geometry']
            if neighbor_geom.area < inner_geom.area:
                continue
            holes = neighbor_geom.holes
            parts = neighbor_geom.parts
            holes_plus_parts = [piece for piece in holes + parts if piece]
            intersections = [
                envelope(inner_geom).contains_point(Polygon(poly).centroid)
                for poly in holes_plus_parts
            ]
            if any(intersections):
                interlopers.append((x, n))
                visited.extend([x, n])
                break

    return dict(interlopers)


def unify_shapes(mggg_graph):
    """Merge all nodes containing other nodes with the contained nodes
    """

    print('Copying graph')

    tmp = deepcopy(mggg_graph)

    print('Finding weird geometries')
    interlopers_all = find_interlopers(mggg_graph)
    interlopers_filtered = {
        k: v for k, v in interlopers_all.items()
        if k not in interlopers_all.values()
    }
    contained_ids = interlopers_filtered.keys()
    container_ids = interlopers_filtered.values()
    print('Found this many: %s' % len(interlopers_filtered))

    print('Constructing new dataframe')
    new_df_dict = {}
    dtypes = mggg_graph.shape_df.dtypes
    for i, ind in enumerate(mggg_graph.shape_df.index):
        if ind in container_ids:
            continue
        elif ind in contained_ids:
            new_df_dict[interlopers_filtered[ind]] = merge_rows(
                mggg_graph.get_vertex_attrs(ind),
                mggg_graph.get_vertex_attrs(interlopers_filtered[ind]), dtypes)
            tmp.neighbors[interlopers_filtered[ind]] = set([
                x for x in tmp.neighbors[interlopers_filtered[ind]] if x != ind
            ])
        else:
            new_df_dict[ind] = mggg_graph.get_vertex_attrs(ind)
#        if i % 1000 == 0 and i != 0:
#            tmp.shape_df = pd.DataFrame.from_dict(new_df_dict, orient='index')
#            plot_gr(tmp)

    tmp.shape_df = pd.DataFrame.from_dict(new_df_dict, orient='index')
    return tmp
#    new_interlopers = find_interlopers(tmp)
#    if new_interlopers:
#        return unify_shapes(tmp)
#    else:
#        return tmp
