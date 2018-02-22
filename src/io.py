from collections import Counter
import random
from uuid import uuid4

import fiona
from fiona.crs import from_epsg
import numpy as np
from numpy import float64, int64, object_
import pandas as pd
from shapely.geometry import mapping
from adjacency_graphs.algorithms import TwoStepGraph
from unify_shapes import unify_shapes

visited = set()


def row_to_geojson(row, geometry_col='geometry'):
    """Convert a row of data into a geojson record

    Args:
        row (pd.Series): the row to write
        geometry_col (str): the column name containing the geometry

    Returns:
        dict
    """

    geom_dict = mapping(row[geometry_col])
    properties = {
        k: v
        for k, v in row.to_dict().iteritems() if k != geometry_col
    }
    properties[u'GEOID'] = row.name

    return {
        'geometry': geom_dict,
        'properties': properties,
        # The index value of a series is the series' name
        'id': row.name
    }


def schema_from_graph(mggg_graph, geometry_col='geometry'):
    """Create a schema for fiona to use with output from a graph

    Args:
        mggg_graph (MgggGraph): the graph to create the schema from
        geometry_col (str): the column name of the records' geometry column

    Returns:
        dict
    """

    type_lookup = {float64: 'float:15.2', int64: 'int:64', object_: 'str'}
    dtypes = mggg_graph.shape_df.dtypes
    schema = {
        'properties': {
            k: type_lookup[v.type]
            for k, v in dtypes.to_dict().iteritems() if k != geometry_col
        }
    }
    schema['properties'][u'GEOID'] = 'str'
    schema[geometry_col] = 'Polygon'
    return schema


def write_graph(path, mggg_graph, geometry_col='geometry'):
    """Write an MgggGraph to a shapefile

    Args:
        path (str): path to write the output file to
        mggg_graph (MgggGraph): the graph to write

    Returns:
        None
    """

    records = [
        row_to_geojson(row, geometry_col)
        for _, row in mggg_graph.shape_df.iterrows()
    ]
    schema = schema_from_graph(mggg_graph, geometry_col)
    with fiona.open(
            path, 'w', driver='ESRI Shapefile', schema=schema,
            crs=from_epsg(4326)) as sink:
        for record in records:
            sink.write(record)


def load_graph(shape_path):
    """Return an adjacency graph from shape_path with geoid in GEOID

    GEOID _must_ be unique (sort of obviously), otherwise the adjacency
    graph construction will drop some records, which is bad.

    Args:
        shape_path (str): path to the input shapefile
    """
    return TwoStepGraph(shape_path, 'GEOID')


def find_start(gr, x_column, y_column, iteration=0):
    filtered = gr.shape_df[pd.isnull(gr.shape_df['DISTRICT'])
                           & ~(gr.shape_df.index.isin(visited))]
    if iteration % 6 == 0:
        ind = (filtered[x_column]**2 + filtered[y_column]**2).idxmin()
    elif iteration % 6 == 1:
        ind = (filtered[x_column]**2 + filtered[y_column]**2).idxmax()
    elif iteration % 6 == 2:
        ind = filtered[x_column].idxmax()
    elif iteration % 6 == 3:
        ind = filtered[x_column].idxmin()
    elif iteration % 6 == 4:
        ind = filtered[y_column].idxmax()
    else:
        ind = filtered[y_column].idxmin()

    return ind


def fill_district(district_id, mggg_graph, df, start, target_pop, upper_bound,
                  population_col):
    # Get all the neighbors for the district
    neighbors = mggg_graph.neighbors[start]
    # Get the start polygon at the "corner" index
    record = mggg_graph.get_vertex_attrs(start)
    record['DISTRICT'] = district_id

    total_pop = record[population_col]

    keep_going = True
    while keep_going:
        # Calculate available population neighboring this district
        # .query('@pd.isnull(DISTRICT)') evaluates the dataframe object it's
        # called on for whether the DISTRICT column is null without having
        # had to name the intermediate filtered dataframe
        available_pop = (df.loc[neighbors].query('@pd.isnull(DISTRICT)')[
            population_col].sum())

        # Add all the neighbors to this district if the population is below
        # target
        if total_pop + available_pop < target_pop:
            df.loc[neighbors, 'DISTRICT'] = district_id
            neighbor_ids = reduce(lambda x, y: x | y,
                                  [mggg_graph.neighbors[x] for x in neighbors])
            neighbors = df.loc[neighbor_ids].query(
                '@pd.isnull(DISTRICT)').index.tolist()
            total_pop += available_pop
        # Otherwise, add some of the neighbors until the target population is
        # approximately reached
        else:
            ordered = df.loc[neighbors].query(
                '@pd.isnull(DISTRICT)').sort_values(population_col)
            ordered_idx = ordered.index
            for ind in ordered_idx:
                this_pop = ordered.loc[ind, population_col]
                # if under target population after adding this one, definitely
                # add this one to the district
                if this_pop + total_pop < target_pop:
                    df.loc[[ind], 'DISTRICT'] = district_id
                    total_pop += this_pop
                # if over the target population after adding this one but
                # within the tolerance, flip a coin to decide whether to add
                # this one
                elif (target_pop < this_pop + total_pop < upper_bound
                      and random.random() > 0.5):
                    df.loc[[ind], 'DISTRICT'] = district_id
                    total_pop += this_pop
                    keep_going = False
                    break
                # This should only be reached in practice when we opt out of
                # adding the district in the check above but would also fall
                # through when the geometry in question would add too much
                # population I guess
                else:
                    keep_going = False
                    break


def build_districts(mggg_graph,
                    n_districts,
                    population_col='TOTAL_POP',
                    x_column='CENTROID_XCOORDINATES',
                    y_column='CENTROID_YCOORDINATES',
                    tolerance=0.01):
    """Assign all geometries in the loaded graph to a district
    """
    # strategy
    # x start in a corner
    # get populations for all neighbors
    # figure out closest it's possible to get to target population with
    # available neighbors add units from bottom to top until either no more
    # units or target population reached

    # Set a more convenient reference for the shape dataframe
    df = mggg_graph.shape_df

    # Calculate the equipopulation targets
    target_pop = df[population_col].sum() / n_districts
    upper_bound = target_pop * (1 + tolerance)

    # Initialize the DISTRICT column with nulls
    df['DISTRICT'] = np.nan

    # Find a corner by getting the index that's least far from 0
    # It might not end up being a literal corner, but it will probably be a
    # polygon on the edge of the set of polygons
    counter = 0

    tries_at_numbers = Counter([])

    while len(df['DISTRICT'].unique() < n_districts):
        # Choose a random ID
        district_id = str(uuid4())

        try:
            corner = find_start(mggg_graph, x_column, y_column, counter)
        except ValueError:
            break

        try:
            fill_district(district_id, mggg_graph, df, corner, target_pop,
                          upper_bound, population_col)

        except TypeError:
            df['DISTRICT'] = df['DISTRICT'].map(
                lambda x: np.nan if x == district_id else x)
            length_of_dist = len(df['DISTRICT'].unique())
            tries_at_numbers[length_of_dist] += 1
            if tries_at_numbers[length_of_dist] > 50:
                percentage_filled = '%.2f' % ((
                    1 - pd.isnull(df['DISTRICT']).mean()) * 100)
                df['DISTRICT'] = np.nan
                tries_at_numbers = Counter([])
                print 'Starting over after s districts... %s' % percentage_filled

            global visited
            visited |= {corner}

        finally:
            counter += 1


def main():
    gr = unify_shapes(load_graph('data/unique_id.shp'))
    try:
        build_districts(gr, 15)
    except KeyboardInterrupt:
        return gr
    return gr
