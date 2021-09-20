from yellowcab import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable


# could be deleted; Same functionalities are possible with heatmap()
def create_map(gdf, filter_borough="Brooklyn", file_type="png", mapname='map'):
    # filter
    gdf_filtered = gdf.to_crs(3857)[gdf['borough'] == filter_borough]

    # Set boundaries etc.
    gdf_filtered["area"] = gdf_filtered.area
    gdf_filtered['boundary'] = gdf_filtered.boundary
    gdf_filtered['centroid'] = gdf_filtered.centroid

    # Create plot
    fig = plt.figure(1, figsize=(15, 15))
    ax = fig.add_subplot()
    gdf_filtered.apply(
        lambda x: ax.annotate(text=x["location_id"], xy=x.geometry.centroid.coords[0], ha='center', fontsize=14,
                              weight='bold'),
        axis=1)
    gdf_filtered.boundary.plot(ax=ax, color='Black', linewidth=0.75)
    gdf_filtered.plot(ax=ax, cmap='Pastel2', figsize=(12, 12), alpha=0.4)
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)
    ax.text(
        0.875, 0.98, 'Brooklyn', transform=ax.transAxes,
        fontsize=30, color='black', alpha=0.85,
        ha='center', va='top', rotation='0'
    )

    io.save_fig(fig, mapname, file_type)
    return fig, gdf_filtered


def heatmap(trip_data, gdf, location='PULocationID', count_by="duration",
            group_type='sum', borough='Brooklyn', file_type='png',
            figsize=15, title_fontsize=25, vis_data=False, bubbles=False, basemap=True):
    # filter the right dfs for higher performance
    gdf = gdf.to_crs(3857)[gdf['borough'] == borough]
    if location == 'PULocationID':
        trip_data = trip_data[trip_data['PUBorough'] == borough]
    elif location == 'DOLocationID':
        trip_data = trip_data[trip_data['DOBorough'] == borough]
    else:
        raise ValueError('Filter-Error')

    # groups data by count_by method
    data_By = getattr(trip_data.groupby(location), group_type)()[[count_by]].reset_index().rename(
        {location: "location_id"}, axis=1)

    # join gdf and trip_data -Frames
    gdf['location_id'] = pd.to_numeric(gdf["location_id"])
    data_heat = pd.merge(gdf, data_By, on="location_id")

    # create fig
    figsize_t = (figsize, figsize)
    fig, ax = plt.subplots(1, 1, figsize=figsize_t)
    title = 'The ' + count_by + ' grouped by ' + group_type + ' which applies to ' + location
    ax.set_title(title, fontsize=title_fontsize)

    # fits legend bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # vis. trip amount on map
    if not vis_data:
        data_heat.apply(
            lambda x: ax.annotate(text=x["location_id"], xy=x.geometry.centroid.coords[0],
                                  ha='center', fontsize=figsize - 1, weight='bold', color='grey'), axis=1)
    elif vis_data:
        data_heat.apply(
            lambda x: ax.annotate(
                text='ID: ' + str(x["location_id"]) + '\n' + group_type + ':' + str(round(x[count_by], 1)),
                xy=x.geometry.centroid.coords[0], ha='center', fontsize=figsize - 6, weight='bold', color='grey'),
            axis=1)
    else:
        raise ValueError('group_type must be set to ''count''')

    # plotting
    data_heat.plot(column=count_by, ax=ax, legend=True, cax=cax, alpha=0.95)

    # add basemap
    if basemap:
        ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    # creating bubbles
    if bubbles:
        gdf_points = data_heat.copy()
        gdf_points['geometry'] = gdf_points['geometry'].centroid
        gdf_points.plot(ax=ax, markersize=count_by, alpha=0.2, cmap='Reds', categorical=False, legend=True)

    mapname = 'heatmaps/' + count_by + '_g.by_' + group_type + '_from_' + location
    io.save_fig(fig, mapname, file_type)
    return data_heat


# creating multiple heatmaps each with PU and DO Loc by trip_distance and duration
def create_all_heatmaps(trip_data, gdf, group_type='count', borough='Brooklyn',
                        file_type='svg', basemap=True, vis_data=True, bubbles=True,
                        figsize=15, title_fontsize=25):
    heatmap(trip_data, gdf, count_by='duration', location='DOLocationID', borough=borough,
            file_type=file_type, figsize=figsize, title_fontsize=title_fontsize,
            group_type=group_type, basemap=basemap, vis_data=vis_data, bubbles=bubbles)
    print('1. heatmap processed')
    heatmap(trip_data, gdf, count_by='duration', location='PULocationID', borough=borough,
            file_type=file_type, figsize=figsize, title_fontsize=title_fontsize,
            group_type=group_type, basemap=basemap, vis_data=vis_data, bubbles=bubbles)
    print('2. heatmap processed')
    heatmap(trip_data, gdf, count_by='trip_distance', location='DOLocationID', borough=borough,
            file_type=file_type, figsize=figsize, title_fontsize=title_fontsize,
            group_type=group_type, basemap=basemap, vis_data=vis_data, bubbles=bubbles)
    print('3. heatmap processed')
    heatmap(trip_data, gdf, count_by='trip_distance', location='PULocationID', borough=borough,
            file_type=file_type, figsize=figsize, title_fontsize=title_fontsize,
            group_type=group_type, basemap=basemap, vis_data=vis_data, bubbles=bubbles)
    print('4. heatmap processed')

    return


def create_passenger_overview(gdf):
    # Group
    week_groups = gdf.groupby([gdf["start_week"]])['passenger_count'].sum()/1000

    # Create plot
    x = week_groups.index
    y = week_groups
    fig, ax = plt.subplots()
    ax.bar(x, y)
    # Set Color of Lockdown/Reopening measures
    ax.get_children()[10].set_color("r")
    ax.get_children()[11].set_color("r")
    ax.get_children()[45].set_color("r")
    ax.get_children()[19].set_color("g")
    ax.get_children()[23].set_color("g")
    ax.get_children()[25].set_color("g")
    ax.get_children()[27].set_color("g")
    ax.get_children()[29].set_color("g")
    ax.set(title='Total passengers per Calendar Week', ylabel='Passengers in Thousand', xlabel='Calendar Week (2020)')
    plt.figure(figsize=(30, 15))

    # Save plot and return
    io.save_fig(fig, "passenger_per_week")
    return plt


def create_corona_cases_overview():
    corona_df = pd.read_csv("../../data/input/case_data/time_series_covid19_confirmed_US.csv")
    # Filter relevant data and name colomns
    corona_df_t = corona_df.T[65:356]
    corona_df_t = corona_df_t.reset_index().rename({"index": "date", 0: "corona_cases_abs"}, axis=1)

    # Create Dataframe
    data = pd.Series(range(0, 291))
    corona_cases_rel = pd.DataFrame(data, columns=["rel"])

    # Calculate relative values
    for i in range(1, 291):
        corona_cases_rel.loc[i] = corona_df_t["corona_cases_abs"][i] - corona_df_t["corona_cases_abs"][i - 1]
    corona_df_t["corona_cases_rel"] = corona_cases_rel

    # Create plot
    x = corona_df_t["date"]
    y = corona_df_t["corona_cases_rel"]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(title='Covid-19 Cases in New York City in 2020', ylabel='New Cases', xlabel='Time')
    plt.xticks([10, 70, 140, 210, 290], labels=["26.03", "25.05", "03.08", "12.10", "31.12"])
    plt.figure(figsize=(25, 20))

    # Save plot and return
    io.save_fig(fig, "corona_cases_overview")
    return plt


def create_march_overview(gdf):
    only_march = gdf[gdf["start_month"] == 3]
    # Order by day
    daily_march = only_march.groupby([only_march["start_day"]])['duration'].sum()
    daily_march = daily_march.reset_index().rename({"start_day": "day", "duration": "duration_sec"}, axis=1)

    # Calculate seconds to days
    daily_march["duration_days"] = daily_march["duration_sec"] / 86400

    # Create plot
    x = daily_march.day
    y = daily_march.duration_days
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o', markevery=[11])
    ax.set(title='Total Duration of Taxi rides in 03/2020', ylabel='Duration in days', xlabel='March 2020')
    plt.figure(figsize=(12, 6))

    io.save_fig(fig, "march_overview")
    return plt


def week_comparison(gdf):
    week_10 = gdf[gdf["start_week"] == 10]
    week_13 = gdf[gdf["start_week"] == 13]
    # Order by passengers per hour
    week_10_hourly = week_10.groupby([week_10["start_hour"]])['passenger_count'].sum()
    week_10_hourly = week_10_hourly.reset_index()
    week_13_hourly = week_13.groupby([week_13["start_hour"]])['passenger_count'].sum()
    week_13_hourly = week_13_hourly.reset_index()

    # Calculate in thousand
    week_13_hourly["passenger_count"] = week_13_hourly["passenger_count"] / 1000
    week_10_hourly["passenger_count"] = week_10_hourly["passenger_count"] / 1000

    # Create Plots
    x1 = week_10_hourly.start_hour
    y1 = week_10_hourly.passenger_count
    x2 = week_13_hourly.start_hour
    y2 = week_13_hourly.passenger_count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Total Passenger Count per hour in Calender Week 10 and 13')

    # Create first plot
    ax1.plot(x1, y1, '-X', markevery=[8, 18])
    plt.xticks([0, 8, 16, 23])
    plt.yticks([10, 20, 30, 40, 50])
    ax1.set(ylabel='Total Passenger Count (in thousand)', xlabel='Hour of the day (Week 10)')

    # Create second plot
    ax2.plot(x2, y2, '-X', markevery=[8, 18])
    plt.xticks([0, 4, 8, 12, 16, 20, 23])
    plt.yticks([2, 4, 6, 8, 10])
    ax2.set(xlabel='Hour of the day (Week 13)')

    io.save_fig(fig, "week_comparison")
    return plt
