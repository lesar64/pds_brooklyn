from yellowcab.io.utils import get_stats
import yellowcab.io
import seaborn as sns
import pandas as pd
import calendar
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def double_barplot(df, col, left="mean", right="std"):

    width = 0.35
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots()
    ax.bar(x - width/2, df[left], width, label=left)
    ax.bar(x + width/2, df[right], width, label=right)

    ax.set_ylabel(col)
    ax.set_xlabel("month")
    ax.set_xticks(x)
    ax.legend()

    fig.tight_layout()

    yellowcab.io.save_fig(fig, str("double_barplot_" + col))
    plt.show()


def visualize_trip_lengths(trip_data, monthly=True, daily=True, hourly=True):
  
    # Setting seaborn as default style
    sns.set() 
    
    #trip length consists of 'trip_distance' and 'duration'
    #Checking start_month, start_day and start_hour
    trip_data_filtered = trip_data[['start_month', 'start_day', 'start_hour', 'duration', 'trip_distance']] 
    #filter out negative trip length values and 0
    trip_data_filtered = trip_data_filtered[trip_data_filtered['duration'] > 0] 
    trip_data_filtered = trip_data_filtered[trip_data_filtered['trip_distance'] > 0]
    
    if monthly == True:
        
        fig, axes = plt.subplots(1, 2,  figsize=(10, 5), gridspec_kw={'wspace':0.2})
        
        axes[0].set_title('Monthly trip duration')
        axes[1].set_title('Monthly trip distance')
        
        sns.boxplot(ax=axes[0], data=trip_data_filtered, x='start_month', y ='duration', showfliers=False, color="#5699c6")
        sns.boxplot(ax=axes[1], data=trip_data_filtered, x='start_month', y ='trip_distance', showfliers=False, color="#5699c6")
        
        yellowcab.io.save_fig(fig, "visualize_monthly_trip_lengths")
        
        plt.show()
     
    if daily == True:
        
        fig, axes = plt.subplots(1, 2,  figsize=(16, 5), gridspec_kw={'wspace':0.2})
        
        axes[0].set_title('Daily trip duration')
        axes[1].set_title('Daily trip distance')
        
        sns.boxplot(ax=axes[0], data=trip_data_filtered, x='start_day', y ='duration', showfliers=False, color="#5699c6")
        sns.boxplot(ax=axes[1], data=trip_data_filtered, x='start_day', y ='trip_distance', showfliers=False, color="#5699c6")
        
        yellowcab.io.save_fig(fig, "visualize_daily_trip_lengths")
        
        plt.show()
        
    if hourly == True:
        
        #hourly trip duration plot
        fig, axes = plt.subplots(1, 1,  figsize=(10, 4), gridspec_kw={'wspace':0.2})
   
        sns.boxplot(data=trip_data_filtered, x='start_hour', y ='duration', showfliers=False, color="#5699c6").set_title('Hourly trip duration', fontsize=12)
        
        yellowcab.io.save_fig(fig, "visualize_hourly_trip_duration")
        
        plt.show()
        
        #hourly trip duration plot
        fig, axes = plt.subplots(1, 1,  figsize=(10, 4), gridspec_kw={'wspace':0.2})
        
        sns.boxplot(data=trip_data_filtered, x='start_hour', y ='trip_distance', showfliers=False, color="#5699c6").set_title('Hourly trip distance', fontsize=12)
        
        yellowcab.io.save_fig(fig, "visualize_hourly_trip_distance")
        
        plt.show()
    
    
def monthly_hist(df, col="duration", log_scale=True, bins=40):

    try:
        df = df[[col, "start_month"]]
        df = df[df[col] > 0]
    except KeyError:
        raise ValueError("dataframe doesn't have required columns (" + col + " or start_month)")

    df_list = []

    for i in range(1, 13):
        df_list.append(df[df["start_month"] == i])

    fig, axes = plt.subplots(3, 4, figsize=(12,6))

    index = 0

    for i in range(3):
        for j in range(4):
            sns.histplot(ax=axes[i, j], data=df_list[index][col], log_scale=log_scale, bins=bins)
            axes[i, j].set_title(calendar.month_name[index+1])
            index += 1
    
    plt.tight_layout()
    yellowcab.io.save_fig(fig, str("monthly_hist_" + col))
    plt.show()


def monthly_kde(df, col="duration"):

    try:
        df = df[[col, "tpep_pickup_datetime", "start_month"]]
        df = df[df[col] > 0]
    except KeyError:
        raise ValueError("dataframe doesn't have required columns (" + col + " or start_month)")

    df_list = []

    for i in range(1, 13):
        df_list.append(df[df["start_month"] == i])

    df_stats = pd.DataFrame()
    df_stats["mean"] = get_stats(df, metric=col, stat="mean")[col]
    df_stats["std"] = get_stats(df, metric=col, stat="std")[col]

    fig, axes = plt.subplots(3, 4, figsize=(12,6))

    index = 0

    x = np.linspace(0, df[col].max(), 1000)

    for i in range(3):
        for j in range(4):
            sns.kdeplot(ax=axes[i, j], data=df_list[index][col])
            axes[i, j].plot(x, stats.norm.pdf(x, df_stats["mean"][index+1], df_stats["std"][index+1]))
            axes[i, j].set_title(calendar.month_name[index+1])
            index += 1
    
    plt.tight_layout()
    yellowcab.io.save_fig(fig, str("monthly_kde_" + col))
    plt.show()


def split_hist(df, col="duration", split=14400):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={'wspace':0.2})

    sns.histplot(data=df[col], ax=ax1, log_scale=True)

    ax1.annotate('spike',
        xy=(86400, 9000),
        xytext=(0.4, 0.7),
        textcoords='figure fraction',
        fontsize=16,
        arrowprops=dict(facecolor='blue', shrink=0.1)
        )

    data=df[df[col] > split][col]

    sns.histplot(data=data, ax=ax2)
    yellowcab.io.save_fig(fig, str("split_hist_" + col))
    plt.show()


def compare_hist(df, col="duration", leftcol="start_hour", rightcol="end_hour", bins=24, split=14400):

    data=df[df["duration"] > split]

    print("Plotting", len(data), "instances with duration above", split)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={'wspace':0.2})

    sns.histplot(data[leftcol], ax=ax1, bins=bins)

    sns.histplot(data[rightcol], ax=ax2, bins=bins)

    yellowcab.io.save_fig(fig, str("compare_hist_" + col))
    plt.show()


def compare_distr(df, col="duration"):

    data = df[df[col] > 0]
    data = data[col]

    fig, ax = plt.subplots()

    label_patches = []

    sns.kdeplot(ax=ax, data=data)

    label_patches.append(mpatches.Patch(
        label=col
    ))

    x = np.linspace(data.min(), data.max(), 1000)

    ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()), color="orange")

    label_patches.append(mpatches.Patch(
        label="normal distribution",
        color="orange"
    ))

    if col == "duration":
        plt.xlim([0, 20000])

    ax.legend(handles=label_patches, loc="upper right")

    yellowcab.io.save_fig(fig, str("compare_distr_" + col))
    plt.show()


def log_distr(df, col="duration", bins = 100):

    data = df[col]
    data = data[data > 0]

    N_bins = bins

    # source = https://stackoverflow.com/questions/35001607/scaling-and-fitting-to-a-log-normal-distribution-using-a-logarithmic-axis-in-pyt

    # make a fit to the data
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    x_fit       = np.linspace(data.min(), data.max(), 100)
    data_fit = stats.lognorm.pdf(x_fit, shape, loc=loc, scale=scale)

    # plot a histrogram with linear x-axis
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.2})
    counts, bin_edges, ignored = ax1.hist(data, N_bins, histtype='stepfilled', alpha=0.4,
                                        label='histogram')

    # calculate area of histogram (area under PDF should be 1)
    area_hist = ((bin_edges[1:] - bin_edges[:-1]) * counts).sum()

    # plot fit into histogram
    ax1.plot(x_fit, data_fit*area_hist, label='fitted and area-scaled PDF', linewidth=2)
    ax1.legend()

    # equally sized bins in log10-scale and centers
    bins_log10 = np.logspace(np.log10(data.min()), np.log10(data.max()), N_bins)
    bins_log10_cntr = (bins_log10[1:] + bins_log10[:-1]) / 2

    # histogram plot
    counts, bin_edges, ignored = ax2.hist(data, bins_log10, histtype='stepfilled', alpha=0.4,
                                        label='histogram')

    # calculate length of each bin and its centers(required for scaling PDF to histogram)
    bins_log_len = np.r_[bin_edges[1:] - bin_edges[: -1], 0]
    bins_log_cntr = bin_edges[1:] - bin_edges[:-1]

    # get pdf-values for same intervals as histogram
    data_fit_log = stats.lognorm.pdf(bins_log10, shape, loc=loc, scale=scale)

    # pdf-values for centered scale
    data_fit_log_cntr = stats.lognorm.pdf(bins_log10_cntr, shape, loc=loc, scale=scale)

    # pdf-values using cdf 
    data_fit_log_cntr2_ = stats.lognorm.cdf(bins_log10, shape, loc=loc, scale=scale)
    data_fit_log_cntr2 = np.diff(data_fit_log_cntr2_)

    # plot fitted and scaled PDFs into histogram
    ax2.plot(bins_log10, 
            data_fit_log * bins_log_len * counts.sum(), '-', 
            label='PDF with edges',  linewidth=2)

    ax2.plot(bins_log10_cntr, 
            data_fit_log_cntr * bins_log_cntr * counts.sum(), '-', 
            label='PDF with centers', linewidth=2)

    ax2.plot(bins_log10_cntr, 
            data_fit_log_cntr2 * counts.sum(), 'b-.', 
            label='CDF with centers', linewidth=2)

    ax2.set_xscale('log')
    ax2.set_xlim(bin_edges.min(), bin_edges.max())
    ax2.legend(loc=3)

    yellowcab.io.save_fig(fig, str("log_distr_" + col))
    plt.show()
