# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:47:16 2025

@author: RafaelSuñé
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from scipy.special import gammaln, logsumexp
from scipy.optimize import root_scalar
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformWeibullSite
from py_wake import BastankhahGaussian
from py_wake.wind_turbines import WindTurbine, WindTurbines
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import urllib.request as urllib
from windrose import WindroseAxes
import os











#########################################################################################################################
###################################### TO REPLACE #######################################################################
#########################################################################################################################



plantname = "Wogas-Wohlsdorf"
locationname = "Wohlsdorf"
json_file = "../data/wind_turbines_models.json"
modelname = "E175_7000kW"
wind_turbine_name='Enercon E-175 EP5 7 MW'
num_iterations = 500
x_coord = [9.468773, 9.459934, 9.463604,9.468993]
y_coord = [53.115252,53.115317,53.112773,53.11152]

################




# plantname = "Grohnde-Nord"
# locationname = "Grohnde"
# json_file = "../data/wind_turbines_models.json"
# modelname = "V172_7200kW"
# wind_turbine_name='Vestas V-172 7.2 MW'
# num_iterations = 1000
# x_coord = [9.385293, 9.380588, 9.396692]
# y_coord = [52.022409, 52.01845, 52.024889]





































filename = f"WeatherData_{locationname}_Repowering_NEWA_wind_bats"


# Define root paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def read_turbine_model(json_file, modelname):
    """Reads wind turbine model parameters from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            turbines = json.load(f)
        turbine_data = turbines.get(modelname, None)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return None

    if not turbine_data:
        print(f"Error: Model '{modelname}' not found in JSON file.")
        return None
    
    # Extract relevant data
    u_values = turbine_data.get("u_values", [])
    power_values = turbine_data.get("power_values", [])
    air_density = 1.225  # Standard air density

    # Create DataFrame for power curve
    df_power_curve = pd.DataFrame({
        "Wind speed oscillating speed [m/s]": u_values,
        "Power in [kW]": power_values
    })

    # # Insert a formatted header row at the top
    # header_df = pd.DataFrame({
    #     "Wind speed oscillating speed [m/s]": [f"{modelname}"],
    #     "Power in [kW]": [f"Air density [kg/m³] {air_density}"]
    # })

    # Combine formatted header with data
    df_final = pd.concat([df_power_curve], ignore_index=True)

    # Save to CSV
    df_final.to_csv(os.path.join(TABLES_DIR, f"{modelname}_table.csv"), index=False)
    
    return turbine_data

def add_wind_speed_hub_height(df, hub_height):
    
    if hub_height > 150:
        
        a = 1 - ((hub_height - 150) / 50)
        b = 1 - ((200 - hub_height) / 50)
        
        h1 = 150
        h2 = 200

        print(a + b)
    
    elif 100 < hub_height < 150:
        
        a = 1 - ((hub_height - 100) / 50)
        b = 1 - ((150 - hub_height) / 50)
        
        h1 = 100
        h2 = 150
        
        print(a + b)
        
    elif hub_height < 100:
        
        a = 1 - ((hub_height - 50) / 50)
        b = 1 - ((100 - hub_height) / 50)
        
        h1 = 50
        h2 = 100
        
        print(a + b)
        
    else:
        
        print("No need to interpolate")
    
    try:
    
        df[f"wind_speed_{hub_height}m"] = a * df[f"wind_speed_{h1}m"].values + b * df[f"wind_speed_{h2}m"].values
        
        direction_h1_rad = np.deg2rad(df[f'wind_direction_{h1}m'])
        direction_h2_rad = np.deg2rad(df[f'wind_direction_{h2}m'])
    
        sin_avg = a * np.sin(direction_h1_rad) + b * np.sin(direction_h2_rad)
        cos_avg = a * np.cos(direction_h1_rad) + b * np.cos(direction_h2_rad)
    
        avg_direction_rad = np.arctan2(sin_avg, cos_avg)
        avg_direction_deg = np.rad2deg(avg_direction_rad)
        avg_direction_deg = (avg_direction_deg + 360) % 360
        df[f'wind_direction_{hub_height}m'] = avg_direction_deg
    
    except NameError:
        
        print("No need to interpolate")
    
    return(df)


def wind_rose(df, hub_height, plantname):
    
    """
    Generate a wind rose plot based on wind speed and direction data.

    The wind rose plot visualizes the distribution of wind speed and direction
    for a specific location.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing wind speed and direction data.
    info : dict
        Additional information related to the wind rose plot, such as the plant name.

    Returns
    -------
    None
        The function generates and displays the wind rose plot.

    """    
    mean_speed = np.nanmean(df[f"wind_speed_{hub_height}m"])
    
    
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    #ax.set_title(info["plantName"])
    ax.set_title(f"Wind rose with wind speed mean {mean_speed:.2f} m/s at height {hub_height} m")
    ax.bar(df[f"wind_direction_{hub_height}m"], df[f"wind_speed_{hub_height}m"], opening=0.8, edgecolor='white', normed=True)
    ax.set_legend(title="Wind Speed (m/s)", loc="lower right")
    
    plt.savefig(os.path.join(FIGURES_DIR, "wind_rose_"+plantname+".png"))
    plt.show()
    
    return(np.round(mean_speed, 2))

def correlation_heatmap(data, hub_height, plantname, plot_title="Correlation heatmap",
                        fig_size=(10, 10)):
    """
    Draw correaltion graph for data using heatmap.

    Parameters
    ----------
    data : DataFrame
            Data to plot correlation graph.
    plot_title : str, optional
            Title to set for Graph. The default is "".
    fig_size : series of int, optional
            Figure size to plot. The default is (10,10).

    Returns
    -------
    None.

    """
    data = data[['temperature', 'pressure', 'windspeed', 'winddirection',
           'precipitation', 'wind_speed_100m', 'wind_direction_100m',
           'temperature_100m', f'wind_speed_{hub_height}m', f'wind_direction_{hub_height}m']]
    
    correlations = data.corr()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        correlations,
        vmax=1.0,
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.25,
        annot=True,
        cbar_kws={"shrink": 0.70},
    )
    plt.title(plot_title, size=20)
    plt.savefig(os.path.join(FIGURES_DIR, "heatmap_"+plantname+".png"))

    plt.show()

def weibull_distribution(data_speed, bins=25):
    """Computes Weibull distribution parameters and plots histogram."""
    def weibull_params(mean, std):
        def equation(c, mean, std):
            return np.real(logsumexp([gammaln(1 + 2/c) - 2*gammaln(1+1/c), np.pi*1j]) - 2*np.log(std) + 2*np.log(mean))
        res = root_scalar(equation, args=(mean, std), method='bisect', bracket=[1e-300, 1e300], maxiter=2000)
        c = res.root
        scale = np.exp(np.log(mean) - gammaln(1 + 1/c))
        return c, scale
    
    k, c = weibull_params(np.nanmean(data_speed), np.nanstd(data_speed))
    return k, c



def weib_distribution_plot(data_speed, bins, plantname):
    """
    Plot the Weibull distribution of wind speeds.

    Calculates the Weibull distribution parameters from the mean and standard deviation
    of the provided wind speed data and plots the Weibull distribution curve along with
    a histogram of the wind speed data.

    Parameters
    ----------
    data_speed : array-like
        Wind speed data.
    bins : int
        Number of bins for the histogram.

    Returns
    -------
    tuple
        Tuple containing the Weibull shape parameter (k_s) and scale parameter (c_s).

    """
    
    # Function to calculate Weibull parameters from mean and std
    def weibull_c_scale_from_mean_std(mean, std):
        log_mean, log_std = np.log(mean), np.log(std)
    
        def r(c, mean, std):
            logratio = (logsumexp([gammaln(1 + 2/c) - 2*gammaln(1+1/c), np.pi*1j])
                        - 2*log_std + 2*log_mean)
            return np.real(logratio)
    
        # Solve for the Weibull shape parameter using root finding
        res = root_scalar(r, args=(mean, std), method='bisect',
                          bracket=[1e-300, 1e300], maxiter=2000, xtol=1e-16)
        assert res.converged
        c = res.root
        scale = np.exp(log_mean - gammaln(1 + 1/c))
        return (c, scale)
    
    # Calculate Weibull parameters from mean and std of the wind speed data
    k_s, c_s = weibull_c_scale_from_mean_std(np.nanmean(data_speed), np.nanstd(data_speed))
    
    # Weibull distribution function
    def weib(v, k_shape, c_scale):
        return (k_shape / c_scale * (v / c_scale)**(k_shape - 1) * np.exp(-(v / c_scale)**k_shape))
    
    # Generate an array of wind speeds
    speed = np.linspace(np.min(data_speed), np.max(data_speed), bins)
    
    # Calculate histogram counts and bin edges
    counts, bins = np.histogram(data_speed, bins)
    
    # Calculate Weibull distribution values
    w = weib(speed, k_s, c_s)
        
    # Create the plot
    fig, ax1 = plt.subplots()
    plt.title("Wind speed weibull distribution")
    ax1.set_xlabel("Wind speed at hub's height [m/s]")
    ax1.set_ylabel("Bins", color="g")
    plt.hist(speed, bins, weights=counts, color="g")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Prob. [%]", color="b")
    ax2.plot(speed, w, color="b")
    ax2.text(np.nanmax(speed) * 0.75, np.nanmax(w) * 0.75, f"k_s = {round(k_s,3)} \nc_s = {round(c_s,3)}")
    plt.savefig(os.path.join(FIGURES_DIR, f"weibull_distribution_{plantname}.png"))
    plt.show()
    
    
    # Return Weibull shape and scale parameters
    return (k_s, c_s)


def weibull_distribution_sections(df, hub_height, plantname):
    
    lim1 = np.arange(0, 331, 30)
    lim2 = np.arange(30, 361, 30)

    wind_speed_mean = []
    wind_speed_max = []
    wind_speed_max_percentage = []

    
    a_list = []
    k_list = []
    percentage = []
    ws_list = []
    
    
    for i in range(len(lim1)):
    # #for i in range(1):
        
        if i == -1:
            
            df_section = df[(df[f'wind_direction_{hub_height}m'] > lim1[i]+360) | (df[f'wind_direction_{hub_height}m'] <= lim2[i])]
    
            k, a = weibull_distribution(df[f'wind_speed_{hub_height}m'])
        
        else:
            
        
            df_section = df[(df[f'wind_direction_{hub_height}m'] > lim1[i]) & (df[f'wind_direction_{hub_height}m'] <= lim2[i])]
        
        
            k, a = weibull_distribution(df[f'wind_speed_{hub_height}m'])
            perc = (len(df_section)/len(df))*100
            
            a_list.append(a.round(3))
            k_list.append(np.float64(k).round(3))
            percentage.append(np.float64(perc).round(3))
            ws_list.append(np.nanmean(df_section[f'wind_speed_{hub_height}m']).round(3))
            
    df_weibull = pd.DataFrame(data = {"From (deg)":lim1+1, "To (deg)":lim2,"Wind Speed mean (m/s)":ws_list, "K":k_list, "A":a_list, "Percentage section (%)":percentage})
    wind_speed_mean.append(np.nanmean(ws_list).round(3))
    wind_speed_max.append(np.nanmax(ws_list).round(3))
    wind_speed_max_percentage.append(np.nanmax(percentage))
    df_weibull.to_csv(os.path.join(TABLES_DIR, f"Weibull_distribution_{plantname}_{hub_height}.csv"), index = False)


def plot_monthly_wind_speed(df, hub_height, plantname):
    """
    Processes wind data to calculate and plot the monthly mean wind speed
    with standard deviation as error bars.
    
    Parameters:
    wind_data (pd.DataFrame): A DataFrame with two columns: 'timestamp' and 'wind_speed'.
    """
    # Rename columns for clarity
    df= df[["timestamp", f"wind_speed_{hub_height}m"]]
    
    # Convert timestamp to datetime and set as index
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.set_index('timestamp', inplace=True)
    
    # Group by month across all years
    monthly_mean_std = df.groupby(df.index.month).agg(['mean', 'std'])
    monthly_mean_std.columns = ['mean_wind_speed', 'std_wind_speed']
    monthly_mean_std.index = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(
        monthly_mean_std.index, 
        monthly_mean_std['mean_wind_speed'], 
        yerr=monthly_mean_std['std_wind_speed'], 
        capsize=5, 
        color="#1E90FF", 
        alpha=0.7, 
        label="Average wind speed"
    )
    
    ax.set_title("Monthly Average Wind Speed with Variability", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Wind Speed [m/s]", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"wind_speed_month_{plantname}.png"))
    plt.show()

def generate_TMY(df):
    """Generates Typical Meteorological Year (TMY) data, accounting for leap years."""
    unique_years = df['timestamp'].dt.year.unique()
    tmy_data = pd.DataFrame()
    for month in range(1, 13):
        selected_year = random.choice(unique_years)
        selected_data = df[(df['timestamp'].dt.month == month) & (df['timestamp'].dt.year == selected_year)]
        
        # Handle February for leap years
        if month == 2:
            if (selected_year % 4 == 0 and (selected_year % 100 != 0 or selected_year % 400 == 0)):
                selected_data = selected_data[selected_data['timestamp'].dt.day != 29]
                tmy_data = pd.concat([tmy_data, selected_data], ignore_index=True)
            else:
                tmy_data = pd.concat([tmy_data, selected_data], ignore_index=True)
        else:
            tmy_data = pd.concat([tmy_data, selected_data], ignore_index=True)
    return tmy_data

def setup_wind_farm(df, turbine_data, modelname, hub_height, x_pos, y_pos, plantname):
    """Sets up the wind farm model."""
    wind_turbine = WindTurbine(
        name=modelname,
        diameter=turbine_data['blade_diameter'],
        hub_height=hub_height,
        powerCtFunction=PowerCtTabular(turbine_data['u_values'], turbine_data['power_values'], 'kW', turbine_data['ct_values'])
    )
    
    k, c = weibull_distribution(df[f'wind_speed_{hub_height}m'])
    weib_distribution_plot(df[f'wind_speed_{hub_height}m'], 25, plantname)
    site = UniformWeibullSite(p_wd=[.25], a=[c], k=[k], ti=0.1)
    return BastankhahGaussian(site, wind_turbine), x_pos, y_pos, wind_turbine

def run_simulation(df, wf_model, hub_height, x_pos, y_pos, num_iterations):
    """Runs simulation using TMY data."""
    data = []
    wind_speed_data = pd.DataFrame()
    wind_direction_data = pd.DataFrame()
    
    for i in range(num_iterations):
        
        tmy_data = generate_TMY(df)
        ws = tmy_data[f'wind_speed_{hub_height}m'].values
        wd = tmy_data[f'wind_direction_{hub_height}m'].values
        time_stamp = np.arange(1, len(wd)+1)
        sim_res = wf_model(x_pos, y_pos, h=hub_height, wd=wd, ws=ws, time = time_stamp)
        power_wake = np.sum(sim_res.aep(with_wake_loss=True).values * 1e6)
        
        data.append([i, power_wake])
        wind_speed_data[f'{i}'] = ws
        wind_direction_data[f'{i}'] = wd
        
        data_df = pd.DataFrame(data = data, columns=['iteration', 'sum_power'])


        
    return data_df, wind_speed_data, wind_direction_data

def save_results(data_df, filename_prefix):
    """Saves simulation results to CSV files."""
    data_df.to_csv(os.path.join(TABLES_DIR, f"{filename_prefix}_summary.csv"), index=False)
    print(f"Results saved: {filename_prefix}_summary.csv")

    
def plot_percentiles(df, plantname):
    df_sorted = df.sort_values(by='sum_power')
    
    p50 = df_sorted['sum_power'].median()
    p75 = df_sorted['sum_power'].quantile(0.25)
    p90 = df_sorted['sum_power'].quantile(0.10)
    p10 = df_sorted['sum_power'].quantile(0.90)
    std_dev = df_sorted['sum_power'].std()
    mean = p50
    
    # Convert to numpy arrays explicitly
    x_values = df_sorted['sum_power'].to_numpy()
    y_gaussian = (1 / (std_dev * np.sqrt(2 * np.pi)) * 
                 np.exp(- (x_values - mean) ** 2 / (2 * std_dev ** 2)))
    
    plt.figure(figsize=(10, 5))
    plt.hist(x_values, bins=50, density=True, alpha=0.5, 
             label='Wind Plant Production')
    
    # Plot using numpy arrays
    plt.plot(x_values, y_gaussian, 
             label='Gaussian Distribution', 
             linewidth=2, color='red')
    
    plt.title(f'Yearly production at {plantname}')
    plt.xlabel('Yearly production [kWh/y]')
    plt.axvline(p10, color='yellow', linestyle='--', 
                label=f'P10: {p10/1e6:.2f} M. kWh')
    plt.axvline(p50, color='blue', linestyle='--', 
                label=f'P50: {p50/1e6:.2f} M. kWh')
    plt.axvline(p75, color='green', linestyle='--', 
                label=f'P75: {p75/1e6:.2f} M. kWh')
    plt.axvline(p90, color='purple', linestyle='--', 
                label=f'P90: {p90/1e6:.2f} M. kWh')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"Percentile_distribution_{plantname}_{modelname}_{hub_height}.png"), 
                format="png", dpi=300)
    plt.show()
    
    return(p10, p50, p75, p90)
    
def percentiles(df, wind_speed_df, wind_direction_df, wf_model, plantname):
    
    df_sorted = df.sort_values(by='sum_power')
    
    p50 = df_sorted['sum_power'].median()
    p75 = df_sorted['sum_power'].quantile(0.25)
    p90 = df_sorted['sum_power'].quantile(0.10)
    
    i_p50 = df_sorted.iloc[(df_sorted['sum_power'] - p50).abs().argsort()[:1]]['iteration'].values[0]
    i_p75 = df_sorted.iloc[(df_sorted['sum_power'] - p75).abs().argsort()[:1]]['iteration'].values[0]
    i_p90 = df_sorted.iloc[(df_sorted['sum_power'] - p90).abs().argsort()[:1]]['iteration'].values[0]
    
    i_array = [i_p50, i_p75, i_p90]
    
    data_percentiles = []
    df_percentiles = pd.DataFrame()
    df_percentiles_WEA0 = []
    df_percentiles_WEA1 = []
    df_percentiles_WEA2 = []
    
    for per in i_array:
        
        ws = wind_speed_df[str(per)]
        wd = wind_direction_df[str(per)]
        time_stamp = np.arange(1, len(wd)+1)
        sim_res = wf_model(x_pos, y_pos, h=hub_height, wd=wd, ws=ws, time = time_stamp)
        power_wake = sim_res.aep(with_wake_loss=True).values * 1e6
        #power_wake = power_wake.round(2)
        
        data_percentiles.append(([np.sum(power_wake).round(2)]))
        df_percentiles_WEA0.append(([np.sum(power_wake[0]).round(2)]))
        df_percentiles_WEA1.append(([np.sum(power_wake[1]).round(2)]))
        df_percentiles_WEA2.append(([np.sum(power_wake[2]).round(2)]))
        
        if per == i_p50:
            
            time_simulation = pd.date_range("2001", "2002", freq="30min")[:-1]
            df_sim_wake = pd.DataFrame(data={"timestamp":time_simulation, "wind_speed":ws, "wind_direction":wd,
                                  "WEA0":power_wake[0].round(2), "WEA1":power_wake[1].round(2), "WEA2":power_wake[2].round(2), "Total": np.sum(power_wake.round(2), axis=0)
                                  })
            
            cols_to_round = ['wind_speed', 'wind_direction', 'WEA0', 'WEA1', 'WEA2', 'Total']
            df_sim_wake[cols_to_round] = df_sim_wake[cols_to_round].round(2)
            df_sim_wake.to_csv(os.path.join(TABLES_DIR, "P50_"+plantname+'.csv'), index = False)

    
    #df_percentiles = pd.DataFrame(data = data_percentiles, columns=['power'])    
    df_percentiles = pd.DataFrame(data = {'power':data_percentiles, "WEA0":df_percentiles_WEA0,
                                          "WEA1":df_percentiles_WEA1, "WEA2":df_percentiles_WEA2})    
    cols_to_round = ['power', 'WEA0', 'WEA1', 'WEA2']
    df_percentiles[cols_to_round] = df_percentiles[cols_to_round].round(2)
    df_percentiles.to_csv(os.path.join(TABLES_DIR, "percentiles_sum_"+plantname+'.csv'), index = False)

    
    return(i_p50, i_p75, i_p90, df_sim_wake, df_percentiles, df_percentiles_WEA0, df_percentiles_WEA1, df_percentiles_WEA2)

def percentiles_gen(df, wind_speed_df, wind_direction_df, wf_model, plantname, x_pos, y_pos, hub_height):
    """Calculate energy production percentiles for variable number of turbines.
    
    Args:
        df: DataFrame with simulation results
        wind_speed_df: DataFrame with wind speed data
        wind_direction_df: DataFrame with wind direction data
        wf_model: Wake model instance
        plantname: Name of the wind plant
        x_pos: Array of x positions for turbines
        y_pos: Array of y positions for turbines
        hub_height: Hub height in meters
    
    Returns:
        tuple: (i_p50, i_p75, i_p90, df_sim_wake, df_percentiles, *turbine_arrays)
    """
    # Sort and calculate percentiles
    df_sorted = df.sort_values(by='sum_power')
    
    p50 = df_sorted['sum_power'].median()
    p75 = df_sorted['sum_power'].quantile(0.25)
    p90 = df_sorted['sum_power'].quantile(0.10)
    
    i_p50 = df_sorted.iloc[(df_sorted['sum_power'] - p50).abs().argsort()[:1]]['iteration'].values[0]
    i_p75 = df_sorted.iloc[(df_sorted['sum_power'] - p75).abs().argsort()[:1]]['iteration'].values[0]
    i_p90 = df_sorted.iloc[(df_sorted['sum_power'] - p90).abs().argsort()[:1]]['iteration'].values[0]
    
    i_array = [i_p50, i_p75, i_p90]
    num_turbines = len(x_pos)
    
    # Initialize data structures
    data_percentiles = []
    df_percentiles_data = {'power': []}
    for i in range(num_turbines):
        df_percentiles_data[f'WEA{i}'] = []
    
    # Process each percentile
    for per in i_array:
        ws = wind_speed_df[str(per)]
        wd = wind_direction_df[str(per)]
        time_stamp = np.arange(1, len(wd)+1)
        sim_res = wf_model(x_pos, y_pos, h=hub_height, wd=wd, ws=ws, time=time_stamp)
        power_wake = sim_res.aep(with_wake_loss=True).values * 1e6
        
        # Store total power
        data_percentiles.append(np.sum(power_wake).round(2))
        df_percentiles_data["power"].append(np.sum(power_wake).round(2))

        
        # Store individual turbine data
        for i in range(num_turbines):
            df_percentiles_data[f'WEA{i}'].append(np.sum(power_wake[i]).round(2))
        
        # Create detailed DataFrame for P50 scenario
        if per == i_p50:
            time_simulation = pd.date_range("2001", "2002", freq="30min")[:-1]
            data_dict = {
                "timestamp": time_simulation,
                "wind_speed": ws,
                "wind_direction": wd
            }
            for i in range(num_turbines):
                data_dict[f'WEA{i}'] = power_wake[i].round(2)
            data_dict['Total'] = np.sum(power_wake.round(2), axis=0)
            
            df_sim_wake = pd.DataFrame(data=data_dict)
            numeric_cols = ['wind_speed', 'wind_direction', 'Total'] + \
                          [f'WEA{i}' for i in range(num_turbines)]
            df_sim_wake[numeric_cols] = df_sim_wake[numeric_cols].round(2)
            df_sim_wake.to_csv(os.path.join(TABLES_DIR, f"P50_{plantname}.csv"), index=False)
    
    # Create final DataFrames
    df_percentiles = pd.DataFrame(df_percentiles_data)
    #df_percentiles['power'] = data_percentiles
    cols_to_round = ['power'] + [f'WEA{i}' for i in range(num_turbines)]
    df_percentiles[cols_to_round] = df_percentiles[cols_to_round].round(2)
    df_percentiles.to_csv(os.path.join(TABLES_DIR, f"percentiles_sum_{plantname}.csv"), index=False)
    
    # Prepare return values
    # for i in range(num_turbines):
    #     return_values += (df_percentiles[f'WEA{i}'].tolist(),)
    
    return i_p50, i_p75, i_p90, df_sim_wake, df_percentiles

def plots_wind_farm(i_p50, wind_speed_df, wind_direction_df, wf_model, wind_turbine, plantname, modelname, hub_height):
    
    ws = wind_speed_df[str(i_p50)]
    wd = wind_direction_df[str(i_p50)]
    time_stamp = np.arange(1, len(wd)+1)
    sim_res = wf_model(x_pos, y_pos, h=hub_height, wd=wd, ws=ws, time = time_stamp)
    
    x = np.array([(i-np.min(x_pos)) for i in x_pos]) #transform to meters
    y = np.array([(i-np.min(y_pos)) for i in y_pos])
    
    x_max, x_min = np.nanmax(x), np.nanmin(x)
    y_max, y_min = np.nanmax(y), np.nanmin(y)
    
    
    aep_wt = sim_res.aep().sum(axis=1).values
    plt.figure(figsize=(12,8))
    plt.title(f"WP-{plantname} Annual Energy Production (AEP) with Wake Effect")
    #aep = sim_res.aep()
    wind_turbine.plot(x, y)
    c = plt.scatter(x, y, c=aep_wt)
    plt.colorbar(c, label='AEP [GWh]')
    plt.ylim(y_min - (y_max-y_min)*0.30, y_max + (y_max-y_min)*0.30)
    plt.xlim(x_min - (x_max-x_min)*0.30, x_max + (x_max-x_min)*0.30)
    plt.legend().remove()
    plt.xlabel("Distance [m]")
    plt.ylabel("Distance [m]")
    plt.savefig(os.path.join(FIGURES_DIR, f"AEP_{plantname}_{modelname}_{hub_height}.png"), format = "png")
    plt.show()
    
    aep_wt_nowake = sim_res.aep(with_wake_loss=False).sum(axis=1).values
    plt.figure(figsize=(12,8))
    plt.title(f"WP-{plantname} Annual Energy Production (AEP) without Wake Effect")
    #aep = sim_res.aep()
    wind_turbine.plot(x, y)
    c = plt.scatter(x, y, c=aep_wt_nowake)
    plt.colorbar(c, label='AEP [GWh]')
    plt.ylim(y_min - (y_max-y_min)*0.30, y_max + (y_max-y_min)*0.30)
    plt.xlim(x_min - (x_max-x_min)*0.30, x_max + (x_max-x_min)*0.30)
    plt.legend().remove()
    plt.xlabel("Distance [m]")
    plt.ylabel("Distance [m]")
    plt.savefig(os.path.join(FIGURES_DIR, f"AEP_nowake_{plantname}_{modelname}_{hub_height}.png"), format = "png")
    plt.show()
    
    
    
    aep_wt_gwh = np.round(aep_wt * 1e6, 2)
    aep_wt_nowake_gwh = np.round(aep_wt_nowake * 1e6, 2)

    # Compute percentage difference
    percentage_loss = np.round(((aep_wt_nowake_gwh - aep_wt_gwh) / aep_wt_nowake_gwh) * 100, 2)

    # Create DataFrame
    column_names = [f"WEA{i}" for i in range(len(aep_wt))]
    df_aep = pd.DataFrame(
        data=[aep_wt_nowake_gwh, aep_wt_gwh, percentage_loss],
        index=["No wake effect [GWh]", "With wake effect [GWh]", "Percentage [\%]"],
        columns=column_names
    )
    
    if len(aep_wt) >= 5:
        
        df_aep.iloc[:,:5].to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_{hub_height}_aep_table.csv"))
        df_aep.iloc[:,:].to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_{hub_height}_aep_all_table.csv"))
    else:
        
        df_aep.to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_{hub_height}_aep_table.csv"))
    
    return df_aep
    

def plot_resampled_power(df, plantname, modelname, hub_height):
    """
    Plot resampled power of plant.
    
    Resample the plant power on daily, weekly,
    monthly and yearly basis by taking sum.
    Then plot all resampled data in one plot.
    
    TODO:Production column Or column to plot
    should be renamed as "power" in dataframe

    Parameters
    ----------
    data : DataFrame
            Plant production data.
    unit : str
        Actual unit of power in format [<unit>]
        e.g. [kW], [MW], [kWh], [MWh]

    Returns
    -------
    None.

    """
    data = df.copy()
    data["Day of year"] = data["timestamp"].dt.dayofyear

    plt.figure(figsize=(12,8))
    plt.title(f"WP-{plantname} hourly power behavior")
    plt.plot(data["Day of year"].to_numpy(), data["Total"].to_numpy())
    plt.xlabel("Day of the year")
    plt.ylabel("Power (kWh)")
    plt.savefig(os.path.join(FIGURES_DIR, f"WP_{plantname}_yearly_production_timeseries_{modelname}_{hub_height}.png"))
    plt.show()
    
    
    
    data = df.copy()
    data.rename(columns = {"Total":"power"}, inplace = True)
    
    data.set_index('timestamp', inplace=True)
    # fig = plt.figure(figsize=(18,16))
    # fig.subplots_adjust(hspace=.4)
    # ax1 = fig.add_subplot(5,1,1)
    
    daily = data['power'].resample('D').sum().to_frame()
    daily.reset_index(inplace=True)
    daily["Day of year"] = daily["timestamp"].dt.dayofyear
    daily["year"] = daily["timestamp"].dt.strftime('-%y')
    daily["Day of year"] = daily["Day of year"].astype(str)+daily["year"].astype(str)
    # ax1.set_title('Sum of estimated power resampled over day',size=15)
    # sns.barplot(x=daily["Day of year"], y=daily["power"], data=daily, ax=ax1)
    # ax1.set_ylabel("Power ")
    # plt.gca().yaxis.set_major_formatter(
    #     FormatStrFormatter("%d")
    #     )
    # plt.gca().xaxis.set_major_formatter(
    #     FormatStrFormatter("")
    #     )
    
    # ax2 = fig.add_subplot(5,1,2)
    weekly = data['power'].resample('W').sum().to_frame()
    weekly.reset_index(inplace=True)
    weekly["Week of year"] = weekly["timestamp"].dt.strftime('%U')
    # ax2.set_title('Sum of estimated power resampled over week',size=15)
    # sns.barplot(x=weekly["Week of year"], y=weekly["power"], data=weekly, ax=ax2)
    # ax2.set( ylabel="Power [kWh]")
    # plt.gca().yaxis.set_major_formatter(
    #     FormatStrFormatter("%d")
    #     )
    
    # ax3 = fig.add_subplot(5,1,3)
    monthly = data['power'].resample('M').sum().to_frame()
    monthly.reset_index(inplace=True)
    monthly["Month"] = monthly["timestamp"].apply(lambda x: x.strftime('%b'))    
    # ax3.set_title('Sum of estimated power resampled over month',size=15)
    # sns.barplot(x=monthly["Month"], y=monthly["power"], data=monthly, ax=ax3)
    # ax3.set_ylabel("Power [kWh]")
    # plt.gca().yaxis.set_major_formatter(
    #     FormatStrFormatter("%d")
    #     )
    
    # ax4  = fig.add_subplot(5,1,4)
    # ax4.set_title('Sum of estimated power resampled over quarter',size=15)
    quarterly = data['power'].resample('Q').sum().to_frame()
    quarterly.reset_index(inplace=True)
    quarterly["Quarter of year"] = quarterly["timestamp"].dt.to_period('Q')
    #If to remove year from column
    quarterly["Quarter of year"] = quarterly["Quarter of year"].apply(lambda x: x.strftime('Q%q'))     
    # sns.barplot(x=quarterly["Quarter of year"], y=quarterly["power"], data=quarterly, ax=ax4)
    # ax4.set_ylabel("Power [kWh]")
    # plt.gca().yaxis.set_major_formatter(
    #     FormatStrFormatter("%d")
    #     )
    
    
    # ax5  = fig.add_subplot(5,1,5)
    yearly = data['power'].resample('A').sum().to_frame()
    yearly.reset_index(inplace=True)
    yearly["Year"] = yearly["timestamp"].dt.year
    # #If to remove year from column
    yearly["Year"] = yearly["timestamp"].apply(lambda x: x.strftime(''))    
    # ax5.set_title('Sum of estimated power resampled over year',size=15)
    # sns.barplot(x=yearly["Year"], y=yearly["power"], data=yearly, ax=ax5)
    # # yearly[["Year","power"]].plot(ax=ax5, kind='bar')
    # ax5.set_ylabel("Power [kWh]")
    # plt.gca().yaxis.set_major_formatter(
    #     FormatStrFormatter("%d")
    #     )
    # fig.suptitle(plantname+" resampled power",
    #              size=20)
    
    
    fig = plt.figure(figsize=(18,16))
    fig.subplots_adjust(hspace=.4)
    
    ax3 = fig.add_subplot(3,1,1)    
    ax3.set_title('Sum of estimated power resampled over month',size=15)
    sns.barplot(x=monthly["Month"], y=monthly["power"], data=monthly, ax=ax3)
    ax3.set_ylabel("Power [kWh]")
    plt.gca().yaxis.set_major_formatter(
        FormatStrFormatter("%d")
        )
    
    ax4  = fig.add_subplot(3,1,2)
    ax4.set_title('Sum of estimated power resampled over quarter',size=15)
    sns.barplot(x=quarterly["Quarter of year"], y=quarterly["power"], data=quarterly, ax=ax4)
    ax4.set_ylabel("Power [kWh]")
    plt.gca().yaxis.set_major_formatter(
        FormatStrFormatter("%d")
        )
    
    ax5  = fig.add_subplot(3,1,3)
    ax5.set_title('Sum of estimated power resampled over year',size=15)
    sns.barplot(x=yearly["Year"], y=yearly["power"], data=yearly, ax=ax5)
    # yearly[["Year","power"]].plot(ax=ax5, kind='bar')
    ax5.set_ylabel("Power [kWh]")
    plt.gca().yaxis.set_major_formatter(
        FormatStrFormatter("%d")
        )
    fig.suptitle(plantname+" resampled power",
                 size=20)
    
    data.reset_index(inplace=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"WP_{plantname}_{modelname}_monthly_quarterly_yearly_{hub_height}.png"))
    plt.show()
    
    # Plot pie graph of resampled production
    monthly_pie  = monthly[["Month", "power"]]
    monthly_pie.set_index("Month", inplace=True)
    monthly_pie.plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(8, 8))
    plt.title('Monthly contribution of estimated power')
    plt.ylabel("Power [kWh]")
    plt.savefig(os.path.join(FIGURES_DIR, f"WP_{plantname}_{modelname}_monthly_pie_{hub_height}.png"))
    plt.show()
    monthly_pie.reset_index(inplace=True)
    
    # Plot pie graph of resampled production
    quarterly_pie  = quarterly[["Quarter of year", "power"]]
    quarterly_pie.set_index("Quarter of year", inplace=True)
    quarterly_pie.plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(8, 8))
    plt.title('Quarterly contribution of estimated power')
    plt.ylabel("Power [kWh]")
    plt.savefig(os.path.join(FIGURES_DIR, f"WP_{plantname}_{modelname}_quarterly_pie_{hub_height}.png"))
    plt.show()
    quarterly_pie.reset_index(inplace=True)
    


    return daily,weekly,monthly,quarterly,yearly

def plot_power_curve(data, unit, plantname, hub_height, modelname):
    """
    Plot theoretcial curve with actual produced power against windspeed.

    Parameters
    ----------
    data : DataFrame
        Original data.
    info : dict
        Plant information.
    data_type : str, optional
        Sometimes different data to plot. The default is " ".
    hubHeight : str, optional
        Use windspeed at what height. The default is "100m".
    turbine : str, optional
        Turbine type. The default is " ".

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Better control over figure size
    scatter = ax.scatter(
        data["wind_speed"].to_numpy(), 
        data["WEA0"].to_numpy(),
    )
    
    # Labels and title
    ax.set_xlabel("Wind speed [m/s]", fontsize=12)
    ax.set_ylabel(f"Power {unit}", fontsize=12)
    ax.set_title(f"{plantname} WEA0 Power Curve", fontsize=14, pad=20)
    plt.savefig(os.path.join(FIGURES_DIR, f"Power_curve_{plantname}_{modelname}_{hub_height}.png"))
    plt.show()  
    
    data["Day of year"] = data["timestamp"].dt.dayofyear
    plt.figure(figsize=(12,8))
    plt.title(f"WP-{plantname} WEA0 hourly power behavior")
    plt.plot(data["Day of year"].to_numpy(), data["WEA0"].to_numpy())
    plt.xlabel("Day of the year")
    plt.ylabel("Power (kWh)")
    plt.savefig(os.path.join(FIGURES_DIR, f"WP_{plantname}_WEA0_yearly_production_timeseries_{modelname}_{hub_height}.png"))
    plt.show()


# def build_final_csv(monthly, quarterly, df_percentiles, plantname, modelname):
#     """
#     Merges monthly, quarterly, and percentile data into a structured CSV file for Word formatting.
    
#     - `monthly`: DataFrame with 'Month' and 'power' columns.
#     - `quarterly`: DataFrame with 'Quarter of year' and 'power' columns.
#     - `df_percentiles`: DataFrame with yearly P50, P75, P90 values.
#     - `output_filename`: Name of the output CSV file.
#     """

#     # # Prepare Monthly Data
#     # monthly = data[['power']].resample('M').sum().to_frame()
#     # monthly.reset_index(inplace=True)
#     monthly["Month"] = monthly["timestamp"].apply(lambda x: x.strftime('%b'))  
#     monthly_table = monthly[['Month', 'power']].rename(columns={'power': 'Sum of estimated power plant [kWh]'})
    
#     # Prepare Quarterly Data
#     quarterly_table = quarterly[['Quarter of year', 'power']].rename(columns={'power': 'Sum of estimated power plant [kWh]', 'Quarter of year':'Month'})
    
#     yearly_table_name = pd.DataFrame({
#         'Month': 'Yearly',
#         'Sum of estimated power plant [kWh]': ['']
#     }) 
#     # Prepare Yearly Percentile Data
#     yearly_table = pd.DataFrame({
#         'Month': ['P50', 'P75', 'P90'],
#         'Sum of estimated power plant [kWh]': df_percentiles['power'].round(2).values
#     })

#     # Combine DataFrames with Spacing for Readability
#     spacer = pd.DataFrame({'Month': [''], 'Sum of estimated power plant [kWh]': ['']})  # Empty row for spacing
#     combined = pd.concat([
#         monthly_table, spacer, quarterly_table, yearly_table_name, yearly_table
#     ], ignore_index=True)

#     # Save the Final CSV
#     combined.to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_montly_quarterly_yearly_table.csv", index=False)


def summary_table(df_percentiles, plantname, modelname,wind_turbine_name, hub_height):
    # Ensure wind_turbine_name is a string and not a tuple
    if isinstance(wind_turbine_name, tuple):
        wind_turbine_name = wind_turbine_name[0]  # Extract string if inside a tuple
    
    print("Debug - df_percentiles['power'] structure:")
    print(df_percentiles['power'].iloc[0])  # Check if it's a list, array, or scalar
    print(type(df_percentiles['power'].iloc[0]))
    
    

    df_summary = pd.DataFrame({
        "Turbine Type": [f"P50 {wind_turbine_name}", f"P75 {wind_turbine_name}", f"P90 {wind_turbine_name}"],
        "Yearly estimated energy production [kWh]": [
            f"{df_percentiles['power'].iloc[0].round(2)}",
            f"{df_percentiles['power'].iloc[1].round(2)}",
            f"{df_percentiles['power'].iloc[2].round(2)}"
        ],
        "Hub height [m]": [hub_height, hub_height, hub_height]
    })

    # Create a clean filename by removing unwanted characters
    safe_wind_turbine_name = str(wind_turbine_name).replace(" ", "_").replace(",", "")
    filename = f"WP_{plantname}_{modelname}_summary_table.csv"
    
    print(os.path.join(TABLES_DIR, filename))
    
    df_summary.to_csv(os.path.join(TABLES_DIR, filename), index=False)
    print(f"Saved CSV as: {filename}")

def build_final_csv(df_sim, df_percentiles , plantname, modelname, hub_height):
    """
    Merges monthly, quarterly, and percentile data into a structured CSV file for Word formatting.
    
    - `monthly`: DataFrame with 'Month' and 'power' columns.
    - `quarterly`: DataFrame with 'Quarter of year' and 'power' columns.
    - `df_percentiles`: DataFrame with yearly P50, P75, P90 values.
    - `output_filename`: Name of the output CSV file.
    """

    # Prepare Monthly Data
    data = df_sim.copy()
    data.rename(columns = {"Total":"power"}, inplace = True)
    
    data.set_index('timestamp', inplace=True)
    
    monthly = data.resample('M').sum()
    monthly.reset_index(inplace=True)
    monthly["Month"] = monthly["timestamp"].apply(lambda x: x.strftime('%b'))  
    monthly_table = monthly[['Month', 'power', 'WEA0', 'WEA1', 'WEA2']].rename(columns={'power': 'Sum of estimated power plant [kWh]', 
                                                                                        'WEA0' : 'Sum of estimated WEA0 [kWh]',
                                                                                        'WEA1' : 'Sum of estimated WEA1 [kWh]',
                                                                                        'WEA2' : 'Sum of estimated WEA2 [kWh]',})
    
    quarterly_table_name = pd.DataFrame({
        'Month': 'Quarterly',
        'Sum of estimated power plant [kWh]': [''],
        'Sum of estimated WEA0 [kWh]': [''],
        'Sum of estimated WEA1 [kWh]': [''],
        'Sum of estimated WEA2 [kWh]': [''],
    })
    # Prepare Quarterly Data
    quarterly = data.resample('Q').sum()
    quarterly.reset_index(inplace=True)
    quarterly["Quarter of year"] = quarterly["timestamp"].dt.to_period('Q')
    quarterly["Quarter of year"] = ["Q1", "Q2", "Q3", "Q4"]
    quarterly_table = quarterly[['Quarter of year', 'power', 'WEA0', 'WEA1', 'WEA2']].rename(columns={'power': 'Sum of estimated power plant [kWh]', 'Quarter of year':'Month',
                                                                                                      'WEA0' : 'Sum of estimated WEA0 [kWh]',
                                                                                                      'WEA1' : 'Sum of estimated WEA1 [kWh]',
                                                                                                      'WEA2' : 'Sum of estimated WEA2 [kWh]',})
    yearly_table_name = pd.DataFrame({
        'Month': 'Yearly',
        'Sum of estimated power plant [kWh]': [''],
        'Sum of estimated WEA0 [kWh]': [''],
        'Sum of estimated WEA1 [kWh]': [''],
        'Sum of estimated WEA2 [kWh]': [''],
    }) 
    
    
    
    
    # Prepare Yearly Percentile Data
    # yearly_table = pd.DataFrame({
    #     'Month': ['P50', 'P75', 'P90'],
    #     'Sum of estimated power plant [kWh]': df_percentiles['power'],
    #     'Sum of estimated WEA0 [kWh]': df_percentiles['WEA0'],
    #     'Sum of estimated WEA1 [kWh]': df_percentiles['WEA1'],
    #     'Sum of estimated WEA2 [kWh]': df_percentiles['WEA2']
    # })
    
    
    def extract_value(val):
        if isinstance(val, list):
            return float(val[0]) if len(val) > 0 else 0.0
        elif isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return float(val[1:-1])
        return float(val)
    
    # Prepare Yearly Percentile Data with cleaned values
    yearly_table = pd.DataFrame({
        'Month': ['P50', 'P75', 'P90'],
        'Sum of estimated power plant [kWh]': [extract_value(x) for x in df_percentiles['power']],
        'Sum of estimated WEA0 [kWh]': [extract_value(x) for x in df_percentiles['WEA0']],
        'Sum of estimated WEA1 [kWh]': [extract_value(x) for x in df_percentiles['WEA1']],
        'Sum of estimated WEA2 [kWh]': [extract_value(x) for x in df_percentiles['WEA2']]
    })

    # Combine DataFrames with Spacing for Readability
    spacer = pd.DataFrame({'Month': [''], 'Sum of estimated power plant [kWh]': [''],
                           'Sum of estimated WEA0 [kWh]': [''],
                           'Sum of estimated WEA1 [kWh]': [''],
                           'Sum of estimated WEA2 [kWh]': ['']})  # Empty row for spacing
    combined = pd.concat([
        monthly_table, quarterly_table_name, quarterly_table, yearly_table_name, yearly_table
    ], ignore_index=True)

    # Save the Final CSV
    combined.to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_monthly_quarterly_yearly_table_{hub_height}.csv"), index=False)


def build_final_csv_gen(df_sim, df_percentiles , plantname, modelname, hub_height):
    """
    Merges monthly, quarterly, and percentile data into a structured CSV file for Word formatting.
    
    - `monthly`: DataFrame with 'Month' and 'power' columns.
    - `quarterly`: DataFrame with 'Quarter of year' and 'power' columns.
    - `df_percentiles`: DataFrame with yearly P50, P75, P90 values.
    - `output_filename`: Name of the output CSV file.
    """

    # Identify turbine columns (all columns starting with 'WEA')
    turbine_cols = [col for col in df_sim.columns if col.startswith('WEA')]
    num_turbines = len(turbine_cols)
    
    # Prepare column renaming mappings
    rename_map = {'power': 'Sum of estimated power plant [kWh]'}
    for i in range(num_turbines):
        rename_map[f'WEA{i}'] = f'Sum of estimated WEA{i} [kWh]'
    
    # Prepare Monthly Data
    data = df_sim.copy()
    if 'Total' in data.columns:
        data.rename(columns={'Total': 'power'}, inplace=True)
    data.set_index('timestamp', inplace=True)
    
    # Monthly aggregation
    monthly = data.resample('M').sum()
    monthly.reset_index(inplace=True)
    monthly["Month"] = monthly["timestamp"].apply(lambda x: x.strftime('%b'))
    
    # Select and rename columns
    monthly_cols = ['Month', 'power'] + turbine_cols
    monthly_table = monthly[monthly_cols].rename(columns=rename_map)
    
    # Quarterly table header - create with correct number of rows
    quarterly_table_name = pd.DataFrame({
        'Month': ['Quarterly'],
        'Sum of estimated power plant [kWh]': ['']
    })
    # Add turbine columns
    for i in range(num_turbines):
        quarterly_table_name[f'Sum of estimated WEA{i} [kWh]'] = ['']
    
    # Quarterly aggregation
    quarterly = data.resample('Q').sum()
    quarterly.reset_index(inplace=True)
    quarterly["Quarter of year"] = quarterly["timestamp"].dt.to_period('Q')
    quarterly["Quarter of year"] = ["Q1", "Q2", "Q3", "Q4"]
    
    # Select and rename columns
    quarterly_cols = ['Quarter of year', 'power'] + turbine_cols
    quarterly_table = quarterly[quarterly_cols].rename(columns={
        **rename_map,
        'Quarter of year': 'Month'
    })
    
    # Yearly table header
    yearly_table_name = pd.DataFrame({
        'Month': ['Yearly'],
        'Sum of estimated power plant [kWh]': ['']
    })
    # Add turbine columns
    for i in range(num_turbines):
        yearly_table_name[f'Sum of estimated WEA{i} [kWh]'] = ['']
    
    # Prepare Yearly Percentile Data
    def extract_value(val):
        if isinstance(val, list):
            return float(val[0]) if len(val) > 0 else 0.0
        elif isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return float(val[1:-1])
        return float(val)
    
    # Build yearly table data
    yearly_data = {
        'Month': ['P50', 'P75', 'P90'],
        'Sum of estimated power plant [kWh]': [
            extract_value(x) for x in df_percentiles['power']
        ]
    }
    
    for i in range(num_turbines):
        yearly_data[f'Sum of estimated WEA{i} [kWh]'] = [
            extract_value(x) for x in df_percentiles[f'WEA{i}']
        ]
    
    yearly_table = pd.DataFrame(yearly_data)
    
    # Create spacer row with correct columns
    spacer_data = {'Month': ['']}
    for col in rename_map.values():
        spacer_data[col] = ['']
    spacer = pd.DataFrame(spacer_data)
    
    # Combine all sections
    combined = pd.concat([
        monthly_table,
        spacer,
        quarterly_table_name,
        quarterly_table,
        yearly_table_name,
        yearly_table
    ], ignore_index=True)

    # Save the Final CSV
    combined.to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_monthly_quarterly_yearly_table_{hub_height}.csv"), index=False)
    
    if num_turbines >= 5:
        
        combined.iloc[:,:5].to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_monthly_quarterly_yearly_table_{hub_height}.csv"), index=False)
        combined.iloc[:,:].to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_monthly_quarterly_yearly_table_all_{hub_height}.csv"), index=False)
    else:
        
        combined.to_csv(os.path.join(TABLES_DIR, f"{plantname}_{modelname}_monthly_quarterly_yearly_table_{hub_height}.csv"), index=False)


def summary_table(df_percentiles, plantname, modelname,wind_turbine_name, hub_height):
    # Ensure wind_turbine_name is a string and not a tuple
    if isinstance(wind_turbine_name, tuple):
        wind_turbine_name = wind_turbine_name[0]  # Extract string if inside a tuple
        
    print("Debug - df_percentiles['power'] structure:")
    print(df_percentiles['power'].iloc[0])  # Check if it's a list, array, or scalar
    print(type(df_percentiles['power'].iloc[0]))
    
    df_percentiles['power'] = df_percentiles['power'].apply(lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x))

    df_summary = pd.DataFrame({
        "Turbine Type": [f"P50 {wind_turbine_name}", f"P75 {wind_turbine_name}", f"P90 {wind_turbine_name}"],
        "Yearly estimated energy production [kWh]": [
            f"{df_percentiles['power'].iloc[0]}",
            f"{df_percentiles['power'].iloc[1]}",
            f"{df_percentiles['power'].iloc[2]}"
        ],
        "Hub height [m]": [hub_height, hub_height, hub_height]
    })

    # Create a clean filename by removing unwanted characters
    safe_wind_turbine_name = str(wind_turbine_name).replace(" ", "_").replace(",", "")
    filename = f"WP_{plantname}_{modelname}_summary_table_{hub_height}.csv"
    
    print(os.path.join(TABLES_DIR, filename))
    
    df_summary.to_csv(os.path.join(TABLES_DIR, filename), index=False)
    print(f"Saved CSV as: {filename}")
    
    p50 = df_percentiles['power'].iloc[0]
    p75 = df_percentiles['power'].iloc[1]
    p90 = df_percentiles['power'].iloc[2]
    
    #return(f"{p50/1000000:.2f}", f"{p75/1000000:.2f}", f"{p90/1000000:.2f}")


def generate_map_yandex(locationname, x_coord, y_coord):
    
    lon = np.nanmean(x_coord)
    lat = np.nanmean(y_coord)
    
    zoom = 12

    url = f"https://static-maps.yandex.ru/1.x/?lang=en-US&ll={lon},{lat}&size=450,450&z={zoom}&l=map&pt={lon},{lat},vkbkm"
    print("Here is printing where is being saved")
    print(os.path.join(FIGURES_DIR, locationname + "_map.png"))
    
    urllib.urlretrieve(url, os.path.join(FIGURES_DIR, locationname + "_map.png"))
    


#generate_map_yandex(plantname)


# Configuration
# plantname = "Wogas-Wohlsdorf"
# locationname = "Wohlsdorf"
# json_file = "../data/wind_turbines_models.json"
# modelname = "E175_7000kW"
# wind_turbine_name='Enercon E-175 EP5 7 MW',
# hub_height = 175
# num_iterations = 30
# filename = "WeatherData_Wohlsdorf_Repowering_NEWA_wind_bats"
# x_pos = np.array([9.455468, 9.459261, 9.459720]) * 66600  # Convert to meters
# y_pos = np.array([53.119035, 53.117940, 53.115319]) * 111339  # Convert to meters
x_pos = np.array(x_coord) * 66600  # Convert to meters
y_pos = np.array(y_coord)* 111339
################






df = pd.read_csv(f"../data/{filename}.csv")
df['timestamp'] = pd.to_datetime(df['time'])


        

turbine_data = read_turbine_model(json_file, modelname)
capacity = np.nanmax(np.array(turbine_data['power_values'])/1000)
hub_height = turbine_data['hub_height']

df = add_wind_speed_hub_height(df, hub_height)    

if turbine_data:
    #generate_map_yandex(locationname, x_coord, y_coord)
    mean_wind_speed = wind_rose(df, hub_height, plantname)
    #plot_monthly_wind_speed(df, hub_height, plantname)
    #correlation_heatmap(df, hub_height, plantname)
    #weibull_distribution_sections(df, hub_height, plantname)
    wf_model, x_pos, y_pos, wind_turbine = setup_wind_farm(df, turbine_data, modelname, hub_height, x_pos, y_pos, plantname)
    data_df, wind_speed_df, wind_direction_df = run_simulation(df, wf_model, hub_height, x_pos, y_pos, num_iterations)
    # #save_results(data_df, f"{plantname}_{modelname}_{hub_height}")
    p10, p50, p75, p90 = plot_percentiles(data_df, plantname)
    # #i_p50, i_p75, i_p90, df_sim_wake, df_percentiles, df_percentiles_WEA0, df_percentiles_WEA1, df_percentiles_WEA2 = percentiles(data_df, wind_speed_df, wind_direction_df, wf_model, plantname)
    i_p50, i_p75, i_p90, df_sim_wake, df_percentiles = percentiles_gen(data_df, wind_speed_df, wind_direction_df, wf_model, plantname, x_pos, y_pos, hub_height)
    df_wake = plots_wind_farm(i_p50, wind_speed_df, wind_direction_df, wf_model, wind_turbine, plantname, modelname, hub_height)
    # daily,weekly,monthly,quarterly,yearly =  plot_resampled_power(df_sim_wake, plantname, modelname, hub_height)
    # plot_power_curve(df_sim_wake, "kWh", plantname, hub_height, modelname)
    # #build_final_csv(monthly, quarterly, df_percentiles, plantname, modelname)
    # #build_final_csv(df_sim_wake, df_percentiles, plantname, modelname, hub_height)
    # build_final_csv_gen(df_sim_wake, df_percentiles, plantname, modelname, hub_height)
    # summary_table(df_percentiles, plantname, modelname,wind_turbine_name, hub_height)

wind_turbine_df = pd.DataFrame()
wind_turbine = [f"WEA{i}" for i in range(len(x_pos))]
wind_turbine_df["Wind Turbine"] = wind_turbine
wind_turbine_df["Longitude [degrees]"] = x_coord
wind_turbine_df["Latitude [degrees]"] = y_coord
wind_turbine_df["Altitude [m]"] = 25
wind_turbine_df["Hub Height [m]"] = hub_height
wind_turbine_df["Timestamp [min]"] = 30
wind_turbine_df["Mean Wind Speed [m/s]"] = mean_wind_speed
wind_turbine_df["Air Density [kg/m**3]"] = 1.225
yield_energy = [np.nansum(df_sim_wake[f"WEA{i}"])/1000 for i in range(len(x_pos))]
wind_turbine_df["Energy Yield [MWh/a]"] = yield_energy
wind_turbine_df["Percentage losses [%]"] = df_wake.iloc[-1].values
wea_3500_counts = [int((df_sim_wake[f'WEA{i}'] == 3500.0).sum()/2) for i in range(len(x_pos))]
wind_turbine_df["Full Load [h/a]"] = wea_3500_counts

print(wea_3500_counts)
# Create DataFrame
wind_turbine_df.to_csv(os.path.join(TABLES_DIR, f"wind_turbines_data.csv"), index=False)

wind_turbine_df = pd.DataFrame()
wind_turbine = [f"WEA{i}" for i in range(len(x_pos))]
wind_turbine_df[""] = wind_turbine
wind_turbine_df["[degrees from north]"] = x_coord
wind_turbine_df["[degrees from east]"] = y_coord
wind_turbine_df["[m from sea level]"] = 25
wind_turbine_df["[m from ground]"] = hub_height
wind_turbine_df["[min]"] = 30
wind_turbine_df["[m/s]"] = mean_wind_speed
wind_turbine_df["[kg/m\textsuperscript{3}]"] = 1.225
yield_energy = [np.nansum(df_sim_wake[f"WEA{i}"])/1000 for i in range(len(x_pos))]
wea_3500_counts = [int((df_sim_wake[f'WEA{i}'] == 3500.0).sum()/2) for i in range(len(x_pos))]
wind_turbine_df["[h/a]"] = wea_3500_counts

print(wea_3500_counts)
# Create DataFrame
wind_turbine_df.to_csv(os.path.join(TABLES_DIR, f"wind_turbines_data2.csv"), index=False)



# Create a list of dictionaries
data_list = [
    {'Description': 'Wind turbine type', 'Value': f'{wind_turbine_name}'},
    {'Description': 'Park energy yield', 'Value': f'{np.round(np.nansum(df_wake.iloc[0])/1000,2)} MWh/a'},
    {'Description': 'Yield loss due to noise', 'Value': '0.000 MWh/a'},
    {'Description': 'Yield loss due to shadow casting', 'Value': f'{np.round(np.nansum(df_wake.iloc[0])/1000 - np.nansum(df_wake.iloc[1])/1000,2)} MWh/a'},
    {'Description': 'Yield loss due to radar', 'Value': '0.000 MWh/a'},
    {'Description': 'Yield loss due to sector management', 'Value': '0.000 MWh/a'},
    {'Description': 'Yield loss due to bat protection', 'Value': '0.000 MWh/a'},
    {'Description': 'Loss of yield due to bird protection', 'Value': '0.000 MWh/a'},
    {'Description': 'Net energy yield', 'Value': f'{np.round(np.nansum(df_wake.iloc[0])/1000,2)} MWh/a'},
    {'Description': 'Other yield losses', 'Value': '0.000 MWh/a'},
    {'Description': 'Real net energy yield', 'Value': f'{np.round(np.nansum(df_wake.iloc[1])/1000,2)} MWh/a'},
    {'Description': 'Total yield losses (excluding shading losses)', 'Value': f'{np.round(np.nanmean(df_wake.iloc[2]),2)} %'},
    {'Description': 'Relative reduction (excluding shading losses)', 'Value': '0 %'}
]

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(TABLES_DIR, f"wind_turbines_data_summary.csv"), index=False)

# Display the DataFrame
print(df)


wind_turbine_df = pd.DataFrame()
wind_turbine = [f"WEA{i}" for i in range(len(x_pos))]
wind_turbine_df[""] = wind_turbine
yield_energy = [df_wake[f"WEA{i}"].iloc[0]/1000 for i in range(len(x_pos))]
wind_turbine_df["[MWh/a]"] = yield_energy
wind_turbine_df["1[MWh/a]"] = 0
shadow_energy = [df_wake[f"WEA{i}"].iloc[0]/1000 - df_wake[f"WEA{i}"].iloc[1]/1000 for i in range(len(x_pos))]
wind_turbine_df["2[MWh/a]"] = shadow_energy
wind_turbine_df["3[MWh/a]"] = 0
wind_turbine_df["4[MWh/a]"] = 0
wind_turbine_df["5[MWh/a]"] = 0
percentage_energy = [df_wake[f"WEA{i}"].iloc[2] for i in range(len(x_pos))]
wind_turbine_df["[%]"] = percentage_energy
net_energy = [df_wake[f"WEA{i}"].iloc[1] for i in range(len(x_pos))]
wind_turbine_df["6[MWh/a]"] = net_energy
# wind_turbine_df["Other losses [MWh/a]"] = 0
# wind_turbine_df["Real Net Energy Yield [MWh/a]"] = net_energy


# Create DataFrame
wind_turbine_df.to_csv(os.path.join(TABLES_DIR, f"wind_turbines_data_info2.csv"), index=False)












print("simulation done")

# # Define your template and output filenames
# TEMPLATE_FILE = "main.tex"  # Your LaTeX template file
# OUTPUT_FILE = "output.tex"  # The final LaTeX file with values replaced

# # Dictionary of placeholders and values
# replacements = {
#     "plantname": plantname,
#     "locationname": locationname,
#     "hub_height": f"{turbine_data['hub_height']}",
#     "modelname": modelname,
#     "wind_turbine_name": wind_turbine_name,
#     "P50_value": f"{p50/1000000:.2f}",
#     "P10_value": f"{p10/1000000:.2f}",
#     "P90_value": f"{p90/1000000:.2f}",
#     "P75_value": f"{p75/1000000:.2f}",
#     "mean_wind_speed": f"{mean_wind_speed}",
#     "blade_diameter": f"{turbine_data['blade_diameter']}",
#     "wind_turbine_capacity": f"{capacity}",
#     "LAT":str(round(np.nanmean(y_coord), 2)),
#     "LON": str(round(np.nanmean(x_coord), 2))
    
# }


# # template_path = os.path.join(PROJECT_ROOT, 'templates', 'main.tex')
# # with open(template_path, 'r') as f:
# #     latex_content = f.read()

# # # # Read the template
# # # with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
# # #     latex_content = f.read()

# # # Replace placeholders
# # for key, value in replacements.items():
# #     latex_content = latex_content.replace(key, value)

# # # # Save the modified LaTeX file
# # # with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
# # #     f.write(latex_content)

# # # print(f"LaTeX file generated: {OUTPUT_FILE}")


# # # Define output path
# # output_tex_path = os.path.join(REPORTS_DIR, "output.tex")

# # # Save the modified LaTeX file
# # with open(output_tex_path, "w", encoding="utf-8") as f:
# #     f.write(latex_content)

# # print(f"LaTeX file generated at: {output_tex_path}")

# num_turbines = len(x_pos)


# def generate_longtable_start_aep(num_turbines):
#     base_columns = ['l']  # AEP
#     turbine_columns = [f'c' for _ in range(num_turbines)]
#     column_format = '|'.join([''] + base_columns + turbine_columns + [''])  # add leading/trailing pipes
#     return f"\\begin{{longtable}}[H]{{{column_format}}}"


# def generate_longtable_start(num_turbines):
#     base_columns = ['p{1.85cm}', 'p{2.5cm}']  # 'Month' and plant total
#     turbine_columns = [f'p{{2.5cm}}' for _ in range(num_turbines)]
#     column_format = '|'.join([''] + base_columns + turbine_columns + [''])  # add leading/trailing pipes
#     return f"\\begin{{longtable}}[H]{{{column_format}}}"


# def generate_table_header(num_turbines):
#     base_columns = [
#         r'\color{white}\textbf{Month}',
#         r'\color{white}\textbf{Sum of estimated power plant [kWh]}'
#     ]
#     turbine_columns = [
#         rf'\color{{white}}\textbf{{Sum of estimated power WEA{i} [kWh]}}'
#         for i in range(num_turbines)
#     ]
    
#     all_columns = base_columns + turbine_columns
#     header_line = r'\rowcolor{tableblue} ' + ' & '.join(all_columns) + r'\\'
#     return header_line

# def generate_table_header_aep(num_turbines):
#     base_columns = [
#         r'\color{white}\textbf{AEP}'
#     ]
#     turbine_columns = [
#         rf'\color{{white}}\textbf{{WEA{i}}}'
#         for i in range(num_turbines)
#     ]
    
#     all_columns = base_columns + turbine_columns
#     header_line = r'\rowcolor{tableblue} ' + ' & '.join(all_columns) + r'\\'
#     return header_line

# def generate_csvcol_row(turbine_count):
#     # Map index to Roman numerals
#     def roman(n):
#         romans = {
#             1: 'i', 2: 'ii', 3: 'iii', 4: 'iv', 5: 'v',
#             6: 'vi', 7: 'vii', 8: 'viii', 9: 'ix', 10: 'x',
#             11: 'xi', 12: 'xii', 13: 'xiii', 14: 'xiv', 15: 'xv'
#         }
#         return romans.get(n, f'col{n}')  # fallback

#     num_columns = 2 + turbine_count
#     cols = [rf'\csvcol{roman(i)}' for i in range(1, num_columns + 1)]
#     return ' & '.join(cols)

# def generate_csvcol_row_aep(turbine_count):
#     # Map index to Roman numerals
#     def roman(n):
#         romans = {
#             1: 'i', 2: 'ii', 3: 'iii', 4: 'iv', 5: 'v',
#             6: 'vi', 7: 'vii', 8: 'viii', 9: 'ix', 10: 'x',
#             11: 'xi', 12: 'xii', 13: 'xiii', 14: 'xiv', 15: 'xv'
#         }
#         return romans.get(n, f'col{n}')  # fallback

#     num_columns = 1 + turbine_count
#     cols = [rf'\csvcol{roman(i)}' for i in range(1, num_columns + 1)]
#     return ' & '.join(cols)

# # Read the template
# template_path = os.path.join(PROJECT_ROOT, 'templates', 'main.tex')
# with open(template_path, 'r', encoding='utf-8') as f:
#     latex_content = f.read()


# longtable_line = generate_longtable_start_aep(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_LONGTABLE_START>>", longtable_line)

# longtable_line2 = generate_longtable_start(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_LONGTABLE_START2>>", longtable_line2)

# header_row2 = generate_table_header(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_HEADER_ROW2>>", header_row2)

# header_row = generate_table_header_aep(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_HEADER_ROW>>", header_row)

# csvcol_row2 = generate_csvcol_row(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_CSVCOL_ROW2>>", csvcol_row2)

# csvcol_row = generate_csvcol_row_aep(num_turbines)
# latex_content = latex_content.replace("<<DYNAMIC_CSVCOL_ROW>>", csvcol_row)

# # Replace placeholders
# for key, value in replacements.items():
#     print(key)
#     print(value)
#     latex_content = latex_content.replace(key, value)
    


# output_tex_path = os.path.join(REPORTS_DIR, "output.tex")

# # Save the modified LaTeX file
# with open(output_tex_path, "w", encoding="utf-8") as f:
#     f.write(latex_content)

# print(f"LaTeX file generated: {OUTPUT_FILE}")



