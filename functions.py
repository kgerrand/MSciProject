'''
Functions for setting up variables and evaluating models, including making predictions, calculating statistics, and plotting results.

'''
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from joblib import load
import calendar

import config

data_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'


## BASELINE SETUP FUNCTIONS
#=======================================================================
def read_manning(site):
    """
    Extracting Alistair's baseline flags for a given site

    Args:
    - site (str): Site code (e.g., MHD)

    Returns:
    - df (pandas.DataFrame): DataFrame with baseline flags as a binary variable
    """
    
    site_translator = {"MHD":"MH", "CGO":"CG", "GSN":"GS", "JFJ":"J1", "CMN":"M5", "THD":"TH", "ZEP":"ZE", "RPB":"BA", "SMO":"SM"}

    # Filtering so only including data relevant to the given site
    files = (data_path / "manning_baselines").glob(f"{site_translator[site]}*.txt")

    dfs = []

    # Looping through each of the files for the given site
    for file in files:

        # Read the data, skipping metadata, putting into pandas dataframe
        data = pd.read_csv(file, skiprows=6, delim_whitespace=True)

        # Setting the index of the dataframe to be the extracted datetime and naming it time
        data.index = pd.to_datetime(data['YY'].astype(str) + "-" + \
                                    data['MM'].astype(str) + "-" + \
                                    data['DD'].astype(str) + " " + \
                                    data['HH'].astype(str) + ":00:00")

        data.index.name = "time"
        
        # Adding the 'Ct' column to the previously created empty list
        dfs.append(data[["Ct"]])
    
    # Creating a dataframe from the list containing all the 'Ct' values
    df = pd.concat(dfs)

    df.sort_index(inplace=True)

    # Replace all values in Ct column less than 10 or greater than 20 with 0
    # not baseline values
    df.loc[(df['Ct'] < 10) | (df['Ct'] >= 20), 'Ct'] = 0

    # Replace all values between 10 and 19 with 1
    # baseline values
    df.loc[(df['Ct'] >= 10) & (df['Ct'] < 20), 'Ct'] = 1

    # Rename Ct column to "baseline"
    df.rename(columns={'Ct': 'baseline'}, inplace=True)

    return df

#=======================================================================
def balance_baselines(ds, minority_ratio): 
    """
    Balances the dataset by randomly undersampling non-baseline data points.

    Args:
    - ds (xarray.Dataset): The dataset to be balanced.
    - minority_ratio (float): The desired ratio of baseline (minority class) data points in the final dataset. 
                            For example, 0.4 means 40% of data points will be baseline.

    Returns:
    - xarray.Dataset: The balanced dataset where the ratio of baseline to non-baseline data points is as specified by the `minority_ratio` argument.

    Raises:
    - ValueError: If the counts of baseline and non-baseline values are not in the expected ratio (within a tolerance of 1%).

    """
    np.random.seed(42)

    # counting number of baseline&non-baseline data points
    baseline_count = ds['baseline'].where(ds['baseline']==1).count()
    non_baseline_count = ds['baseline'].where(ds['baseline']==0).count()
    # print(f"ORIGINAL baseline count: {baseline_count}, non-baseline count: {non_baseline_count}")

    # calculating the minority class count (expected to be baseline)
    minority_count = int(min(baseline_count, non_baseline_count))

    # calculating the majority class count based on majority_ratio and minority_count
    majority_ratio = 1 - minority_ratio
    majority_count = int(minority_count * (majority_ratio/minority_ratio))

    # subsetting the non-baseline data points
    undersampled_non_baseline = ds.where(ds['baseline'] == 0, drop=True)

    # creating an array of time indices & randomly selecting some
    time_indices = undersampled_non_baseline['time'].values
    selected_indices = np.random.choice(time_indices, majority_count, replace=False)
    selected_indices = np.sort(selected_indices)

    # setting the non-baseline data points to only include the randomly selected indices
    undersampled_non_baseline = undersampled_non_baseline.sel(time=selected_indices)

    # combining the the undersampled non-baseline with the baseline values
    balanced_ds = xr.merge([ds.sel(time=(ds['baseline'] == 1)), undersampled_non_baseline])
    balanced_ds = balanced_ds.sortby('time')

    # checking balance
    new_baseline_count = balanced_ds['baseline'].where(balanced_ds['baseline']==1).count()
    new_non_baseline_count = balanced_ds['baseline'].where(balanced_ds['baseline']==0).count()
    # print(f"NEW baseline count: {new_baseline_count}, non-baseline count: {new_non_baseline_count}")

    # verifying that the ratio of baseline:non-baseline data points is as expected (within a tolerance of 1%)
    tolerance = 0.01
    upper_bound = (1+tolerance)*(majority_ratio/minority_ratio)
    lower_bound = (1-tolerance)*(majority_ratio/minority_ratio)

    if(lower_bound <= (new_non_baseline_count/new_baseline_count) <= upper_bound):
        return balanced_ds
    else:
        raise ValueError("The counts of baseline and non-baseline values are not in the expected ratio.")

#=======================================================================
def add_shifted_time(df, points):
    """
    Adds columns with wind data shifted by 6 hours (up three index rows) to the input dataframe.

    Args:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with shifted time columns.
    """

    # copying dataframe
    df_ = df.copy()   


    # extracting wind colunmns
    u10_columns = [f"u10_{point}" for point in points]
    v10_columns = [f"v10_{point}" for point in points]
    u850_columns = [f"u850_{point}" for point in points]
    v850_columns = [f"v850_{point}" for point in points]
    u500_columns = [f"u500_{point}" for point in points]
    v500_columns = [f"v500_{point}" for point in points]
    wind_columns = u10_columns + v10_columns + u850_columns + v850_columns + u500_columns + v500_columns

    # checking if adding a shifted time column has already been done - in which case, remove it before adding it again
    if f'u10_0_past' in df_.columns:
        df_ = df_.drop(columns=[col + f'_past' for col in wind_columns])
        print("Shifted time columns already exist and have been removed. Note that redoing this function will remove additional columns.")

    # create shifted columns
    shifted_columns = [col + '_past' for col in wind_columns]

    # Create a dictionary for the shifted columns
    shifted_dict = {}

    for col, shifted_col in zip(wind_columns, shifted_columns):
        # Shift the column values up by two rows
        shifted_dict[shifted_col] = df_[col].shift(3)

    # Convert the dictionary to a DataFrame
    df_shifted = pd.DataFrame(shifted_dict)

    # Concatenate the original DataFrame with the new DataFrame
    df_ = pd.concat([df_, df_shifted], axis=1)

    # dropping the first three rows as NaN values
    df_ = df_.iloc[3:]

    return df_

#=======================================================================



## MODEL EVALUATION FUNCTIONS
#=======================================================================
def access_info(model_name=None):
    """
    Accesses information about the site, site name, and compound

    Args:
    - model_name (str): The name of the model to load

    Returns:
    - site (str): The site as defined in config.py
    - site_name (str): The name of the site corresponding to the site
    - compound (str): The compound as defined in config.py
    - model: The loaded model
    """
    site = config.site
    site_name = config.site_dict[site]
    compound = config.compound

    if model_name is None:
        return site, site_name, compound
    
    else:
        model = load(data_path/'saved_files'/model_name)
        return site, site_name, compound, model
        
#=======================================================================
def quantify_noise(results):
    """
    Quantifies the noise in the true baselines, by calculating the coefficient of variation of the true baseline values based on aggregate data.
    This is a relative measure of dispersion, calculated as the standard deviation divided by the mean, allowing for comparison between different datasets. 
    A higher coefficient of variation indicates a higher level of dispersion.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.

    Returns:
    - None
    """

    _, _, compound = access_info()

    # extracting true baseline values
    df_actual = results.where(results["flag"] == 1).dropna()
    df_actual.index = pd.to_datetime(df_actual.index)

    # resampling to monthly averages
    df_actual_monthly = df_actual.resample('M').mean()
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    # calculating monthly standard deviation
    df_actual_std = df_actual.resample('M').std()
    df_actual_std.index = df_actual_std.index.to_period('M')

    overall_cv = []
    
    # calculating coefficient of variation
    for idx, row in df_actual_std.iterrows():
        cv = row['mf'] / df_actual_monthly.loc[idx, 'mf']
        overall_cv.append(cv)

    # removing nans
    overall_cv = [x for x in overall_cv if str(x) != 'nan']

    print(f'Overall Coefficient of Variation of True Baseline Values for {compound}: \033[1m{np.mean(overall_cv):.3f}\033[0m')

#=======================================================================
def make_predictions(model):
    """
    Make predictions based on the previously trained model, taking into account model type.

    Returns:
    - results (pandas.DataFrame): DataFrame containing the predicted flags, actual flags, and mf values.
    """

    site, _, compound = access_info()

    # load in data from baseline_setup.ipynb
    data_balanced_df = pd.read_csv(data_path/'saved_files'/f'for_model_{compound}_{site}.csv', index_col='time')
    data_pca = pd.read_csv(data_path/'saved_files'/f'for_model_pca_{compound}_{site}.csv', index_col='time')
    data_balanced_ds = xr.open_dataset(data_path/'saved_files'/f'data_balanced_ds_{compound}_{site}.nc')

    # removing top three values from index of data_balanced_ds to match the length of the predicted flags
    # this is due to the data balancing process
    data_balanced_ds = data_balanced_ds.isel(time=slice(3, None))

    # making predictions based on model
    # remove predicted_flag if it already exists
    if "predicted_flag" in data_balanced_df.columns:
        data_balanced_df.drop(columns=["predicted_flag"], inplace=True)


    # making predictions based on model type
    model_type = model.__class__.__name__

    # if model is NEURAL NETWORK () - predict normally using meteorological dataset
    if model_type == 'MLPClassifier':
        df_predict = data_balanced_df.copy()
        df_predict.drop(columns=["flag"], inplace=True)
        
        print("Predictons made using neural network model.")
        class_probabilities_predict = model.predict_proba(df_predict.reset_index(drop=True))
        threshold = config.confidence_threshold
        y_pred = (class_probabilities_predict[:,1] >= threshold).astype(int)
        data_balanced_df["predicted_flag"] = y_pred

    # if model is RANDOM FOREST - predict based on class probabilities using PCA dataset
    if model_type == 'RandomForestClassifier':
        df_predict = data_pca.copy()
        df_predict.drop(columns=["flag"], inplace=True)

        print("Predictions made using random forest model.")
        class_probabilities_predict = model.predict_proba(df_predict.reset_index(drop=True))

        threshold = config.confidence_threshold
        y_pred = (class_probabilities_predict[:,1] >= threshold).astype(int)

        data_balanced_df["predicted_flag"] = y_pred


    # add mf values to results
    columns_to_keep = ["flag", "predicted_flag"]
    results = data_balanced_df[columns_to_keep].copy()
    results["mf"] = data_balanced_ds.mf.values

    return results

#=======================================================================
def calc_statistics(results):
    """
    Calculates statistics to compare model to true flags.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.

    Returns:
    - None
    """

    bold = '\033[1m'
    end = '\033[0m'
    
    # finds the number of predicted baseline and non-baseline values
    num_baseline = results["predicted_flag"].value_counts()[1]
    num_not_baseline = results["predicted_flag"].value_counts()[0]

    print(f'Number of predicted baseline values: {bold}{num_baseline}{end}, Number of predicted non-baseline values: {bold}{num_not_baseline}{end}')
    

    # finds mean and standard deviation of mf values for predicted and true baseline values
    actual_values = results["mf"].where((results["flag"] == 1)).dropna()
    predicted_values = results["mf"].where(results["predicted_flag"] == 1).dropna()

    actual_mean = actual_values.mean()
    predicted_mean = predicted_values.mean()

    actual_std = actual_values.std()
    predicted_std = predicted_values.std()

    print(f'True Mean: {bold}{actual_mean:.3f}{end}, Model Mean: {bold}{predicted_mean:.3f}{end}')
    print(f'True Std Dev: {bold}{actual_std:.3f}{end}, Model Std Dev: {bold}{predicted_std:.3f}{end}')


    # finds MAE, RMSE and MAPE of model monthly means
    df_pred = results.where(results["predicted_flag"] == 1).dropna()
    df_actual = results.where(results["flag"] == 1).dropna()

    df_pred.index = pd.to_datetime(df_pred.index)
    df_actual.index = pd.to_datetime(df_actual.index)
    df_pred_monthly = df_pred.resample('M').mean()
    df_actual_monthly = df_actual.resample('M').mean()
    df_pred_monthly.index = df_pred_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    mae = np.mean(np.abs(df_pred_monthly["mf"] - df_actual_monthly["mf"]))
    rmse = np.sqrt(np.mean((df_pred_monthly["mf"] - df_actual_monthly["mf"])**2))
    mape = np.mean(np.abs((df_actual_monthly["mf"] - df_pred_monthly["mf"]) / df_actual_monthly["mf"])) * 100

    print(f'Mean Absolute Error: {bold}{mae:.3f}{end}')
    print(f'Root Mean Squared Error: {bold}{rmse:.3f}{end}')
    print(f'Mean Absolute Percentage Error: {bold}{mape:.3f}{end}')

#=======================================================================
def plot_predictions(results):
    """
    Plots mole fraction against time, with the predicted baselines and true baselines highlighted.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.
    
    Returns:
    - None
    """

    _, site_name, compound = access_info()

    fig, axes = plt.subplots(2,1, figsize=(15,10))
    sns.set_theme(style='ticks', font='Arial')

    # plot 1 - true baselines
    results["mf"].plot(ax=axes[0], label="All Data", color='grey', linewidth=1, alpha=0.8)
    results["mf"].where(results["flag"] == 1).plot(ax=axes[0], label="True Baselines", color='darkgreen', linewidth=1.5)
    axes[0].legend(loc="lower right", fontsize=12)

    # plot 2 - predicted baselines
    results["mf"].plot(ax=axes[1], label="All Data", color='grey', linewidth=1, alpha=0.8)
    results["mf"].where(results["predicted_flag"] == 1).plot(ax=axes[1], label="Predicted Baselines", color='darkblue', linewidth=1.5)
    axes[1].legend(loc="lower right", fontsize=12)

    # setting overall title and labels
    fig.suptitle(f"{compound} at {site_name}", fontsize=20, y=0.92)

    for ax in axes:
        ax.set_xlabel("Time", fontsize=12, fontstyle='italic')
        ax.set_ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
    
#=======================================================================
def plot_predictions_monthly(results, model_name, start_year=None, end_year=None):
    """
    Plots the predicted baselines and their standard deviations against the true baselines and their standard deviations, highlighting any points outside three standard deviations.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.

    Returns:
    - None
    """    
    results = results.dropna()
    site, site_name, compound = access_info()

    # extracting flags and predicted flags
    df_pred = results.where(results["predicted_flag"] == 1).dropna()
    df_actual = results.where(results["flag"] == 1).dropna()

    df_pred.index = pd.to_datetime(df_pred.index)
    df_actual.index = pd.to_datetime(df_actual.index)

    # filtering to only show the years specified
    if start_year and end_year:
        df_pred = df_pred.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
        df_actual = df_actual.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    # resampling to monthly averages
    df_pred_monthly = df_pred.resample('M').mean()
    df_actual_monthly = df_actual.resample('M').mean()
    # setting index to year and month only
    df_pred_monthly.index = df_pred_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    # calculating standard deviation
    std_pred_monthly = df_pred.groupby(df_pred.index.to_period('M'))["mf"].std().reset_index()
    std_pred_monthly.set_index('time', inplace=True)
    std_actual_monthly = df_actual.groupby(df_actual.index.to_period('M'))["mf"].std().reset_index()
    std_actual_monthly.set_index('time', inplace=True)


    # plotting
    fig, ax = plt.subplots(figsize=(12,5))
    sns.set_theme(style='ticks', font='Arial')
    ax.minorticks_on()

    df_actual_monthly["mf"].plot(ax=ax, label="True Baselines", color='darkgreen', alpha=0.75, linewidth=1.5)
    if site == 'GSN':
        df_pred_monthly["mf"].plot(ax=ax, label="Predicted Baselines", color='blue', linestyle='--', marker='s', markersize=3, linewidth=1.5)
    else:
        df_pred_monthly["mf"].plot(ax=ax, label="Predicted Baselines", color='blue', linestyle='--', linewidth=1.5)

    # adding standard deviation shading
    upper_actual = df_actual_monthly["mf"] + std_actual_monthly['mf']
    lower_actual = df_actual_monthly["mf"] - std_actual_monthly['mf']

    ax.fill_between(df_actual_monthly.index, lower_actual, upper_actual, color='green', alpha=0.2, label="True Baseline Standard Deviation")

    # adding shading for training/validation/finetuning sets if visualised in chosen time period

    model_type = model_name[:2]

    if start_year and end_year:
        pass
    else:
        # Mace Head model - trained on 2013-2018, validated on 2019 (NN) or 2016-2018, validated on 2019 (RF)
        if site == 'MHD' and model_type == 'nn':
            ax.axvspan(datetime(2013,1,1), datetime(2018,12,31), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2019,1,1), datetime(2019,12,31), alpha=0.2, label="Validation Set", color='purple')

        elif site == 'MHD' and model_type == 'rf':
            ax.axvspan(datetime(2016,1,1), datetime(2018,12,31), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2019,1,1), datetime(2019,12,31), alpha=0.2, label="Validation Set", color='purple')

        # Gosan model
        elif site == 'GSN' and "finetuned" not in model_name and model_type == 'nn':
            ax.axvspan(datetime(2009,1,1), datetime(2013,12,31), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2014,1,1), datetime(2014,12,31), alpha=0.2, label="Validation Set", color='purple')

        elif site == 'GSN' and "finetuned" not in model_name and model_type == 'rf':
            ax.axvspan(datetime(2011,1,1), datetime(2013,12,31), alpha=0.3, label="Training Set", color='grey')
            ax.axvspan(datetime(2014,1,1), datetime(2014,12,31), alpha=0.2, label="Validation Set", color='purple')

        # Gosan finetuned model - finetuned on 2011-2014 for NN and 2014 for RF
        elif site == 'GSN' and "finetuned" in model_name and model_type == 'nn':
            ax.axvspan(datetime(2011,1,1), datetime(2014,12,31), alpha=0.2, label="Fine-tuning Set", color='orange')

        elif site == 'GSN' and "finetuned" in model_name and model_type == 'rf':
            ax.axvspan(datetime(2014,1,1), datetime(2014,12,31), alpha=0.2, label="Fine-tuning Set", color='orange')


    # adding tolerance range based on 3 standard deviations
    upper_range = df_actual_monthly["mf"] + 3*(std_actual_monthly['mf'])
    lower_range = df_actual_monthly["mf"] - 3*(std_actual_monthly['mf'])

    # calculating overall standard deviation for arrows
    overall_std = df_actual_monthly["mf"].std()


    # adding labels to points outside tolerance range
    # looping through in this way as indexes don't always match up (i.e. in the case that no predictions are made in a month)
    anomalous_months = []
    
    for idx, row in df_pred_monthly.iterrows():
        if idx in upper_range.index and row["mf"] >= upper_range.loc[idx]:
            arrow_end = row["mf"] + (overall_std * 0.5)
            ax.annotate(idx.strftime('%B %Y'),
                        xy=(idx, row["mf"]),
                        xytext=(idx, arrow_end),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = idx.strftime('%Y-%m')
            anomalous_months.append(date)
        
        elif idx in upper_range.index and row["mf"] <= lower_range.loc[idx]:
            arrow_end = row["mf"] - (overall_std * 0.5)
            ax.annotate(idx.strftime('%B %Y'),
                        xy=(idx, row["mf"]),
                        xytext=(idx, arrow_end),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = idx.strftime('%Y-%m')
            anomalous_months.append(date)

    plt.ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
    plt.xlabel("")
    # plt.title(f"Comparing True and Predicted Baseline Monthly Means for {compound} at {site_name}", fontsize=15)
    plt.legend(loc="best", fontsize=12)
    plt.show()

    print(f"Number of anomalies: {len(anomalous_months)}")
    print(f"Anomalous months: {anomalous_months}")

    # calculates percentage of non-anomalous months
    total_months = len(df_pred_monthly)
    non_anomalous_months = total_months - len(anomalous_months)
    percentage = non_anomalous_months / total_months * 100
    print(f"Percentage of non-anomalous months: {percentage:.1f}%")

#=======================================================================
def analyse_anomalies(results, anomalies_list):
    """
    Plots a given set of anomalous months, comparing the predicted baselines to the true baselines.
    Works when considering two or more anomalous months.

    Args:
    - results (pandas.DataFrame): Dataframe containing the predicted flags, true flags, and mf values.
    - anomalies_list (list): A list of strings representing the anomalous months in the format 'YYYY-MM'.

    Returns:
    - None
    """   

    _, site_name, compound = access_info()

    def get_days(month):
        """
        Returns the start and end dates of a given month. Only used for plotting anomalies.

        Args:
        - month (str): A string representing the month in the format 'YYYY-MM'.

        Returns:
        - tuple: A tuple containing the start date and end date of the month in the format 'YYYY-MM-DD'.
        """
        start_date = datetime.strptime(month, "%Y-%m")
        end_date = (start_date + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    

    # getting start and end dates for each anomalous month
    anomalies_range_list = [day for month in anomalies_list for day in get_days(month)]

    # plotting
    figsize_list = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    if len(anomalies_list) == 1:
        fig, axs = plt.subplots(1, 2, figsize=(15,6))
        sns.set_theme(style='ticks', font='Arial')

        start, end = anomalies_range_list[0], anomalies_range_list[1]
        month = results.loc[start:end]
        month.index = pd.to_datetime(month.index)
        # calculating some stats
        pred_mean = month["mf"].where(month["predicted_flag"] == 1).mean()
        actual_mean = month["mf"].where(month["flag"] == 1).mean()
        pred_count = month["mf"].where(month["predicted_flag"] == 1).count()
        actual_count = month["mf"].where(month["flag"] == 1).count()

        # plotting baseline points
        axs[0].scatter(month.index, month["mf"].where(month["flag"] == 1),
                    color='darkgreen', label=f"Baselines (count={actual_count})", marker='x', s=75)
        axs[0].scatter(month.index, month["mf"].where(month["predicted_flag"] == 1),
                    color='blue', label=f"Predicted Baselines (count={pred_count})", marker='x', s=75)
        # plotting mean lines
        axs[0].axhline(y=actual_mean, color='darkgreen', linestyle='-', label='Actual Mean')
        axs[0].axhline(y=pred_mean, color='blue', linestyle='-', label='Predicted Mean')

        # adding tolerance limit line
        actual_std = month["mf"].where(month["flag"] == 1).std()
        upper_limit = actual_mean + (3*actual_std)
        lower_limit = actual_mean - (3*actual_std)
        axs[0].axhline(y=upper_limit, color='darkgreen', linestyle=':')
        axs[0].axhline(y=lower_limit, color='darkgreen', linestyle=':')
        axs[0].fill_between(month.index, lower_limit, upper_limit, color='green', alpha=0.1, label="Tolerance Range")

        # formatting plot
        start_date = datetime.strptime(start, '%Y-%m-%d')
        formatted_date = start_date.strftime('%b %Y')
        # axs[0].set_title(f"{formatted_date}", fontsize=20)
        axs[0].set_ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
        axs[0].legend(fontsize=12, loc='upper left')

        # comparing counts
        counts = []
        true_positive_count = month["mf"].where((month["predicted_flag"] == 1) & (month["flag"] == 1)).count()
        false_postive_count = month["mf"].where((month["predicted_flag"] == 1) & (month["flag"] == 0)).count()
        true_negative_count = month["mf"].where((month["predicted_flag"] == 0) & (month["flag"] == 0)).count()
        false_negative_count = month["mf"].where((month["predicted_flag"] == 0) & (month["flag"] == 1)).count()
        counts.append(true_positive_count)
        counts.append(false_postive_count)
        counts.append(true_negative_count)
        counts.append(false_negative_count)

        axs[1].bar(["True Positive", "False Positive", "True Negative", "False Negative"], counts, color='grey')

        for i in range(len(counts)):
            percentage = counts[i] / month["mf"].count() * 100
            axs[1].text(i, counts[i], f"{counts[i]} ({percentage:.1f}%)", fontsize=10, ha='center', va='bottom')


    else:
        fig, axs = plt.subplots(len(anomalies_list), 2, 
                                figsize=(15,figsize_list[len(anomalies_list)-1]))
        
        sns.set(style='whitegrid')
        
        for n in range(0, len(anomalies_list)):
            start, end = anomalies_range_list[2*n], anomalies_range_list[2*n+1]
            month = results.loc[start:end]
            month.index = pd.to_datetime(month.index)
            # calculating some stats
            pred_mean = month["mf"].where(month["predicted_flag"] == 1).mean()
            actual_mean = month["mf"].where(month["flag"] == 1).mean()
            pred_count = month["mf"].where(month["predicted_flag"] == 1).count()
            actual_count = month["mf"].where(month["flag"] == 1).count()

            # plotting baseline points
            axs[n,0].scatter(month.index, month["mf"].where(month["flag"] == 1), 
                        color='darkgreen', label=f"True Baselines (count={actual_count})", marker='x', s=75)
            axs[n,0].scatter(month.index, month["mf"].where(month["predicted_flag"] == 1), 
                        color='blue', label=f"Predicted Baselines (count={pred_count})", marker='x', s=75)
            # plotting mean lines
            axs[n,0].axhline(y=actual_mean, color='darkgreen', linestyle='-', label='True Mean')
            axs[n,0].axhline(y=pred_mean, color='red', linestyle='-', label='Predicted Mean') 

            # adding tolerance limit line
            actual_std = month["mf"].where(month["flag"] == 1).std()
            upper_limit = actual_mean + (3*actual_std)
            lower_limit = actual_mean - (3*actual_std)
            axs[n,0].axhline(y=upper_limit, color='darkgreen', linestyle=':')
            axs[n,0].axhline(y=lower_limit, color='darkgreen', linestyle=':')
            axs[n,0].fill_between(month.index, lower_limit, upper_limit, color='green', alpha=0.1, label="Tolerance Range")

            # formatting plot
            start_date = datetime.strptime(start, '%Y-%m-%d')
            formatted_date = start_date.strftime('%b %Y')
            axs[n,0].set_title(f"{formatted_date}", fontsize=20)
            axs[n,0].set_ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
            axs[n,0].legend(fontsize=14)
            
            # comparing counts
            counts = []
            true_positive_count = month["mf"].where((month["predicted_flag"] == 1) & (month["flag"] == 1)).count()
            false_postive_count = month["mf"].where((month["predicted_flag"] == 1) & (month["flag"] == 0)).count()
            true_negative_count = month["mf"].where((month["predicted_flag"] == 0) & (month["flag"] == 0)).count()
            false_negative_count = month["mf"].where((month["predicted_flag"] == 0) & (month["flag"] == 1)).count()
            counts.append(true_positive_count)
            counts.append(false_postive_count)
            counts.append(true_negative_count)
            counts.append(false_negative_count)

            axs[n,1].bar(["True Positive", "False Positive", "True Negative", "False Negative"], counts, color='grey')

            for i in range(len(counts)):
                percentage = counts[i] / month["mf"].count() * 100
                axs[n,1].text(i, counts[i], f"{counts[i]} ({percentage:.1f}%)", fontsize=10, ha='center', va='bottom')


    # fig.suptitle(f"Anomalous Months for {compound} at {site_name}", fontsize=25, y=1.01)
    fig.set_tight_layout(True)
    plt.show()



## BENCHMARKING FUNCTIONS
#=======================================================================
def calc_benchmark(percentile):
    """
    Calculates benchmark flags based on the given percentile.

    Args:
    - percentile (float): The percentile value used to determine the baseline flag.

    Returns:
    - df_benchmark (pandas.DataFrame): A DataFrame containing the benchmark flags.

    Raises:
    - AssertionError: If no data has been marked as baseline.
    """
    site, _, compound = access_info()

    # load in data from manning_baselines.ipynb
    data_ds = xr.open_dataset(data_path/'saved_files'/f'data_ds_{compound}_{site}.nc')

    # Create a dataframe with molefraction and baseline flags
    df_benchmark = pd.DataFrame({"flag": data_ds.baseline.values, "mf": data_ds.mf.values},
                                index=data_ds.time.values)
    
    # adding empty column
    df_benchmark["benchmark_flag"] = np.zeros(len(df_benchmark))

    # assigning a value within empty column for baseline flag (baseline if within lowest n% of mole fraction values, per month [n=percentile])
    # loop through each year
    start_year = data_ds.time.dt.year[0].item()
    end_year = data_ds.time.dt.year[-1].item()

    for year in range(start_year, end_year + 1):
        # loop through each month
        for month in range(1, 13):
            key = f"{year}-{month:02d}"
            # skip if month or year not in data
            if key not in df_benchmark.index:
                continue
            else:
                # extract data for year and month
                df_month = df_benchmark.loc[f"{year}-{month}"].dropna()
                # skip month if contains no data
                if len(df_month) == 0:
                    continue

                # calculate the percentile values (within the lowest n% of mole fraction values)
                percentile_values = np.percentile(df_month["mf"], percentile)
                # add baseline flag if mole fraction value is less than calculated percentile
                df_month["benchmark_flag"] = np.where(df_month["mf"] <= percentile_values, 1, 0)
                # update the dataframe with the new values
                df_benchmark.loc[f"{year}-{month}"] = df_month

    # checking that data has been marked baseline
    assert df_benchmark["benchmark_flag"].where(df_benchmark["benchmark_flag"] == 1).count() > 0

    return df_benchmark

#=======================================================================
def plot_benchmark(df_benchmark, percentile):
    """
    Plots the benchmarked baselines and their standard deviations against the Manning baselines and their standard deviations, highlighting any points outside three standard deviations.
    
    Args:
    - df_benchmark (DataFrame): DataFrame containing the benchmarked and Manning baselines data.
    - percentile (float): The percentile value used to determine the baseline flag.
    
    Returns:
    - None
    """
    _, site_name, compound = access_info()
    
    df_benchmark_baselines = df_benchmark.where(df_benchmark["benchmark_flag"] == 1).dropna()
    df_benchmark_baselines.drop(columns=["flag"], inplace=True)
    df_actual_baselines = df_benchmark.where(df_benchmark["flag"] == 1).dropna()
    df_actual_baselines.drop(columns=["benchmark_flag"], inplace=True)

    # resampling to monthly averages
    df_benchmark_monthly = df_benchmark_baselines.resample('M').mean()
    df_actual_monthly = df_actual_baselines.resample('M').mean()

    # setting index to year and month only
    df_benchmark_monthly.index = df_benchmark_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')

    # calculating standard deviation
    std_benchmark_monthly = df_benchmark_baselines.groupby(df_benchmark_baselines.index.to_period('M'))["mf"].std().reset_index()
    std_benchmark_monthly.set_index('index', inplace=True)
    std_actual_monthly = df_actual_baselines.groupby(df_actual_baselines.index.to_period('M'))["mf"].std().reset_index()
    std_actual_monthly.set_index('index', inplace=True)


    # plotting
    fig, ax = plt.subplots(figsize=(12,5))
    sns.set_theme(style='ticks', font='Arial')

    df_actual_monthly["mf"].plot(ax=ax, label="True Baselines", color='darkgreen', alpha=0.75)
    df_benchmark_monthly["mf"].plot(ax=ax, label=f"Benchmarked Baselines (bottom {percentile} percent)", color='purple', linestyle='-.')

    # adding standard deviation shading
    upper_actual = df_actual_monthly["mf"] + std_actual_monthly['mf']
    lower_actual = df_actual_monthly["mf"] - std_actual_monthly['mf']
    ax.fill_between(df_actual_monthly.index, lower_actual, upper_actual, color='green', alpha=0.2, label="True Baseline Standard Deviation")

    upper_pred = df_benchmark_monthly["mf"] + std_benchmark_monthly['mf']
    lower_pred = df_benchmark_monthly["mf"] - std_benchmark_monthly['mf']
    # ax.fill_between(df_benchmark_monthly.index, lower_pred, upper_pred, color='purple', alpha=0.2, label="Benchmarked Baseline Standard Deviation")


    # adding tolerance range based on 3 standard deviations
    upper_range = df_actual_monthly["mf"] + 3*(std_actual_monthly['mf'])
    lower_range = df_actual_monthly["mf"] - 3*(std_actual_monthly['mf'])

    # calculating overall std for arrows
    overall_std = df_actual_monthly["mf"].std()

    
    anomalous_months = []

    # adding labels to points outside tolerance range
    for i in range(len(df_actual_monthly)):
        # if df_benchmark_monthly["mf"].iloc[i].item() >= upper_range.iloc[i]:
        if df_benchmark_monthly.index[i] in upper_range.index and df_benchmark_monthly["mf"].loc[df_benchmark_monthly.index[i]].item() >= upper_range.loc[df_benchmark_monthly.index[i]]:
            arrow_end = df_benchmark_monthly["mf"].iloc[i] + (overall_std*0.5)
            ax.annotate(df_benchmark_monthly.index[i].strftime('%B %Y'), 
                        xy=(df_benchmark_monthly.index[i], df_benchmark_monthly["mf"].iloc[i]), 
                        xytext=(df_benchmark_monthly.index[i], arrow_end), 
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = df_benchmark_monthly.index[i].strftime('%Y-%m')
            anomalous_months.append(date)

        elif df_benchmark_monthly.index[i] in lower_range.index and df_benchmark_monthly["mf"].loc[df_benchmark_monthly.index[i]].item() <= lower_range.loc[df_benchmark_monthly.index[i]]:
            arrow_end = df_benchmark_monthly["mf"].iloc[i] - (overall_std*0.5)
            ax.annotate(df_benchmark_monthly.index[i].strftime('%B %Y'), 
                        xy=(df_benchmark_monthly.index[i], df_benchmark_monthly["mf"].iloc[i]), 
                        xytext=(df_benchmark_monthly.index[i], arrow_end), 
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center', verticalalignment='bottom')
            date = df_benchmark_monthly.index[i].strftime('%Y-%m')
            anomalous_months.append(date)

    plt.ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
    plt.title(f"{compound} at {site_name}", fontsize=15)
    plt.legend(loc="best", fontsize=12)
    plt.show()

    print(f"Number of anomalies: {len(anomalous_months)}")
    print(f"Anomalous months: {anomalous_months}")

#=======================================================================
def compare_benchmark_to_model(df_benchmark, percentile, model, start_year=None, end_year=None):
    """
    Compares the benchmarked baselines to the actual baselines and those predicted by the model.

    Args:
    - df_benchmark (DataFrame): DataFrame containing the benchmarked baselines.
    - percentile (float): The percentile value used to determine the baseline flag.

    Returns:
    - None
    """
    site, site_name, compound = access_info()
        
    # load in data from manning_baselines.ipynb
    data_balanced_ds = xr.open_dataset(data_path/'saved_files'/f'data_balanced_ds_{compound}_{site}.nc')
    data_pca = pd.read_csv(data_path/'saved_files'/f'for_model_pca_{compound}_{site}.csv', index_col='time')
    data_balanced_df = pd.read_csv(data_path/'saved_files'/f'for_model_{compound}_{site}.csv', index_col='time')

    # removing top three values from index of data_balanced_ds to match the length of the predicted flags
    # this is due to the data balancing process
    data_balanced_ds = data_balanced_ds.isel(time=slice(3, None))

    # making predictions based on model
    if "predicted_flag" in data_balanced_df.columns:
        data_balanced_df.drop(columns=["predicted_flag"], inplace=True)

    model_type = model.__class__.__name__

    # if model is NEURAL NETWORK () - predict normally using meteorological dataset
    if model_type == 'MLPClassifier':
        df_predict = data_balanced_df.copy()
        df_predict.drop(columns=["flag"], inplace=True)
        
        print("Predcitons made using neural network model.")
        y_pred = model.predict(df_predict.reset_index(drop=True))
        data_balanced_df["predicted_flag"] = y_pred

    # if model is RANDOM FOREST - predict based on class probabilities using PCA dataset
    if model_type == 'RandomForestClassifier':
        df_predict = data_pca.copy()
        df_predict.drop(columns=["flag"], inplace=True)

        print("Predictions made using class probabilities from random forest model.")
        class_probabilities_predict = model.predict_proba(df_predict.reset_index(drop=True))

        threshold = config.confidence_threshold
        y_pred = (class_probabilities_predict[:,1] >= threshold).astype(int)

        data_balanced_df["predicted_flag"] = y_pred

    # add mf values to results
    columns_to_keep = ["flag", "predicted_flag"]
    df_plot = data_balanced_df[columns_to_keep].copy()
    df_plot["mf"] = data_balanced_ds.mf.values


    # extracting flags
    df_pred = df_plot.where(df_plot["predicted_flag"] == 1).dropna()
    df_actual = df_plot.where(df_plot["flag"] == 1).dropna()
    df_benchmark = df_benchmark.where(df_benchmark["benchmark_flag"] == 1).dropna()

    df_pred.index = pd.to_datetime(df_pred.index)
    df_actual.index = pd.to_datetime(df_actual.index)
    df_benchmark.index = pd.to_datetime(df_benchmark.index)

    # filtering to only show the years specified
    if start_year and end_year:
        df_pred = df_pred.loc[f"{start_year}":f"{end_year}"]
        df_actual = df_actual.loc[f"{start_year}":f"{end_year}"]
        df_benchmark = df_benchmark.loc[f"{start_year}":f"{end_year}"]
        
    # resampling to monthly averages
    df_pred_monthly = df_pred.resample('M').mean()
    df_actual_monthly = df_actual.resample('M').mean()
    df_benchmark_monthly = df_benchmark.resample('M').mean()
    # setting index to year and month only
    df_pred_monthly.index = df_pred_monthly.index.to_period('M')
    df_actual_monthly.index = df_actual_monthly.index.to_period('M')
    df_benchmark_monthly.index = df_benchmark_monthly.index.to_period('M')

    # calculating standard deviation
    std_pred_monthly = df_pred.groupby(df_pred.index.to_period('M'))["mf"].std().reset_index()
    std_pred_monthly.set_index('time', inplace=True)
    std_actual_monthly = df_actual.groupby(df_actual.index.to_period('M'))["mf"].std().reset_index()
    std_actual_monthly.set_index('time', inplace=True)
    std_benchmark_monthly = df_benchmark.groupby(df_benchmark.index.to_period('M'))["mf"].std().reset_index()
    std_benchmark_monthly.set_index('index', inplace=True)

    # plotting
    fig, ax = plt.subplots(figsize=(12,5))
    sns.set_theme(style='ticks', font='Arial')

    df_actual_monthly["mf"].plot(ax=ax, label="True Baselines", color='darkgreen', alpha=0.75)
    df_pred_monthly["mf"].plot(ax=ax, label="Predicted Baselines", color='blue', linestyle='--')
    df_benchmark_monthly["mf"].plot(ax=ax, label=f"Benchmarked Baselines (bottom {percentile} percent)", color='purple', linestyle='-.')

    # adding standard deviation shading
    upper_actual = df_actual_monthly["mf"] + std_actual_monthly['mf']
    lower_actual = df_actual_monthly["mf"] - std_actual_monthly['mf']
    ax.fill_between(df_actual_monthly.index, lower_actual, upper_actual, color='darkgreen', alpha=0.2, label="Baseline Standard Deviation")

    upper_pred = df_pred_monthly["mf"] + std_pred_monthly['mf']
    lower_pred = df_pred_monthly["mf"] - std_pred_monthly['mf']
    # ax.fill_between(df_pred_monthly.index, lower_pred, upper_pred, color='blue', alpha=0.2, label="Predicted Baseline Standard Deviation")

    upper_benchmark = df_benchmark_monthly["mf"] + std_benchmark_monthly['mf']
    lower_benchmark = df_benchmark_monthly["mf"] - std_benchmark_monthly['mf']
    # ax.fill_between(df_benchmark_monthly.index, lower_benchmark, upper_benchmark, color='purple', alpha=0.2, label="Benchmarked Baseline Standard Deviation")
    
    plt.ylabel("mole fraction in air / ppt", fontsize=12, fontstyle='italic')
    # plt.title(f"{compound} at {site_name}", fontsize=15)
    plt.legend(loc="best", fontsize=12)
    plt.show()

    # calculating some statistics for numerical comparison
    # mean absolute error (MAE)
    mae_model = np.mean(np.abs(df_pred_monthly["mf"] - df_actual_monthly["mf"]))
    mae_benchmark = np.mean(np.abs(df_benchmark_monthly["mf"] - df_actual_monthly["mf"]))
    print(f"MAE for model: {mae_model:.4f}, MAE for benchmark: {mae_benchmark:.4f}")

    # root mean squared error (RMSE)
    rmse_model = np.sqrt(np.mean((df_pred_monthly["mf"] - df_actual_monthly["mf"])**2))
    rmse_benchmark = np.sqrt(np.mean((df_benchmark_monthly["mf"] - df_actual_monthly["mf"])**2))
    print(f"RMSE for model: {rmse_model:.4f}, RMSE for benchmark: {rmse_benchmark:.4f}")

    # mean absolute percentage error (MAPE)
    mape_model = np.mean(np.abs((df_actual_monthly["mf"] - df_pred_monthly["mf"]) / df_actual_monthly["mf"])) * 100
    mape_benchmark = np.mean(np.abs((df_actual_monthly["mf"] - df_benchmark_monthly["mf"]) / df_actual_monthly["mf"])) * 100
    print(f"MAPE for model: {mape_model:.2f}%, MAPE for benchmark: {mape_benchmark:.2f}%")

#=======================================================================
    

## EDA FUNCTIONS
#=======================================================================
def one_month(month, resampled_ecmwf_mhd, resampled_met_mhd):
    """
    Retrieves wind speed, wind direction, and Met Office data for a given month.

    Parameters:
    month (int): The month of interest.

    Returns:
    tuple: A tuple containing the ECMWF wind speed, ECMWF wind direction, and Met Office data for the specified month.
    """

    _, last_day = calendar.monthrange(2015, int(month))
    start_date = pd.Timestamp(f'2015-{month}-01')
    end_date = pd.Timestamp(f'2015-{month}-{last_day}')

    # ECMWF
    month_ecmwf_mhd = resampled_ecmwf_mhd[start_date:end_date]

    month_ecmwf_u = month_ecmwf_mhd['u10']
    month_ecmwf_v = month_ecmwf_mhd['v10']

    month_ecmwf_wind_speed = np.sqrt(month_ecmwf_u**2 + month_ecmwf_v**2)
    month_ecmwf_wind_direction = (np.arctan2(month_ecmwf_u, month_ecmwf_v) * 180 / np.pi) + 180

    # Met Office
    month_met_mhd = resampled_met_mhd[start_date:end_date]

    return month_ecmwf_wind_speed, month_ecmwf_wind_direction, month_met_mhd

#=======================================================================
def find_closest_lat_lon(coords, dataset):
    """
    Finds the closest latitude and longitude coordinates in the dataset to the given coordinates.

    Args:
    - coords (tuple): A tuple containing the latitude and longitude coordinates.
    - dataset (xarray.Dataset): The dataset containing latitude and longitude coordinates.

    Returns:
    - closest_lat (float): The closest latitude coordinate.
    - closest_lon (float): The closest longitude coordinate.

    Raises:
    - ValueError: If latitude and longitude coordinates are not found in the dataset.
    """

    if 'lat' in dataset.coords:
        # NCEP dataset
        lats = dataset.lat
        lons = dataset.lon

        site_lat = coords[0]
        site_lon = coords[1] + 360

    elif 'latitude' in dataset.coords:
        # ECMWF dataset
        lats = dataset.latitude
        lons = dataset.longitude

        site_lat = coords[0]
        site_lon = coords[1]

    else:
        raise ValueError('Latitude and longitude coordinates not found in dataset')    
    
    lat_diff = abs(lats - site_lat)
    lon_diff = abs(lons - site_lon)
    
    lat_index = np.where(lat_diff == min(lat_diff))[0][0]
    lon_index = np.where(lon_diff == min(lon_diff))[0][0]
    
    closest_lat = lats[lat_index]
    closest_lon = lons[lon_index]

    if closest_lon > 180:
        closest_lon -= 360
    
    return closest_lat.values, closest_lon.values

#=======================================================================


## RADON FUNCTIONS
#=======================================================================
def extract_radon_baselines(site):
    """
    Extracts radon baselines by computing the fifth percentile, per month.

    Returns:
    - ds_radon_flags (Dataset): A DataFrame containing the radon data and the radon baselines.
    """

    ds_radon_flags = xr.open_mfdataset((data_path/'radon_data').glob(f"{site}*.nc"))  

    # Loop through all unique years in the dataset
    for year in np.unique(ds_radon_flags.time.dt.year.values):
        # Filter the dataset for the current year
        year_data = ds_radon_flags.sel(time=ds_radon_flags.time.dt.year == year)
        
        # Loop through all unique months in the filtered dataset for the current year
        for month in np.unique(year_data.time.dt.month.values):

            month_data = year_data.sel(time=year_data.time.dt.month == month)

            # creating a 5th percentile for the given month
            rechunked_month_data = month_data['radon'].chunk({'time': -1})
            month_data_percentile = float(rechunked_month_data.quantile(0.05))

            # adding another data variable to dataset stating baseline/no baseline based on percentile
            month_data['baseline'] = month_data['radon'] < month_data_percentile
            ds_radon_flags = xr.concat([ds_radon_flags, month_data], dim='time')

    return ds_radon_flags
#=======================================================================