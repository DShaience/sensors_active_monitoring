import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analytical_utils import extract_reasons, read_sensors_data_from_file, add_nan_values_on_time_gap, calc_statistics
from visualization_utils import plot_reasons_bar_chart, plot_reasons_correlation_matrix

from analytical_utils import PERF_CAP_REASON


sns.set(color_codes=True)


def main(file):
    sensors_all = read_sensors_data_from_file(file)
    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)
    target_columns = ['CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                      'GPU Chip Power Draw [W]', 'GPU Load [%]', 'Fan 1 Speed (%) [%]', 'System Memory Used [MB]']
    n_rows = 4
    n_cols = max(1, int(round(len(target_columns) / n_rows + 0.5, 0)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    features_list = []
    for i in range(len(target_columns)):
        target_col = target_columns[i]
        ax = axes.flatten()[i]
        data = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
        ax.plot(range(len(data)), data)
        ax.title.set_text(target_col)
        features_list.append(calc_statistics(data, target_col.replace(' ', '_')))
    if n_cols * n_rows > len(target_columns):
        for ax in axes.flatten()[len(target_columns):]:
            ax.remove()
    sensors_features = pd.DataFrame(features_list)
    print(sensors_features.to_string())
    fig.tight_layout(h_pad=-1)
    plt.show()
    print(n_rows)
    sensors_all['PerfCap Reason []'] = sensors_all['PerfCap Reason []'].fillna(value=0).astype(int)
    sensors_all['Reasons'] = sensors_all['PerfCap Reason []'].apply(extract_reasons)
    # Flatten the list of reasons
    reasons_list = [reason for sublist in sensors_all['Reasons'] for reason in sublist]
    from collections import Counter
    # Count occurrences of each reason
    reasons_count = Counter(reasons_list)
    # Convert the Counter to a DataFrame for plotting
    reasons_df = pd.DataFrame.from_dict(reasons_count, orient='index', columns=['Count'])
    reasons_df.sort_values(by='Count', ascending=True, inplace=True)
    plot_reasons_bar_chart(reasons_df)
    PLOT_CORRELATION = True
    if PLOT_CORRELATION:
        plot_reasons_correlation_matrix(sensors_all, list(PERF_CAP_REASON.values()))


if __name__ == '__main__':
    file = r"C:\Users\shayt\OneDrive\Desktop\Sensors\Spider-Man Morales.txt"

    main(file)








