import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

PRINT_ALLOWED = True
MAX_MISSES_PERCENTS = 45
COLS = 1
ROWS = 0


def show_corr(df: DataFrame):
    width = 30
    height = 10
    sns.set(rc={'figure.figsize': (width, height)})
    sns.heatmap(df.corr(), annot=True, linewidths=3, cbar=False)
    plt.show()


def show_distributions(df: DataFrame, df_stat: DataFrame):
    for i in df.columns:
        plt.figure(i)
        sns.histplot(df[i], kde=True, stat="density")
        interquantile_range = df_stat.loc['Interquantile range'][i]
        plt.axvline(df_stat.loc['Quantile 1'][i] - 1.5 * interquantile_range, color="indigo", ls='--')
        plt.axvline(df_stat.loc['Quantile 1'][i], color="dodgerblue", ls='--')
        plt.axvline(df_stat.loc['Average'][i], color="red", ls='--')
        plt.axvline(df_stat.loc['Median'][i], color="goldenrod", ls='--')
        plt.axvline(df_stat.loc['Quantile 3'][i], color="dodgerblue", ls='--')
        plt.axvline(df_stat.loc['Quantile 3'][i] + 1.5 * interquantile_range, color="indigo", ls='--')
        plt.show()


def get_data_frame() -> DataFrame:
    df = pd.read_excel('ID_data_mass_18122012.xlsx', sheet_name='VU', skiprows=[0, 2])
    df = df.drop(['Unnamed: 0', 'Unnamed: 1'], axis=COLS)
    # weekday_col = 'Номер дня недели'
    # df = df.rename(columns={'Unnamed: 1': weekday_col})
    # for key in df[weekday_col].keys():
    #     d = df[weekday_col][key]
    #     df[weekday_col][key] = int((d - np.datetime64('1969-12-29', 'D')) / np.timedelta64(1, 'D') % 7)
    return df


def get_frame_statistics(df: DataFrame) -> DataFrame:
    col_len = len(df.index)
    col_filled_len = df.count()
    filled_part = ((col_len - col_filled_len) / col_len) * 100
    minimum = df.min()
    q1 = df.quantile(q=0.25, )
    average = df.mean()
    median = df.median()
    q3 = df.quantile(q=0.75, )
    maximum = df.max()
    standard_deviation = df.std()
    unique_count = df.nunique()
    interquantile_range = q3 - q1
    frame = pd.concat([col_filled_len, filled_part, minimum, q1, average, median, q3, maximum, standard_deviation,
                       unique_count, interquantile_range], axis=1, join="inner")
    frame = frame.T
    f = pd.DataFrame(frame)

    f.index = ['Count', 'Unfilled percentage', 'Minimum', 'Quantile 1', 'Average', 'Median', 'Quantile 3',
               'Maximum', 'Standard deviation', 'Unique count', 'Interquantile range']
    return f


def remove_cols_with_many_misses(df: DataFrame, targets: list[str]) -> DataFrame:
    too_little_data_cols = []
    for col in df.columns:
        unfilled = df[col]['Unfilled percentage']
        if unfilled >= MAX_MISSES_PERCENTS and not targets.__contains__(col):
            too_little_data_cols.append(col)
    if PRINT_ALLOWED:
        print('Drop columns with many misses: ', too_little_data_cols)
    return df.drop(too_little_data_cols, axis=COLS)


def remove_cols_with_little_unique(df: DataFrame, targets: list[str]) -> DataFrame:
    too_little_unique_cols = []
    for col in df.columns:
        unique_count = df[col]['Unique count']
        if unique_count == 1 and not targets.__contains__(col):
            too_little_unique_cols.append(col)
    if PRINT_ALLOWED:
        print('Drop columns with little unique: ', too_little_unique_cols)
    return df.drop(too_little_unique_cols, axis=COLS)


def fill_blanks(df: DataFrame, df_stat: DataFrame, targets: [str]) -> DataFrame:
    for col in df.columns:
        if col in targets:
            continue
        for i in df[col].keys():
            if df[col][i] is None or np.isnan(df[col][i]):
                df[col][i] = df_stat[col]['Median']
    return df


# returns list like [a, b, c, d] with intervals [a, b], [b, c], [c, d]
def get_sturges_intervals(minimum: float, maximum: float, count: int) -> list[float]:
    ints_count = 1 + math.floor(math.log2(count))
    if ints_count < 1 or maximum <= minimum:
        return []
    current = minimum
    int_len = (maximum - minimum) / ints_count
    ints = [minimum]
    for i in range(ints_count):
        current += int_len
        ints.append(current)
    ints[ints_count] = maximum
    return ints


def get_interval_id(ints: list[float], val: float):
    ints_count = len(ints) - 1
    for i in range(1, ints_count + 1):
        if val <= ints[i]:
            return i


# calculates (H(y) - H(y|x)) / SplitInformation(y) ,
# where x, y -- names of columns in the data frame
def calc_information_gain_ratio(df: DataFrame, df_stat: DataFrame, y: str, x: str) -> float:
    samples_count = len(df[x])
    # calculating probabilities...
    x_ints = get_sturges_intervals(df_stat[x]['Minimum'], df_stat[x]['Maximum'], samples_count)
    # ex: [3276.0, 3360.5, 3445.0, 3529.5, 3614.0, 3698.5, 3783.0, 3867.5, 3952.0]
    y_ints = get_sturges_intervals(df_stat[y]['Minimum'], df_stat[y]['Maximum'], samples_count)
    ints_count = len(x_ints) - 1
    p_x = {i: 0 for i in range(1, ints_count + 1)}
    p_y = {i: 0 for i in range(1, ints_count + 1)}
    for val in df[x].values:
        p_x[get_interval_id(x_ints, val)] += 1
    # ex: {1: 17, 2: 14, 3: 11, 4: 70, 5: 69, 6: 0, 7: 0, 8: 3}
    for val in df[y].values:
        p_y[get_interval_id(y_ints, val)] += 1
    # scoring target entropy...
    h_y = 0
    for y_count in p_y:
        current_p_y = y_count / samples_count
        h_y -= current_p_y * math.log2(current_p_y)
    # scoring conditional probabilities...
    h_yx = 0
    p_y_x = {i: {j: 0 for j in p_y.keys()} for i in p_x.keys()}
    for row_num in df[x].keys():
        x_int = get_interval_id(x_ints, df[x][row_num])
        p_y_x[x_int][get_interval_id(y_ints, df[y][row_num])] += 1
    for x_int in p_x.keys():
        current_p_x = p_x[x_int] / samples_count
        entropy = 0
        for val in p_y_x[x_int].values():
            if val == 0:
                continue
            current_p_y = val / samples_count
            entropy -= current_p_y * math.log2(current_p_y)
        h_yx += current_p_x * entropy
    # calculating split information(y)
    si = 0
    for y_int in p_y.keys():
        fraction = p_y[y_int] / samples_count
        if fraction == 0:
            continue
        si -= fraction * math.log2(fraction)
    return (h_y - h_yx) / si


def combine_kgf(df: DataFrame) -> DataFrame:
    for row_num in df['КГФ.1'].keys():
        if not np.isnan(df['КГФ.1'][row_num]):
            df['КГФ'][row_num] = df['КГФ.1'][row_num] * 1000
    return df.drop('КГФ.1', axis=COLS)


def remove_empty_target(df: DataFrame, targets: [str]) -> DataFrame:
    to_remove = []
    for i in df['КГФ'].keys():
        no_targets = True
        for col in targets:
            if not (df[col][i] is None or np.isnan(df[col][i])):
                no_targets = False
        if no_targets:
            to_remove.append(i)
    print("Drop ", len(to_remove), " rows: ", to_remove)
    for row in to_remove:
        df = df.drop(row, axis=ROWS)
    return df


def show_gain_ratio(df: DataFrame, df_stat: DataFrame, target: str, targets: list[str]):
    df_igr = pd.DataFrame({
        key: [calc_information_gain_ratio(df, df_stat, target, key)] for key in df.keys() if not targets.__contains__(key)
    })
    sns.barplot(df_igr, orient='horizontal') # Рсб, Рсб.1
    plt.show()


def main():
    mpl.use('TkAgg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = get_data_frame()
    df = combine_kgf(df)
    # print(df.describe())
    targets = ['G_total', 'КГФ']
    df_stat = get_frame_statistics(df)
    df_stat = remove_cols_with_many_misses(df_stat, targets)
    df_stat = remove_cols_with_little_unique(df_stat, targets)
    df = df.filter(items=df_stat)
    df = remove_empty_target(df, targets)
    df_stat = df_stat.filter(items=df)
    df = fill_blanks(df, df_stat, targets)
    # show_distributions(df, df_stat) # trash in Ro_c
    # df = df.drop(151, axis=ROWS)
    # show_corr(df)
    df_stat = get_frame_statistics(df)
    # show_corr(df)
    # print(df_stat)
    # Рзаб and Рзаб1 have almost the same correlations
    # Also Руст and Руст1
    # By IGR defined that Руст.1 > Руст, Рзаб.1 > Рзаб, Дебит воды > Дебит воды.1
    # target = 'КГФ'
    # obs_cols = ['Руст', 'Руст.1', 'Рзаб', 'Рзаб.1', 'Дебит воды', 'Дебит воды.1', 'Дебит газа', 'Дебит гааз', 'Рсб', 'Рсб.1']
    # for column in obs_cols:
    #     print(calc_information_gain_ratio(df, df_stat, target, column))
    df = df.drop(['Рзаб', 'Руст', 'Дебит воды.1'], axis=COLS)
    # df_stat = df_stat.filter(items=df)

    # show_gain_ratio(df, df_stat, 'G_total', targets) # Рсб, Рсб.1
    # show_gain_ratio(df, df_stat, 'КГФ', targets) # Рсб, Рсб.1
    # show_distributions(df, df_stat)
    df_stat = df_stat.filter(items=df)
    df.to_excel("result.xlsx")
    df_stat.to_excel("statistics.xlsx")
    # Рлин, Рсб .1


if __name__ == '__main__':
    main()
