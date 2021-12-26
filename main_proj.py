import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sqlite3

def load_data():
    all_dataframes = []
    for root, dirs, files in (os.walk('./names')):
        for f in sorted(files):
            if f.startswith('yob'):
                img_path = os.path.join(root, f)
                temp = pd.read_csv(img_path, header=None, names=['Name', 'Sex', f[3:-4]], index_col=['Name', 'Sex'],
                                   squeeze=True)
                all_dataframes.append(temp)

    df = pd.concat(all_dataframes, axis=1)
    # df = df.reindex(pd.MultiIndex.from_product([df.index.levels[0], ['F', 'M']], names=['Name', 'Sex']))
    df = df.fillna(0).astype(int)
    return df


def todo2():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    df.reset_index(inplace=True)
    no_unique_names = len(set(df['Name'].values))
    print(no_unique_names)
    print(df)


def todo3():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    f_name_no = df.groupby(level=1, axis=0).count().values[0][0]
    m_name_no = df.groupby(level=1, axis=0).count().values[1][0]
    print('Female names:', f_name_no, ', Male names: ', m_name_no)


def todo4():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    years = df.columns.values
    frequency_male = []
    frequency_female = []
    all_female_birth, all_male_birth = df.groupby(level=1)[years].sum().sum(axis=1)

    print(all_female_birth, all_male_birth)
    d = df[years].sum(axis=1).reset_index()
    for row in d.values:
        if row[1] == 'M':
            freq_of_name_male = row[2] / all_male_birth
            freq_of_name_female = 0
            frequency_male.append(freq_of_name_male)
            frequency_female.append(freq_of_name_female)
        elif row[1] == 'F':
            freq_of_name_male = 0
            freq_of_name_female = row[2] / all_female_birth
            frequency_male.append(freq_of_name_male)
            frequency_female.append(freq_of_name_female)

    df['frequency_male'] = frequency_male
    df['frequency_female'] = frequency_female

    return df

def todo5():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    years = df.columns.values
    no_birth_each_year = list(df.groupby(level=1)[years].sum().sum())
    female_birth_each_year, male_birth_each_year = list(df.groupby(level=1)[years].sum().values)
    ratio_each_year = [female_birth_each_year[i] / male_birth_each_year[i] for i in range(len(female_birth_each_year))]

    possible_max_difference1 = min(ratio_each_year)
    possible_max_difference2 = max(ratio_each_year)
    if possible_max_difference1 > possible_max_difference2:
        index_year = ratio_each_year.index(possible_max_difference1)
        print('Najwieksza roznica w ', years[index_year])
    elif possible_max_difference2 > possible_max_difference1:
        index_year = ratio_each_year.index(possible_max_difference2)
        print('Najwieksza roznica w ', years[index_year])

    min_differences = [abs(1 - ratio) for ratio in ratio_each_year]
    index_of_min = min_differences.index(min(min_differences))
    print('Najmniejsza roznica w', years[index_of_min])

    fig, ax = plt.subplots(2)
    ax[0].plot(years, ratio_each_year)
    ax[1].plot(years, no_birth_each_year)
    plt.show()


def todo6():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    years = df.columns.values
    df = df[years].sum(axis=1).reset_index(name='Number').sort_values(by=['Number'], ascending=False)
    female_rank = df.loc[df['Sex'] == 'F'].iloc[0:1000]
    male_rank = df.loc[df['Sex'] == 'M'].iloc[0:1000]


def todo7():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    years = list(df.columns.values)

    # birth each year of those 2 namess
    rank = df[years].sum(axis=1).reset_index(name='Number').sort_values(by=['Number'], ascending=False)
    first_female_name = rank.loc[rank['Sex'] == 'F'].iloc[0]['Name']
    john_male_birth_each_year = list(df.loc['John', 'M'])
    name_female_birth_each_year = list(df.loc['Mary', 'F'])
    # popularity each year of those 2 names
    female_birth_each_year = df.groupby(level=1)[years].sum().loc['F']
    male_birth_each_year = df.groupby(level=1)[years].sum().loc['M']
    popularity_john = john_male_birth_each_year / male_birth_each_year
    popularity_mary = name_female_birth_each_year / female_birth_each_year

    highlight_years = ['1930', '1970', '2015']
    indexes = [years.index(elem) for elem in highlight_years]

    fig, ax = plt.subplots(2)
    ax[0].plot(years, john_male_birth_each_year)
    ax[0].plot(years, name_female_birth_each_year)
    ax[1].plot(years, popularity_john)
    ax[1].plot(years, popularity_mary)
    for index in indexes:
        ax[0].plot(years[index], john_male_birth_each_year[index], 'g*')
        ax[0].plot(years[index], name_female_birth_each_year[index], 'g*')
        ax[1].plot(years[index], popularity_john[index], 'g*')
        ax[1].plot(years[index], popularity_mary[index], 'g*')
    plt.show()


def todo8():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])

    years = list(df.columns.values)
    female_birth_each_year = list(df.groupby(level=1)[years].sum().loc['F'])
    male_birth_each_year = list(df.groupby(level=1)[years].sum().loc['M'])
    top1000_all = df[years].sum(axis=1).reset_index(name='Number').sort_values(by=['Number'], ascending=False)
    female_rank = list(top1000_all.loc[top1000_all['Sex'] == 'F'].iloc[0:1000]['Name'])
    male_rank = list(top1000_all[top1000_all['Sex'] == 'M'].iloc[0:1000]['Name'])

    sexes = ['M', 'F']
    male_diff_names = []
    female_diff_names = []
    for i, year in enumerate(years):
        temp_df = df[year].reset_index(name='Number')

        for sex in sexes:
            temp_df_spec_sex = temp_df.loc[temp_df['Sex'] == sex].loc[temp_df['Number'] > 0]
            names = list(temp_df_spec_sex['Name'])
            if sex == 'M':
                result_names = list(set(male_rank).intersection(names))
                result_names_df = temp_df_spec_sex.loc[temp_df_spec_sex['Name'].isin(result_names)]
                number_of_male_names_in_top = result_names_df['Number'].sum()
                percent = (number_of_male_names_in_top / male_birth_each_year[i]) * 100
                male_diff_names.append(percent)

            elif sex == 'F':
                result_names = list(set(female_rank).intersection(names))
                result_names_df = temp_df_spec_sex.loc[temp_df_spec_sex['Name'].isin(result_names)]
                number_of_female_names_in_top = result_names_df['Number'].sum()
                percent = (number_of_female_names_in_top / female_birth_each_year[i]) * 100
                female_diff_names.append(percent)

    differences = [abs(male_diff_names[i] - female_diff_names[i]) for i in range(len(years))]
    max_diff_year = years[differences.index(max(differences))]
    print(max_diff_year)

    width = 0.3
    plt.bar(np.array(years, dtype=int) - width / 2, male_diff_names, width, label="Man")
    plt.bar(np.array(years, dtype=int) + width / 2, female_diff_names, width, label="Woman")
    plt.show()


def todo9():
    df = pd.read_csv('name_dataset.csv')
    # dodanie ostatniej litery
    df['LastLetter'] = df['Name'].str[-1:]
    df = df.set_index(['Name', 'Sex', 'LastLetter'])
    years = list(df.columns.values)
    # df z obserwowanymi latami
    obs_years = ['1915', '1965', '2018']
    df_obs = df[obs_years]

    # normalizacja
    normalized_df = ((df_obs - df_obs.min()) / (df_obs.max() - df_obs.min())).reset_index()
    male_df = normalized_df.loc[normalized_df['Sex'] == 'M']
    male_df = male_df.groupby(by='LastLetter').sum()
    print(male_df)
    x = np.arange(0, len(np.array(male_df.index.tolist())))
    width = 0.3
    diff = abs(male_df['1915'] - male_df['2018'])

    # get popularity for 3 letters in list
    top3_ll = diff.nlargest(3).index.tolist()
    df_by_letter = df.groupby(by='LastLetter').sum()
    # print(df_by_letter.loc[top3_ll])
    birth_each_year = df.sum()
    desire_df = df_by_letter.loc[top3_ll] / birth_each_year * 100
    print(desire_df)
    # plot
    fig, ax = plt.subplots(2)
    print(f'Najwieksza zmiana dla litery: {diff.idxmax()}')
    print(f'Najmniejsza zmiana dla litery: {diff.idxmin()}')
    ax[0].bar(x - width, np.array(male_df['1915']), width, label='1915')
    ax[0].bar(x, np.array(male_df['1965']), width, label='1965')
    ax[0].bar(x + width, np.array(male_df['2018']), width, label='2018')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(np.array(male_df.index.tolist()))

    ax[1].plot(years, desire_df.iloc[0])  # dla n
    ax[1].plot(years, desire_df.iloc[1])
    ax[1].plot(years, desire_df.iloc[2])
    plt.show()


def todo10():
    df = pd.read_csv('name_dataset.csv')
    df = df.set_index(['Name', 'Sex'])
    multi_sex_df = df.groupby(level=0).size().reset_index(name='Count')
    multi_sex_names = list(multi_sex_df.loc[multi_sex_df['Count'] == 2]['Name'])
    new_df = df.loc[multi_sex_names].sum(axis=1)
    ratio_df = (new_df.loc[:, 'M'] / new_df.loc[:, 'F']).reset_index(name='Ratio').set_index(['Name'])
    closest_ratio_df = ratio_df.query("1.01 > Ratio > 0.99").copy()
    closest_ratio_df['Diff'] = closest_ratio_df['Ratio'] - 1
    print(closest_ratio_df.query("Diff > 0")['Diff'].idxmin())


def todo12():
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    female_db = conn.execute(f'SELECT * FROM USA_fltper_1x1')
    male_db = conn.execute(f'SELECT * FROM USA_mltper_1x1')
    female_df = pd.DataFrame(female_db, columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx',
                                                 'LLx', 'Tx', 'ex'])
    male_df = pd.DataFrame(male_db, columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx',
                                             'LLx', 'Tx', 'ex'])

    all_df = pd.concat([female_df, male_df], ignore_index=True)
    all_df.set_index(['Sex', 'Year'], inplace=True)
    all_sum_stat = all_df.groupby(level=0).sum().sum()
    print('Przyrost naturalny w calym zakresie to:', all_sum_stat['lx'] - all_sum_stat['dx'])
    print('Wskaznik przezywalnosci dla osob 1 lat to: ', all_df.groupby(by='Age').sum().loc[1]['mx'])


todo9()
