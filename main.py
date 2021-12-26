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
    df = df.fillna(0).astype(int)

    return df


def sql_load_data():
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    female_db = conn.execute(f'SELECT * FROM USA_fltper_1x1')
    male_db = conn.execute(f'SELECT * FROM USA_mltper_1x1')
    female_df = pd.DataFrame(female_db, columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx',
                                                 'LLx', 'Tx', 'ex'])
    male_df = pd.DataFrame(male_db, columns=['PopName', 'Sex', 'Year', 'Age', 'mx', 'qx', 'ax', 'lx', 'dx',
                                             'LLx', 'Tx', 'ex'])

    all_df = pd.concat([female_df, male_df], ignore_index=True)
    all_df.set_index(['Sex', 'Year'], inplace=True)

    return all_df


class Project:
    def __init__(self):
        self.data = load_data()
        self.sql_data = sql_load_data()
        self.years = list(self.data.columns.values)
        self.all_female_birth, self.all_male_birth = self.data.groupby(level=1).sum().sum(axis=1)
        self.no_birth_each_year = self.data.sum()
        self.female_birth_each_year, self.male_birth_each_year = list(self.data.groupby(level=1).sum().values)
        self.rank_df = self.data.sum(axis=1).reset_index(name='Number').sort_values(by=['Number'], ascending=False)

    def showing_unique_names(self):
        unique_names = self.data.groupby(level=0, axis=0).sum().index
        unique_female_names = self.data.groupby(level=1, axis=0).count().values[0][0]
        unique_male_names = self.data.groupby(level=1, axis=0).count().values[1][0]
        print('Liczba unikalnych imion kobiet to:', unique_female_names, ', Liczba unikalnych imion mezczyzn to: ',
              unique_male_names)
        print(f"Liczba unikalnych imion to {len(unique_names)}")

    def add_frequency_columns(self):

        sum_years_name_df = self.data.sum(axis=1).to_frame()
        freq_male = sum_years_name_df.query('Sex == "M"') / self.all_male_birth
        freq_female = sum_years_name_df.query('Sex == "F"') / self.all_male_birth

        self.data['frequency_male'] = freq_male
        self.data['frequency_female'] = freq_female
        self.data['frequency_female'] = self.data['frequency_female'].fillna(0)
        self.data['frequency_male'] = self.data['frequency_male'].fillna(0)
        print(self.data)

    def show_plot_todo5(self):
        ratio_each_year = list(self.female_birth_each_year / self.male_birth_each_year)
        possible_max_difference1 = min(ratio_each_year)
        possible_max_difference2 = max(ratio_each_year)
        max_diff = 0
        if possible_max_difference1 > possible_max_difference2:
            max_diff = possible_max_difference1
            index_of_max = ratio_each_year.index(max_diff)
            print('Najwieksza roznica w ', self.years[index_of_max])
        elif possible_max_difference1 < possible_max_difference2:
            max_diff = possible_max_difference2
            index_of_max = ratio_each_year.index(max_diff)
            print('Najwieksza roznica w ', self.years[index_of_max])

        min_differences = [abs(1 - ratio) for ratio in ratio_each_year]
        index_of_min = min_differences.index(min(min_differences))
        print('Najmniejsza roznica w', self.years[index_of_min])

        fig, ax = plt.subplots(2)
        x = np.array(self.years, dtype=int)
        ax[0].plot(x, self.no_birth_each_year, label='Number of birth each year')
        ax[0].title.set_text('Number of birth each year')
        ax[1].plot(x, ratio_each_year, label='Ratio: number of birth female to male each year')
        ax[1].plot(x[index_of_max], max_diff, 'g.')
        ax[1].plot(x[index_of_min], ratio_each_year[index_of_min], 'r.')
        ax[1].title.set_text('Ratio: number of birth female to male each year')
        fig.tight_layout(pad=3.0)
        plt.show()

    def top1000_names(self):
        female_rank = self.rank_df.loc[self.rank_df['Sex'] == 'F'].iloc[0:1000]
        male_rank = self.rank_df.loc[self.rank_df['Sex'] == 'M'].iloc[0:1000]
        print(female_rank)
        print(male_rank)

    def name_changes_in_top1_female_name(self, name='John', highlight_years=['1930', '1970', '2015']):
        top1_female_name = self.rank_df.loc[self.rank_df['Sex'] == 'F'].iloc[0]['Name']
        top1_birth_each_year = list(self.data[self.years].loc[top1_female_name, 'F'])
        name_birth_each_year = list(self.data[self.years].loc[name, 'M'])
        popularity_of_name = name_birth_each_year / self.male_birth_each_year
        popularity_of_top1 = top1_birth_each_year / self.female_birth_each_year

        indexes = [self.years.index(elem) for elem in highlight_years]

        x = np.array(self.years, dtype=int)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].plot(x, name_birth_each_year, label='Input name')
        ax[0].plot(x, top1_birth_each_year, label='TOP1 female name')
        ax[0].title.set_text('Number of birth each year for 2 names')
        ax[0].legend(loc='upper right')
        ax[1].plot(x, popularity_of_name, label='Input name')
        ax[1].plot(x, popularity_of_top1, label='TOP1 female name')
        ax[1].title.set_text('Popularity of 2 names')
        ax[1].legend(loc='upper right')

        for index in indexes:
            ax[0].plot(x[index], name_birth_each_year[index], 'g*')
            ax[0].plot(x[index], top1_birth_each_year[index], 'g*')
        plt.show()

    def show_plot_todo8(self):
        top1000_f_names = list(self.rank_df.loc[self.rank_df['Sex'] == 'F'].iloc[0:1000]['Name'])
        top1000_m_names = list(self.rank_df.loc[self.rank_df['Sex'] == 'M'].iloc[0:1000]['Name'])
        male_diff_names = []
        female_diff_names = []
        for year in self.years:
            temp_m = self.data[year].to_frame().query(f'Sex == "M"').loc[top1000_m_names].sum().values[0]
            temp_f = self.data[year].to_frame().query(f'Sex == "F"').loc[top1000_f_names].sum().values[0]
            male_diff_names.append(temp_m)
            female_diff_names.append(temp_f)

        male_diff_names = male_diff_names / self.male_birth_each_year * 100
        female_diff_names = female_diff_names / self.female_birth_each_year * 100

        differences = list(abs(male_diff_names - female_diff_names))
        max_diff_year = self.years[differences.index(max(differences))]
        print("Year of max differance of top1000 names between men and women:", max_diff_year)

        x = np.array(self.years, dtype=int)
        width = 0.3
        plt.bar(x - width / 2, male_diff_names, width, label="Man")
        plt.bar(x + width / 2, female_diff_names, width, label="Woman")
        plt.title('% of top1000 names in all each year')
        plt.legend()
        plt.show()

    def distribution_of_last_letters_in_years(self, obs_years=['1915', '1965', '2018']):
        df = self.data[self.years].reset_index()
        df['LastLetter'] = df['Name'].str[-1:]
        df = df.set_index(['Name', 'Sex', 'LastLetter'])

        df_obs = df[obs_years]
        normalized_df_obs = df_obs / self.no_birth_each_year[obs_years]
        last_letter_df_obs = normalized_df_obs.query('Sex == "M"').groupby(by='LastLetter').sum()
        diff_from_1_to_2_obs = abs(last_letter_df_obs[obs_years[0]] - last_letter_df_obs[obs_years[-1]])
        top3_ll = diff_from_1_to_2_obs.nlargest(3).index.tolist()
        popularity_of_top3_ll = df.groupby(by='LastLetter').sum().loc[top3_ll] / self.no_birth_each_year * 100

        print(f'Najwieksza zmiana dla litery: {diff_from_1_to_2_obs.idxmax()}')
        print(f'Najmniejsza zmiana dla litery: {diff_from_1_to_2_obs.idxmin()}')

        x_0 = np.arange(0, len(np.array(last_letter_df_obs.index.tolist())))
        x_1 = np.array(self.years, dtype=int)
        width = 0.3
        fig, ax = plt.subplots(2)

        for i, year in enumerate(obs_years):
            x_offset = (i - len(obs_years) / 2) * width + width / 2
            ax[0].bar(x_0 + x_offset, np.array(last_letter_df_obs[year]), width, label=year)

        for index in popularity_of_top3_ll.index:
            ax[1].plot(x_1, popularity_of_top3_ll.loc[index], label=index)  # dla n

        ax[0].set_xticks(x_0)
        ax[0].set_title('Distribution of last letters in observed years')
        ax[0].set_xticklabels(np.array(last_letter_df_obs.index.tolist()))
        ax[0].legend()
        ax[1].set_title('Popularity of top3 changing last letters')
        ax[1].legend()
        fig.tight_layout(pad=0.9)
        plt.show()

    def same_names_for_mf(self, tolerance=0.01):
        multi_sex_df = self.data[self.years].groupby(level=0).size().reset_index(name='Count')
        multi_sex_names = list(multi_sex_df.loc[multi_sex_df['Count'] == 2]['Name'])
        new_df = self.data[self.years].loc[multi_sex_names].sum(axis=1)
        ratio_df = (new_df.loc[:, 'M'] / new_df.loc[:, 'F']).reset_index(name='Ratio').set_index(['Name'])
        closest_ratio_df = ratio_df.query(f"{1 + tolerance} > Ratio > {1 - tolerance}").copy()
        closest_ratio_df['Diff'] = closest_ratio_df['Ratio'] - 1
        print(closest_ratio_df)
        print('Name with a little more male names than female names: ',
              closest_ratio_df.query("Diff > 0")['Diff'].idxmin())

    def calc_natural_increase(self):
        all_sum_stat = self.sql_data.groupby(level=0).sum().sum()
        natural_increase = all_sum_stat['lx'] - all_sum_stat['dx']
        print('Przyrost naturalny w calym zakresie to:', natural_increase)

    def survivability_by_age(self, age=1):
        mx = self.sql_data.groupby(by='Age').sum().loc[age]['mx']
        print(f'Survivability at age {age}: {mx}')


cc = Project()

cc.showing_unique_names()
cc.add_frequency_columns()
cc.show_plot_todo5()
cc.top1000_names()
cc.name_changes_in_top1_female_name()
cc.show_plot_todo8()
cc.distribution_of_last_letters_in_years()
cc.same_names_for_mf()
cc.calc_natural_increase()
cc.survivability_by_age()

