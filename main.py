import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error

# The latest date for the data worldwide and U.S. is 12/10/2021
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
confirmed_df_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
deaths_df_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
latest_data_all = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_daily_reports/12-10-2021.csv')
latest_data_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_daily_reports_us/12-10-2021.csv')

confirm = confirmed_df.loc[:, '1/22/20': '12/10/21']
death = deaths_df.loc[:, '1/22/20': '12/10/21']
recoveries = recoveries_df.loc[:, '1/22/20': '12/10/21']

# Q: 怎样提取不连续的column
confirmed_df_us['d'] = 0
confirmed_us = confirmed_df_us.iloc[:, np.r_[6, 12:701]]
deaths_us = deaths_df_us.iloc[:, np.r_[6, 12:701]]

confirmed_state = confirmed_us.groupby('Province_State').sum()
deaths_state = deaths_us.groupby('Province_State').sum()


def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


dates = confirm.keys()
world_cases = []
world_death = []
world_recoveries = []
for i in dates:
    confirmed_sum = confirm[i].sum()
    death_sum = death[i].sum()
    recoveries_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    world_death.append(death_sum)
    world_recoveries.append(recoveries_sum)

world_daily_confirmed = daily_increase(world_cases)
world_daily_deaths = daily_increase(world_death)
world_daily_recoveries = daily_increase(world_recoveries)

world_daily_reports = {
    'Increases': world_daily_confirmed,
    'Deaths': world_daily_deaths,
    'Recoveries': world_daily_recoveries
}
# transform lists into a dataframe
df_world_daily_reports = pd.DataFrame(world_daily_reports)

# time is the x-axis
time = pd.to_datetime(list(confirm))


# visualization for daily increases/deaths/recoveries worldwide
def plot_worldwide_cases(cases):
    plt.plot(time, df_world_daily_reports[cases])
    plt.title('Daily Increase for {} Worldwide'.format(cases))
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.ticklabel_format(axis='y', style='plain')  # transform scientific notation into normal version
    plt.show()


# for i in ['Increases', 'Deaths', 'Recoveries']:
#     plot_worldwide_cases(i)

# visualization for daily confirmed/deaths/recoveries in US
def plot_us_cases(state):
    plt.plot(time, confirmed_state.loc[state], label='Confirmed')  # 为啥最后运行出来没有label？
    plt.plot(time, deaths_state.loc[state], label='Deaths')
    plt.title('Daily Confirmed & Deaths in {}'.format(state))
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.ticklabel_format(axis='y', style='plain')
    plt.show()


states = confirmed_df_us.iloc[:, 6].drop_duplicates().values.tolist()


# for i in states:
#     plot_us_cases(i)

# visualizion for daily increased confirmed/deaths/recoveries in US
def plot_us_increased_cases(state):
    increase_confirmed = daily_increase(confirmed_state.loc[state].values.tolist())
    increase_deaths = daily_increase(deaths_state.loc[state].values.tolist())
    plt.plot(time, increase_confirmed, label='Confirmed Increase')
    plt.plot(time, increase_deaths, label='Deaths Increase')
    plt.title('Daily Increased Confirmed & Deaths in {}'.format(state))
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.ticklabel_format(axis='y', style='plain')
    plt.show()


# for i in states:
#     plot_us_increased_cases(i)

# print(confirmed_state)


state_in_us = confirmed_state.drop(
    ['American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico',
     'Virgin Islands']
)
code = {'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'}

# transform column name to a column
state_in_us = state_in_us.rename_axis('Province_State').reset_index()
# add a new column "Code" whose values is equal to dic 'code'
state_in_us['Code'] = state_in_us['Province_State'].map(code)

state_in_us['Sum'] = state_in_us.iloc[:, 1:-1].sum(axis=1)

fig = px.choropleth(state_in_us,
                    locations='Code',
                    color='Sum',
                    color_continuous_scale='spectral_r',
                    hover_name='Province_State',
                    locationmode='USA-states',
                    labels={'Sum': 'Confirmed Cases in US'},
                    scope='usa')
fig.add_scattergeo(
    locations=state_in_us['Code'],
    locationmode='USA-states',
    text=state_in_us['Code'],
    mode='text')

fig.update_layout(
    title={'text': 'Confirmed Cases in US by State',
           'xanchor': 'center',
           'yanchor': 'top',
           'x': 0.5})

# fig.show()


confirmed_df = confirmed_df.rename(columns={"Province/State": "state", "Country/Region": "country"})

latest_data_all = latest_data_all.rename(columns={"Country_Region": "country"})

# Changing the conuntry names as required by pycountry_convert Lib
latest_data_all.loc[latest_data_all['country'] == "US", "country"] = "USA"
latest_data_all.loc[latest_data_all['country'] == "Korea, South", "country"] = "South Korea"
latest_data_all.loc[latest_data_all['country'] == "Taiwan*", "country"] = "Taiwan"
latest_data_all.loc[latest_data_all['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"
latest_data_all.loc[latest_data_all['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
latest_data_all.loc[latest_data_all['country'] == "Reunion", "country"] = "Réunion"
latest_data_all.loc[latest_data_all['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"
latest_data_all.loc[latest_data_all['country'] == "Bahamas, The", "country"] = "Bahamas"
latest_data_all.loc[latest_data_all['country'] == "Gambia, The", "country"] = "Gambia"
latest_data_all = latest_data_all.groupby('country').sum()
latest_data_all = latest_data_all.rename_axis('country').reset_index()

# visualizion for worldwide confirmed cases
fig = px.choropleth(latest_data_all,
                    locations="country",
                    color='Confirmed',
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    title={'text': 'Confirmed Cases Heat Map',
           'xanchor': 'center',
           'yanchor': 'top',
           'x': 0.5})
fig.update_coloraxes(colorbar_title="Confirmed Cases", colorscale="Blues")

# fig.show()

# SVM for prediction
# Select the number of cases from 11/01/2021 to 11/30/2021
confirmed_df = confirmed_df.iloc[:, np.r_[1, 653:683]]
deaths_df = deaths_df.iloc[:, np.r_[1, 653:683]]
# recoveries_df = recoveries_df.iloc[:, np.r_[1, 653:683]]

confirmed_by_countries = confirmed_df.groupby('country').sum()
deaths_by_countries = deaths_df.groupby('Country/Region').sum()
# recoveries_by_countries = recoveries_df.groupby('Country/Region').sum()

confirmed_by_countries = confirmed_by_countries.diff(axis=1)
deaths_by_countries = deaths_by_countries.diff(axis=1)
# recoveries_by_countries = recoveries_by_countries.diff(axis=1)

# print(deaths_by_countries)

confirmed_by_countries['confirmed_mean'] = confirmed_by_countries.mean(axis=1)
deaths_by_countries['deaths_mean'] = deaths_by_countries.mean(axis=1)

confirmed_by_countries['confirmed_sum'] = confirmed_by_countries.iloc[:, 1:30].sum(axis=1)
deaths_by_countries['deaths_sum'] = deaths_by_countries.iloc[:, 1:30].sum(axis=1)
result = pd.concat([confirmed_by_countries, deaths_by_countries], axis=1)

result['Case_fatality_ratio'] = result.iloc[:, 63] / result.iloc[:, 31]

# print(result)
# result.info()

# derive the first three features: avg confirmed/deaths cases/Case_fatality_ratio per countries in November
df_features = result.iloc[:, [30, 62, 64]]
# print(df_features)


density_df = pd.read_csv('./data/csvData.csv')
# rename the first column in the density_df to join with the dataframe df_features
density_df = density_df.rename(columns={'name': 'Countries'})

# Convert row names into a column
df_features.index.name = 'Countries'
df_features.reset_index(inplace=True)

# check which rows have NA values
# check_na = df_features[df_features.isnull().T.any()]
# print(check_na)
# delete NA
df_features.dropna(axis=0, how='any', inplace=True)

# find which countries names are not matched with in the two dataframe
# df_features_1 = pd.merge(df_features, density_df, how='outer', on='Countries')
# a = df_features_1[df_features_1.isnull().T.any()]
# print(a)

df_features.loc[df_features['Countries'] == "US", "Countries"] = "United States"
df_features.loc[df_features['Countries'] == "Congo (Kinshasa)", "Countries"] = "DR Congo"
df_features.loc[df_features['Countries'] == "Korea, South", "Countries"] = "South Korea"
df_features.loc[df_features['Countries'] == "Taiwan*", "Countries"] = "Taiwan"
df_features.loc[df_features['Countries'] == "Congo (Brazzaville)", "Countries"] = "Republic of the Congo"
df_features.loc[df_features['Countries'] == "Bahamas, The", "Countries"] = "Bahamas"
df_features.loc[df_features['Countries'] == "Gambia, The", "Countries"] = "Gambia"

df_features.loc[df_features['Countries'] == "Burma", "Countries"] = "Myanmar"
df_features.loc[df_features['Countries'] == "Czechia", "Countries"] = "Czech Republic"

# df_features_1 = pd.merge(df_features, density_df, how='outer', on='Countries')
# a = df_features_1[df_features_1.isnull().T.any()]

df_features = pd.merge(df_features, density_df, how='inner', on='Countries')
df_features = df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean', 'Case_fatality_ratio', 'Density']]

continents_df = pd.read_csv('./data/continents2.csv')

continents_df = continents_df.rename(columns={'name': 'Countries'})
continents_df = continents_df.rename(columns={'region': 'Continents'})

# find the unmatched
# df_features_2 = pd.merge(df_features, continents_df, how='outer', on='Countries')
# df_features_3 = df_features_2.loc[:, ['Countries', 'Continents']]
# df_features_4 = df_features_2.loc[:, ['Countries', 'confirmed_mean']]
# b = df_features_3[df_features_3.isnull().T.any()]
# c = df_features_4[df_features_4.isnull().T.any()]
# print(b)
# print(c)
continents_df.loc[continents_df['Countries'] == "Congo (Democratic Republic Of The)", "Countries"] = "DR Congo"
continents_df.loc[continents_df['Countries'] == "Congo", "Countries"] = "Republic of the Congo"
continents_df.loc[continents_df['Countries'] == "Bosnia And Herzegovina", "Countries"] = "Bosnia and Herzegovina"
continents_df.loc[continents_df['Countries'] == "Brunei Darussalam", "Countries"] = "Brunei"

df_features = pd.merge(df_features, continents_df, how='inner', on='Countries')
df_features = \
    df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean', 'Case_fatality_ratio', 'Density', 'Continents']]
# display all the columns
# pd.set_option('display.max_columns', None)


vaccinations = pd.read_csv('./data/owid-covid-data.csv')
vaccinations = vaccinations.loc[:, ['location', 'date', 'people_fully_vaccinated_per_hundred']]
vaccinations.dropna(axis=0, how='any', inplace=True)

vaccinations['date'] = vaccinations['date'].astype('string')

vaccinations = vaccinations[vaccinations["date"].str.contains('2021-11-')]
vaccinations['date'] = vaccinations['date'].str.replace(r'2021-11-', '')
vaccinations['date'] = vaccinations['date'].astype('int')

vaccinations = vaccinations.groupby('location')
vaccinations_max = vaccinations.max()

vaccinations_max.index.name = 'Countries'
vaccinations_max.reset_index(inplace=True)
vaccinations_max = vaccinations_max.rename(columns={'location': 'Countries'})

# df_features_5 = pd.merge(vaccinations_max, df_features, how='outer', on='Countries')

# df_features_6 = df_features_5.loc[:, ['Countries', 'people_fully_vaccinated_per_hundred']]
# e = df_features_6[df_features_6.isnull().T.any()]
# print(e)

# some of the countries don't have the full vaccination rate in the dataset so we have to impute the rate manually
vaccinations_max.loc[vaccinations_max['Countries'] == "Singapore", "people_fully_vaccinated_per_hundred"] = 86.00
vaccinations_max.loc[vaccinations_max['Countries'] == "Monaco", "people_fully_vaccinated_per_hundred"] = 62.20

vaccinations_max.loc[vaccinations_max['Countries'] == "Democratic Republic of Congo", "Countries"] = "DR Congo"
vaccinations_max.loc[vaccinations_max['Countries'] == "Czechia", "Countries"] = "Czech Republic"
vaccinations_max.loc[vaccinations_max['Countries'] == "Congo", "Countries"] = "Republic of the Congo"

df_features = pd.merge(df_features, vaccinations_max, how='inner', on='Countries')
df_features = \
    df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean',
                        'Case_fatality_ratio', 'Density', 'Continents', 'people_fully_vaccinated_per_hundred']]

gni = pd.read_csv('./data/UNdata_Export_GNI_2019.csv')
gni = gni.loc[:, ['Country or Area', 'Value']]
gni = gni.rename(columns={'Country or Area': 'Countries'})
gni = gni.rename(columns={'Value': 'GNI'})

# df_features_7 = pd.merge(df_features, gni, how='outer', on='Countries')
# g = df_features_7[df_features_7.isnull().T.any()]
# pd.set_option('display.max_rows', None)
# print(g)
gni.loc[gni['Countries'] == "Bolivia (Plurinational State of)", "Countries"] = "Bolivia"
gni.loc[gni['Countries'] == "Brunei Darussalam", "Countries"] = "Brunei"
gni.loc[gni['Countries'] == "China, People's Republic of", "Countries"] = "China"
gni.loc[gni['Countries'] == "Congo", "Countries"] = "Republic of the Congo"
gni.loc[gni['Countries'] == "Democratic Republic of the Congo", "Countries"] = "DR Congo"
gni.loc[gni['Countries'] == "Czechia", "Countries"] = "Czech Republic"
gni.loc[gni['Countries'] == "Kingdom of Eswatini", "Countries"] = "Eswatini"
gni.loc[gni['Countries'] == "Iran, Islamic Republic of", "Countries"] = "Iran"
gni.loc[gni['Countries'] == "Republic of Korea", "Countries"] = "South Korea"
gni.loc[gni['Countries'] == "Republic of Moldova", "Countries"] = "Moldova"
gni.loc[gni['Countries'] == "Russian Federation", "Countries"] = "Russia"
gni.loc[gni['Countries'] == "Syrian Arab Republic", "Countries"] = "Syria"
gni.loc[gni['Countries'] == "United Republic of Tanzania: Mainland", "Countries"] = "Tanzania"
gni.loc[gni['Countries'] == "United Kingdom of Great Britain and Northern Ireland", "Countries"] = "United Kingdom"
gni.loc[gni['Countries'] == "Venezuela (Bolivarian Republic of)", "Countries"] = "Venezuela"
df_features = pd.merge(df_features, gni, how='inner', on='Countries')
df_features = df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean',
                                  'Case_fatality_ratio', 'Density', 'Continents', 'people_fully_vaccinated_per_hundred',
                                  'GNI']]
pd.set_option('display.max_columns', None)

# index the Continents
index_continents = {elem: index + 1 for index, elem in enumerate(set(df_features["Continents"]))}
df_features['Continents'] = df_features['Continents'].map(index_continents)

# add a column to index the countries names
df_features = df_features.reset_index()

df_features['Countries_indexes'] = range(len(df_features))

# print(df_features)

# build SVM model to make predictions

X = df_features.loc[:, ['index',
                        'confirmed_mean',
                        'deaths_mean',
                        'Case_fatality_ratio',
                        'Density', 'Continents',
                        'people_fully_vaccinated_per_hundred',
                        'GNI',
                        'Countries_indexes']].values  # numpy.ndarray
df_features['whether'] = 0


df_features.loc[df_features['confirmed_mean'] >= 1500, 'whether'] = 1

outbreak_countries = df_features.loc[df_features['whether'] == 1, 'Countries']
rows = outbreak_countries.shape[0]
print(outbreak_countries)
print(rows)

y = df_features.loc[:, 'whether'].values

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = \
    train_test_split(X, y, test_size=0.2, shuffle=False)

# use this to find the optimal parameters for SVR
svm_confirmed = SVC()
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(X_test_confirmed)

print(svm_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_confirmed, svm_pred)
print(cm)
accuracy_score(y_test_confirmed, svm_pred)

