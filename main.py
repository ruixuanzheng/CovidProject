import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Time series data (confirmed_df, deaths_df, recoveries_df, confirmed_df_us, deaths_df_us) are updated automatically
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

# The latest date for the data worldwide and U.S. is 12/10/2021 and need updating manually so we need some
# transformations to get the latest data automatically
# get the current date and the day before the current date because the updating frequency for the data sets we use are
# once a day and the date for updating is one day later than the current day so we need previous_before_current_date
current_date = datetime.datetime.now()
previous_before_current_date = current_date + datetime.timedelta(days=-1)
previous_before_current_date = previous_before_current_date.strftime('%m-%d-%Y')

# form the changeable URL when date changes so we can get the up-to-date dataset
url_all = \
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' \
    + str(previous_before_current_date) + '.csv'
url_us = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/\
csse_covid_19_daily_reports_us/' + str(previous_before_current_date) + '.csv'
latest_data_all = pd.read_csv(url_all)
latest_data_us = pd.read_csv(url_us)

# select the data that contain the data from 01/22/2020 to the day before the current date
confirm = confirmed_df.loc[:, '1/22/20':]
death = deaths_df.loc[:, '1/22/20':]
recoveries = recoveries_df.loc[:, '1/22/20':]

# We have to select the states (index 6) and the the data from 01/22/2020 to the day before the current date
# Since the columns we want to select are discontinuous so we select the columns we want separately and combine them
# select the 'Province_State' column
confirmed_us1 = confirmed_df_us.loc[:, 'Province_State']
# select the the data from 01/22/2020 to the day before the current date
confirmed_us2 = confirmed_df_us.loc[:, '1/22/20':]
# combine the two dataframes together as 'confirmed_us' and the 'deaths_us' dataframe is done as follows
confirmed_us = pd.concat([confirmed_us1, confirmed_us2], axis=1)
deaths_us1 = deaths_df_us.loc[:, 'Province_State']
deaths_us2 = deaths_df_us.loc[:, '1/22/20':]
deaths_us = pd.concat([deaths_us1, deaths_us2], axis=1)

# group the dataframe by states and sum the total number of confirmed and death cases
confirmed_state = confirmed_us.groupby('Province_State').sum()
deaths_state = deaths_us.groupby('Province_State').sum()


# write a function to calculate the daily increase of confirmed/death/recovery cases world wide
def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


# use dates as keys for later iteration
dates = confirm.keys()
# create new null lists for daily total number of confirmed/death/recovery (add all countries together) cases worldwide
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

# use the daily_increase function to calculate daily increase of confirmed/death/recovery cases worldwide
world_daily_confirmed = daily_increase(world_cases)
world_daily_deaths = daily_increase(world_death)
world_daily_recoveries = daily_increase(world_recoveries)

# combine lists into a dictionary
world_daily_reports = {
    'Increases': world_daily_confirmed,
    'Deaths': world_daily_deaths,
    'Recoveries': world_daily_recoveries
}

# transform the dictionary into a dataframe
df_world_daily_reports = pd.DataFrame(world_daily_reports)

# create a variable called 'time' as the x-axis for the later data visualization
time = pd.to_datetime(list(confirm))


# So far we've cleaned and prepared for the majority of datasets and variables we'll need for visualization work
# **************First types of plots: daily increases/deaths/recoveries in U.S. and worldwide*************************
# visualization for daily increases/deaths/recoveries worldwide

def plot_worldwide_cases(cases):
    plt.plot(time, df_world_daily_reports[cases])
    plt.title('Daily Increase for {} Worldwide'.format(cases))  # the title of the plots changes according to data
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.ticklabel_format(axis='y', style='plain')  # transform scientific notation into normal version
    plt.show()


# for i in ['Increases', 'Deaths', 'Recoveries']:
#     plot_worldwide_cases(i)

# visualization for daily confirmed/deaths/recoveries in US
def plot_us_cases(state):
    plt.plot(time, confirmed_state.loc[state], label='Confirmed')  # the label changes as states change
    plt.plot(time, deaths_state.loc[state], label='Deaths')
    plt.title('Daily Confirmed & Deaths in {}'.format(state))
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.ticklabel_format(axis='y', style='plain')
    plt.show()


# create a list containing all state names in U.S. for iteration
# states = confirmed_df_us.iloc[:, 6].drop_duplicates().values.tolist()
# for i in states:
#     plot_us_cases(i)

# visualization for daily increased confirmed/deaths/recoveries in US
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


# ***************Second types of plots: Visualization on Map********************
# delete some areas in the given datasets to match for px.choropleth
state_in_us = confirmed_state.drop(
    ['American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Puerto Rico',
     'Virgin Islands']
)
# encode the states in U.S. for px.choropleth
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

# transform column name 'Province_State' into a column
state_in_us = state_in_us.rename_axis('Province_State').reset_index()
# add a new column "Code" whose values is equal to the dictionary 'code'
state_in_us['Code'] = state_in_us['Province_State'].map(code)
# create a new column called 'sum' and assign the latest date of confirmed/deaths cases (the last two) as the sum cases
state_in_us['Sum'] = state_in_us.iloc[:, -2]

# draw the choropleth map for confirmed cases since 1/22/20 in U.S
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

# rename the columns
confirmed_df = confirmed_df.rename(columns={"Province/State": "state", "Country/Region": "country"})
latest_data_all = latest_data_all.rename(columns={"Country_Region": "country"})

# Changing the country names as required by pycountry_convert Lib
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

# visualization for worldwide confirmed cases
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

fig.show()


# Since then, we have completed all parts of data visualization work and let's move forward to the prediction part.


# We use SVM model for predicting classification.
# The basic logics are choose features from one month (avg confirmed/deaths cases/Case_fatality_ratio per countries),
# together with other features (Density, Continents, people_fully_vaccinated_per_hundred, 'GNI') as features,
# and use average increased confirmed cases per day as label to make classifications. According to label, we split the
# countries into two type: countries with intense COVID outbreak and countries without. Then we train the model.
# In the end, we test the model and optimize the model.

# Input the year and the month that we want to predict
year = 2021
month = 12

# We use the data of month before the month that we want to predict. When month we want to predict is January,
# The data of month we use is the last year's December .
if month == 1:
    month_trained = str(12)
    year_trained = str(year-1)
else:
    month_trained = str(month-1)
    year_trained = str(year)

if month in [1, 2, 4, 6, 8, 9, 11]:
    last_day = 31
elif month in [5, 7, 10, 12]:
    last_day = 30
elif month == 3:
    if year % 4 == 0:
        last_day = 29
    else:
        last_day = 28

last_day = str(last_day)
# Here we use the data from 11/01/2021 to 11/30/2021
# Select the number of cases from 11/01/2021 to 11/30/2021
# create changeable strings to select the columns we want
begin_date = month_trained + '/1/' + year_trained[2:]
end_date = month_trained + '/' + last_day + '/' + year_trained[2:]
confirmed_df1 = confirmed_df.loc[:, 'country']
confirmed_df2 = confirmed_df.loc[:, begin_date: end_date]
confirmed_df = pd.concat([confirmed_df1, confirmed_df2], axis=1)
deaths_df1 = deaths_df.loc[:, 'Country/Region']
deaths_df2 = deaths_df.loc[:, begin_date: end_date]
deaths_df = pd.concat([deaths_df1, deaths_df2], axis=1)

# group the data by countries and calculate the sum
confirmed_by_countries = confirmed_df.groupby('country').sum()
deaths_by_countries = deaths_df.groupby('Country/Region').sum()

# calculate the increase number of confirmed/deaths cases per day (the next column minus the previous column)
# We should pay attention that after the 'diff' function the first column becomes NA
confirmed_by_countries = confirmed_by_countries.diff(axis=1)
deaths_by_countries = deaths_by_countries.diff(axis=1)

# calculate the avg increased confirmed/deaths cases and add it as a new column 'confirmed_mean'
confirmed_by_countries['confirmed_mean'] = confirmed_by_countries.mean(axis=1)
deaths_by_countries['deaths_mean'] = deaths_by_countries.mean(axis=1)

# calculate the total sum confirmed/deaths cases and add it as a new column 'confirmed_sum'
confirmed_by_countries['confirmed_sum'] = confirmed_by_countries.loc[:, begin_date: end_date].sum(axis=1)
deaths_by_countries['deaths_sum'] = deaths_by_countries.loc[:, begin_date: end_date].sum(axis=1)

# concat the confirmed/deaths dataframe together
result = pd.concat([confirmed_by_countries, deaths_by_countries], axis=1)

# We want to calculate 'Case_fatality_ratio' which equals 'deaths_sum'/ 'confirmed_sum'
result['Case_fatality_ratio'] = result.loc[:, 'deaths_sum'] / result.loc[:, 'confirmed_sum']

# derive the first three features: avg confirmed/deaths cases/Case_fatality_ratio per countries in November
df_features = result.loc[:, ['confirmed_mean', 'deaths_mean', 'Case_fatality_ratio']]

# We read another dataset to add the feature 'density'
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
# We use outer join to merge two dataframe and find rows containing NAs which would be the unmatched countries names
# df_features_1 = pd.merge(df_features, density_df, how='outer', on='Countries')
# a = df_features_1[df_features_1.isnull().T.any()]
# print(a)

# change the unmatched country names and drop the countries with NA values
df_features.loc[df_features['Countries'] == "US", "Countries"] = "United States"
df_features.loc[df_features['Countries'] == "Congo (Kinshasa)", "Countries"] = "DR Congo"
df_features.loc[df_features['Countries'] == "Korea, South", "Countries"] = "South Korea"
df_features.loc[df_features['Countries'] == "Taiwan*", "Countries"] = "Taiwan"
df_features.loc[df_features['Countries'] == "Congo (Brazzaville)", "Countries"] = "Republic of the Congo"
df_features.loc[df_features['Countries'] == "Bahamas, The", "Countries"] = "Bahamas"
df_features.loc[df_features['Countries'] == "Gambia, The", "Countries"] = "Gambia"
df_features.loc[df_features['Countries'] == "Burma", "Countries"] = "Myanmar"
df_features.loc[df_features['Countries'] == "Czechia", "Countries"] = "Czech Republic"

# inner join the cleaned dataframes and select rows containing features we want
df_features = pd.merge(df_features, density_df, how='inner', on='Countries')
df_features = df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean', 'Case_fatality_ratio', 'Density']]

# We read another dataset to add the feature 'Continents'
continents_df = pd.read_csv('./data/continents2.csv')
continents_df = continents_df.rename(columns={'name': 'Countries'})
continents_df = continents_df.rename(columns={'region': 'Continents'})

# find the unmatched country names as we did in 'density' feature
# df_features_2 = pd.merge(df_features, continents_df, how='outer', on='Countries')
# df_features_3 = df_features_2.loc[:, ['Countries', 'Continents']]
# df_features_4 = df_features_2.loc[:, ['Countries', 'confirmed_mean']]
# b = df_features_3[df_features_3.isnull().T.any()]
# c = df_features_4[df_features_4.isnull().T.any()]
# print(b)
# print(c)

# change the the unmatched country names and drop the countries with NA values
continents_df.loc[continents_df['Countries'] == "Congo (Democratic Republic Of The)", "Countries"] = "DR Congo"
continents_df.loc[continents_df['Countries'] == "Congo", "Countries"] = "Republic of the Congo"
continents_df.loc[continents_df['Countries'] == "Bosnia And Herzegovina", "Countries"] = "Bosnia and Herzegovina"
continents_df.loc[continents_df['Countries'] == "Brunei Darussalam", "Countries"] = "Brunei"

# inner join the cleaned dataframes and select rows containing features we want
df_features = pd.merge(df_features, continents_df, how='inner', on='Countries')
df_features = \
    df_features.loc[:, ['Countries', 'confirmed_mean', 'deaths_mean', 'Case_fatality_ratio', 'Density', 'Continents']]
# display all the columns for easier and clearer check
pd.set_option('display.max_columns', None)

# We read another dataset to add the feature 'people_fully_vaccinated_per_hundred'
vaccinations = pd.read_csv('./data/owid-covid-data.csv')
vaccinations = vaccinations.loc[:, ['location', 'date', 'people_fully_vaccinated_per_hundred']]
vaccinations.dropna(axis=0, how='any', inplace=True)

# Processing and cleaning the vaccinations dataset are much more tedious than the previous datasets:
# The vaccinations dataset doesn't have the feature 'people_fully_vaccinated_per_hundred' directly.
# The data we want to use to calculate the feature are not recorded on the daily basis.
# And the frequency of updating vaccination records for every country are not the same.
# Since we want to get the data within one month, we have to select the dates on which update the records manually.
# The logic is as follows: first, find the dates that have records on the month we use; second, clean the data and
# leave the last date of the month; third, use vaccination rate on the last day as the vaccination rate for the month

# change the data type into 'string'
vaccinations['date'] = vaccinations['date'].astype('string')

# locate which dates have vaccination records on the month we want to use
day_for_vaccination = year_trained + '-' + month_trained + '-'
vaccinations = vaccinations[vaccinations["date"].str.contains(day_for_vaccination)]

# remove the useless 'month' and left with the dates
vaccinations['date'] = vaccinations['date'].str.replace(day_for_vaccination, '')

# find the last date that has records in the month and use the vaccination rate on that date as the rate for the month
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


# We read another dataset to add the feature 'GNI'
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

# index the Continents. Assign every continent a number
index_continents = {elem: index + 1 for index, elem in enumerate(set(df_features["Continents"]))}
df_features['Continents'] = df_features['Continents'].map(index_continents)

# add a column to index the countries names
df_features = df_features.reset_index()
# index the names as the natural number order
df_features['Countries_indexes'] = range(len(df_features))


# Now we move to the final stage, we train the SVM model and make classifications
X = df_features.loc[:, ['index',
                        'confirmed_mean',
                        'deaths_mean',
                        'Case_fatality_ratio',
                        'Density', 'Continents',
                        'people_fully_vaccinated_per_hundred',
                        'GNI',
                        'Countries_indexes']].values  # convert pandas dataframe into numpy.ndarray

# create the label: to classify whether a country belongs to a country that will have a Covid outbreak
df_features['label'] = 0
# we assign a country which has under 1500 average increased confirmed cases in the given month as 0 (impossible to have
# a Covid outbreak) while a country having over 1500 cases as 1 (possible to have a Covid outbreak)
df_features.loc[df_features['confirmed_mean'] >= 1500, 'label'] = 1
outbreak_countries = df_features.loc[df_features['label'] == 1, 'Countries']
y = df_features.loc[:, 'label'].values

# split the data into tested and trained sets
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = \
    train_test_split(X, y, test_size=0.2, shuffle=False)

# use this to find the optimal parameters for SVR
svm_confirmed = SVC()
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(X_test_confirmed)
print(svm_pred)
# create a confusion matrix to test the model
cm = confusion_matrix(y_test_confirmed, svm_pred)
print(cm)
print(accuracy_score(y_test_confirmed, svm_pred))
# From the confusion matrix, we can see that in the test set, there are 5 countries that are correctly predicted as
# 'covid outbreak'; 23 countries that are correctly predicted as 'no covid outbreak'; 6 countries that are incorrectly
# predicted as 'no covid outbreak'; 0 country that is incorrectly predicted as 'covid outbreak'.
# Precision = 100%, which means that out of those predicted 1, 100% of them are actually 1.
# Recall = 45.5%, which means out of those actually 1, 45.5% of them are predicted as 1 correctly.
# F1 Score = 62.50%
# For our model, it's more important to have a higher Recall rate rather than a higher Precision rate.