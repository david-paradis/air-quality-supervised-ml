# Introduction To Supervised Learning Final Project: Air Quality Prediction 
By David Paradis, for CSCA 5622: Introduction to Machine Learning: Supervised Learning

Link to full notebook :
https://github.com/david-paradis/air-quality-supervised-ml/blob/main/Final%20Project.ipynb

## Introduction
In this notebook, I will study the correlation between Canadian forest fires, climate, and the effect on air quality by predicting air quality indices (labels).

### Problem Statement
Climate experts indicate that Canada is experiencing a warming rate twice that of the global average, primarily due to its northern geographical position. This accelerated warming has led to a surge in forest fire activity, reaching an unprecedented scale in 2023 with numerous devastating wildfires. 

These fires have significantly contributed to the release of smoke particles into the atmosphere, deteriorating air quality in many major cities across both Canada and the United States. The Air Quality Index (AQI), a tool designed to inform the public about daily air quality levels, has frequently indicated unhealthy conditions, forcing the cancellation of numerous outdoor activities in 2023.

With forecasts suggesting an increase in forest fire activity, it is expected that air quality will continue to worsen during the fire season, which spans from May to October, posing ongoing health risks to populations throughout Canada.

![statistic_id553520_area-burned-by-wildfires-in-canada-2000-2023.png](Final Project_files/statistic_id553520_area-burned-by-wildfires-in-canada-2000-2023.png)

#### Connecting wildfires, global warming and air pollution
Human activities like burning fossil fuels elevate greenhouse gas levels, intensifying climate change and increasing wildfire risks. These elevated temperatures and changing weather patterns make forests more susceptible to fires. Wildfires themselves release harmful pollutants like carbon monoxide and particulate matter, exacerbating air pollution and forming a damaging feedback loop. This cycle of worsening climate conditions and frequent, intense wildfires poses significant risks to both human health and the environment.

#### Smoke particles 





















![pm25-wildfire-smoke.png](Final Project_files/pm25-wildfire-smoke.png)


### Objectives
In this study, we will classify whether a given day in a Canadian province is likely to have breathable air, using historical data. This approach allows us to identify patterns in air quality based on past readings, weather data as well as forest fires data. The target value will be the AQI index, from Hazardous to Good, which is a label on a scale that is shown below. 

Numerous scientists have developed models that predict forest fire probabilities based on factors such as location, forest condition, rising temperatures, and decreasing humidity.  This model, while based on historical data, could also be utilized alongside forest fire forecasts to examine potential future impacts on air quality. 



## Data Collection and Description
This study utilizes a combination of three datasets to analyze the impact of wildfire activity on air quality and weather patterns across Canada. The data covers various temporal spans, with specific focus on the year 2023 for weather and air quality metrics, and a broader historical perspective of the last ten years for wildfire occurrences.

### Air Quality Data
Source: OpenAQ platform, which provides real-time air quality information.


Temporal Coverage: Data for the year 2023.

Spatial Coverage: Nationwide, covering all 10 provinces and territories of Canada.

Data Details: Hourly readings of air quality indices such as PM2.5 were averaged to produce a single daily reading per location. This process helps in reducing the noise and focusing on daily trends across different regions.

To collect air quality data from OpenAQ, I've used another notebook that gathered the site information and then queried the API for those sites data for 2023.


[Link to AQ data collection and preprocessing](<AQ data collection.ipynb>)

### Weather Data
Source: Climate Weather Canada's historical data portal.


Temporal Coverage: Data for the year 2023.

Spatial Coverage: Includes comprehensive data from selected weather stations across all provinces and territories. I've selected stations in the most populous and geographically diverse cities for each province, then aggregated the data by province.

Data Details: The dataset includes daily metrics such as temperature, humidity, precipitation, and wind speed. Similar to air quality data, weather readings are averaged by province to simplify the analysis and enhance the clarity of the correlations with wildfire and air quality data.

### Forest Fire Data
Source: Canadian National Fire Database (CNFDB).

Temporal Coverage: Data spanning the last 10 years, providing a historical context to recent trends.

Spatial Coverage: Nationwide

Data Details: Number of forest fires per province per year. No distinction is made between fires that have a burned area of over 200 Ha, for simplicity purpose of gathering this data.



## Exploratory Data Analysis (EDA)



### Library imports



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns


```

### Data Cleaning


### Air quality data


```python
locations_df = pd.read_csv('canada_aq_locations.csv')
locations_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>entity</th>
      <th>country</th>
      <th>sources</th>
      <th>isMobile</th>
      <th>isAnalysis</th>
      <th>parameters</th>
      <th>sensorType</th>
      <th>lastUpdated</th>
      <th>firstUpdated</th>
      <th>measurements</th>
      <th>bounds</th>
      <th>manufacturers</th>
      <th>coordinates.latitude</th>
      <th>coordinates.longitude</th>
      <th>province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>953</td>
      <td>Pickle Lake</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[{'id': 10, 'unit': 'ppm', 'count': 46565, 'av...</td>
      <td>NaN</td>
      <td>2024-04-21T11:00:00+00:00</td>
      <td>2016-03-06T19:00:00+00:00</td>
      <td>46565</td>
      <td>[-90.2175, 54.4494, -90.2175, 54.4494]</td>
      <td>[{'modelName': 'Government Monitor', 'manufact...</td>
      <td>54.449400</td>
      <td>-90.217500</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Wagner2</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[{'id': 7, 'unit': 'ppm', 'count': 33591, 'ave...</td>
      <td>NaN</td>
      <td>2024-04-21T11:00:00+00:00</td>
      <td>2016-03-10T07:00:00+00:00</td>
      <td>67197</td>
      <td>[-114.449722, 53.493889, -114.449722, 53.493889]</td>
      <td>[{'modelName': 'Government Monitor', 'manufact...</td>
      <td>53.493889</td>
      <td>-114.449722</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[{'id': 9, 'unit': 'ppm', 'count': 17146, 'ave...</td>
      <td>NaN</td>
      <td>2024-04-21T11:00:00+00:00</td>
      <td>2016-03-10T07:00:00+00:00</td>
      <td>69668</td>
      <td>[-111.50264, 54.216473, -111.50264, 54.216473]</td>
      <td>[{'modelName': 'Government Monitor', 'manufact...</td>
      <td>54.216473</td>
      <td>-111.502640</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>3</th>
      <td>297</td>
      <td>Steeper</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[{'id': 7, 'unit': 'ppm', 'count': 31206, 'ave...</td>
      <td>NaN</td>
      <td>2024-04-21T11:00:00+00:00</td>
      <td>2016-03-10T07:00:00+00:00</td>
      <td>127847</td>
      <td>[-117.09111, 53.1325, -117.09111, 53.1325]</td>
      <td>[{'modelName': 'Government Monitor', 'manufact...</td>
      <td>53.132500</td>
      <td>-117.091110</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7975</td>
      <td>Vanderhoof Courthous</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[{'id': 1, 'unit': 'µg/m³', 'count': 9064, 'av...</td>
      <td>NaN</td>
      <td>2024-04-21T11:00:00+00:00</td>
      <td>2018-10-04T21:00:00+00:00</td>
      <td>17852</td>
      <td>[-124.0061, 54.0163, -124.0061, 54.0163]</td>
      <td>[{'modelName': 'Government Monitor', 'manufact...</td>
      <td>54.016300</td>
      <td>-124.006100</td>
      <td>BC</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make an average pm25 count per location per day to have a baseline and reduce row count
aq_historical = pd.read_csv('openaq_historical_data.csv')

# Merge with locations to add province column 
locations_df['locationId'] = locations_df['id']
aq_historical = pd.merge(aq_historical, locations_df[['locationId', 'province']], how='left', on='locationId')

aq_historical.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locationId</th>
      <th>location</th>
      <th>parameter</th>
      <th>value</th>
      <th>date</th>
      <th>unit</th>
      <th>coordinates</th>
      <th>country</th>
      <th>city</th>
      <th>isMobile</th>
      <th>isAnalysis</th>
      <th>entity</th>
      <th>sensorType</th>
      <th>province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>{'utc': '2023-12-31T00:00:00+00:00', 'local': ...</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>1</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>{'utc': '2023-12-30T23:00:00+00:00', 'local': ...</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>{'utc': '2023-12-30T22:00:00+00:00', 'local': ...</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>3</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>{'utc': '2023-12-30T21:00:00+00:00', 'local': ...</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>{'utc': '2023-12-30T20:00:00+00:00', 'local': ...</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert date 
def extract_and_convert_utc(date_dict):
    utc_str = date_dict['utc']
    return pd.to_datetime(utc_str)

def convert_to_dict(str):
    str = str.replace("'", '"')
    return json.loads(str)

# Convert date field to dict 
aq_historical['date'] = aq_historical['date'].apply(convert_to_dict)
# Apply the function to the 'date' column to get the UTC datetime
aq_historical['datetime'] = aq_historical['date'].apply(extract_and_convert_utc)

# Extract date from datetime
aq_historical['date'] = aq_historical['datetime'].dt.date

aq_historical.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locationId</th>
      <th>location</th>
      <th>parameter</th>
      <th>value</th>
      <th>date</th>
      <th>unit</th>
      <th>coordinates</th>
      <th>country</th>
      <th>city</th>
      <th>isMobile</th>
      <th>isAnalysis</th>
      <th>entity</th>
      <th>sensorType</th>
      <th>province</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>2023-12-31</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
      <td>2023-12-31 00:00:00+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>2023-12-30</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
      <td>2023-12-30 23:00:00+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>2023-12-30</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
      <td>2023-12-30 22:00:00+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>2023-12-30</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
      <td>2023-12-30 21:00:00+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>297</td>
      <td>Steeper</td>
      <td>pm25</td>
      <td>0.2</td>
      <td>2023-12-30</td>
      <td>µg/m³</td>
      <td>{'latitude': 53.1325, 'longitude': -117.09111}</td>
      <td>CA</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>Governmental Organization</td>
      <td>reference grade</td>
      <td>AB</td>
      <td>2023-12-30 20:00:00+00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Clean up, group data and get mean value for day
grouped_data = aq_historical.groupby([
    'locationId', 'location', 'date', 'parameter', 'province'
]).agg({'value': 'mean'}).reset_index()


grouped_data['date'] = pd.to_datetime(grouped_data['date'])

grouped_data['month'] = pd.to_datetime(grouped_data['date']).dt.month
grouped_data['day_of_month'] = pd.to_datetime(grouped_data['date']).dt.day


grouped_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locationId</th>
      <th>location</th>
      <th>date</th>
      <th>parameter</th>
      <th>province</th>
      <th>value</th>
      <th>month</th>
      <th>day_of_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-01</td>
      <td>pm25</td>
      <td>AB</td>
      <td>24.491304</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-02</td>
      <td>pm25</td>
      <td>AB</td>
      <td>6.675000</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-03</td>
      <td>pm25</td>
      <td>AB</td>
      <td>5.678261</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-04</td>
      <td>pm25</td>
      <td>AB</td>
      <td>5.627273</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-05</td>
      <td>pm25</td>
      <td>AB</td>
      <td>10.283333</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



![AQ-scale.png](Final Project_files/AQ-scale.png)


```python
# Apply scale to generate true labels 
def generate_label(pm25_value):
    if pm25_value <= 12.0:
        return 'Good'
    elif pm25_value <= 35.4:
        return 'Moderate'
    elif pm25_value <= 55.4:
        return 'Unhealthy for Sensitive Groups'
    elif pm25_value <= 150.4:
        return 'Unhealthy'
    elif pm25_value <= 250.4:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'
    
grouped_data['air_quality_label'] = grouped_data['value'].apply(generate_label)

# Optional: Convert air quality labels from categorical to numerical if needed
# This step is optional and depends on your specific needs
quality_mapping = {'Good': 1, 'Moderate': 2, 'Unhealthy for Sensitive Groups': 3, 'Unhealthy': 4, 'Very Unhealthy': 5, 'Hazardous': 6}
grouped_data['air_quality_numerical'] = grouped_data['air_quality_label'].map(quality_mapping)

air_quality_df = grouped_data
air_quality_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locationId</th>
      <th>location</th>
      <th>date</th>
      <th>parameter</th>
      <th>province</th>
      <th>value</th>
      <th>month</th>
      <th>day_of_month</th>
      <th>air_quality_label</th>
      <th>air_quality_numerical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-01</td>
      <td>pm25</td>
      <td>AB</td>
      <td>24.491304</td>
      <td>1</td>
      <td>1</td>
      <td>Moderate</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-02</td>
      <td>pm25</td>
      <td>AB</td>
      <td>6.675000</td>
      <td>1</td>
      <td>2</td>
      <td>Good</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-03</td>
      <td>pm25</td>
      <td>AB</td>
      <td>5.678261</td>
      <td>1</td>
      <td>3</td>
      <td>Good</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-04</td>
      <td>pm25</td>
      <td>AB</td>
      <td>5.627273</td>
      <td>1</td>
      <td>4</td>
      <td>Good</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>287</td>
      <td>St. Lina</td>
      <td>2023-01-05</td>
      <td>pm25</td>
      <td>AB</td>
      <td>10.283333</td>
      <td>1</td>
      <td>5</td>
      <td>Good</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add average values for baseline
average_pm25_df = pd.read_csv('complete_pm25_data.csv')

average_pm25_df.head()

merged_df = pd.merge(air_quality_df, average_pm25_df[['id', 'pm25_average']], left_on='locationId', right_on='id', how='left')

# Drop the extra 'id' column as it is redundant
merged_df.drop('id', axis=1, inplace=True)


# Calculate the mean pm25 value for each province
province_means = merged_df.groupby('province')['pm25_average'].transform('mean')

# Now, fill NaN values in 'average_pm25' with their province's mean
merged_df['pm25_average'] = merged_df['pm25_average'].fillna(province_means)

# Calculate distance to average
merged_df['pm25_distance_to_average'] = merged_df['value'] - merged_df['pm25_average'] 

air_quality_df = merged_df
air_quality_df.sample(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>locationId</th>
      <th>location</th>
      <th>date</th>
      <th>parameter</th>
      <th>province</th>
      <th>value</th>
      <th>month</th>
      <th>day_of_month</th>
      <th>air_quality_label</th>
      <th>air_quality_numerical</th>
      <th>pm25_average</th>
      <th>pm25_distance_to_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43882</th>
      <td>8867</td>
      <td>Sudbury</td>
      <td>2023-06-18</td>
      <td>pm25</td>
      <td>ON</td>
      <td>20.125000</td>
      <td>6</td>
      <td>18</td>
      <td>Moderate</td>
      <td>2</td>
      <td>5.378036</td>
      <td>14.746964</td>
    </tr>
    <tr>
      <th>24374</th>
      <td>1138</td>
      <td>Houston Firehall</td>
      <td>2023-08-08</td>
      <td>pm25</td>
      <td>BC</td>
      <td>3.379167</td>
      <td>8</td>
      <td>8</td>
      <td>Good</td>
      <td>1</td>
      <td>11.686010</td>
      <td>-8.306843</td>
    </tr>
    <tr>
      <th>8939</th>
      <td>491</td>
      <td>Qu�bec - Vieux-Limoi</td>
      <td>2023-03-30</td>
      <td>pm25</td>
      <td>QC</td>
      <td>4.591667</td>
      <td>3</td>
      <td>30</td>
      <td>Good</td>
      <td>1</td>
      <td>7.526511</td>
      <td>-2.934844</td>
    </tr>
    <tr>
      <th>28297</th>
      <td>1506</td>
      <td>Belleville</td>
      <td>2023-05-26</td>
      <td>pm25</td>
      <td>ON</td>
      <td>4.750000</td>
      <td>5</td>
      <td>26</td>
      <td>Good</td>
      <td>1</td>
      <td>6.099971</td>
      <td>-1.349971</td>
    </tr>
    <tr>
      <th>47646</th>
      <td>236027</td>
      <td>Auclair</td>
      <td>2023-06-21</td>
      <td>pm25</td>
      <td>QC</td>
      <td>4.800000</td>
      <td>6</td>
      <td>21</td>
      <td>Good</td>
      <td>1</td>
      <td>3.883310</td>
      <td>0.916690</td>
    </tr>
    <tr>
      <th>40680</th>
      <td>8695</td>
      <td>Marystown/Burin</td>
      <td>2023-10-10</td>
      <td>pm25</td>
      <td>NL</td>
      <td>1.629167</td>
      <td>10</td>
      <td>10</td>
      <td>Good</td>
      <td>1</td>
      <td>4.556462</td>
      <td>-2.927296</td>
    </tr>
    <tr>
      <th>2359</th>
      <td>390</td>
      <td>RED DEER RIVERSIDE D</td>
      <td>2023-09-15</td>
      <td>pm25</td>
      <td>AB</td>
      <td>28.366667</td>
      <td>9</td>
      <td>15</td>
      <td>Moderate</td>
      <td>2</td>
      <td>7.200242</td>
      <td>21.166425</td>
    </tr>
    <tr>
      <th>44729</th>
      <td>230086</td>
      <td>Kitimat Haul Road</td>
      <td>2023-06-03</td>
      <td>pm25</td>
      <td>BC</td>
      <td>5.916667</td>
      <td>6</td>
      <td>3</td>
      <td>Good</td>
      <td>1</td>
      <td>3.622010</td>
      <td>2.294657</td>
    </tr>
    <tr>
      <th>44126</th>
      <td>8867</td>
      <td>Sudbury</td>
      <td>2024-02-17</td>
      <td>pm25</td>
      <td>ON</td>
      <td>2.869565</td>
      <td>2</td>
      <td>17</td>
      <td>Good</td>
      <td>1</td>
      <td>5.378036</td>
      <td>-2.508471</td>
    </tr>
    <tr>
      <th>3803</th>
      <td>447</td>
      <td>Saint-Hilaire-de-Dor</td>
      <td>2023-04-11</td>
      <td>pm25</td>
      <td>QC</td>
      <td>10.095833</td>
      <td>4</td>
      <td>11</td>
      <td>Good</td>
      <td>1</td>
      <td>4.550745</td>
      <td>5.545089</td>
    </tr>
    <tr>
      <th>11380</th>
      <td>506</td>
      <td>Tingwick</td>
      <td>2024-01-15</td>
      <td>pm25</td>
      <td>QC</td>
      <td>2.904167</td>
      <td>1</td>
      <td>15</td>
      <td>Good</td>
      <td>1</td>
      <td>5.471960</td>
      <td>-2.567793</td>
    </tr>
    <tr>
      <th>36540</th>
      <td>8567</td>
      <td>FORT ST JOHN LEARNIN</td>
      <td>2023-05-09</td>
      <td>pm25</td>
      <td>BC</td>
      <td>55.390909</td>
      <td>5</td>
      <td>9</td>
      <td>Unhealthy for Sensitive Groups</td>
      <td>3</td>
      <td>6.867819</td>
      <td>48.523090</td>
    </tr>
    <tr>
      <th>60810</th>
      <td>1275802</td>
      <td>Tiverton</td>
      <td>2024-02-09</td>
      <td>pm25</td>
      <td>ON</td>
      <td>9.434783</td>
      <td>2</td>
      <td>9</td>
      <td>Good</td>
      <td>1</td>
      <td>14.000000</td>
      <td>-4.565217</td>
    </tr>
    <tr>
      <th>3615</th>
      <td>439</td>
      <td>Saint-Faustin-Lac-Ca</td>
      <td>2023-11-23</td>
      <td>pm25</td>
      <td>QC</td>
      <td>3.558333</td>
      <td>11</td>
      <td>23</td>
      <td>Good</td>
      <td>1</td>
      <td>5.045388</td>
      <td>-1.487055</td>
    </tr>
    <tr>
      <th>18555</th>
      <td>748</td>
      <td>CHARLOTTETOWN</td>
      <td>2023-06-06</td>
      <td>pm25</td>
      <td>PE</td>
      <td>2.304167</td>
      <td>6</td>
      <td>6</td>
      <td>Good</td>
      <td>1</td>
      <td>6.927930</td>
      <td>-4.623763</td>
    </tr>
    <tr>
      <th>9093</th>
      <td>491</td>
      <td>Qu�bec - Vieux-Limoi</td>
      <td>2023-09-01</td>
      <td>pm25</td>
      <td>QC</td>
      <td>7.687500</td>
      <td>9</td>
      <td>1</td>
      <td>Good</td>
      <td>1</td>
      <td>7.526511</td>
      <td>0.160989</td>
    </tr>
    <tr>
      <th>51921</th>
      <td>268739</td>
      <td>Golden Helipad</td>
      <td>2023-11-02</td>
      <td>pm25</td>
      <td>BC</td>
      <td>51.383333</td>
      <td>11</td>
      <td>2</td>
      <td>Unhealthy for Sensitive Groups</td>
      <td>3</td>
      <td>11.472739</td>
      <td>39.910594</td>
    </tr>
    <tr>
      <th>27396</th>
      <td>1364</td>
      <td>Hope</td>
      <td>2024-01-07</td>
      <td>pm25</td>
      <td>BC</td>
      <td>3.345833</td>
      <td>1</td>
      <td>7</td>
      <td>Good</td>
      <td>1</td>
      <td>6.434420</td>
      <td>-3.088587</td>
    </tr>
    <tr>
      <th>22989</th>
      <td>988</td>
      <td>Langdale Elementary</td>
      <td>2024-01-29</td>
      <td>pm25</td>
      <td>BC</td>
      <td>2.587500</td>
      <td>1</td>
      <td>29</td>
      <td>Good</td>
      <td>1</td>
      <td>5.523749</td>
      <td>-2.936249</td>
    </tr>
    <tr>
      <th>62341</th>
      <td>1289474</td>
      <td>North Bay</td>
      <td>2024-01-03</td>
      <td>pm25</td>
      <td>ON</td>
      <td>4.913043</td>
      <td>1</td>
      <td>3</td>
      <td>Good</td>
      <td>1</td>
      <td>3.000000</td>
      <td>1.913043</td>
    </tr>
    <tr>
      <th>3425</th>
      <td>439</td>
      <td>Saint-Faustin-Lac-Ca</td>
      <td>2023-05-11</td>
      <td>pm25</td>
      <td>QC</td>
      <td>13.112500</td>
      <td>5</td>
      <td>11</td>
      <td>Moderate</td>
      <td>2</td>
      <td>5.045388</td>
      <td>8.067112</td>
    </tr>
    <tr>
      <th>43577</th>
      <td>8817</td>
      <td>Sarnia</td>
      <td>2024-04-05</td>
      <td>pm25</td>
      <td>ON</td>
      <td>1.458333</td>
      <td>4</td>
      <td>5</td>
      <td>Good</td>
      <td>1</td>
      <td>7.395996</td>
      <td>-5.937662</td>
    </tr>
    <tr>
      <th>21994</th>
      <td>836</td>
      <td>Abbotsford Airport</td>
      <td>2023-06-22</td>
      <td>pm25</td>
      <td>BC</td>
      <td>5.287500</td>
      <td>6</td>
      <td>22</td>
      <td>Good</td>
      <td>1</td>
      <td>5.909470</td>
      <td>-0.621970</td>
    </tr>
    <tr>
      <th>37959</th>
      <td>8642</td>
      <td>KITIMAT WHITESAIL</td>
      <td>2024-01-04</td>
      <td>pm25</td>
      <td>BC</td>
      <td>4.444444</td>
      <td>1</td>
      <td>4</td>
      <td>Good</td>
      <td>1</td>
      <td>4.000465</td>
      <td>0.443980</td>
    </tr>
    <tr>
      <th>40648</th>
      <td>8695</td>
      <td>Marystown/Burin</td>
      <td>2023-09-08</td>
      <td>pm25</td>
      <td>NL</td>
      <td>3.212500</td>
      <td>9</td>
      <td>8</td>
      <td>Good</td>
      <td>1</td>
      <td>4.556462</td>
      <td>-1.343962</td>
    </tr>
  </tbody>
</table>
</div>



### Climate data
The goal is to generate provincial weather average for each day, and add to the air quality data


```python
climate_df = pd.read_csv('climate-daily.csv')


# Ensure the date column is a datetime type
climate_df['LOCAL_DATE'] = pd.to_datetime(climate_df['LOCAL_DATE'])

# Extract year and month from the date for grouping
climate_df['YEAR'] = climate_df['LOCAL_DATE'].dt.year
climate_df['MONTH'] = climate_df['LOCAL_DATE'].dt.month
climate_df['DAY_OF_MONTH'] = climate_df['LOCAL_DATE'].dt.day
climate_df = climate_df[climate_df['YEAR'] == 2023]
# Select relevant features
relevant_data = climate_df[['PROVINCE_CODE', 'YEAR', 'MONTH', 'DAY_OF_MONTH', 'MAX_TEMPERATURE', 'MIN_TEMPERATURE', 
                      'MAX_REL_HUMIDITY', 'MIN_REL_HUMIDITY', 'TOTAL_PRECIPITATION', 
                      'SPEED_MAX_GUST']]

print(relevant_data['PROVINCE_CODE'].unique())

# Calculate averages and sums by province, year, and month
daily_aggregates = relevant_data.groupby(['PROVINCE_CODE', 'YEAR', 'MONTH', 'DAY_OF_MONTH']).agg({
    'MAX_TEMPERATURE': 'mean',  # Average of maximum temperatures
    'MIN_TEMPERATURE': 'mean',  # Average of minimum temperatures
    'MAX_REL_HUMIDITY': 'mean',  # Average of maximum relative humidity
    'MIN_REL_HUMIDITY': 'mean',  # Average of minimum relative humidity
    'TOTAL_PRECIPITATION': 'sum',  # Total precipitation
    'SPEED_MAX_GUST': 'mean'  # Average of maximum wind gust speeds
}).reset_index()

daily_aggregates.sample(25)

```

    ['AB' 'BC' 'MB' 'NB' 'NL' 'NS' 'NT' 'NU' 'ON' 'PE' 'QC' 'SK' 'YT']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROVINCE_CODE</th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>DAY_OF_MONTH</th>
      <th>MAX_TEMPERATURE</th>
      <th>MIN_TEMPERATURE</th>
      <th>MAX_REL_HUMIDITY</th>
      <th>MIN_REL_HUMIDITY</th>
      <th>TOTAL_PRECIPITATION</th>
      <th>SPEED_MAX_GUST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1485</th>
      <td>NL</td>
      <td>2023</td>
      <td>1</td>
      <td>27</td>
      <td>7.50</td>
      <td>-2.00</td>
      <td>98.0</td>
      <td>71.0</td>
      <td>11.4</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>206</th>
      <td>AB</td>
      <td>2023</td>
      <td>7</td>
      <td>26</td>
      <td>19.45</td>
      <td>7.95</td>
      <td>91.0</td>
      <td>52.0</td>
      <td>8.4</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>QC</td>
      <td>2023</td>
      <td>8</td>
      <td>5</td>
      <td>23.95</td>
      <td>14.10</td>
      <td>93.5</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>NB</td>
      <td>2023</td>
      <td>1</td>
      <td>18</td>
      <td>1.00</td>
      <td>-1.10</td>
      <td>96.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>AB</td>
      <td>2023</td>
      <td>4</td>
      <td>3</td>
      <td>2.95</td>
      <td>-4.55</td>
      <td>93.5</td>
      <td>51.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>MB</td>
      <td>2023</td>
      <td>11</td>
      <td>4</td>
      <td>1.30</td>
      <td>-7.40</td>
      <td>92.0</td>
      <td>76.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>MB</td>
      <td>2023</td>
      <td>10</td>
      <td>19</td>
      <td>17.10</td>
      <td>5.90</td>
      <td>100.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>354</th>
      <td>AB</td>
      <td>2023</td>
      <td>12</td>
      <td>21</td>
      <td>6.10</td>
      <td>-4.95</td>
      <td>93.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>MB</td>
      <td>2023</td>
      <td>10</td>
      <td>31</td>
      <td>-4.40</td>
      <td>-11.70</td>
      <td>93.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>4670</th>
      <td>YT</td>
      <td>2023</td>
      <td>10</td>
      <td>19</td>
      <td>7.50</td>
      <td>-1.70</td>
      <td>92.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>4068</th>
      <td>SK</td>
      <td>2023</td>
      <td>2</td>
      <td>24</td>
      <td>-20.10</td>
      <td>-29.90</td>
      <td>77.5</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4618</th>
      <td>YT</td>
      <td>2023</td>
      <td>8</td>
      <td>28</td>
      <td>27.50</td>
      <td>11.80</td>
      <td>78.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2426</th>
      <td>NT</td>
      <td>2023</td>
      <td>8</td>
      <td>26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2799</th>
      <td>NU</td>
      <td>2023</td>
      <td>9</td>
      <td>3</td>
      <td>5.70</td>
      <td>3.00</td>
      <td>100.0</td>
      <td>70.0</td>
      <td>0.0</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>1585</th>
      <td>NL</td>
      <td>2023</td>
      <td>5</td>
      <td>7</td>
      <td>10.70</td>
      <td>0.40</td>
      <td>100.0</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>3049</th>
      <td>ON</td>
      <td>2023</td>
      <td>5</td>
      <td>11</td>
      <td>26.35</td>
      <td>9.20</td>
      <td>68.5</td>
      <td>28.5</td>
      <td>0.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>4026</th>
      <td>SK</td>
      <td>2023</td>
      <td>1</td>
      <td>13</td>
      <td>-8.60</td>
      <td>-15.90</td>
      <td>90.0</td>
      <td>83.5</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4206</th>
      <td>SK</td>
      <td>2023</td>
      <td>7</td>
      <td>12</td>
      <td>23.75</td>
      <td>10.60</td>
      <td>93.0</td>
      <td>44.5</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>MB</td>
      <td>2023</td>
      <td>4</td>
      <td>3</td>
      <td>-2.60</td>
      <td>-10.90</td>
      <td>93.0</td>
      <td>62.0</td>
      <td>0.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>2543</th>
      <td>NT</td>
      <td>2023</td>
      <td>12</td>
      <td>21</td>
      <td>-10.80</td>
      <td>-20.10</td>
      <td>91.0</td>
      <td>84.0</td>
      <td>0.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1887</th>
      <td>NS</td>
      <td>2023</td>
      <td>3</td>
      <td>5</td>
      <td>2.00</td>
      <td>-9.80</td>
      <td>78.0</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4686</th>
      <td>YT</td>
      <td>2023</td>
      <td>11</td>
      <td>4</td>
      <td>1.00</td>
      <td>-2.20</td>
      <td>98.0</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4233</th>
      <td>SK</td>
      <td>2023</td>
      <td>8</td>
      <td>8</td>
      <td>24.15</td>
      <td>8.20</td>
      <td>94.0</td>
      <td>31.5</td>
      <td>0.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>MB</td>
      <td>2023</td>
      <td>9</td>
      <td>26</td>
      <td>24.20</td>
      <td>10.60</td>
      <td>99.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2531</th>
      <td>NT</td>
      <td>2023</td>
      <td>12</td>
      <td>9</td>
      <td>-9.70</td>
      <td>-12.50</td>
      <td>92.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>37.0</td>
    </tr>
  </tbody>
</table>
</div>



### Forest fire data
Data is not easily available, but we do have access to the number of forest fires per province for the year 2023, as well as a 10 year average.


```python
fire_data = pd.read_csv('fires_by_province.csv')

fire_data.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)


fire_data = pd.melt(fire_data, id_vars=['Year'], var_name='province', value_name='nb_fires')
fire_data['province'] = fire_data['province'].str.strip()  # Ensure matching province names

# Split data into separate frames for each year type to facilitate specific merging
fires_current = fire_data[fire_data['Year'] == '23'].drop(columns=['Year']).rename(columns={'nb_fires': 'nb_fires'})
fires_avg = fire_data[fire_data['Year'] == '10-yr avg'].drop(columns=['Year']).rename(columns={'nb_fires': 'nb_fires_average'})

print(fires_avg)
# Merge these frames on the 'province' to unify current and average fire data
fires_data = pd.merge(fires_current, fires_avg, on='province', how='outer')


```

       province  nb_fires_average
    1        BC              1392
    3        YT               116
    5        AB              1181
    7        NT               197
    9        SK               399
    11       MB               342
    13       ON               675
    15       QC               458
    17       NL                96
    19       NB               246
    21       NS               172
    23       PE                 6
    25    Total              5380


### Data Visualization



```python
# Filter data for 2023 only
filtered_df = air_quality_df[(air_quality_df['date'] >= '2023-01-01') & (air_quality_df['date'] <= '2024-01-01')]

filtered_df = filtered_df.sort_values('date')



# Aggregate by province and month
filtered_df = filtered_df.groupby(['province', 'month']).agg({'air_quality_numerical': 'mean'}).reset_index()

# Pivot the table for the plot
df_pivot = filtered_df.pivot("month", "province", "air_quality_numerical")

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_pivot, markers=True, dashes=False)
plt.title('Average Air Quality Value by Month and Province (2023)\nHigher values indicate unhealthier air')
plt.xlabel('Month')
plt.ylabel('Average Air Quality Value')
plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.legend(title='Province')
plt.tight_layout()
plt.show()
```


    
![png](Final%20Project_files/Final%20Project_22_0.png)
    



```python
# Plot the average pm25 level for baseline by province and month
filtered_df = air_quality_df[(air_quality_df['date'] >= '2023-01-01') & (air_quality_df['date'] <= '2024-01-01')]

filtered_df = filtered_df.groupby(['province', 'month']).agg({'value': 'mean'}).reset_index()

# Pivot the table for the plot
df_pivot = filtered_df.pivot("month", "province", "value")

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_pivot, markers=True, dashes=False)
plt.title('Average PM2.5 by Month and Province (2023)\nHigher values indicate unhealthier air')
plt.xlabel('Month')
plt.ylabel('Average PM2.5 Value')
plt.axhline(y=12, color='g', linestyle='--', label='Healthy level')

plt.axhline(y=35.5, color='r', linestyle='--', label='Unhealthy for Sensitive Groups')
plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.legend(title='Province')
plt.tight_layout()
plt.show()
```


    
![png](Final%20Project_files/Final%20Project_23_0.png)
    



```python
# Plot bar chart for labels from May to October
filtered_df = air_quality_df[(air_quality_df['date'] >= '2023-05-01') & (air_quality_df['date'] < '2023-10-01')]

# Group and aggregate data
grouped_df = filtered_df.groupby(['province', 'month', 'air_quality_label']).size()

# Use unstack to pivot the last level of the index to columns, fill missing values with 0
pivot_df = grouped_df.unstack(level=-1, fill_value=0)

# Reset index to make 'province' and 'month' regular columns for easier plotting
pivot_df.reset_index(inplace=True)

# Define the order of air quality labels from best to worst
# This is just an example, modify it according to your actual labels and desired order
labels_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
color_map = {
    'Good': 'blue',
    'Moderate': 'green',
    'Unhealthy for Sensitive Groups': 'yellow',
    'Unhealthy': 'red',
    'Very Unhealthy': 'purple',
    'Hazardous': 'brown'
}
# Filter to include only the columns that exist in your DataFrame to avoid KeyErrors
columns_order = [label for label in color_map.keys() if label in pivot_df.columns]

# Reorder the DataFrame columns according to this order
pivot_df = pivot_df[['province', 'month'] + columns_order]  # Include other necessary columns like 'province' and 'month'

# Plot each province's data in a separate subplot
unique_provinces = pivot_df['province'].unique()

# Create figure with appropriate size
plt.figure(figsize=(15, 5 * len(unique_provinces)))

for i, province in enumerate(unique_provinces, 1):
    ax = plt.subplot(len(unique_provinces), 1, i)
    # Filter data for the current province
    province_data = pivot_df[pivot_df['province'] == province]
    colors = [color_map[label] for label in province_data.columns[2:] if label in color_map]

    # Plot as a bar chart
    province_data.plot(kind='bar', x='month', stacked=True, ax=ax, legend=(i == 1), color=colors)
    
    ax.set_title(f'Air Quality in {province}')
    ax.set_ylabel('Count')
    ax.set_xlabel('Month')

plt.tight_layout()
plt.show()
```


    
![png](Final%20Project_files/Final%20Project_24_0.png)
    


Demonstrate that 2023 was warmer than usual across Canada.


```python
# Plot weather trends by province by month
pivot_data = climate_df.pivot_table(index='PROVINCE_CODE', columns='MONTH', values='MAX_TEMPERATURE', aggfunc='mean')
province_codes = daily_aggregates['PROVINCE_CODE'].unique()
print(province_codes)
# Convert the 'Community' in 'july_highs' to province codes to match 'pivot_data'
province_mapping = {
    'Charlottetown, PE': 'PE',
    'Edmonton, AB': 'AB',
    'Fredericton, NB': 'NB',
    'Halifax, NS': 'NS',
    'Iqaluit, NU': 'NU',
    'Quebec City, QC': 'QC',
    'Regina, SK': 'SK',
    'St. John\'s, NL': 'NL',
    'Toronto, ON': 'ON',
    'Victoria, BC': 'BC',
    'Whitehorse, YT': 'YT',
    'Winnipeg, MB': 'MB',
    'Yellowknife, NT': 'NT'
}

# Data handpicked from Climate Weather Canada 
data = {
    'Community': ['Charlottetown, PE', 'Edmonton, AB', 'Fredericton, NB', 'Halifax, NS', 'Iqaluit, NU',
                  'Quebec City, QC', 'Regina, SK', 'St. John\'s, NL', 'Toronto, ON', 'Victoria, BC',
                  'Whitehorse, YT', 'Winnipeg, MB', 'Yellowknife, NT'],
    'July_Avg_High_C': [23.3, 23.1, 25.5, 23.8, 12.3, 25.0, 25.8, 20.7, 27.1, 22.4, 20.6, 25.9, 21.3]
}

# Create DataFrame
july_highs = pd.DataFrame(data)

july_highs['PROVINCE_CODE'] = july_highs['Community'].map(province_mapping)

palette = sns.color_palette("tab10", n_colors=len(pivot_data.index))

# Map each province to a specific color
color_map = {province: color for province, color in zip(pivot_data.index, palette)}

# Set the plot size
plt.figure(figsize=(14, 10))

# Plotting each province's monthly data
for province in pivot_data.index:
    sns.lineplot(data=pivot_data.loc[province], label=province, color=color_map[province])

# Add July average high temperatures for each capital using the same color map
for index, row in july_highs.iterrows():
    province_code = row['PROVINCE_CODE']
    july_temp = row['July_Avg_High_C']
    if province_code in color_map:
        plt.scatter(7, july_temp, color=color_map[province_code], s=100, zorder=5)  # Month 'July' is the 7th month

plt.title('Monthly Max Temperatures Across Provinces with Capitals\' July Averages')
plt.xlabel('Month')
plt.ylabel('Max Temperature (°C)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Province')
plt.grid(True)

plt.show()
```

    ['AB' 'BC' 'MB' 'NB' 'NL' 'NS' 'NT' 'NU' 'ON' 'PE' 'QC' 'SK' 'YT']



    
![png](Final%20Project_files/Final%20Project_26_1.png)
    



```python
# Filter the data for July 2023
july_2023_data = climate_df[(climate_df['MONTH'] == 7) & (climate_df['YEAR'] == 2023)]

# Calculate the average max temperature for July 2023 by province
july_2023_averages = july_2023_data.groupby('PROVINCE_CODE')['MAX_TEMPERATURE'].mean()

# Calculate differences from the general July averages
july_highs.set_index('PROVINCE_CODE', inplace=True)
differences = july_2023_averages - july_highs['July_Avg_High_C']

# Reset index for plotting
differences = differences.reset_index()
differences.columns = ['PROVINCE_CODE', 'Temperature_Difference']

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='PROVINCE_CODE', y='Temperature_Difference', data=differences)
plt.title('Difference Between General July Average and July 2023 Average Temperatures')
plt.xlabel('Province Code')
plt.ylabel('Temperature Difference (°C)')
plt.axhline(0, color='black', linewidth=0.8)  # Adds a line at zero difference for reference
plt.xticks(rotation=45)

plt.show()
```


    
![png](Final%20Project_files/Final%20Project_27_0.png)
    


## Model Building and Training
### Choice of Model
For this study, the RandomForestClassifier from scikit-learn was selected as the primary model for predicting outcomes based on our labeled data. This decision was influenced by several factors:

Robustness to Overfitting: Random forests are inherently robust against overfitting, which makes them suitable for our dataset with its complex and potentially noisy features derived from multiple sources (air quality indices, weather data, and wildfire metrics).

Ability to Handle Non-linear Relationships: Given the complex nature of environmental data, where relationships between features may not be linear, the random forest's ability to handle non-linearity without explicit model tuning is a significant advantage.

Feature Importance: This model provides straightforward insights into which features are most influential in predicting outcomes, which is crucial for understanding the impact of different environmental factors on air quality and wildfire occurrences.

Flexibility and Scalability: The model is known for its flexibility in handling various types of data and its scalability, which is beneficial for analyzing datasets that could expand in future studies.

### Configuration of the Model
The RandomForestClassifier was configured with specific hyperparameters tailored to optimize performance for our dataset:

n_estimators: Set to 100, indicating the model will use 100 trees. Increasing the number of trees can lead to better performance but requires more computational power and time.

max_depth: Set to None, allowing the trees to expand until all leaves are pure or contain less than min_samples_split samples.

min_samples_split: Set to 2, the smallest number to split a node. Smaller values prevent the model from generalizing well, while larger values might lead to underfitting.

max_features: Set to 'auto', which uses the square root of the number of features at each split, providing a good balance between prediction accuracy and model complexity.

### Model Training
The model was trained using a training dataset comprising 70% of the total data, with the remaining 30% reserved for testing and validation. This split ensures that the model is tested on unseen data, providing a fair assessment of its predictive power.

### Model Validation
Validation was conducted through k-fold cross-validation (with k=10), providing a robust estimate of the model's performance across different subsets of the data. This method helps in identifying any potential issues with model stability and generalizability.


### Preprocessing




```python
from sklearn.preprocessing import OneHotEncoder

# Feature engineering
preprocessed_data = air_quality_df
preprocessed_data['date'] = pd.to_datetime(air_quality_df['date'])
# Keep only fire season 2023
preprocessed_data = preprocessed_data[(preprocessed_data['date'] >= '2023-05-01') & (preprocessed_data['date'] < '2023-10-01')]

weather_df = climate_df
weather_df['date'] = pd.to_datetime(weather_df['YEAR'].astype(str) + '-' + weather_df['MONTH'].astype(str) + '-' + weather_df['DAY_OF_MONTH'].astype(str))

# Rename the 'PROVINCE_CODE' in weather_df to 'province' for a consistent merge key
weather_df.rename(columns={'PROVINCE_CODE': 'province'}, inplace=True)

# Merge air quality measurements and weather readings averages
merged_df = pd.merge(preprocessed_data, weather_df, how='left', on=['date', 'province'])
# Merge fire data
merged_df = pd.merge(merged_df, fires_data, on='province', how='left')

final_df = merged_df[['date', 'day_of_month', 'month', 'air_quality_numerical', 
                        'MAX_TEMPERATURE', 'MIN_REL_HUMIDITY', 'province', 'pm25_distance_to_average',
                        'nb_fires', 'nb_fires_average']]

# Encode province code using one hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(final_df[['province']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['province']))
final_df = pd.concat([final_df, encoded_df], axis=1)
final_df.drop(['province'], axis=1, inplace=True)

final_df.set_index('date', inplace=True)

# Interpolate missing values using the time method
final_df.interpolate(method='time', inplace=True)

final_df.sample(20)

```

    /opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_of_month</th>
      <th>month</th>
      <th>air_quality_numerical</th>
      <th>MAX_TEMPERATURE</th>
      <th>MIN_REL_HUMIDITY</th>
      <th>pm25_distance_to_average</th>
      <th>nb_fires</th>
      <th>nb_fires_average</th>
      <th>province_AB</th>
      <th>province_BC</th>
      <th>province_MB</th>
      <th>province_NB</th>
      <th>province_NL</th>
      <th>province_NS</th>
      <th>province_ON</th>
      <th>province_PE</th>
      <th>province_QC</th>
      <th>province_SK</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-06-30</th>
      <td>30</td>
      <td>6</td>
      <td>1</td>
      <td>22.0</td>
      <td>53.0</td>
      <td>-1.759420</td>
      <td>2217</td>
      <td>1392</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-08-26</th>
      <td>26</td>
      <td>8</td>
      <td>2</td>
      <td>23.4</td>
      <td>56.0</td>
      <td>18.468463</td>
      <td>2217</td>
      <td>1392</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-06-12</th>
      <td>12</td>
      <td>6</td>
      <td>2</td>
      <td>29.6</td>
      <td>38.0</td>
      <td>9.423048</td>
      <td>690</td>
      <td>458</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-06-04</th>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>-3.225395</td>
      <td>731</td>
      <td>675</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-07-17</th>
      <td>17</td>
      <td>7</td>
      <td>3</td>
      <td>22.1</td>
      <td>60.0</td>
      <td>35.719100</td>
      <td>952</td>
      <td>1181</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-28</th>
      <td>28</td>
      <td>5</td>
      <td>1</td>
      <td>25.7</td>
      <td>24.0</td>
      <td>-2.723342</td>
      <td>952</td>
      <td>1181</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-07-31</th>
      <td>31</td>
      <td>7</td>
      <td>1</td>
      <td>28.2</td>
      <td>19.0</td>
      <td>0.695255</td>
      <td>952</td>
      <td>1181</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-13</th>
      <td>13</td>
      <td>5</td>
      <td>1</td>
      <td>21.9</td>
      <td>53.0</td>
      <td>0.358697</td>
      <td>2217</td>
      <td>1392</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-06</th>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>17.9</td>
      <td>29.0</td>
      <td>1.099195</td>
      <td>731</td>
      <td>675</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-07-27</th>
      <td>27</td>
      <td>7</td>
      <td>2</td>
      <td>29.6</td>
      <td>51.0</td>
      <td>6.794686</td>
      <td>731</td>
      <td>675</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-10</th>
      <td>10</td>
      <td>5</td>
      <td>2</td>
      <td>20.6</td>
      <td>39.0</td>
      <td>6.256664</td>
      <td>952</td>
      <td>1181</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-06-08</th>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>15.0</td>
      <td>76.0</td>
      <td>-4.080677</td>
      <td>690</td>
      <td>458</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-08-10</th>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>24.8</td>
      <td>39.0</td>
      <td>-1.985003</td>
      <td>952</td>
      <td>1181</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-10</th>
      <td>10</td>
      <td>5</td>
      <td>1</td>
      <td>23.0</td>
      <td>25.0</td>
      <td>-0.528418</td>
      <td>690</td>
      <td>458</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-01</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>9.9</td>
      <td>76.0</td>
      <td>-3.962055</td>
      <td>690</td>
      <td>458</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-09-24</th>
      <td>24</td>
      <td>9</td>
      <td>1</td>
      <td>19.9</td>
      <td>47.0</td>
      <td>-2.082021</td>
      <td>2217</td>
      <td>1392</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-06-26</th>
      <td>26</td>
      <td>6</td>
      <td>1</td>
      <td>18.7</td>
      <td>87.0</td>
      <td>1.276446</td>
      <td>216</td>
      <td>172</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-08-28</th>
      <td>28</td>
      <td>8</td>
      <td>1</td>
      <td>22.8</td>
      <td>47.0</td>
      <td>-14.625000</td>
      <td>731</td>
      <td>675</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-05-04</th>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>19.1</td>
      <td>68.0</td>
      <td>3.247390</td>
      <td>2217</td>
      <td>1392</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2023-09-17</th>
      <td>17</td>
      <td>9</td>
      <td>1</td>
      <td>24.1</td>
      <td>50.0</td>
      <td>-0.021875</td>
      <td>690</td>
      <td>458</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# See if there are correlated features
correlations = final_df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix')
plt.show()
```


    
![png](Final%20Project_files/Final%20Project_31_0.png)
    



### Model Training


```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data
data = final_df 

X_train, X_test, y_train, y_test = train_test_split(data.drop('air_quality_numerical', axis=1), data['air_quality_numerical'], test_size=0.3, random_state=42)

# Create model
clf = RandomForestClassifier(
    n_estimators=100,    # Number of trees in the forest
    max_depth=None,      # Max depth of each tree. 'None' means nodes are expanded until all leaves are pure or contain less than min_samples_split samples
    min_samples_split=2, # Minimum number of samples required to split an internal node
    max_features='auto', # Number of features to consider when looking for the best split: "auto" means sqrt(n_features)
    random_state=42      # Ensures a deterministic behavior when fitting
)

# Fit the model on the training data
clf.fit(X_train, y_train)

```




    RandomForestClassifier(random_state=42)



## Results 


```python

train_preds = clf.predict(X_train)

# Check model performance on the training set
print(f"Training Accuracy: {accuracy_score(y_train, train_preds):.2f}")

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
print(f"Cross-Validated Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.2f}")

# Analyze class distribution
print("Class distribution in training set:")
print(y_train.value_counts(normalize=True))

y_pred = clf.predict(X_test)

# Generate the confusion matrix
labels = [1, 2, 3, 4, 5, 6]

cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()
```

    Training Accuracy: 1.00
    Cross-Validated Scores: [0.93723029 0.94468419 0.93369949 0.9376226  0.92703021 0.9376226
     0.92977638 0.93720565 0.93210361 0.9399529 ]
    Mean CV Score: 0.94
    Class distribution in training set:
    1    0.685291
    2    0.233374
    3    0.036215
    4    0.035391
    5    0.006238
    6    0.003492
    Name: air_quality_numerical, dtype: float64



    
![png](Final%20Project_files/Final%20Project_35_1.png)
    


We obtain a very good accuracy at around 0.94, with a confusion matrix demonstrating that the model can correctly predict across all 6 labels.


```python
feature_importances = clf.feature_importances_

# Get the feature names from X_test
feature_names = X_test.columns

# Create a DataFrame to hold the feature importances
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})

# Sort the DataFrame to see the most important features at the top
importances_df = importances_df.sort_values(by='importance', ascending=False)

# Plot the feature importances for better visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances_df.head(10))
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

```


    
![png](Final%20Project_files/Final%20Project_37_0.png)
    


As we can see, the PM2.5 distance to the average contributes very significantly to the model. Therefore, it is safe to conclude that the wildfires have a great incidency on the air quality in a province. It is interesting to note that the maximum temperature as well as the relative humidity also contribute, which is understandable given that fires happen when temperatures are warm and relative humidity low.

### Predict future dates

To test the model, we generate random synthetic data with a relatively pessimistic value. Temperatures are between 18 and 35 degrees Celsius, the PM2.5 distance from the average is between -10 and +35, and the number of fires is between 700 and 1500 which is higher than the past 10 years average. 


```python
# Constants
provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK']
months_fire_season = [6, 7, 8, 9]  # June to September
np.random.seed(42)


def generate_random_row():
    # Generate random data
    data = {
        'day_of_month': np.random.randint(1, 30),  # Random day of the month
        'month': np.random.choice(months_fire_season),  # Random month in fire season
        'MAX_TEMPERATURE': np.random.uniform(18, 35),  # Simulate temperature between 18°C and 35°C
        'MIN_REL_HUMIDITY': np.random.uniform(20, 60),  # Simulate humidity between 20% and 60%
        'pm25_distance_to_average': np.random.uniform(-10, 35),  # PM2.5 distance to average
        'nb_fires': np.random.randint(700, 1500),  # Number of fires, simulate a strong season
        'nb_fires_average': np.random.randint(500, 1400)  # Average number of fires
    }

    # Add a column for each province as one-hot encoding
    row_province = np.random.choice(provinces)
    for province in provinces:
        data[f'province_{province}'] = 1 if province == row_province else 0

    return data

# Create DataFrame
future_data = []
for i in range(30): 
    row = generate_random_row()
    future_data.append(row)
future_data_df = pd.DataFrame(future_data)

predicted_labels = clf.predict(future_data_df)
future_data_df['predicted_air_quality'] = predicted_labels

future_data_df.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_of_month</th>
      <th>month</th>
      <th>MAX_TEMPERATURE</th>
      <th>MIN_REL_HUMIDITY</th>
      <th>pm25_distance_to_average</th>
      <th>nb_fires</th>
      <th>nb_fires_average</th>
      <th>province_AB</th>
      <th>province_BC</th>
      <th>province_MB</th>
      <th>province_NB</th>
      <th>province_NL</th>
      <th>province_NS</th>
      <th>province_ON</th>
      <th>province_PE</th>
      <th>province_QC</th>
      <th>province_SK</th>
      <th>predicted_air_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>9</td>
      <td>34.162143</td>
      <td>49.279758</td>
      <td>16.939632</td>
      <td>1314</td>
      <td>621</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23</td>
      <td>8</td>
      <td>25.807231</td>
      <td>33.348344</td>
      <td>-3.570993</td>
      <td>830</td>
      <td>1161</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>32.151525</td>
      <td>28.493564</td>
      <td>-1.817876</td>
      <td>976</td>
      <td>660</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>6</td>
      <td>25.343065</td>
      <td>31.649166</td>
      <td>17.533380</td>
      <td>1381</td>
      <td>975</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>8</td>
      <td>34.714925</td>
      <td>38.670516</td>
      <td>28.697318</td>
      <td>1346</td>
      <td>520</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>7</td>
      <td>34.131054</td>
      <td>58.625281</td>
      <td>26.377881</td>
      <td>1476</td>
      <td>845</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>9</td>
      <td>22.097433</td>
      <td>47.330541</td>
      <td>17.449850</td>
      <td>1475</td>
      <td>534</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>7</td>
      <td>29.262879</td>
      <td>32.468443</td>
      <td>13.403061</td>
      <td>805</td>
      <td>1271</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>29</td>
      <td>8</td>
      <td>32.318841</td>
      <td>37.990165</td>
      <td>7.781761</td>
      <td>969</td>
      <td>1362</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14</td>
      <td>8</td>
      <td>26.854182</td>
      <td>58.446881</td>
      <td>28.004023</td>
      <td>1448</td>
      <td>837</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24</td>
      <td>7</td>
      <td>34.409340</td>
      <td>44.281370</td>
      <td>2.419963</td>
      <td>856</td>
      <td>514</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>25</td>
      <td>8</td>
      <td>25.197825</td>
      <td>35.795261</td>
      <td>3.206968</td>
      <td>762</td>
      <td>638</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>17</td>
      <td>9</td>
      <td>30.393122</td>
      <td>50.850814</td>
      <td>-6.667991</td>
      <td>930</td>
      <td>540</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9</td>
      <td>9</td>
      <td>25.640661</td>
      <td>23.816405</td>
      <td>6.686821</td>
      <td>1273</td>
      <td>1227</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>9</td>
      <td>28.838477</td>
      <td>55.488510</td>
      <td>11.249672</td>
      <td>1312</td>
      <td>961</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>6</td>
      <td>27.541712</td>
      <td>50.838687</td>
      <td>12.220802</td>
      <td>1092</td>
      <td>706</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10</td>
      <td>9</td>
      <td>19.834154</td>
      <td>21.257167</td>
      <td>18.638469</td>
      <td>1263</td>
      <td>595</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>8</td>
      <td>28.275095</td>
      <td>41.593644</td>
      <td>-0.862245</td>
      <td>1224</td>
      <td>659</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27</td>
      <td>8</td>
      <td>20.740762</td>
      <td>57.187906</td>
      <td>26.365417</td>
      <td>1345</td>
      <td>1295</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>11</td>
      <td>9</td>
      <td>31.726483</td>
      <td>55.843652</td>
      <td>4.310156</td>
      <td>924</td>
      <td>884</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>27</td>
      <td>7</td>
      <td>18.008846</td>
      <td>34.102754</td>
      <td>3.715157</td>
      <td>1409</td>
      <td>955</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>6</td>
      <td>20.037711</td>
      <td>33.504607</td>
      <td>32.430937</td>
      <td>1073</td>
      <td>1171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9</td>
      <td>9</td>
      <td>24.181703</td>
      <td>58.871283</td>
      <td>33.310128</td>
      <td>1209</td>
      <td>1306</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>6</td>
      <td>20.517478</td>
      <td>59.909619</td>
      <td>2.005146</td>
      <td>701</td>
      <td>1141</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>23</td>
      <td>6</td>
      <td>23.866211</td>
      <td>45.374054</td>
      <td>20.631745</td>
      <td>752</td>
      <td>1183</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6</td>
      <td>9</td>
      <td>28.075844</td>
      <td>23.234133</td>
      <td>6.634451</td>
      <td>1100</td>
      <td>1266</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>24</td>
      <td>6</td>
      <td>28.749199</td>
      <td>45.341188</td>
      <td>14.109861</td>
      <td>1451</td>
      <td>643</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>9</td>
      <td>26.639379</td>
      <td>47.832512</td>
      <td>28.626146</td>
      <td>1463</td>
      <td>902</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>8</td>
      <td>21.850428</td>
      <td>45.806912</td>
      <td>-2.153511</td>
      <td>1339</td>
      <td>1050</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8</td>
      <td>6</td>
      <td>29.486732</td>
      <td>49.408645</td>
      <td>-0.591777</td>
      <td>1114</td>
      <td>797</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a 'province' column from the one-hot encoded province columns
for province in provinces:
    future_data_df.loc[future_data_df[f'province_{province}'] == 1, 'province'] = province

# Convert 'month' and 'day_of_month' to string before concatenation
future_data_df['date'] = future_data_df['month'].astype(str) + '-' + future_data_df['day_of_month'].astype(str)
future_data_df['date_province'] = future_data_df['date'] + ' (' + future_data_df['province'] + ')'

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a horizontal bar chart
plt.figure(figsize=(10, 15))  # Adjust size as needed for clarity
chart = sns.barplot(y='date_province', x='predicted_air_quality', data=future_data_df, palette='viridis', orient='h')
plt.title('Predicted Air Quality Labels for Future Dates')
plt.ylabel('Date and Province')
plt.xlabel('Predicted Air Quality Label')

# Improve label readability
chart.set_yticklabels(chart.get_yticklabels(), rotation=0, horizontalalignment='right')

plt.show()
```


    
![png](Final%20Project_files/Final%20Project_41_0.png)
    


As we can see, the outputted labels globally trend higher, with more Moderate than we normally have, as seen in the class distribution.

## Discussion and Conclusion
By using relatively pessimistic randomized data for the future, we can show that the air quality has a tendency to degrade from Good to Moderate. While it is maybe not alarming, it shows that it is concerning given the implications of global warming and its impact on the wildfires. 

It is important to note that many shortcuts were taken in averaging the data, which do not reflect the vast climate differences across different canadian cities in a same province, as well as the complexity of forest fire patterns and behaviors. A more in-depth look with access to better data might yield different results, although the model seems to be well generalized. 

While this model is very simplistic, it can show that there is a tendency towards degrading air quality as the climate warms up.

### Future Work
For future work, one thing that would be interesting would be to insert more precise geographical data into the model, allowing for better predictions base on location. Another idea would be to use weather data from before the fire season, showing the impact of a warmer spring on the future air quality, for instance. 

## References
[https://www.epa.gov/wildfire-smoke-course/why-wildfire-smoke-health-concern#:~:text=Fine particles](https://www.epa.gov/wildfire-smoke-course/why-wildfire-smoke-health-concern#:~:text=Fine%20particles%20) 

https://cdnsciencepub.com/doi/10.1139/cjfr-2018-0293

https://cwfis.cfs.nrcan.gc.ca/report/graphs

https://natural-resources.canada.ca/climate-change/what-adaptation/10025

https://aqicn.org/scale/

https://www.cleanairfund.org/news-item/wildfires-climate-change-and-air-pollution-a-vicious-cycle/

https://openaq.org/

https://climate.weather.gc.ca/ 

https://cwfis.cfs.nrcan.gc.ca/ha/nfdb

https://publicinterestnetwork.org/wp-content/uploads/2021/10/CA-Trouble-in-the-Air-Web.pdf

https://github.com/david-paradis/air-quality-supervised-ml/
