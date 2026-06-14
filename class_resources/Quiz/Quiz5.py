import pandas as pd
print(f'pandas version: {pd.__version__}')

# Grouping by neighborhood 
df = pd.read_csv('Quiz5_Air_BnB_in_NYC.csv')

counts = df["neighbourhood_group"].value_counts()
print(counts)

# defining most popular by number of reviews
top_twenty = df.sort_values("number_of_reviews", ascending=False).head(20)
avg_reviews = (top_twenty.groupby("neighbourhood")["number_of_reviews"].mean().reset_index(name="avg_reviews")
    .sort_values(by="avg_reviews", ascending=False))
print(f'Average number of reviews for top 20 listings: {avg_reviews}')

# getting cheapest recommendation for upper east side
import numpy as np

ues = df[df["neighbourhood"] == "Upper East Side"]
p85 = np.quantile(ues["number_of_reviews"], 0.85)
filtered = ues[ues["number_of_reviews"] >= p85]
cheapest_private = (filtered[filtered["room_type"] == "Private room"].sort_values("price", ascending=True).iloc[0]["price"])
cheapest_home = (filtered[filtered["room_type"] == "Entire home/apt"].sort_values("price", ascending=True).iloc[0]["price"])
price_gap = abs(cheapest_private - cheapest_home)
print(f'Price gap between cheapest entire home/apt and cheapest private room: {price_gap}')

# uber data
from datetime import datetime

def date_convertion(df, cols):
  for col in cols:
    df[col] = df[col].apply(lambda x: x.replace(' +0000 UTC', ''))
    df[col] = pd.to_datetime(df[col])
  return df

uber = pd.read_csv('UberData.csv')

uber = date_convertion(uber, ['Request Time', 'Begin Trip Time', 'Dropoff Time'])

uber['year'] = uber['Begin Trip Time'].map(lambda x: datetime.strftime(x,"%Y"))
uber['month'] = uber['Begin Trip Time'].map(lambda x: datetime.strftime(x,"%b"))
uber['day'] = uber['Begin Trip Time'].map(lambda x: datetime.strftime(x,"%d"))

# find shortest distance of completed UberX trips on Jan 14, 2020
shortest_distance = uber[
    (uber["Trip or Order Status"] == "COMPLETED") &
    (uber["Product Type"] == "UberX") &
    (uber["year"] == "2020") &
    (uber["month"] == "Jan") &
    (uber["day"] == "14")
]

distance = shortest_distance.iloc[0]["Distance (miles)"]
print(f'shortest distance of completed UberX trips: {round(distance)} miles')

# find peak hour of the day for Uber requests
uber['Hour'] = uber['Request Time'].map(lambda x: datetime.strftime(x,"%H"))

hourly_requests = uber.groupby('Hour').size().reset_index(name='Request Count')

peak_hour = hourly_requests.loc[hourly_requests['Request Count'].idxmax()]
print(f"Peak Hour of the Day: {int(peak_hour['Hour'])}:00 with {int(peak_hour['Request Count'])} requests")

# find peak day of the week for Uber requests
uber['weekday'] = uber['Request Time'].map(lambda x: datetime.strftime(x,"%a"))

weekly_requests = uber.groupby('weekday').size().reset_index(name='Request Count')
peak_day = weekly_requests.loc[weekly_requests['Request Count'].idxmax()]
print(f"Peak Day of the Week: {peak_day['weekday']} with {int(peak_day['Request Count'])} requests")
