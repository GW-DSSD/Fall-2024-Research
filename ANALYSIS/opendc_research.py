#%% [markdown]
# # Analyzing Energy and Water Usage Across George Washington University Buildings
# 
# This analysis explores total measurements of water and electricity usage across various buildings. We aim to understand the patterns in water and energy usage, identifying high and low usage properties and examining the relationship between energy consumption and greenhouse gas emissions.

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('University_Buildings.csv')

#%% [markdown]
# ## Data Preparation
# 
# Filtering data for George Washington University buildings and checking for null values in key columns related to energy and water usage.

#%%

# Filter for GWU buildings
GWU_data = df[df['OWNEROFRECORD'] == 'George Washington University']

# Columns to check for null values
columns_to_check = [
    "WATERUSE_ALLWATERSOURCES_KGAL",
    "ELECTRICITYUSE_RENEWABLE_KWH",
    "ELECTRICITYUSE_GRID_KWH",
    "NATURALGASUSE_THERMS",
    "FUELOILANDDIESELFUELUSEKBTU",
    "TOTGHGEMISSIONS_METRICTONSCO2E",
    "SOURCEEUI_KBTU_FT",
    "ENERGYSTARSCORE"
]

# Checking null values in key columns
null_values = GWU_data[columns_to_check].isnull().sum()
print(null_values)

# Count the unique properties
unique_properties = GWU_data['PROPERTYNAME'].nunique()
print(f"Unique buildings in dataset: {unique_properties}")

#%% [markdown]
# ## Water Usage Analysis
# 
# Total annual water usage (in kilogallons) for each building. We'll look at the highest and lowest usage to understand resource distribution.

#%%

# Convert 'WATERUSE_ALLWATERSOURCES_KGAL' to numeric and handle any errors
GWU_data['WATERUSE_ALLWATERSOURCES_KGAL'] = pd.to_numeric(GWU_data['WATERUSE_ALLWATERSOURCES_KGAL'], errors='coerce').fillna(0)

# Calculate average water usage by building
GWU_WATERUSE_ALLWATERSOURCES = GWU_data.groupby('PROPERTYNAME')['WATERUSE_ALLWATERSOURCES_KGAL'].mean().reset_index()
GWU_water_usage_sorted = GWU_WATERUSE_ALLWATERSOURCES.sort_values(by='WATERUSE_ALLWATERSOURCES_KGAL', ascending=False)

# Create a Plotly table for average water usage by building
fig = go.Figure(data=[go.Table(
    header=dict(values=['Building Name', 'Average Water Usage (KGAL)'],
                fill_color='paleturquoise', align='left'),
    cells=dict(values=[GWU_water_usage_sorted['PROPERTYNAME'], GWU_water_usage_sorted['WATERUSE_ALLWATERSOURCES_KGAL']],
               fill_color='lavender', align='left'))
])

# Add title
fig.update_layout(title_text="Average Water Usage for GWU Buildings Across All Years")
fig.show()

#%% [markdown]
# ### Water Usage by Year - (last few years considered)
# 
# Water usage trends from 2016 onwards, allowing selection of specific years to observe annual changes.

#%%

# Filter data for years from 2016 onwards
filtered_data = GWU_data[GWU_data['REPORTINGYEAR'] >= 2016]
unique_years = sorted(filtered_data['REPORTINGYEAR'].unique())

# Create an interactive bar chart for water usage per year
fig = go.Figure()
for year in unique_years:
    year_data = filtered_data[filtered_data['REPORTINGYEAR'] == year]
    water_usage_summary = year_data.groupby('PROPERTYNAME')['WATERUSE_ALLWATERSOURCES_KGAL'].sum().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=water_usage_summary['PROPERTYNAME'],
            y=water_usage_summary['WATERUSE_ALLWATERSOURCES_KGAL'],
            name=f"Year {year}",
            visible=(year == unique_years[0])  # Show first year initially
        )
    )

# Dropdown menu for year selection
dropdown_buttons = [
    {
        "label": str(year),
        "method": "update",
        "args": [{"visible": [year == y for y in unique_years]},
                 {"title": f"Water Usage by Property - (Latest Years)- {year}"}]
    } for year in unique_years
]

# Layout adjustments
fig.update_layout(
    updatemenus=[{"buttons": dropdown_buttons, "direction": "down", "showactive": True}],
    title=f"Water Usage (All Water Sources) by Property - (Latest Years)-{unique_years[0]}",
    xaxis_title="Property Name",
    yaxis_title="Water Usage (KGal) [Log Scale]",
    yaxis_type="log",
    xaxis_tickangle=-45,
    height=600
)
fig.show()

#%% [markdown]
# ### Focused Water Usage Trends: High and Low Usage Buildings
# 
# Let's examine the water usage over the years for the buildings with the highest and lowest total usage: Ross Hall Group and Law Clinic Townhouses.

#%%

# Filter data for Ross Hall Group and Law Clinic Townhouses
rosshall_group = GWU_data[GWU_data['PROPERTYNAME'] == 'Ross Hall Group']
law_clinic = GWU_data[GWU_data['PROPERTYNAME'] == 'Law Clinic Townhouses']

# Aggregate water usage over years for each
rossgroup_water_usage_summary = rosshall_group.groupby('REPORTINGYEAR')['WATERUSE_ALLWATERSOURCES_KGAL'].sum().reset_index()
law_clinic_water_usage_summary = law_clinic.groupby('REPORTINGYEAR')['WATERUSE_ALLWATERSOURCES_KGAL'].sum().reset_index()

# Plotting water usage trends for each building
fig = go.Figure()

# Ross Hall Group
fig.add_trace(
    go.Bar(
        x=rossgroup_water_usage_summary['REPORTINGYEAR'],
        y=rossgroup_water_usage_summary['WATERUSE_ALLWATERSOURCES_KGAL'],
        name="Ross Hall Group",
        visible=True
    )
)

# Law Clinic Townhouses
fig.add_trace(
    go.Bar(
        x=law_clinic_water_usage_summary['REPORTINGYEAR'],
        y=law_clinic_water_usage_summary['WATERUSE_ALLWATERSOURCES_KGAL'],
        name="Law Clinic Townhouses",
        visible=False
    )
)

# Dropdown for switching between buildings
dropdown_buttons = [
    {"label": "Ross Hall Group", "method": "update", "args": [{"visible": [True, False]}, {"title": "Water Usage for Ross Hall Group by Year"}]},
    {"label": "Law Clinic Townhouses", "method": "update", "args": [{"visible": [False, True]}, {"title": "Water Usage for Law Clinic Townhouses by Year"}]}
]

fig.update_layout(
    updatemenus=[{"buttons": dropdown_buttons, "direction": "down", "showactive": True}],
    title="Water Usage by Property",
    xaxis_title="Year",
    yaxis_title="Water Usage (KGal)",
    xaxis_tickangle=-45,
    height=500,
    xaxis=dict(type="category")
)
fig.show()

#%% [markdown]
# ## Relationship Between Water Usage and Greenhouse Gas Emissions
# 
# Next, let's explore if there's a relationship between water usage and greenhouse gas emissions, focusing on the Ross Hall Group.

#%%

fig = px.scatter(
    rosshall_group,
    x="TOTGHGEMISSIONS_METRICTONSCO2E",
    y="WATERUSE_ALLWATERSOURCES_KGAL",
    color="REPORTINGYEAR",
    size="ELECTRICITYUSE_GRID_KWH",
    title="GHG Emissions vs Water Usage for Ross Hall Group",
    labels={
        "TOTGHGEMISSIONS_METRICTONSCO2E": "Total GHG Emissions (Metric Tons CO2e)",
        "WATERUSE_ALLWATERSOURCES_KGAL": "Water Usage (KGal)",
        "ELECTRICITYUSE_GRID_KWH": "Electricity Usage (kWh)"
    },
    trendline="ols"
)
fig.show()

#%% [markdown]
# ## Electricity Usage Analysis
# 
# Finally, let's examine electricity usage, both grid and renewable, across GWU buildings to understand energy consumption patterns.

#%%

# Convert ELECTRICITYUSE_GRID_KWH to numeric
GWU_data['ELECTRICITYUSE_GRID_KWH'] = pd.to_numeric(GWU_data['ELECTRICITYUSE_GRID_KWH'], errors='coerce').fillna(0)

# Calculating average grid electricity usage by building
gwu_electricity_usage_avg = GWU_data.groupby('PROPERTYNAME')['ELECTRICITYUSE_GRID_KWH'].mean().reset_index()
gwu_electricity_usage_avg_sorted = gwu_electricity_usage_avg.sort_values(by='ELECTRICITYUSE_GRID_KWH', ascending=False)

# Plotly table for average electricity usage
fig = go.Figure(data=[go.Table(
    header=dict(values=['Building Name', 'Average Electricity Usage (Grid) in kWh'],
                fill_color='paleturquoise', align='left'),
    cells=dict(values=[gwu_electricity_usage_avg_sorted['PROPERTYNAME'], gwu_electricity_usage_avg_sorted['ELECTRICITYUSE_GRID_KWH']],
               fill_color='lavender', align='left'))
])

fig.update_layout(title_text="Average Electricity Usage (Grid) for GWU Buildings Across All Years")
fig.show()

#%% [markdown]
# ### Electricity Usage Over the Years for High and Low Usage Buildings
# 
# We will focus on Ross Hall, with high usage, and 2109 F St, with low usage.

#%%

# Filter data for Ross Hall and 2109 F St
rosshall = GWU_data[GWU_data['PROPERTYNAME'] == "Ross Hall"]
f_St = GWU_data[GWU_data['PROPERTYNAME'] == '2109 F St']

# Summing grid electricity usage over years
rosshall_electricity_usage_summary = rosshall.groupby('REPORTINGYEAR')['ELECTRICITYUSE_GRID_KWH'].sum().reset_index()
f_St_electricity_usage_summary = f_St.groupby('REPORTINGYEAR')['ELECTRICITYUSE_GRID_KWH'].sum().reset_index()

# Plot for electricity usage by property
fig = go.Figure()

# Ross Hall
fig.add_trace(
    go.Bar(
        x=rosshall_electricity_usage_summary['REPORTINGYEAR'],
        y=rosshall_electricity_usage_summary['ELECTRICITYUSE_GRID_KWH'],
        name="Ross Hall",
        visible=True
    )
)

# 2109 F St
fig.add_trace(
    go.Bar(
        x=f_St_electricity_usage_summary['REPORTINGYEAR'],
        y=f_St_electricity_usage_summary['ELECTRICITYUSE_GRID_KWH'],
        name="2109 F St",
        visible=False
    )
)

# Dropdown menu
dropdown_buttons = [
    {"label": "Ross Hall", "method": "update", "args": [{"visible": [True, False]}, {"title": "Electricity (Grid) Usage for Ross Hall by Year"}]},
    {"label": "2109 F St", "method": "update", "args": [{"visible": [False, True]}, {"title": "Electricity (Grid) Usage for 2109 F St by Year"}]}
]

fig.update_layout(
    updatemenus=[{"buttons": dropdown_buttons, "direction": "down", "showactive": True}],
    title="Electricity (Grid) Usage by Property",
    xaxis_title="Year",
    yaxis_title="Electricity Usage (KWH)",
    xaxis_tickangle=-45,
    height=500,
    xaxis=dict(type="category")
)
fig.show()

#%% [markdown]
# ### Annual Electricity Usage Across GWU Buildings
# 
# Explore electricity usage by year for all buildings, focusing on grid electricity from 2016 onwards.

#%%

filtered_data = GWU_data[(GWU_data['REPORTINGYEAR'] >= 2016) & (GWU_data["ELECTRICITYUSE_GRID_KWH"] > 0)]
unique_years = sorted(filtered_data['REPORTINGYEAR'].unique())

# Plotting annual electricity usage
fig = go.Figure()
for year in unique_years:
    year_data = filtered_data[filtered_data['REPORTINGYEAR'] == year]
    electricity_usage_summary = year_data.groupby('PROPERTYNAME')['ELECTRICITYUSE_GRID_KWH'].sum().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=electricity_usage_summary['PROPERTYNAME'],
            y=electricity_usage_summary['ELECTRICITYUSE_GRID_KWH'],
            name=f"Year {year}",
            visible=(year == unique_years[0])
        )
    )

# Dropdown for year selection
dropdown_buttons = [
    {"label": str(year), "method": "update", "args": [{"visible": [year == y for y in unique_years]}, {"title": f"Electricity (Grid) Usage by Property - (Latest Years)- {year}"}]}
    for year in unique_years
]

# Layout adjustments
fig.update_layout(
    updatemenus=[{"buttons": dropdown_buttons, "direction": "down", "showactive": True}],
    title=f"Electricity (Grid) Usage by Property - (Latest Years)- {unique_years[0]}",
    xaxis_title="Property Name",
    yaxis_title="Electricity Usage (KWH) [Log Scale]",
    yaxis_type="log",
    yaxis_rangemode="tozero",
    xaxis_tickangle=-45,
    height=600,
    margin=dict(l=80, r=50, t=80, b=200),
    legend=dict(x=0.85, y=0.95)
)
fig.show()
