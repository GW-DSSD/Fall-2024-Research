#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
df = pd.read_csv('Building_Energy_Benchmarking.csv',low_memory=False)
print(df)


# In[57]:


print(df.isna().sum())


# In[58]:


print(df.shape)


# In[59]:


print(df.columns)


# In[60]:


print(df.isna())


# In[61]:


print(df['UBID'].isna().sum())


# In[62]:


#focusing on university level eco labels in DC
print(df['PRIMARYPROPERTYTYPE_SELFSELECT'].value_counts())


# Subset the dataset By university

# In[63]:


university_buildings = df[df['PRIMARYPROPERTYTYPE_SELFSELECT'] == 'College/University']

print(university_buildings)


# In[64]:


# Select relevant columns
relevant_columns = [
    'PROPERTYNAME', 'OWNEROFRECORD', 'REPORTINGYEAR', 'ENERGYSTARSCORE',
    'SITEEUI_KBTU_FT', 'WEATHERNORMALZEDSITEEUI_KBTUFT', 'SOURCEEUI_KBTU_FT',
    'WEATHERNORMALZEDSOUREUI_KBTUFT', 'TOTGHGEMISSIONS_METRICTONSCO2E',
    'TOTGHGEMISSINTENSITY_KGCO2EFT', 'WATERUSE_ALLWATERSOURCES_KGAL',
    'ELECTRICITYUSE_KBTU_JANUARY', 'ELECTRICITYUSE_KBTU_FEBRUARY',
    'ELECTRICITYUSE_KBTU_MARCH', 'ELECTRICITYUSE_KBTU_APRIL',
    'ELECTRICITYUSE_KBTU_MAY', 'ELECTRICITYUSE_KBTU_JUNE',
    'ELECTRICITYUSE_KBTU_JULY', 'ELECTRICITYUSE_KBTU_AUGUST',
    'ELECTRICITYUSE_KBTU_SEPTEMBER', 'ELECTRICITYUSE_KBTU_OCTOBER',
    'ELECTRICITYUSE_KBTU_NOVEMBER', 'ELECTRICITYUSE_KBTU_DECEMBER',
    'NATURALGAS_KBTU_JANUARY', 'NATURALGAS_KBTU_FEBRUARY',
    'NATURALGAS_KBTU_MARCH', 'NATURALGAS_KBTU_APRIL', 'NATURALGAS_KBTU_MAY',
    'NATURALGAS_KBTU_JUNE', 'NATURALGAS_KBTU_JULY', 'NATURALGAS_KBTU_AUGUST',
    'NATURALGAS_KBTU_SEPTEMBER', 'NATURALGAS_KBTU_OCTOBER',
    'NATURALGAS_KBTU_NOVEMBER', 'NATURALGAS_KBTU_DECEMBER'
]
university_performance = university_buildings[relevant_columns]

# Display the first few rows of the subsetted data
print(university_performance.head())


# In[65]:


print(university_buildings.isna().sum())


# In[66]:


university_performance_columns= print(university_performance.columns)
print(university_performance_columns)


# In[67]:


print(university_performance.shape)


# In[68]:


#dropping the null value in university perfomance 
university_performance = university_performance.drop(columns=['ENERGYSTARSCORE'], axis=1)
print(university_performance.isna().sum())


# In[69]:


print(university_performance['PROPERTYNAME'])


# converting the numeric column 

# In[72]:


# Convert the columns to numeric types
#List of columns to convert to numeric
numeric_columns = [
     'SITEEUI_KBTU_FT', 'WEATHERNORMALZEDSITEEUI_KBTUFT',
    'SOURCEEUI_KBTU_FT', 'WEATHERNORMALZEDSOUREUI_KBTUFT',
    'TOTGHGEMISSIONS_METRICTONSCO2E', 'TOTGHGEMISSINTENSITY_KGCO2EFT',
    'WATERUSE_ALLWATERSOURCES_KGAL', 'ELECTRICITYUSE_KBTU_JANUARY',
    'ELECTRICITYUSE_KBTU_FEBRUARY', 'ELECTRICITYUSE_KBTU_MARCH',
    'ELECTRICITYUSE_KBTU_APRIL', 'ELECTRICITYUSE_KBTU_MAY',
    'ELECTRICITYUSE_KBTU_JUNE', 'ELECTRICITYUSE_KBTU_JULY',
    'ELECTRICITYUSE_KBTU_AUGUST', 'ELECTRICITYUSE_KBTU_SEPTEMBER',
    'ELECTRICITYUSE_KBTU_OCTOBER', 'ELECTRICITYUSE_KBTU_NOVEMBER',
    'ELECTRICITYUSE_KBTU_DECEMBER', 'NATURALGAS_KBTU_JANUARY',
    'NATURALGAS_KBTU_FEBRUARY', 'NATURALGAS_KBTU_MARCH',
    'NATURALGAS_KBTU_APRIL', 'NATURALGAS_KBTU_MAY', 'NATURALGAS_KBTU_JUNE',
    'NATURALGAS_KBTU_JULY', 'NATURALGAS_KBTU_AUGUST',
    'NATURALGAS_KBTU_SEPTEMBER', 'NATURALGAS_KBTU_OCTOBER',
    'NATURALGAS_KBTU_NOVEMBER', 'NATURALGAS_KBTU_DECEMBER'
]

# Convert the columns to numeric types
university_performance[numeric_columns] = university_performance[numeric_columns].apply(pd.to_numeric, errors='coerce')


# In[73]:


# Select only numeric columns for aggregation
numeric_data = university_performance[numeric_columns + ['OWNEROFRECORD']]

# Group by university and calculate the mean
grouped_data = numeric_data.groupby('OWNEROFRECORD').mean()

# Display the grouped data
print(grouped_data.head())


# 

# In[74]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set the style correctly using a valid style
plt.style.use('seaborn-v0_8')  # or use 'seaborn-darkgrid' depending on your matplotlib version
# Alternative styles you can try:
# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-whitegrid')
# plt.style.use('default')

def analyze_university_energy(university_performance):
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 25))
    gs = GridSpec(4, 2, figure=fig)
    
    # 1. Site EUI vs Source EUI by University
    ax1 = fig.add_subplot(gs[0, :])
    university_means = university_performance.groupby('OWNEROFRECORD')[
        ['SITEEUI_KBTU_FT', 'SOURCEEUI_KBTU_FT']
    ].mean().reset_index()
    
    bar_width = 0.35
    x = np.arange(len(university_means))
    
    ax1.bar(x - bar_width/2, university_means['SITEEUI_KBTU_FT'], bar_width, label='Site EUI')
    ax1.bar(x + bar_width/2, university_means['SOURCEEUI_KBTU_FT'], bar_width, label='Source EUI')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(university_means['OWNEROFRECORD'], rotation=45, ha='right')
    ax1.set_title('Average Site EUI vs Source EUI by University')
    ax1.set_ylabel('EUI (kBTU/ft²)')
    ax1.legend()
    
    # 2. Monthly Electricity Usage Patterns
    ax2 = fig.add_subplot(gs[1, :])
    
    try:
        # Calculate monthly averages for each university
        electricity_columns = [col for col in university_performance.columns if 'ELECTRICITY' in col]
        monthly_electricity = university_performance.groupby('OWNEROFRECORD')[electricity_columns].mean()
        
        # Reshape data for plotting
        monthly_electricity = monthly_electricity.T
        monthly_electricity.index = [i.split('_')[-1] for i in monthly_electricity.index]
        
        monthly_electricity.plot(ax=ax2, marker='o')
        ax2.set_title('Average Monthly Electricity Usage by University')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Electricity Usage (kBTU)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    except Exception as e:
        print(f"Error plotting electricity usage: {e}")
        ax2.text(0.5, 0.5, 'Could not plot electricity data', ha='center', va='center')
    
    # 3. GHG Emissions vs Water Usage
    ax3 = fig.add_subplot(gs[2, 0])
    sns.scatterplot(data=university_performance, 
                   x='TOTGHGEMISSIONS_METRICTONSCO2E', 
                   y='WATERUSE_ALLWATERSOURCES_KGAL',
                   hue='OWNEROFRECORD',
                   ax=ax3)
    ax3.set_title('GHG Emissions vs Water Usage')
    ax3.set_xlabel('GHG Emissions (Metric Tons CO2e)')
    ax3.set_ylabel('Water Usage (kGal)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Energy Star Score Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    try:
        sns.boxplot(data=university_performance, 
                   x='OWNEROFRECORD', 
                   y='ENERGYSTARSCORE',
                   ax=ax4)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_title('Distribution of Energy Star Scores')
        ax4.set_xlabel('University')
        ax4.set_ylabel('Energy Star Score')
    except Exception as e:
        print(f"Error plotting Energy Star scores: {e}")
        ax4.text(0.5, 0.5, 'Could not plot Energy Star data', ha='center', va='center')
    
    # 5. Natural Gas Usage Patterns
    ax5 = fig.add_subplot(gs[3, :])
    
    try:
        # Calculate monthly averages for each university
        gas_columns = [col for col in university_performance.columns if 'NATURALGAS' in col]
        monthly_gas = university_performance.groupby('OWNEROFRECORD')[gas_columns].mean()
        
        # Reshape data for plotting
        monthly_gas = monthly_gas.T
        monthly_gas.index = [i.split('_')[-1] for i in monthly_gas.index]
        
        monthly_gas.plot(ax=ax5, marker='o')
        ax5.set_title('Average Monthly Natural Gas Usage by University')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Natural Gas Usage (kBTU)')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    except Exception as e:
        print(f"Error plotting natural gas usage: {e}")
        ax5.text(0.5, 0.5, 'Could not plot natural gas data', ha='center', va='center')
    
    plt.tight_layout()
    
    # Generate summary statistics
    try:
        summary_stats = pd.DataFrame({
            'Avg_Site_EUI': university_performance.groupby('OWNEROFRECORD')['SITEEUI_KBTU_FT'].mean(),
            'Avg_Source_EUI': university_performance.groupby('OWNEROFRECORD')['SOURCEEUI_KBTU_FT'].mean(),
            'Total_GHG_Emissions': university_performance.groupby('OWNEROFRECORD')['TOTGHGEMISSIONS_METRICTONSCO2E'].sum(),
            'Avg_Water_Usage': university_performance.groupby('OWNEROFRECORD')['WATERUSE_ALLWATERSOURCES_KGAL'].mean(),
            'Avg_Energy_Star': university_performance.groupby('OWNEROFRECORD')['ENERGYSTARSCORE'].mean()
        }).round(2)
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
        summary_stats = pd.DataFrame()
    
    return fig, summary_stats

# Function to print insights
def print_energy_insights(summary_stats):
    if summary_stats.empty:
        print("No summary statistics available")
        return
        
    print("\nKey Insights:")
    print("-" * 50)
    
    try:
        # Most efficient university by Site EUI
        best_site_eui = summary_stats['Avg_Site_EUI'].idxmin()
        print(f"Most efficient university (Site EUI): {best_site_eui}")
        
        # Highest Energy Star scorer
        best_energy_star = summary_stats['Avg_Energy_Star'].idxmax()
        print(f"Highest average Energy Star score: {best_energy_star}")
        
        # Largest GHG emitter
        largest_emitter = summary_stats['Total_GHG_Emissions'].idxmax()
        print(f"Largest GHG emitter: {largest_emitter}")
        
        # Highest water user
        highest_water = summary_stats['Avg_Water_Usage'].idxmax()
        print(f"Highest average water usage: {highest_water}")
    except Exception as e:
        print(f"Error generating insights: {e}")

# :
try:
    # Verify data is ready
    print("Shape of university_performance:", university_performance.shape)
    print("\nColumns available:", university_performance.columns.tolist())
    
    # Generating visualizations and summary statistics
    fig, summary_stats = analyze_university_energy(university_performance)
    
    # Print insights
    print_energy_insights(summary_stats)
    
    # Display the plots
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")



print(university_performance)
# Assuming your DataFrame is named university_performance

university_performance.to_csv('university_performance.csv', index=False)



#%%
# Analysis of GWU and Howard University Building

import pandas as pd 
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go 

def university_analyze(university_performance):
    #filtering for GWU snd howard
    universities =university_performance[university_performance['OWNEROFRECORD'].str.contains('HOWARD| GEORGE WASHINGTON', case=False, na=False)]
    
    #create time series comparison
    fig=px.line(universities, x='REPORTINGYEAR', y='SOURCEEUI_KBTU_FT', color='PROPERTYNAME', title='Energy Usage Intensity Over Time')
    
    #create scatter plot
    fig2=px.scatter(universities, x='SORCEEUI_KBTU_FT', 
                    Y='TOTGHGEMISSINTENSITY_KGCO2EFT', 
                    color='PROPERTYNAME',
                    hover_data=['REPORTINGYEAR'],
                    title='Energy Usage Intensity Over Time')
    
    return fig, fig2

    
#%%

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add page configuration with app title
st.set_page_config(
    page_title="University Buildings Energy Analysis",
    page_icon="🏛️",
    layout="wide"
)

def load_data():
    # Replace with your actual data loading
    df = pd.read_csv('university_performance.csv')  # Update with your actual dataset name
    return df

def main():
    st.title("University Buildings Energy Analysis")
    st.subheader("Energy and Sustainability Performance Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_year = st.sidebar.selectbox(
        "Select Year",
        sorted(df['REPORTINGYEAR'].unique())
    )
    
    selected_university = st.sidebar.multiselect(
        "Select Universities",
        df['OWNEROFRECORD'].unique(),
        default=['HOWARD UNIVERSITY']
    )
    
    # Filter data
    filtered_df = df[
        (df['REPORTINGYEAR'] == selected_year) &
        (df['OWNEROFRECORD'].isin(selected_university))
    ]
    
    # Main content
    st.header("Energy Performance Analysis")
    
    # Energy Usage Over Time
    fig1 = px.line(
        filtered_df,
        x='REPORTINGYEAR',
        y='SOURCEEUI_KBTU_FT',
        color='PROPERTYNAME',
        title='Energy Usage Intensity Over Time'
    )
    st.plotly_chart(fig1)
    
    # Emissions vs Energy Usage
    fig2 = px.scatter(
        filtered_df,
        x='SOURCEEUI_KBTU_FT',
        y='TOTGHGEMISSINTENSITY_KGCO2EFT',
        color='PROPERTYNAME',
        hover_data=['REPORTINGYEAR'],
        title='Energy Usage vs GHG Emissions'
    )
    st.plotly_chart(fig2)
    
    # Building Performance Metrics Table
    st.header("Building Performance Metrics")
    metrics_df = filtered_df[['PROPERTYNAME', 'SOURCEEUI_KBTU_FT', 
                            'TOTGHGEMISSINTENSITY_KGCO2EFT', 
                            'WATERUSE_ALLWATERSOURCES_KGAL']].copy()
    st.dataframe(metrics_df)

if __name__ == "__main__":
    main()

#%%
##%%
#Research Questions
#How do different university buildings compare in energy efficiency?
#What are the trends in water and energy consumption over time?
#Which buildings meet or exceed ENERGY STAR performance standards?
#What are the primary sources of greenhouse gas emissions?


