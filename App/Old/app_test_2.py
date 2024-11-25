import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.gridspec import GridSpec

# Function to analyze university energy performance
def analyze_university_energy(university_performance):
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 25))
    gs = GridSpec(4, 2, figure=fig)

    # 1. Site EUI vs Source EUI by University
    ax1 = fig.add_subplot(gs[0, :])
    university_means = university_performance.groupby('OWNEROFRECORD')[['SITEEUI_KBTU_FT', 'SOURCEEUI_KBTU_FT']].mean().reset_index()

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

    plt.tight_layout()
    
    # Generate summary statistics
    summary_stats = pd.DataFrame({
        'Avg_Site_EUI': university_performance.groupby('OWNEROFRECORD')['SITEEUI_KBTU_FT'].mean(),
        'Avg_Source_EUI': university_performance.groupby('OWNEROFRECORD')['SOURCEEUI_KBTU_FT'].mean(),
        'Total_GHG_Emissions': university_performance.groupby('OWNEROFRECORD')['TOTGHGEMISSIONS_METRICTONSCO2E'].sum(),
        'Avg_Water_Usage': university_performance.groupby('OWNEROFRECORD')['WATERUSE_ALLWATERSOURCES_KGAL'].mean(),
        'Avg_Energy_Star': university_performance.groupby('OWNEROFRECORD')['ENERGYSTARSCORE'].mean()
    }).round(2)
    
    return fig, summary_stats

# Function to print insights
def print_energy_insights(summary_stats):
    if summary_stats.empty:
        print("No summary statistics available")
        return
        
    print("\nKey Insights:")
    print("-" * 50)
    
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

def load_data():
    # Load your actual data here
    df = pd.read_csv('/Users/samharris/Desktop/Fall-2024-Research/App/Data/University_Buildings.csv')  
    return df

def create_map(df):
    # Create a Plotly mapbox scatter plot
    fig = px.scatter_mapbox(df,
                            lat='LATITUDE',
                            lon='LONGITUDE',
                            hover_name='PROPERTYNAME',
                            hover_data=['SITEEUI_KBTU_FT', 'SOURCEEUI_KBTU_FT'],
                            color='OWNEROFRECORD',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='Building Locations in Washington D.C.',
                            mapbox_style='open-street-map',
                            zoom=10,
                            center={"lat": 38.89511, "lon": -77.03637})  # Center on Washington D.C.
    fig.update_layout(
        height=700,  # Set height (in pixels)
        width=1400   # Set width (in pixels)
    )
    
    return fig

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
        default=['George Washington University']
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

    # Call the analyze function
    fig, summary_stats = analyze_university_energy(filtered_df)
    st.pyplot(fig)  # Display matplotlib figures in Streamlit
    
    # Print insights
    insights = print_energy_insights(summary_stats)

    # Create and display interactive map
    st.header("Interactive Building Map")
    map_fig = create_map(filtered_df)
    st.plotly_chart(map_fig)

if __name__ == "__main__":
    main()