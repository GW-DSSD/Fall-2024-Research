import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import os
import numpy as np

def display_html(html_file_name):
    # Get the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))
    html_file_path = os.path.join(current_directory, html_file_name)

    # Check if the file exists
    if os.path.exists(html_file_path):
        with open(html_file_path, "r") as f:
            plotly_map_html = f.read()

        # Remove leading and trailing whitespaces in the HTML content
        plotly_map_html = plotly_map_html.strip()

        # Embed custom CSS directly within the HTML content to eliminate white space
        custom_css = """
        <style>
            body, html {
                margin: 0;
                padding: 0;
                overflow: hidden;
                width: 100%;
                height: 100%;
            }
            iframe {
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                width: 100%;
                height: 100%;
            }
        </style>
        """
        plotly_map_html = custom_css + plotly_map_html
        
        # Use components.html to embed the HTML content with the CSS included
        components.html(plotly_map_html, width=800, height=600, scrolling=False)
    else:
        st.error("HTML file not found. Please make sure it exists in the same directory as your app.py.")
        st.write("File not found:", html_file_path)  # Debug print


@st.cache_data
def load_data():
    file_path = "App/University_Buildings.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File {file_path} not found in {os.getcwd()}")
        return pd.DataFrame()

# Add column
def add_beps_column(df):
    def meets_beps(row):
        if row['PRIMARYPROPERTYTYPE_SELFSELECT'] == 'Residence Hall/Dormitory':
            if pd.notnull(row['ENERGYSTARSCORE']) and row['ENERGYSTARSCORE'] > 56:
                return 'Yes'
            elif pd.isnull(row['ENERGYSTARSCORE']) and row['SOURCEEUI_KBTU_FT'] < 141.4:
                return 'Yes'
            else:
                return 'No'
        else:
            if row['SOURCEEUI_KBTU_FT'] < 180.6:
                return 'Yes'
            else:
                return 'No'
    
    # Apply the function to each row
    df['Meets_BEPS_2021'] = df.apply(meets_beps, axis=1)
    return df

# Map of Buildings
def create_map(df):
    fig = px.scatter_mapbox(
        df,
        lat='LATITUDE',
        lon='LONGITUDE',
        hover_name='PROPERTYNAME',
        hover_data={'LATITUDE': False, 
        'LONGITUDE': False,  
        'REPORTINGYEAR': True,  
        'OWNEROFRECORD': True,  
        'REPORTEDADDRESS': True, 
        'SOURCEEUI_KBTU_FT': True,  
        'TOTGHGEMISSIONS_METRICTONSCO2E': True,  
        'WATERUSE_ALLWATERSOURCES_KGAL': True,  
        'Meets_BEPS_2021': True},
        color='Meets_BEPS_2021',
        color_discrete_map={'Yes': 'green', 'No': 'red'},
        mapbox_style='carto-positron',  
        zoom=11.2,  
        center={"lat": 38.92, "lon": -77.03637},
        title="Building Locations in Washington D.C."
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7)) 
    fig.update_layout(
        height=500,
        width=1200,
        title="Building Locations in Washington D.C.",
        margin={"r":0,"t":40,"l":0,"b":0}  
    )
    return fig

# Function to aggregate and create the single variable plot for SOURCEEUI_KBTU_FT by university
def single_variable_eui_plot(df, variable, title):
    # Aggregate by university (OWNEROFRECORD)
    aggregated_df = df.groupby('OWNEROFRECORD').agg(
        total_eui=(variable, 'sum'),  # Sum of SOURCEEUI_KBTU_FT
        num_buildings=('PID', 'nunique')  # Count of unique PIDs (buildings)
    )
    
    # Calculate average EUI per building
    aggregated_df['average_eui'] = aggregated_df['total_eui'] / aggregated_df['num_buildings']
    
    # Reset index for plotting
    aggregated_df = aggregated_df.reset_index()

    # Create a bar plot with the average EUI by university
    fig = px.bar(
        aggregated_df, 
        x='OWNEROFRECORD', 
        y='average_eui', 
        color='OWNEROFRECORD', 
        title=title,
        labels={'average_eui': 'Source EUI (kBtu/ft²)', 'OWNEROFRECORD': 'University'},
        color_continuous_scale='Viridis'
    )
    
    return fig

# Function to aggregate and create the single variable plot for TOTGHGEMISSIONS_METRICTONSCO2E by university
def single_variable_ghg_plot(df, variable, title):
    # Aggregate by university (OWNEROFRECORD)
    aggregated_df = df.groupby('OWNEROFRECORD').agg(
        total_ghg=(variable, 'sum'),  # Sum of TOTGHGEMISSIONS_METRICTONSCO2E
        num_buildings=('PID', 'nunique')  # Count of unique PIDs (buildings)
    )
    
    # Calculate average GHG per building
    aggregated_df['average_ghg'] = aggregated_df['total_ghg'] / aggregated_df['num_buildings']
    
    # Reset index for plotting
    aggregated_df = aggregated_df.reset_index()

    # Create a bar plot with the average GHG by university
    fig = px.bar(
        aggregated_df, 
        x='OWNEROFRECORD', 
        y='average_ghg', 
        color='OWNEROFRECORD', 
        title=title,
        labels={'average_ghg': 'Emissions (KG of CO₂e / ft²)', 'OWNEROFRECORD': 'University'},
        color_continuous_scale='Viridis'
    )
    
    return fig

# Function to aggregate and create the single variable plot for WATERUSE_ALLWATERSOURCES_KGAL by university
def single_variable_water_plot(df, variable, title):
    # Aggregate by university (OWNEROFRECORD)
    aggregated_df = df.groupby('OWNEROFRECORD').agg(
        total_water=(variable, 'sum'),  # Sum of WATERUSE_ALLWATERSOURCES_KGAL
        num_buildings=('PID', 'nunique')  # Count of unique PIDs (buildings)
    )
    
    # Calculate average water usage per building
    aggregated_df['average_water'] = aggregated_df['total_water'] / aggregated_df['num_buildings']
    
    # Reset index for plotting
    aggregated_df = aggregated_df.reset_index()

    # Create a bar plot with the average water usage by university
    fig = px.bar(
        aggregated_df, 
        x='OWNEROFRECORD', 
        y='average_water', 
        color='OWNEROFRECORD', 
        title=title,
        labels={'average_water': 'Water Usage (KGAL)', 'OWNEROFRECORD': 'University'},
        color_continuous_scale='Blues'
    )
    
    return fig


def main():
    # Load data
    df = load_data()
    df = add_beps_column(df)  # Add the Meets_BEPS_2021 column

    # Sidebar Tab navigation (unchanged)
    tab_selection = st.sidebar.radio("Select a Page", ["Introduction", "University Comparison", "GWU Buildings Comparison", "Conclusion and Recommendations"])

    # Get all unique years that have been reported in the dataset
    all_years = sorted(df['REPORTINGYEAR'].unique(), reverse=True)

    # Get the most recent year reported for each property (PID)
    most_recent_df = df.loc[df.groupby('PID')['REPORTINGYEAR'].idxmax()]

    # Add the "Most Recent Year" option to the list of years
    options = all_years + ['Most Recent Year Reported']

    # Create the selectbox for year and multiselect for universities (now on top of Overview tab)
    if tab_selection == "Introduction":
        st.title("University Buildings Energy and Sustainability Analysis")
        st.subheader("An Overview of Energy, Water, and Emissions Data for University Buildings in Washington D.C.")
        # Filters for University Comparison tab
        st.sidebar.header("Filters")
        selected_year = st.sidebar.selectbox("Select Year", options, index=options.index('Most Recent Year Reported'))
        if selected_year == 'Most Recent Year Reported':
            filtered_df = most_recent_df
        else:
            filtered_df = df[df['REPORTINGYEAR'] == selected_year]

        selected_universities = st.sidebar.multiselect("Select Universities", df['OWNEROFRECORD'].unique(), default=df['OWNEROFRECORD'].unique())
        filtered_df = filtered_df[filtered_df['OWNEROFRECORD'].isin(selected_universities)]

        # Introduction
        st.header("Introduction")
        st.write("This report provides insights into water usage, emissions, and energy usage of university buildings in Washington D.C. \
                 The data is from Open Data DC's dataset, Building Energy Benchmarking. This dataset contains yearly metrics on water usage, \
                 emissions, and energy usage for qualified buildings in DC. For this analysis, we focus exclusively on university buildings to evaluate how \
                 well universities align with current sustainability standards. We specifically highlight buildings at GWU which are meeting and not meeting standards. \
                 In additon, we provide the reader with anlaysis tools in this app \
                 that compare univeristies and buildings individually.")

        # Map of all buildings
        st.header("Map of University Buildings in Washington D.C.")
        st.write('The map below offers an initial overview of university buildings in Washington, D.C. These buildings are categorized by their compliance with Building Energy Performance Standards (BEPS). \
                 In short, BEPS provide a minimum standard for the energy consumption of a specific building type. These standards are continually updated \
                 to provide a snapshot of where buildings should be as time progresses. For the map below, 2021 standards are used.')
        st.write('Overall, we provide the map along with percentage of buildings by university which meet BEPS below. We further address building differences by university in the next section.' )
        st.plotly_chart(create_map(filtered_df), use_container_width=True)
        beps_summary = filtered_df.groupby('OWNEROFRECORD').apply(
        lambda group: (group['Meets_BEPS_2021'] == 'Yes').mean() * 100
        ).reset_index(name='Percentage_Meeting_BEPS')

        # Display the result
        st.dataframe(beps_summary)

    elif tab_selection == "University Comparison":
        st.sidebar.header("Filters")
        selected_year = st.sidebar.selectbox("Select Year", options, index=options.index(2023))
        if selected_year == 'Most Recent Year Reported':
            filtered_df = most_recent_df
        else:
            filtered_df = df[df['REPORTINGYEAR'] == selected_year]

        selected_universities = st.sidebar.multiselect("Select Universities", df['OWNEROFRECORD'].unique(), default=df['OWNEROFRECORD'].unique())
        filtered_df = filtered_df[filtered_df['OWNEROFRECORD'].isin(selected_universities)]

        # Section: Compare all universities
        st.header("Comparison of Universities in Washington D.C.")
        st.write("Here we compare various sustainability metrics across all universities in D.C. Given the data has missing years for most buildingss, most recent year reported provides the best \
                 way to compare universities building metrics. Based on energy consumption, we see that \
                 Trinity College has the highest average energy conspution while Catholic University has the lowest. \
                 Based on GHG, we see that Georgetown has the highest average emission rate while Catholic University has the lowest. \
                 Based on total water usage, we see American University has the higest average total water usage per building while Trinity College has the lowest. \
                 As a caveat, we note that some of these universities have incomplete or limited building data which could affect the overall results.")

        # Single Variable Plots - Energy, Water, Emissions
        st.subheader("Single Variable Comparisons")
        # Source EUI Plot by University
        st.plotly_chart(single_variable_eui_plot(filtered_df, 'SOURCEEUI_KBTU_FT', 'Average Source Energy Use Intensity (EUI) per Building by University'))
        # GHG Plot by University
        st.plotly_chart(single_variable_ghg_plot(filtered_df, 'TOTGHGEMISSINTENSITY_KGCO2EFT', 'Average Greenhouse Gas Emissions (GHG) per Building by University'))
        # Emissions Plot By University
        st.plotly_chart(single_variable_water_plot(filtered_df, 'WATERUSE_ALLWATERSOURCES_KGAL', "Average Water Usage per Building by University"))

        # Time Series
        time_series_options = {
            "SOURCE EUI (kBTU/ft²)": "html_map_files/SOURCEEUI_KBTU_FT_time_series_map.html",
            "Total GHG Emissions (Metric Tons CO₂e)": 'html_map_files/TOTGHGEMISSIONS_METRICTONSCO2E_time_series_map.html',
            "Water Use (All Water Sources in KGAL)": 'html_map_files/WATERUSE_ALLWATERSOURCES_KGAL_time_series_map.html'
        }

        # Allow the user to choose which time series to display
        st.write('In addtion, we provide time series plots of metrics by university. Again, we present this data with caution that there is missing and/or incomplete data \
                 for some univeristies.')
        selected_option = st.selectbox("Select a Time Series to Display", list(time_series_options.keys()))

        # Display the selected time series map
        display_html(time_series_options[selected_option])

    elif tab_selection == "GWU Buildings Comparison":
        # Section: Compare GWU Buildings
        st.header("Analysis of George Washington University Buildings")
        st.write("The scatter plot below visualizes GWU buildings by energy usage and GHG emissions with the size of each point determined by water usage. \
                 We conclude that in the most recent year reported, Ross Hall is the least sustainable building at GW. The most sustainable \
                 building at GWU is the Science and Engineering building which had zero GHG emissions in its most recent reported year. The visualization \
                 ultimately provides an easy way to see the most and least sustainable buildings at GWU.")
        # Filters for University Comparison tab
        st.sidebar.header("Filters")
        selected_year = st.sidebar.selectbox("Select Year", options, index=options.index('Most Recent Year Reported'))
        if selected_year == 'Most Recent Year Reported':
            filtered_df = most_recent_df
        else:
            filtered_df = df[df['REPORTINGYEAR'] == selected_year]

        selected_universities = st.sidebar.multiselect("Select Universities", df['OWNEROFRECORD'].unique(), default='George Washington University')
        filtered_df = filtered_df[filtered_df['OWNEROFRECORD'].isin(selected_universities)]

        gwu_df = filtered_df[filtered_df['OWNEROFRECORD'] == 'George Washington University']

        # Emissions vs Energy Usage
        fig2 = px.scatter(
            gwu_df,
            x='SOURCEEUI_KBTU_FT',
            y='TOTGHGEMISSINTENSITY_KGCO2EFT',
            color='PROPERTYNAME',
            size='WATERUSE_ALLWATERSOURCES_KGAL',
            hover_data=['REPORTINGYEAR'],
            title='Energy Usage vs GHG Emissions'
        )
        fig2.update_traces(marker=dict(
            opacity=0.9  
        ))

        st.plotly_chart(fig2)

        st.title("Building Comparison Tool")

        st.write("The interactive tool below allows for comparison of buildings individually from any university. \
                 The tool displays three time series for each variable and plots line plots for each building selected. \
                ")

        # Embed the Dash app using an iframe
        st.write("Below is the interactive map created using Dash:")
        st.components.v1.iframe(src="http://localhost:8052", height=700, width=900, scrolling=True) 

        st.write("Use the interactive map above to select buildings and view data comparisons.")
        
    elif tab_selection == "Conclusion and Recommendations":
        # Conclusion
        st.header("Conclusion")
        st.write("This dashboard provides insights into energy, water, and emissions metrics for university buildings in D.C. All in all, GWU and Howard \
                 have the most significant samples of buildings reporting data. GWU had one of the lowest percentage of buildings meet BEPS with 50% while \
                 Howard had one of the highest at 96%. Furthermore, for GWU, the results indicated that Ross Hall was the least sustainable building and \
                 the science and engineering hall was the most sustainable.")
        st.header("Recommendations")
        st.write("Overall, for GWU, we recommend looking at the sustainability practices at science and engineering building and applying those practices to \
                 other buildings. Specifically, reducing greenhouse gas emissions for buildings at GWU could be a long term goal. In addition, we recommend examining \
                 the sustainability practices of Ross Hall. The sustanaibilty metrics of Ross Hall are a significant outlier among buildings at GW and could be improved." )


if __name__ == "__main__":
    main()