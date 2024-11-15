import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import os

def display_html(html_file_name):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    html_file_path = os.path.join(current_directory, html_file_name)
    if os.path.exists(html_file_path):
        with open(html_file_path, "r") as f:
            plotly_map_html = f.read()
        # Apply custom CSS to remove white space
        st.markdown(
            f"""
            <style>
                .stComponent.e1e1i5i31 {{
                    width: 100%;
                    height: 100%;
                    margin: 0;
                    padding: 0;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        components.html(plotly_map_html, width=800, height=600)
    else:
        st.error("HTML file not found. Please make sure it exists in the same directory as your app.py.")
        st.write("File not found:", html_file_path)  # Debug print

# Load Data
def load_data():
    return pd.read_csv('Data/University_Buildings.csv')  

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
        hover_data=['REPORTINGYEAR','OWNEROFRECORD', 'REPORTEDADDRESS', 'SOURCEEUI_KBTU_FT', 'TOTGHGEMISSIONS_METRICTONSCO2E', 'WATERUSE_ALLWATERSOURCES_KGAL', 'Meets_BEPS_2021'],
        color='Meets_BEPS_2021',
        color_discrete_map={'Yes': 'green', 'No': 'red'},
        mapbox_style='carto-positron',  
        zoom=10.9,  
        center={"lat": 38.89511, "lon": -77.03637},
        title="Building Locations in Washington D.C."
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7)) 
    fig.update_layout(
        height=600,
        width=1200,
        title="Building Locations in Washington D.C.",
        margin={"r":0,"t":40,"l":0,"b":0}  #
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
        labels={'average_eui': 'Average Source EUI (kBtu/ft²)', 'OWNEROFRECORD': 'University'},
        color_continuous_scale='Viridis'
    )
    
    return fig



# Main App
def main():
    st.title("University Buildings Energy and Sustainability Analysis")
    st.subheader("An Overview of Energy, Water, and Emissions Data for University Buildings in Washington D.C.")

    # Load data
    df = load_data()
    df = add_beps_column(df)  # Add the Meets_BEPS_2021 column

    # Sidebar filters
    st.sidebar.header("Filters")

    # Get all unique years that have been reported in the dataset
    all_years = sorted(df['REPORTINGYEAR'].unique(), reverse=True)

    # Get the most recent year reported for each property (PID)
    most_recent_df = df.loc[df.groupby('PID')['REPORTINGYEAR'].idxmax()]

    # Add the "Most Recent Year" option to the list of years
    options = all_years + ['Most Recent Year']

    # Create the selectbox for year
    selected_year = st.sidebar.selectbox(
        "Select Year", 
        options,  # Include all years + the "Most Recent Year" option
        index=options.index('Most Recent Year')  # Default to "Most Recent Year"
    )

    # Adjust the filtered data based on the selected year
    if selected_year == 'Most Recent Year':
        # Filter data for the most recent year for each property (PID)
        filtered_df = most_recent_df
    else:
        # Filter data based on the selected year
        filtered_df = df[df['REPORTINGYEAR'] == selected_year]

    selected_universities = st.sidebar.multiselect("Select Universities", df['OWNEROFRECORD'].unique(), default=df['OWNEROFRECORD'].unique())

    # Filter data for selected universities
    filtered_df = filtered_df[filtered_df['OWNEROFRECORD'].isin(selected_universities)]

    # Introduction
    st.header("Introduction")
    st.write("This report provides insights into energy and water usage, emissions, and energy efficiency of university buildings in Washington D.C.")

    # Map of all buildings
    st.header("Map of University Buildings in Washington D.C.")
    st.plotly_chart(create_map(filtered_df), use_container_width=True)

    # Section: Compare all universities
    st.header("Comparison of Universities in Washington D.C.")
    st.write("Here we compare various sustainability metrics across all universities in D.C.")
    
    # Single Variable Plots - Energy, Water, Emissions
    st.subheader("Single Variable Comparisons")
    # Source EUI Plot by University
    st.plotly_chart(single_variable_eui_plot(filtered_df, 'SOURCEEUI_KBTU_FT', 'Average Source Energy Use Intensity (EUI) by University'))

    display_html("html_map_files/SOURCEEUI_KBTU_FT_time_series_map.html")
    display_html('html_map_files/TOTGHGEMISSIONS_METRICTONSCO2E_time_series_map.html')
    display_html('html_map_files/TOTGHGEMISSIONS_METRICTONSCO2E_time_series_map.html')

    st.header("Building Performance Metrics")
    metrics_df = filtered_df[['PROPERTYNAME', 'SOURCEEUI_KBTU_FT', 
                            'TOTGHGEMISSINTENSITY_KGCO2EFT', 
                            'WATERUSE_ALLWATERSOURCES_KGAL']].copy()
    st.dataframe(metrics_df)

    # Section: Compare GWU Buildings
    st.header("Analysis of George Washington University Buildings")

    gwu_df = filtered_df[filtered_df['OWNEROFRECORD'] == 'George Washington University']

    # Emissions vs Energy Usage
    fig2 = px.scatter(
        gwu_df,
        x='SOURCEEUI_KBTU_FT',
        y='TOTGHGEMISSINTENSITY_KGCO2EFT',
        color='PROPERTYNAME',
        hover_data=['REPORTINGYEAR'],
        title='Energy Usage vs GHG Emissions'
    )
    st.plotly_chart(fig2)

    # Conclusion
    st.header("Conclusion")
    st.write("This dashboard provides insights into energy, water, and emissions metrics for university buildings in D.C., aiding in data-driven decision making towards energy efficiency and sustainability improvements.")

if __name__ == "__main__":
    main()

