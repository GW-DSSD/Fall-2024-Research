import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/samharris/Desktop/Fall-2024-Research/App/Data/University_Buildings.csv')

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


# Your existing data setup
gwu_df = df[['PROPERTYNAME', 'LATITUDE', 'LONGITUDE', 'REPORTINGYEAR', 
             'WATERUSE_ALLWATERSOURCES_KGAL', 'TOTGHGEMISSIONS_METRICTONSCO2E', 'ELECTRICITYUSE_GRID_KWH']]

app = Dash(__name__)

# Store selected buildings globally (using a list to maintain multiple selections)
selected_buildings = []

app.layout = html.Div([
    html.H1("University Building Data Map"),
    dcc.Graph(id='map', config={'scrollZoom': False}),
    html.Div(id='building-data')
])

@app.callback(
    Output('map', 'figure'),
    Input('map', 'clickData')
)
def update_map(clickData):
    # Map setup remains the same
    fig = px.scatter_mapbox(
        gwu_df,
        lat='LATITUDE',
        lon='LONGITUDE',
        hover_name='PROPERTYNAME',
        hover_data={'LATITUDE': False, 'LONGITUDE': False},
        zoom=11
    )
    fig.update_layout(mapbox_style="carto-positron")
    return fig

@app.callback(
    Output('building-data', 'children'),
    Input('map', 'clickData')
)
def display_building_data(clickData):
    global selected_buildings

    if clickData is None:
        return "Click on a building to see its data."
    
    # Get clicked building name
    building_name = clickData['points'][0]['hovertext']
    
    # Toggle building selection
    if building_name in selected_buildings:
        selected_buildings.remove(building_name)  # Remove if already selected
    else:
        selected_buildings.append(building_name)  # Add if not already selected

    # Filter the DataFrame to include only selected buildings
    selected_info = gwu_df[gwu_df['PROPERTYNAME'].isin(selected_buildings)]

    # Create line charts
    water_fig = px.line(
        selected_info,
        x='REPORTINGYEAR',
        y='WATERUSE_ALLWATERSOURCES_KGAL',
        color='PROPERTYNAME',  # Different line colors for different buildings
        title='Water Usage (KGAL)'
    )
    emissions_fig = px.line(
        selected_info,
        x='REPORTINGYEAR',
        y='TOTGHGEMISSIONS_METRICTONSCO2E',
        color='PROPERTYNAME',
        title='Emissions (Metric Tons CO2e)'
    )
    electricity_fig = px.line(
        selected_info,
        x='REPORTINGYEAR',
        y='ELECTRICITYUSE_GRID_KWH',
        color='PROPERTYNAME',
        title='Electricity Usage (KWH)'
    )

    # Enhance layout appearance
    for fig in [water_fig, emissions_fig, electricity_fig]:
        fig.update_layout(
            paper_bgcolor='white',  # White background
            plot_bgcolor='white',   # White plotting area
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            font=dict(size=14)
        )

    return html.Div([
        html.H3("Selected Buildings: " + ", ".join(selected_buildings) if selected_buildings else "No buildings selected."),
        dcc.Graph(figure=water_fig),
        dcc.Graph(figure=emissions_fig),
        dcc.Graph(figure=electricity_fig)
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)