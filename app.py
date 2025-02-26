import pandas as pd
import plotly.graph_objects as go
import numpy as np
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pickle

with open('module.pkl','rb') as f:
    model = pickle.load(f)

# Load datasets
df = pd.read_csv('Datasets/final_electricity_data.csv')
# Convert Date column to datetime before setting as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df2 = pd.read_csv('Datasets/electrical_appliance_consumption.csv')
df2 = df2[df2['year'] == 2019].reset_index(drop=True)

df3 = pd.read_csv('Datasets/full_pred.csv')

df4 = pd.read_csv('Datasets/electricity_appliance_wise_data.csv')
df4['Date'] = pd.to_datetime(df4['Date']).reset_index(drop=True)
# df4 = df4[df4['Date'].dt.year == 2021].reset_index(drop=True)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = 'Power Track'

# Dropdown options
dropdown_options = [
    {'label': 'Time-Series Plot', 'value': 'Time-Series Plot'},
    {'label': 'Appliance-wise Consumption', 'value': 'Appliance-wise Consumption'},
    {'label': 'Electricity Consumption Forecast', 'value': 'Electricity Consumption Forecast'},
    {'label': 'Faulty Devices', 'value': 'Faulty Devices'}
]

# Layout
app.layout = html.Div(children=[
    html.Div(className='row', children=[
        html.Div(className='four columns div-user-controls', children=[
            html.H2('Power Track Dashboard', style={'font-family': 'Trebuchet MS'}),
            dcc.Dropdown(
                id='stockselector',
                options=dropdown_options,
                value=None,
                style={'backgroundColor': '#1E1E1E'},
                placeholder="Select an Option"
            ),
            html.P(id="graph-info", style={'color': 'white', 'margin-top': '10px'})
        ]),
        html.Div(className='eight columns div-for-charts bg-grey', children=[
            dcc.Graph(id='timeseries', config={'displayModeBar': False})
        ])
    ])
])

# Helper function to calculate z-scores
def calculate_zscore(df, column, window):
    r = df[column].rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    return (df[column] - m) / s

# Callback to update the graph
@app.callback(
    Output('timeseries', 'figure'), [Input('stockselector', 'value')]
    )
def update_timeseries(selected_value):
    if selected_value == 'Electricity Consumption Forecast':
        return electricity_consumption_forecast()
    elif selected_value == 'Faulty Devices':
        return faulty_devices()
    elif selected_value == 'Appliance-wise Consumption':
        return appliance_wise_consumption()
    else:  # Time-Series Plot
        return time_series_plot()
@app.callback(
    Output('graph-info', 'children'),
    Input('stockselector', 'value')
)
def update_graph_info(selected_value):
    descriptions = {
        'Time-Series Plot': "This graph shows the electricity consumption over time for different years. You can select a year from the dropdown in the graph to see the data for that specific year.",
        'Appliance-wise Consumption': "This pie chart represents the electricity consumption of different household appliances for each month. You can select a month to view its breakdown.",
        'Electricity Consumption Forecast': "This graph shows the actual vs predicted electricity consumption. Red markers indicate excessive consumption, which may require attention.",
        'Faulty Devices': "This graph highlights anomalies in power consumption for different appliances. Spikes in the data may indicate faulty devices consuming excess energy."
    }
    return descriptions.get(selected_value, "Select an option from the dropdown to view the corresponding graph.")


def electricity_consumption_forecast():
        df_sub = df3
        df_anoms = df_sub[df_sub['MAE'] >= 15].reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Total_Consumption'], mode='lines', name='Actual Consumption', line_color="#19E2C5"))
        fig.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Predicted_Consumption'], mode='lines', name='Predicted Consumption', line_color="#C6810B"))
        fig.add_trace(go.Scatter(x=df_anoms['Date'], y=df_anoms['Total_Consumption'], mode='markers', name='Excess Consumption', marker=dict(size=5, line=dict(width=5, color='#C60B0B'))))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            autosize=True,
            yaxis_title="Consumption (kWh)",
            xaxis_title="Date",
            title={'text': 'Time-Series Plot & Forecasting Electricity Consumption for this year', 'font': {'color': 'white'}, 'x': 0.5}
        )
        return fig

def faulty_devices():
    df_sub = df4.copy()  # Ensure df4 is not modified directly
    appliances = ['Kitchen Appliances', 'Fridge', 'AC', 'Other Appliances', 'Washing Machine']
    
    for appliance in appliances:
        df_sub[f'{appliance.lower()}_zscore'] = calculate_zscore(df_sub, appliance, 30)

    df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sub.fillna(0, inplace=True)

    fig_anom = go.Figure()
    for appliance in appliances:
        df_anom = df_sub[df_sub[f'{appliance.lower()}_zscore'] > 5]
        fig_anom.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[appliance], mode='lines', name=f'Actual {appliance}', line_color="#19E2C5"))
        fig_anom.add_trace(go.Scatter(x=df_anom['Date'], y=df_anom[appliance], mode='markers', name=f'Fluctuations in {appliance}', marker=dict(size=10)))

    # Dropdown buttons including "All Appliances"
    buttons = [
        {
            'label': 'All Appliances',
            'method': 'update',
            'args': [{'visible': [True] * (len(appliances) * 2)},  # Show all traces
                     {'title': 'Anomalies in power consumption of All Appliances'}]
        }
    ]

    buttons += [
        {
            'label': appliance,
            'method': 'update',
            'args': [{'visible': [i == idx * 2 or i == idx * 2 + 1 for i in range(len(appliances) * 2)]},
                     {'title': f'Anomalies in power consumption of {appliance}'}]
        }
        for idx, appliance in enumerate(appliances)
    ]

    fig_anom.update_layout(
        updatemenus=[{
            'active': 0,
            'buttons': buttons
        }],
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        title_font_color="#90E219",
        font_color="#90E219",
        autosize=True
    )
    
    return fig_anom

def appliance_wise_consumption():
    df_sub = df2
    months = df_sub['month'].unique()
    appliances = ['Fridge', 'Kitchen Appliances', 'AC', 'Washing Machine', 'Other Appliances']
    irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)', 'rgb(175, 49, 35)', 'rgb(36, 73, 147)']

    fig = go.Figure(data=[
        go.Pie(
            name=month,
            labels=appliances,
            values=[df_sub[df_sub['month'] == month][appliance].values[0] for appliance in appliances],
            marker_colors=irises_colors,
            hole=0.3
        ) for month in months
    ])

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=[dict(
                label=month,
                method="update",
                args=[{"visible": [i == idx for i in range(len(months))]},
                        {"title": f"{month} Consumption Distribution (%) by each Appliance"}]
            ) for idx, month in enumerate(months)]
        )],
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        title_font_color="#90E219",
        font_color="#90E219",
        width=700,
        height=700
    )
    return fig

def time_series_plot():
    df_sub = df.copy()  # Create a copy to avoid modifying the original data
    
    # Ensure datetime index is properly formatted
    if not isinstance(df_sub.index, pd.DatetimeIndex):
        df_sub.index = pd.to_datetime(df_sub.index)
    
    # Extract unique years from the dataset
    years = sorted(df_sub.index.year.unique())
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each year (initially all invisible except the first year)
    buttons = []
    for i, year in enumerate(years):
        df_year = df_sub[df_sub.index.year == year]
        visible = [i == j for j in range(len(years))]  # Only the selected year is visible

        fig.add_trace(go.Scatter(
            x=df_year.index,
            y=df_year['Total_Consumption'],
            mode='lines',
            name=f'Electricity Consumption {year}',
            visible=visible[i],  # Set visibility based on the selected year
            line=dict(color="#19E2C5")
        ))

        # Create a dropdown button for each year
        buttons.append(dict(
            label=str(year),
            method='update',
            args=[
                {'visible': visible},  # Update visibility for all traces
                {'title': f'Time-Series Plot of Electricity Consumption for {year}'}
            ]
        ))
    
    # Update layout with dropdown
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        yaxis_title="Consumption (kWh)",
        xaxis_title="Date",
        title={
            'text': f'Time-Series Plot of Electricity Consumption for {years[0]}',
            'font': {'color': 'white'},
            'x': 0.5
        },
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True
            )
        ]
    )
    
    return fig
# Run the app
if __name__ == '__main__':
    app.run(debug=True)