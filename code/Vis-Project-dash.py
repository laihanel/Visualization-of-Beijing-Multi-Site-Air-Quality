'''
DATS 6401 Visualization of Complex Data - Lab5
Name: Aihan Liu
GWID: G45894738
Date: 4/2/2022
'''

from dash import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import pandas as pd
import math
import plotly.graph_objects as go
from datetime import date
import os

'''
# the style arguments for the main content page.
'''
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "6rem",
    "left": 0,
    "bottom": 20,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}



### READ DATA
PATH = os.getcwd()
print(PATH)
DATA_DIR = os.getcwd() + os.path.sep + 'PRSA_Data_20130301-20170228' + os.path.sep

if not os.path.exists(DATA_DIR + 'ALL.csv'):
    frames = []
    for file in os.listdir(DATA_DIR):
        if file[-4:] == '.csv':
            FILE_NAME = DATA_DIR + os.path.sep + file
            dataframe = pd.read_csv(FILE_NAME, header=0)
            frames.append(dataframe)

    result = pd.concat(frames)
    df = result.dropna(axis=0, how='any')

    cols = ["year", "month", "day", "hour"]
    df['date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d-%H')
    df['date_YM'] = df['year'] + df['month'] / 12

    df.to_csv(DATA_DIR + 'ALL.csv', index=False)
    print('Data merged!')
else:
    df = pd.read_csv(DATA_DIR + 'ALL.csv', header=0)
    print('Data read!')



print(len(df))
print(df.columns)
print(df.head())

'''
Dash layout
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('Project', external_stylesheets = [dbc.themes.SANDSTONE])
# my_app = dash.Dash('Project', external_stylesheets = external_stylesheets)

df = pd.read_csv(DATA_DIR + 'ALL.csv', header=0)
my_app.layout = html.Div([html.H1('Beijing Multi-Site Air-Quality', style = {'textAlign':'center'}), # a title
                          html.Div([
                              html.H3('Data Description', style = {'textAlign':'center'}),
                              html.H6('This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites. The air-quality data are from the Beijing Municipal Environmental Monitoring Center. The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration. The time period is from March 1st, 2013 to February 28th, 2017.'),
                              html.H6('PM2.5: PM2.5 concentration (ug/m^3)'),
                              html.H6('PM10: PM10 concentration (ug/m^3'),
                              html.H6('SO2: SO2 concentration (ug/m^3)'),
                              html.H6('NO2: NO2 concentration (ug/m^3)'),
                              html.H6('CO: CO concentration (ug/m^3)'),
                              html.H6('O3: O3 concentration (ug/m^3)'),
                              html.H6('TEMP: temperature (degree Celsius)'),
                              html.H6('PRES: pressure (hPa)'),
                              html.H6('DEWP: dew point temperature (degree Celsius)'),
                              html.H6('RAIN: precipitation (mm)'),
                              html.H6('WD: wind direction'),
                              html.H6('WSPM: wind speed (m/s)'),

                          ], style=SIDEBAR_STYLE,),
                              # style = {'width':'15%', 'display':'inline-block'}),
                                                                                                                                                                  # html.Br(),

                          html.Div([dbc.Tabs(id='layout_tab',
                                             children=[
                                                 dbc.Tab(label='Location-Based', tab_id='Air-Quality'),
                                                 dbc.Tab(label='Measurement-Based', tab_id='Analysis'),
                                                 dbc.Tab(label='Data and References', tab_id='Data and References'),
                                             ]),
                          ], style=CONTENT_STYLE),
                          # style = {'width':'79%', 'display':'inline-block'}),
                          html.Div(id='layout', style=CONTENT_STYLE)
], style= {"margin-top": "6rem"})

controls1 = html.Div(
    [html.H2('Parameters', style=TEXT_STYLE),
     html.Hr(),
     html.P('Choose a place:', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Aotizhongxin', 'value': 'Aotizhongxin'},
            {'label': 'Changping', 'value': 'Changping'},
            {'label': 'Dingling', 'value': 'Dingling'},
            {'label': 'Dongsi', 'value': 'Dongsi'},
            {'label': 'Guanyuan', 'value': 'Guanyuan'},
            {'label': 'Gucheng', 'value': 'Gucheng'},
            {'label': 'Huairou', 'value': 'Huairou'},
            {'label': 'Nongzhanguan', 'value': 'Nongzhanguan'},
            {'label': 'Shunyi', 'value': 'Shunyi'},
            {'label': 'Tiantan', 'value': 'Tiantan'},
            {'label': 'Wanliu', 'value': 'Wanliu'},
            {'label': 'Wanshouxigong', 'value': 'Wanshouxigong'},
        ], value='Aotizhongxin', clearable=False),
    html.Br(),

    html.P('Choose a time period:', style={'textAlign': 'center'}),
    dcc.RangeSlider(id='range_slider',min=2013.25,max=2017.25,step=0.25,
                    marks={2013.25:'2013 Mar', 2013.75: 'Sep',
                            2014.25:'2014 Mar', 2014.75: 'Sep',
                           2015.25:'2015 Mar', 2015.75: 'Sep',
                            2016.25:'2016 Mar',  2016.75: 'Sep',
                            2017.25:'2017 Mar'}),
    html.Div([
    html.Br(),
    html.Button("Download Data", id="download-csv"),
    dcc.Download(id="download-action")
    ])
])

@my_app.callback(
    Output(component_id="download-action", component_property="data"),
    [Input(component_id='download-csv', component_property='n_clicks')],
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "AirQuality.csv")

content_first_row = html.Div([
    html.H2('Analytics Dashboard', style=TEXT_STYLE),
    html.Hr(),
    dbc.Row([
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('PM2.5', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='PM25', children=['PM2.5'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbpm25', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('PM10', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='PM10', children=['PM10'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbpm10', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('SO2', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='SO2', children=['SO2'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbso2', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('NO2', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='NO2', children=['NO2'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbno2', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('CO', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='CO', children=['CO'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbco', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('O3', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='O3', children=['O3'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbo3', label='', value=False)
            ])),
    ]),

    html.Br(),
    dbc.Row([
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('TEMP', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='TEMP', children=['TEMP'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbtemp', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('PRES', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='PRES', children=['PRES'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbpres', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('DEWP', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='DEWP', children=['DEWP'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbdewp', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('RAIN', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='RAIN', children=['RAIN'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbrain', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('WD', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='WD', children=['WD'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbwd', label='', value=False)
            ])),
        dbc.Col(
            dbc.Card([dbc.CardBody(
                [html.H4('WSPM', className='card-title', style=CARD_TEXT_STYLE),
                 html.P(id='WSPM', children=['WSPM'], style=CARD_TEXT_STYLE)]),
                dbc.Switch(id='cbwspm', label='', value=False)
            ])),
    ])
])



@my_app.callback(
    Output(component_id='PM25', component_property='children'),
    Output(component_id='PM10', component_property='children'),
    Output(component_id='SO2', component_property='children'),
    Output(component_id='NO2', component_property='children'),
    Output(component_id='CO', component_property='children'),
    Output(component_id='O3', component_property='children'),
    Output(component_id='TEMP', component_property='children'),
    Output(component_id='PRES', component_property='children'),
    Output(component_id='DEWP', component_property='children'),
    Output(component_id='RAIN', component_property='children'),
    Output(component_id='WD', component_property='children'),
    Output(component_id='WSPM', component_property='children'),
    [Input(component_id='dropdown', component_property='value'),
     Input(component_id='range_slider', component_property='value')]
)

def update_cards(region, timerange):
    df_region = df[df['station'] == region]
    df_date = df_region[(df_region.date_YM >= timerange[0]) & (df_region.date_YM < timerange[1])]
    pm25 = "{:.2f}".format(df_date[['PM2.5']].mean()[0])
    pm10 = "{:.2f}".format(df_date[['PM10']].mean()[0])
    SO2 = "{:.2f}".format(df_date[['SO2']].mean()[0])
    NO2 = "{:.2f}".format(df_date[['NO2']].mean()[0])
    CO = "{:.2f}".format(df_date[['CO']].mean()[0])
    O3 = "{:.2f}".format(df_date[['O3']].mean()[0])
    TEMP = "{:.2f}".format(df_date[['TEMP']].mean()[0])
    PRES = "{:.2f}".format(df_date[['PRES']].mean()[0])
    DEWP = "{:.2f}".format(df_date[['DEWP']].mean()[0])
    RAIN = "{:.2f}".format(df_date[['RAIN']].mean()[0])
    WD = df_date['wd'].mode()
    WSPM = "{:.2f}".format(df_date[['WSPM']].mean()[0])
    return pm25, pm10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WD, WSPM


content_second_row = html.Div([
    html.H3('Timeseries of Select variables', style=TEXT_STYLE),
    dbc.Col(dcc.Graph(id='tsplot')),
])


@my_app.callback(
    Output(component_id='tsplot', component_property='figure'),
    [Input(component_id='dropdown', component_property='value'),
     Input(component_id='range_slider', component_property='value'),
     Input(component_id='cbpm25', component_property='value'),
     Input(component_id='cbpm10', component_property='value'),
     Input(component_id='cbso2', component_property='value'),
     Input(component_id='cbno2', component_property='value'),
     Input(component_id='cbco', component_property='value'),
     Input(component_id='cbo3', component_property='value'),
     Input(component_id='cbtemp', component_property='value'),
     Input(component_id='cbpres', component_property='value'),
     Input(component_id='cbdewp', component_property='value'),
     Input(component_id='cbrain', component_property='value'),
     Input(component_id='cbwd', component_property='value'),
     Input(component_id='cbwspm', component_property='value'),
     ]
)


def display_ts(region, timerange, pm25, pm10, so2, no2, co, o3, temp, pres, dewp, rain, wd, wspm):
    df_region = df[df['station'] == region]
    df_date = df_region[(df_region.date_YM >= timerange[0]) & (df_region.date_YM < timerange[1])]
    checklist = [pm25, pm10, so2, no2, co, o3, temp, pres, dewp, rain, wd, wspm]
    fig = go.Figure()
    var_name = df.columns[5:17]
    if wd:
        fig = go.Figure([go.Bar(x=df_date['wd'], y=df_date['wd'].value_counts())])
    else:
        for k in range(len(checklist)):
            if checklist[k]:
                fig.add_trace(go.Scatter(x=df_date['date'], y=df_date[var_name[k]], mode='lines', opacity=.5, name=var_name[k]))
    return fig


firsttab = html.Div([dbc.Row([dbc.Col(controls1, width=5),
                              dbc.Col(content_first_row, width=7)]),
                     html.Hr(),
                     content_second_row
])

'''
second tab
'''
controls2 = html.Div(
    [html.H2('Parameters', style=TEXT_STYLE),
     html.Hr(),
     html.P('Region Pick:', style={'textAlign': 'left'}),
    dbc.Checklist(
        id='checklist_region',
        options=[
            {'label': 'Aotizhongxin', 'value': 'Aotizhongxin'},
            {'label': 'Changping', 'value': 'Changping'},
            {'label': 'Dingling', 'value': 'Dingling'},
            {'label': 'Dongsi', 'value': 'Dongsi'},
            {'label': 'Guanyuan', 'value': 'Guanyuan'},
            {'label': 'Gucheng', 'value': 'Gucheng'},
            {'label': 'Huairou', 'value': 'Huairou'},
            {'label': 'Nongzhanguan', 'value': 'Nongzhanguan'},
            {'label': 'Shunyi', 'value': 'Shunyi'},
            {'label': 'Tiantan', 'value': 'Tiantan'},
            {'label': 'Wanliu', 'value': 'Wanliu'},
            {'label': 'Wanshouxigong', 'value': 'Wanshouxigong'},
            # {'label': 'ALL', 'value': 'ALL'},
        ], value='Aotizhongxin', inline=True, inputCheckedClassName="border border-success bg-success"),

    html.Br(),
    dbc.Button( "Select all region", id="checkall", className="me-2", outline=True, color="primary", n_clicks=0),
    dbc.Button( "Clear all region", id="uncheck", className="me-2", outline=True, color="primary", n_clicks=0),

    html.Br(),
     html.Br(),
     html.P('Measurement Pick:', style={'textAlign': 'left'}),
     html.Div([
         dbc.RadioItems(
             options=[
                 {"label": "PM2.5", "value": "PM2.5"},
                 {"label": "PM10", "value": "PM10"},
                 {"label": "SO2", "value": "SO2"},
                 {"label": "NO2", "value": "NO2"},
                 {"label": "CO", "value": "CO"},
                 {"label": "O3", "value": "O3"},
                 {"label": "TEMP", "value": "TEMP"},
                 {"label": "PRES", "value": "PRES"},
                 {"label": "DEWP", "value": "DEWP"},
                 {"label": "RAIN", "value": "RAIN"},
                 {"label": "WSPM", "value": "WSPM"},
                 {"label": "WD", "value": "wd"},
             ],
             id="measurement_picker", inline=True, value='PM2.5')
     ]),

    html.Br(),

    html.P('Time Step:', style={'textAlign': 'left'}),
    html.Div(
    [dbc.RadioItems(
        options = [{"label":"Yearly", 'value':'yearly'},
                   {"label":"Monthly", 'value':'monthly'},
                   {"label": "Daily", 'value': 'daily'},
                   {"label": "Hourly", 'value': 'hourly'},],
        id="timestep", inline=True, value='monthly'
    )
    ]),

    html.Br(),
    html.P('Time Period:', style={'textAlign': 'left'}),
    dcc.DatePickerRange(
        id='datepicker',
        start_date=date(2013, 3, 1),
        end_date=date(2017, 2, 28),
        display_format='MMMM Y, DD',
        start_date_placeholder_text='MMMM Y, DD',
        style={'font-size': '8px'}
    ),

    html.Br(),
    html.P('Notes', style={'textAlign': 'left'}),
    dcc.Textarea(
        id='textarea-example',
        value='This is the place where you could take notes',
        style={'width': '100%', 'height': 300},
    ),
    # html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
])
@my_app.callback(Output(component_id='checklist_region', component_property='value'),
                 Output(component_id='uncheck', component_property='n_clicks'),  # reset the number of clicks for uncheck bottom
                 [Input(component_id='checkall', component_property='n_clicks'),
                  Input(component_id='uncheck', component_property='n_clicks')],  # input should be inside a []
                 [State("checklist_region", "options")]
)

def select_all_none(all_selected, all_clean, options):
    all_or_none = []
    if all_selected:
        all_or_none = [option["value"] for option in options]
    if all_clean:
        all_or_none = []
    return all_or_none, 0

# @my_app.callback(Output(component_id='output', component_property='children'),  # reset the number of clicks for uncheck bottom
#                  [Input(component_id='checklist_region', component_property='value'),
#                   Input(component_id='timestep', component_property='value'),
#                   Input(component_id="datepicker", component_property="start_date"),
#                   Input(component_id="datepicker", component_property="end_date"),
#                   Input(component_id="measurement_picker", component_property="value")],  # input should be inside a []
# )
#
# def print_output(region, timestep, start, end, measure):
#     return f'The reigon is {region}, \ntime is {timestep}, \nstart date is {start}, end date is {end}, \nmeasurement is {measure}'

content_third_row = html.Div([
    html.H2('Graphical Analysis', style=TEXT_STYLE),
    dbc.Row([dcc.Graph(id='tsplot2')]),

    html.Br(),
    dbc.Row([
        dbc.Col([dbc.RadioItems(
            options = [{"label":"overlay", 'value':'overlay'},
                       {"label":"stack", 'value':'stack'},
                       {"label": "relative", 'value': 'relative'},
                       {"label": "group", 'value': 'group'},],
            id="histoption", inline=True, value='group'),
                dcc.Graph(id='histogram')]),
        dbc.Col(dcc.Graph(id='boxplot')),])

])

@my_app.callback(Output(component_id='tsplot2', component_property='figure'),  # reset the number of clicks for uncheck bottom
                 Output(component_id='histogram', component_property='figure'),
                 Output(component_id='boxplot', component_property='figure'),
                 [Input(component_id='checklist_region', component_property='value'),
                  Input(component_id='timestep', component_property='value'),
                  Input(component_id="datepicker", component_property="start_date"),
                  Input(component_id="datepicker", component_property="end_date"),
                  Input(component_id="measurement_picker", component_property="value"),
                  Input(component_id="histoption", component_property="value"),],  # input should be inside a []
)

def timeseries_output(region, timestep, start, end, measure, histoption):
    df_region = df[df['station'].isin(region)]
    df_date= df_region.loc[pd.to_datetime(df_region["date"]).between(*pd.to_datetime([start, end]))]
    if measure == 'wd':
        layout = go.Layout(title=f'Barplot for {measure} in {region}')
        fig = go.Figure([go.Bar(x=df_date['wd'], y=df_date['wd'].value_counts())], layout=layout)
    else:
        data = []
        data2 = []
        data3 = []
        for station_name in region:
            df_group = df_date[df_date['station'] == station_name]
            if timestep == 'yearly':
                per = pd.to_datetime(df_group['date']).dt.to_period("Y")
                df_fin = df_group.groupby(per).mean()
                df_fin['date'] = df_fin.index
                df_fin['date'] = df_fin['date'].dt.strftime('%Y')
                binsize = len(df_fin)
            if timestep == 'monthly':
                per = pd.to_datetime(df_group['date']).dt.to_period("M")
                df_fin = df_group.groupby(per).mean()
                df_fin['date'] = df_fin.index
                df_fin['date'] = df_fin['date'].dt.strftime('%Y-%m')
                binsize = len(df_fin)//2
            if timestep == 'daily':
                per = pd.to_datetime(df_group['date']).dt.to_period("D")
                df_fin = df_group.groupby(per).mean()
                df_fin['date'] = df_fin.index
                df_fin['date'] = df_fin['date'].dt.strftime('%Y-%m-%d')
                binsize = 30
            if timestep == 'hourly':
                df_fin = df_group.copy()
                df_fin['date'] = pd.to_datetime(df_fin['date'])
                binsize = 30
            trace = go.Scatter(x=df_fin['date'], y=df_fin[measure], name=station_name, opacity=.3)
            trace2 = go.Histogram(x=df_fin[measure], nbinsx=binsize, name=station_name, opacity=.5)
            trace3 = go.Box(y=df_fin[measure], name=station_name)
            data.append(trace)
            data2.append(trace2)
            data3.append(trace3)
        # Layout of the plot
        layout = go.Layout(title=f'Time Series')
        fig = go.Figure(data=data, layout=layout)
        layout2 = go.Layout(title=f'Histogram', barmode=histoption)
        fig2 = go.Figure(data=data2, layout=layout2)
        layout3 = go.Layout(title=f'Boxplot')
        fig3 = go.Figure(data=data3, layout=layout3)
    return fig, fig2, fig3


secondtab = html.Div([dbc.Row([dbc.Col(controls2, width=3),
                              dbc.Col(content_third_row, width=9)]),
])


# third tab, upload map
thirdtab = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
        html.H5('Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality'
                ' improvement in Beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205), 20170457.'),
        dcc.Link('https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data', href='/')
    ])

@my_app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


####################### Layout callback
@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='layout_tab', component_property='active_tab')]  # input should be inside a []
)


def update_layout(ques):
    if ques == 'Air-Quality':
        return firsttab
    elif ques == 'Analysis':
        return secondtab
    elif ques == 'Data and References':
        return thirdtab

my_app.run_server(
    # debug=True,
    port = 8048,
    host = '0.0.0.0'
)


