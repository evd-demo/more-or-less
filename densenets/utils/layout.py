
import numpy as np
import plotly.graph_objs as go
from collections import OrderedDict 

layouts = {}


layouts['times font 18'] = go.Layout(
    font=dict(
        family="Times New Roman",
        color="black",
        size=18),
)

layouts['times font 17'] = go.Layout(
    font=dict(
        family="Times New Roman",
        color="black",
        size=17),
)

layouts['times font 20 white'] = go.Layout(
    font=dict(
        family="Times New Roman",
        color="black",
        size=20),
        template='simple_white'
)

layouts['triple heatmap'] = go.Layout(
    font=dict(
            family="Times New Roman",
            color="black",
            size=22),
    yaxis = dict(
        titlefont = dict(
            size = 35)),
    xaxis = dict(
        titlefont = dict(
            size = 40)),
    template='simple_white',
    xaxis2 = dict(
        tickangle=90,
    ),
    xaxis3 = dict(
        tickangle=90,
    ),

)

layouts['legend inside'] = go.Layout(
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=.5,
        xanchor="right",
        x=.25, 
        font=dict(
            size=30
        )
    )
)

layouts['triple line plot'] = go.Layout(
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=.675,
        xanchor="right",
        x=.1, 
        font=dict(
            size=30
        )
    ),
    font=dict(
        family="Times New Roman",
        color="black",
        size=20
    ),
    template='simple_white'
)

layouts['triple plot'] = go.Layout(

    font=dict(
        family="Times New Roman",
        color="black",
        size=20
    ),
    
    yaxis = dict(
        titlefont = dict(
            size = 30
        )
    ),

    xaxis2 = dict(
        titlefont = dict(
            size = 30
        )
    ),
    
    template='simple_white'
)

layouts['single plot'] = go.Layout(

    font=dict(
        family="Times New Roman",
        color="black",
        size=20
    ),
    
    yaxis = dict(
        titlefont = dict(
            size = 25
        )
    ),

    xaxis = dict(
        titlefont = dict(
            size = 25
        ),
        tickangle=90,
    ),
    
    template='simple_white', 

    autosize=False, 

    width=666, 

    height=500, 

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.8,
        xanchor="right",
        x=.6, 
        font=dict(
            size=25
        )
    ),
)

layouts['single plot right legend'] = go.Layout(

    font=dict(
        family="Times New Roman",
        color="black",
        size=20
    ),
    
    yaxis = dict(
        titlefont = dict(
            size = 25
        )
    ),

    xaxis = dict(
        titlefont = dict(
            size = 25
        ),
        tickangle=90,
    ),
    
    template='simple_white', 

    autosize=False, 

    width=666, 

    height=500, 

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.8,
        xanchor="right",
        x=.9, 
        font=dict(
            size=25
        )
    ),
)

layouts['vertical double plot'] = go.Layout(

    font=dict(
        family="Times New Roman",
        color="black",
        size=20
    ),
    
    yaxis = dict(
        titlefont = dict(
            size = 25
        )
    ),

    yaxis2 = dict(
        titlefont = dict(
            size = 25
        )
    ),

    xaxis = dict(
        titlefont = dict(
            size = 25
        ),
        tickangle=90,
    ),

    xaxis2 = dict(
        titlefont = dict(
            size = 25
        ),
        tickangle=90,
    ),
    
    template='simple_white', 

    autosize=False, 

    width=666, 

    height=700, 

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.9,
        xanchor="right",
        x=.6, 
        font=dict(
            size=25
        )
    ),
)

layouts['triple bar plot'] = go.Layout(
    
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=.675,
        xanchor="right",
        x=.075, 
        font=dict(
            size=30
        )
    ),

    yaxis = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),

    yaxis2 = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),

    yaxis3 = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),
)

layouts['triple line plot'] = go.Layout(
    
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=.675,
        xanchor="right",
        x=.3, 
        font=dict(
            size=30
        )
    ),

    yaxis = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),

    yaxis2 = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),

    yaxis3 = dict(
        titlefont = dict(
            size = 30
            ),
        tickfont = dict(
            size = 25
        )
    ),
)

layouts['legend inside horizontal'] = go.Layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.9,
        xanchor="left",
        x=0.2
    )
)

colors = OrderedDict()
colors['single'] = '#b2182b'
colors['horizontal'] = '#5ab4ac'
colors['vertical'] = '#7fbf7b'
colors['both'] = '#1b7837'

#wap it up
colors['single'] = '#9a2e54'
colors['horizontal'] = '#c2d4bf'
colors['vertical'] = '#e4bf85'
colors['both'] = '#628a84'

green_scale = ['#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#006837','#004529']


layouts['times font 20'] = go.Layout(
    font=dict(
        family="Times New Roman",
        color="black",
        size=20)
    )

def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale 