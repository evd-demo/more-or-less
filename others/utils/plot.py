"""
August 2020
Plotting functionality for EvD
"""
# 
import plotly
import plotly.graph_objs as go
import plotly.express as px
import sys
sys.path.append('/Users//code/EvD/densenets/utils')
from layout import *


def simple_heatmap(z, x, y=None, xaxis_title="", yaxis_title="", title=""):

    if y==None:
        fig = go.Figure(data=go.Heatmap(z=z, x=x, colorscale = px.colors.sequential.YlGn))
    else:
        fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale = px.colors.sequential.YlGn))
    fig.update_layout(xaxis_title=xaxis_title)
    fig.update_layout(yaxis_title=yaxis_title)
    fig.update_layout(title=title)

    return fig

def combined_heatmap(z, x, y=None, xaxis_title="", yaxis_title="", title="", trace=False):

    # 
    bvals = [-0.5, 0, 0.5, 1, 1.5]
    colors_ = list(colors.values())
    dcolorsc = discrete_colorscale(bvals, colors_)
    colorbar = dict(thickness=20,tickvals=[0.15, 0.5, 1, 1.35],ticktext=['single','deq only', 'weq only', 'both ens.'], tickfont=dict(size=35))
    
    if y==None:
        fig = go.Figure(data=go.Heatmap(z=z, x=x, colorscale = dcolorsc, colorbar=colorbar))
    else:
        fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale = dcolorsc, colorbar=colorbar))
    fig.update_layout(xaxis_title=xaxis_title)
    fig.update_layout(yaxis_title=yaxis_title)
    fig.update_layout(title=title)
    fig.update_layout(layouts['times font 20 white'])

    if trace:
        return fig.data[0]
    return fig

def get_trace(z, x, y=None, xaxis_title="", yaxis_title="", title=""):

    bvals = [-0.5, 0, 0.5, 1, 1.5]
    colors = list(colors.values())
    dcolorsc = discrete_colorscale(bvals, colors)
    colorbar = dict(thickness=20,tickvals=[0.15, 0.5, 1, 1.35],ticktext=['single','hor. only', 'ver. only', 'both ens.'])
    
    if y==None:
        return go.Heatmap(z=z, x=x, colorscale = dcolorsc, colorbar=colorbar)
    else:
        return go.Heatmap(z=z, x=x, y=y, colorscale = px.colors.sequential.YlGn)

from plotly.subplots import make_subplots

# 
import plotly
import plotly.graph_objs as go
import plotly.express as px

def vertical_stack(traces, titles, xlabel, ylabel):

    """
        Vertical stack traces with titles, x, and y label.
    """

    # Make the subplot
    num_rows = len(traces)
    final_fig = make_subplots(rows=num_rows, cols=1)

    for row in range(num_rows):
        
        # Add the subplot
        final_fig.add_trace(traces[row], row=row+1, col=1)

        # disable y ticks
        final_fig.update_xaxes(showticklabels=False, row=row+1, col=1)

    final_fig.update_yaxes(showticklabels=True, row=1, col=1)

    # Add x and y label
    eval('final_fig.update_layout(yaxis'+str(int(num_rows/2)+1)+'_title=ylabel)')
    eval('final_fig.update_layout(xaxis'+str(num_rows)+'_title=xlabel)')

    return final_fig

def horizontal_stack(traces, titles, xlabel, ylabel, showyticklabels=False):

    """
        Horizontal stack traces with titles, x, and y label.
    """

    if showyticklabels:
        horizontal_spacing = 0.03
    else:
        horizontal_spacing = 0.01
    
    # Make the subplot
    num_columns = len(traces)
    assert len(traces) == len(titles)
    final_fig = make_subplots(rows=1, cols=num_columns, horizontal_spacing = horizontal_spacing, subplot_titles=titles)

    for column in range(num_columns):
        
        # Add the subplot
        final_fig.add_trace(traces[column], row=1, col=column + 1)

        # disable y ticks
        final_fig.update_yaxes(showticklabels=showyticklabels, row=1, col=column+1)
        

    final_fig.update_yaxes(showticklabels=True, row=1, col=1)
    
    final_fig.update_yaxes(titlefont = layouts['triple heatmap']['yaxis']['titlefont'], row=1, col=1)
    final_fig.update_xaxes(titlefont = layouts['triple heatmap']['xaxis']['titlefont'], row=1, col=str(int(num_columns/2)+1))
    

    # Add x and y label
    eval('final_fig.update_layout(yaxis_title=ylabel)')
    eval('final_fig.update_layout(xaxis'+str(int(num_columns/2)+1)+'_title=xlabel)')
    
    return final_fig

def horizontal_stack_multiple(traces, titles, xlabel, ylabel, showyticklabels=False, showcolorbar=True):

    """
        Horizontal stack traces with titles, x, and y label.
    """

    if showyticklabels:
        horizontal_spacing = 0.03
    else:
        horizontal_spacing = 0.01
    
    # Make the subplot
    num_columns = len(traces)
    assert len(traces) == len(titles)
    final_fig = make_subplots(rows=1, cols=num_columns, horizontal_spacing = horizontal_spacing, subplot_titles=titles)

    for column in range(num_columns):
        
        # Add the subplot
        for trace in traces[column]:
            final_fig.add_trace(trace, row=1, col=column + 1)

        # disable y ticks
        final_fig.update_yaxes(showticklabels=showyticklabels, row=1, col=column+1)

    final_fig.update_yaxes(showticklabels=True, row=1, col=1)
    
    if not showcolorbar:
        final_fig.update(layout_coloraxis_showscale=False)
    
    final_fig.update_yaxes(titlefont = layouts['triple heatmap']['yaxis']['titlefont'], row=1, col=1)
    final_fig.update_xaxes(titlefont = layouts['triple heatmap']['xaxis']['titlefont'], row=1, col=str(int(num_columns/2)+1))

    # Add x and y label
    eval('final_fig.update_layout(yaxis_title=ylabel)')
    eval('final_fig.update_layout(xaxis'+str(int(num_columns/2)+1)+'_title=xlabel)')
    
    return final_fig