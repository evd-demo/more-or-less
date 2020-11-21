from plotly.subplots import make_subplots

# 
import plotly
import plotly.graph_objs as go
import plotly.express as px
from .layout import *

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

def horizontal_stack(traces, titles, xlabel, ylabel, showyticklabels=False, showcolorbar=True):

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


