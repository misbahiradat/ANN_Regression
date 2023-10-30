

__all__ = ['parity_plot', 'uncertainty_curve', 'residual_plot', 'visualize_SHAP_global', 'visualize_Sobol_global']

# Cell
#export
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display_html, HTML
from scipy.stats import norm

# Cell
def parity_plot(y_target, y_pred, split_type = 'cv', ax = None, interactive = True, fig_height = 5, \
                dpi = 80, imageloc = None, fig = None, return_fig = False, thumbnail = False):

    # Axis limits and ticks
    min_ax = np.min([*y_target, *y_pred]); max_ax = np.max([*y_target, *y_pred])
    ax_lims = [min_ax - 0.05 * (max_ax - min_ax), max_ax + 0.05 * (max_ax - min_ax)]
    axticks = np.linspace(*ax_lims, 5)
    x_ax_label = 'Target'
    y_ax_label = 'Prediction'
    if np.max(np.abs(axticks)) <= 0.1:
        str_format = '{:.3f}'
    elif np.max(np.abs(axticks)) >= 100:
        str_format = '{:.1f}'
    else:
        str_format = '{:.2f}'
    ax_ticks_labels = [str_format.format(i).rstrip('0').rstrip('.') for i in axticks]

    # Setting the color of the markers
    if split_type.lower() in ['training', 'cv']:
        marker_color = np.array([29,150,157])/255
    elif split_type.lower() == 'test':
        marker_color = np.array([244,177,131])/255
    else:
        raise ValueError('split_type must be one of "training", "cv" and "test"')

    # Interactive case (for platform)
    if interactive:

        # Generate plotly figure if None is entered for fig parameter
        if fig is None:
            fig = go.Figure(layout_xaxis_range = ax_lims, layout_yaxis_range = ax_lims)

        hovertextformat = 'Target: %{x:.3f}' + \
                          '<br>Prediction: %{y:.3f}<extra></extra>'

        # Adding parity line and scatter points
        fig.add_trace(go.Scatter(x = ax_lims, y = ax_lims, mode = 'lines', \
                                 line = dict(color = 'rgb(120,120,120)', width = 2), hovertemplate = \
                                 '<extra></extra>', showlegend = False))
        fig.add_trace(go.Scatter(x = y_target, y = y_pred, mode = 'markers', \
                                 marker = dict(color = 'rgb({},{},{})'.format(*marker_color), size = 8), \
                                 hovertemplate = hovertextformat, showlegend = False, xaxis = 'x'))
        layout = go.Layout(xaxis = go.layout.XAxis(tickmode = 'array', range = ax_lims, tickvals = axticks, showgrid = True, \
                                                   title_text = x_ax_label, titlefont = dict(size = 16), \
                                                   tickfont = dict(size = 14), ticktext = ax_ticks_labels, \
                                                   zeroline = False, title_standoff = 10, scaleratio = 1), \
                           yaxis = go.layout.YAxis(tickmode = 'array', range = ax_lims, tickvals = axticks, showgrid = True, \
                                                   title_text = y_ax_label, tickfont = dict(size = 14), \
                                                   titlefont = dict(size = 16), ticktext = ax_ticks_labels, \
                                                   zeroline = False, title_standoff = 10), autosize = True, \
                           showlegend = False, margin = dict(t = 35, b = 10, r = 15), \
                           margin_autoexpand = True)

        fig.update_layout(layout)

        config = {'responsive': True, 'displayModeBar': True, 'displaylogo': False, \
                  'modeBarButtonsToRemove':['lasso2d', 'autoScale2d','toggleSpikelines']}

        if not return_fig:
            fig.show(config = config)


    # Matplotlib non-interactive case
    elif not interactive:

        # Creating axes if it is not specified in the parameters
        if ax is None:
            fig = plt.figure(figsize = (fig_height * 1.2, fig_height))
            ax = fig.gca()
        else:
            fig = ax.get_figure()

        # Removing wide white margins
        ax.get_figure().tight_layout()

        ax.grid(linestyle = '--', linewidth = 0.5)   # Showing the grid lines
        ax.set_aspect('equal','box') # Setting the axis equal

        # Plotting the parity line
        ax.plot([ax_lims[0], ax_lims[1]], [ax_lims[0], ax_lims[1]], 'k-', linewidth = 1)

        # Plotting the predictions with the model built with all training data
        ax.plot(y_target, y_pred, 'o', markeredgecolor = marker_color, \
                markerfacecolor = marker_color, markersize = 8, \
                markeredgewidth = 0.5)

        # Setting the ticks and tick label fontsize
        ax.set_xticks(axticks)
        ax.set_yticks(axticks)
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xticklabels(ax_ticks_labels)
        ax.set_yticklabels(ax_ticks_labels)
        ax.tick_params(axis='both', which = 'major', labelsize = 16)

        # Setting the axes limits of the parity plots
        ax.set(xlim = ax_lims, ylim = ax_lims)

        # Setting the labels for plots
        ax.set_xlabel('Target', fontsize = 20, labelpad = 10)
        ax.set_ylabel('Prediction', fontsize = 20, labelpad = 10)

        # Removing the axes ticks and labels for thumbnail images
        if thumbnail:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Saving the plot to a png file
        if not imageloc is None:  # Save the file to the location
            fig.savefig(imageloc + '.png', dpi = 150, facecolor = 'w', edgecolor = 'k',
                orientation = 'portrait', transparent = False, bbox_inches = 'tight', pad_inches = 0)

        # Returns the fig object as output (used for platform purposes)
        if return_fig:
            return fig

# Cell
def uncertainty_curve(y, yh, sigma, ax = None, interactive = True, xlabel = None, fig_size = (6,4), \
                      imageloc = None):

    x = np.linspace(yh - 2.7 * sigma, yh + 2.7 * sigma, num = 51)
    f = norm.pdf(x, yh, sigma)
    linecolor = np.array([29,150,157])/255

    if ax is None:
        fig = plt.figure(figsize = fig_size)
        ax = fig.gca()

    # Removing wide white margins
    ax.get_figure().tight_layout()

    # Plotting the vcurve
    ax.plot(x, f, '-', color = linecolor, linewidth = 1)

    # Vertical line from curve to the x axis indicating mean prediction
    ax.plot([yh, yh],[0,norm.pdf(yh, yh, sigma)], '-', color = linecolor, linewidth = 3, \
            label = 'Prediction')

    # Plotting dashed vertical lines depicting one and two standard deviations
    ax.plot([yh - sigma, yh - sigma],[0, norm.pdf(yh - sigma, yh, sigma)], '--', color = linecolor)
    ax.plot([yh - 2 * sigma,  yh - 2 * sigma],[0, norm.pdf(yh - 2 * sigma, yh, sigma)], '--', \
            color = linecolor)
    ax.plot([yh + sigma, yh + sigma],[0,norm.pdf(yh + sigma, yh, sigma)], '--', color = linecolor)
    ax.plot([yh + 2 * sigma, yh + 2 * sigma],[0,norm.pdf(yh + 2 * sigma, yh, sigma)], '--', color = linecolor)

    # Plotting the vertical line indicating the target value
    ax.plot([y, y],[0,norm.pdf(y, yh, sigma)], '-', color = np.array([244,177,131])/255, \
            linewidth = 3, label = 'Target (' + np.round(y, 2).astype('str') + ')')

    ax.grid(linestyle = '--', linewidth = 0.5) # Gridlines

    ax.tick_params(axis='both', which = 'major', labelsize = 14)
    ax.set(xlim = [yh - 2.7 * sigma, yh + 2.7 * sigma], ylim = [0, 1.05 * norm.pdf(y, y, sigma)])
    ax.set_xticks(np.round(np.linspace(yh - 2 * sigma, yh + 2 * sigma,5),2))
    ax.set_xticklabels(np.round(np.linspace(yh - 2 * sigma, yh + 2 * sigma, 5), 2).astype('str'))
    ax.legend(fontsize = 14, loc = 'upper right')  # Legend

    # Adding label name to x axis if specified as parameter
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = 14)

    if not imageloc is None:  # Save the file to the location
        plt.savefig(imageloc + '.png', dpi=150, facecolor='w', edgecolor='k',
            orientation='portrait', transparent=False, bbox_inches='tight', pad_inches=0)


# Cell
def residual_plot(x, Res, ax = None, split_type = 'cv', interactive = True, xlabel = None, \
                  fig_height = 5, dpi = 80, fig = None, imageloc = None, return_fig = False, \
                  thumbnail = False):

    # Axis limits and ticks
    axlims_x = [np.min(x) - 0.05 * np.ptp(x), np.max(x) + 0.05 * np.ptp(x)]
    axlims_y = [np.min(Res) - 0.05 * np.ptp(Res), np.max(Res) + 0.05 * np.ptp(Res)]
    axticks_x = np.linspace(*axlims_x, 5)
    axticks_y = np.linspace(*axlims_y, 5)
    x_ax_label = xlabel
    y_ax_label = 'Residual'
    if np.max(np.abs(axticks_x)) <= 0.1:
        str_format = '{:.3f}'
    elif np.max(np.abs(axticks_x)) >= 100:
        str_format = '{:.1f}'
    else:
        str_format = '{:.2f}'
    ax_xticks_labels = [str_format.format(i).rstrip('0').rstrip('.') for i in axticks_x]
    if np.max(np.abs(axticks_y)) <= 0.1:
        str_format = '{:.3f}'
    elif np.max(np.abs(axticks_y)) >= 100:
        str_format = '{:.1f}'
    else:
        str_format = '{:.2f}'
    ax_yticks_labels = [str_format.format(i).rstrip('0').rstrip('.') for i in axticks_y]

    # Setting the color of the markers
    if split_type.lower() in ['training', 'cv']:
        marker_color = np.array([29,150,157])/255
    elif split_type.lower() == 'test':
        marker_color = np.array([244,177,131])/255
    else:
        raise ValueError('split_type must be one of "training", "cv" and "test"')

    # Interactive plot with plot.ly
    if interactive:

        # Generate plotly figure if None is entered for fig parameter
        if fig is None:
            fig = go.Figure()

        hovertextformat = 'Residual : %{y:.4f}' + \
                          '<br>{}'.format(x_ax_label) +  ' : %{x:.4f}<extra></extra>'

        fig.add_trace(go.Scatter(x = axlims_x, y = [0, 0], mode = 'lines', \
                                 line = dict(color = 'rgb(120,120,120)', width = 2), hovertemplate = \
                                 '<extra></extra>', showlegend = False))
        fig.add_trace(go.Scatter(x = x, y = Res, mode = 'markers',
                                 marker = dict(color = 'rgb({},{},{})'.format(*marker_color), size = 8), \
                                 hovertemplate = hovertextformat, showlegend = False))
        layout = go.Layout(xaxis = go.layout.XAxis(tickmode = 'array', range = axlims_x, tickvals = axticks_x, showgrid = True, \
                                           title_text = x_ax_label, titlefont = dict(size = 16), \
                                           tickfont = dict(size = 14), ticktext = ax_xticks_labels, \
                                           zeroline = False, title_standoff = 10), \
                   yaxis = go.layout.YAxis(tickmode = 'array', range = axlims_y, tickvals = axticks_y, showgrid = True, \
                                           title_text = y_ax_label, tickfont = dict(size = 14), \
                                           titlefont = dict(size = 16), ticktext = ax_yticks_labels, \
                                           zeroline = False, title_standoff = 10), autosize = True, \
                   showlegend = False, margin = dict(t = 35, b = 10, r = 15), \
                   margin_autoexpand = True)

        fig.update_layout(layout)

        config = {'responsive': True, 'displayModeBar': True, 'displaylogo': False, \
                  'modeBarButtonsToRemove':['lasso2d', 'autoScale2d','toggleSpikelines']}

        if not return_fig:
            fig.show(config = config)


    # Noninteractive matplotlib plot
    else:

        # Create an axes handle if
        if ax is None:
            fig = plt.figure(figsize = (1.3 * fig_height, fig_height))
            ax = fig.gca()

        # Removing wide white margins
        ax.get_figure().tight_layout()

        # Plotting the horizontal line on y = 0  for reference
        ax.plot(axlims_x, [0, 0], '-', color = [0.5,0.5,0.5])

        # Plotting the curve
        ax.plot(x, Res, 'o', markeredgecolor = marker_color, markerfacecolor = marker_color, markersize = 8)

        ax.grid(linestyle = '--', linewidth = 0.5) # Gridlines

        # Setting the ticks and tick label fontsize
        ax.set_xticks(axticks_x)
        ax.set_yticks(axticks_y)
        ax.set_xlim(axlims_x)
        ax.set_ylim(axlims_y)
        ax.set_xticklabels(ax_xticks_labels)
        ax.set_yticklabels(ax_yticks_labels)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

        # Adding label name to x axis if specified as parameter
        if xlabel is not None:
            ax.set_xlabel(x_ax_label, fontsize = 20, labelpad = 10)
        ax.set_ylabel(y_ax_label, fontsize = 20, labelpad = 10) # y label

        # Removing the axes ticks and labels for thumbnail images
        if thumbnail:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Saving the plot to a png file
        if not imageloc is None:  # Save the file to the location
            fig.savefig(imageloc + '.png', dpi = 150, facecolor = 'w', edgecolor = 'k',
                orientation = 'portrait', transparent = False, bbox_inches = 'tight', pad_inches = 0)

        # Returns the fig object as output (used for platform purposes)
        if return_fig:
            return fig

# Cell
def visualize_SHAP_global(SHAP_results, n_features = None, fig_width = 5, dpi = 80, \
                          fig = None, imageloc = None, return_fig = False, thumbnail = False, \
                          interactive = True, ax = None):

    # Extracting Sobol global and confidence interval from dictionry
    shap_global = SHAP_results['SHAP']
    feature_names = SHAP_results['feature_names']

    # Extraction of first n_features number of features and the sum of others'
    if isinstance(n_features, int) and n_features < feature_names.__len__():
        shap_global = np.concatenate((shap_global[:n_features], \
                                      np.array([np.sum(shap_global[n_features:])])))
        feature_names = [*feature_names[:n_features], 'Sum of others']

    # Axis limits and ticks
    axlims_x = [0, np.max(shap_global) + 0.05 * np.ptp(shap_global)]
    axticks_x = np.linspace(*axlims_x, 5)
    x_ax_label = 'mean(|SHAP value|)'
    if np.max(np.abs(axticks_x)) <= 0.1:
        str_format = '{:.3f}'
    elif np.max(np.abs(axticks_x)) >= 100:
        str_format = '{:.1f}'
    else:
        str_format = '{:.2f}'
    ax_xticks_labels = [str_format.format(i).rstrip('0').rstrip('.') for i in axticks_x]

    # Interactive plotly plot
    if interactive:

        hovertextformat = 'SHAP : %{x:.4f}' + \
                          '<br>Feature : %{y}<extra></extra>'

        # Generate plotly figure if None is entered for fig parameter
        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Bar(x = shap_global[::-1], y = feature_names[::-1], orientation = 'h', \
                             marker = dict(color='rgba(29, 150, 157, 255)'), showlegend = False, \
                             hovertemplate = hovertextformat))
        layout = go.Layout(xaxis = go.layout.XAxis(tickmode = 'array', range = axlims_x, tickvals = axticks_x, \
                                                   showgrid = True, title_text = x_ax_label, titlefont = dict(size = 16), \
                                                   tickfont = dict(size = 14), ticktext = ax_xticks_labels, \
                                                   zeroline = False, title_standoff = 10), \
                           yaxis = go.layout.YAxis(ticks = "outside", tickcolor = 'white', ticklen = 10, \
                                                   tickfont = dict(size = 14)), autosize = True, \
                           margin_autoexpand = True, margin = dict(t = 35, b = 10, r = 15))

        fig.update_layout(layout)

        config = {'responsive': True, 'displayModeBar': True, 'displaylogo': False, \
                  'modeBarButtonsToRemove':['lasso2d', 'autoScale2d','toggleSpikelines']}

        if not return_fig:
            fig.show(config = config)

    # Noninteractive matplotlib case
    elif not interactive:

        # Create an axes handle if
        if ax is None:
            fig = plt.figure(figsize = (fig_width, 0.1 * fig_width * shap_global.shape[0] + 0.5))
            ax = fig.gca()

        # Removing wide white margins
        ax.get_figure().tight_layout()
        ax.grid(linestyle = '--', linewidth = 0.5)   # Showing the grid lines

        # Plotting the bars
        ax.barh(np.arange(feature_names.__len__()-1, -1, -1), shap_global, align = 'center', \
                color = np.array([29,150,157])/255)

        # Setting the ticks and tick label fontsize
        ax.set_xticks(axticks_x)
        ax.set_yticks(np.arange(feature_names.__len__()-1, -1, -1))
        ax.set_xlim(axlims_x)
        ax.set_ylim([-0.6, feature_names.__len__() - 0.4])
        ax.set_xticklabels(ax_xticks_labels)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel(x_ax_label, fontsize = 20, labelpad = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

        # Removing the axes ticks and labels for thumbnail images
        if thumbnail:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Saving the plot to a png file
        if not imageloc is None:  # Save the file to the location
            fig.savefig(imageloc + '.png', dpi = 150, facecolor = 'w', edgecolor = 'k',
                orientation = 'portrait', transparent = False, bbox_inches = 'tight', pad_inches = 0)

        # Returns the fig object as output (used for platform purposes)
        if return_fig:
            return fig

# Cell
def visualize_Sobol_global(sobol_indices, n_features = None, fig_width = 5, dpi = 80, \
                          fig = None, imageloc = None, return_fig = False, thumbnail = False, \
                          interactive = True, ax = None):

    # Extracting Sobol global and confidence interval from dictionry
    sobol_global = sobol_indices['ST']
    sobol_global_conf = sobol_indices['ST_conf']
    feature_names = sobol_indices['feature_names']

    # Extraction of first n_features number of features and the sum of others'
    if isinstance(n_features, int) and n_features < feature_names.__len__():
        sobol_global = np.concatenate((sobol_global[:n_features], \
                                      np.array([np.sum(sobol_global[n_features:])])))
        sobol_global_conf = np.concatenate((sobol_global_conf[:n_features], \
                                            np.array([np.sum(sobol_global_conf[n_features:])])))
        feature_names = [*feature_names[:n_features], 'Sum of others']

    # Axis limits and ticks
    axlims_x = [0, np.max(sobol_global + sobol_global_conf) + \
                0.05 * np.ptp(sobol_global + sobol_global_conf)]
    axticks_x = np.linspace(*axlims_x, 5)
    x_ax_label = 'S<sub>T</sub>, Sobol total effect index'
    if np.max(np.abs(axticks_x)) <= 0.1:
        str_format = '{:.3f}'
    elif np.max(np.abs(axticks_x)) >= 100:
        str_format = '{:.1f}'
    else:
        str_format = '{:.2f}'
    ax_xticks_labels = [str_format.format(i).rstrip('0').rstrip('.') for i in axticks_x]

    # Interactive plotly plot
    if interactive:

        hovertextformat = 'S<sub>T</sub> : %{x:.4f}' + \
                          '<br>Conf : %{text:.4f}' + \
                          '<br>Feature: %{y}<extra></extra>'

        # Generate plotly figure if None is entered for fig parameter
        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Bar(x = sobol_global[::-1], y = feature_names[::-1], orientation = 'h', \
                             marker = dict(color='rgba(29, 150, 157, 255)'), showlegend = False, \
                             hovertemplate = hovertextformat, text = sobol_global_conf[::-1], \
                             error_x = dict(type = 'data', array = sobol_global_conf[::-1])))
        layout = go.Layout(xaxis = go.layout.XAxis(tickmode = 'array', range = axlims_x, tickvals = axticks_x, \
                                                   showgrid = True, title_text = x_ax_label, titlefont = dict(size = 16), \
                                                   tickfont = dict(size = 14), ticktext = ax_xticks_labels, \
                                                   zeroline = False, title_standoff = 10), \
                           yaxis = go.layout.YAxis(ticks = "outside", tickcolor = 'white', ticklen = 10, \
                                                   tickfont = dict(size = 14)), autosize = True, \
                           margin_autoexpand = True, margin = dict(t = 35, b = 10, r = 15))

        fig.update_layout(layout)

        config = {'responsive': True, 'displayModeBar': True, 'displaylogo': False, \
                  'modeBarButtonsToRemove':['lasso2d', 'autoScale2d','toggleSpikelines']}

        if not return_fig:
            fig.show(config = config)


    # Noninteractive matplotlib case
    elif not interactive:

        # Create an axes handle if
        if ax is None:
            fig = plt.figure(figsize = (fig_width, 0.1 * fig_width * sobol_global.shape[0] + 0.5))
            ax = fig.gca()

        # Removing wide white margins
        ax.get_figure().tight_layout()
        ax.grid(linestyle = '--', linewidth = 0.5)   # Showing the grid lines

        # Plotting the bars
        ax.barh(np.arange(feature_names.__len__()-1, -1, -1), sobol_global, align = 'center', \
                color = np.array([29,150,157])/255)

        # Plotting the confidence bars
        ax.errorbar(sobol_global, np.arange(feature_names.__len__()-1, -1, -1), fmt = '.', xerr = sobol_global_conf, color = 'k')

        # Setting the ticks and tick label fontsize
        ax.set_xticks(axticks_x)
        ax.set_yticks(np.arange(feature_names.__len__()-1, -1, -1))
        ax.set_xlim(axlims_x)
        ax.set_ylim([-0.6, feature_names.__len__() - 0.4])
        ax.set_xticklabels(ax_xticks_labels)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('$S_T$, Sobol total effect index', fontsize = 20, labelpad = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

        # Removing the axes ticks and labels for thumbnail images
        if thumbnail:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Saving the plot to a png file
        if not imageloc is None:  # Save the file to the location
            fig.savefig(imageloc + '.png', dpi = 150, facecolor = 'w', edgecolor = 'k',
                orientation = 'portrait', transparent = False, bbox_inches = 'tight', pad_inches = 0)

        # Returns the fig object as output (used for platform purposes)
        if return_fig:
            return fig