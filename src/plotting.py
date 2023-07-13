import matplotlib.pyplot as plt
import autograd.numpy as np
from pathlib import Path
import matplotlib.lines as mlines

from matplotlib.lines import Line2D

from global_ import *


def determine_y_lims(data, minimum=1e-12):
    """Determines how large the y-axis has to be given the data."""
    max_val = 1e-16
    min_val = 1e+16
    for entry in data:
        min_val = max(min(min(entry) / 10, min_val), minimum)
        max_val = max(max(entry) * 10, max_val)
    return min_val, max_val


def only_min(data):
    """
    Replaces every entry in every list in data by the minimum up to said entry.

    Args:
        data: list of lists
    """
    new_data = []
    for sub_data in data:
        new_subdata = [sub_data[0]]
        for i in range(1, len(sub_data)):
            entry = sub_data[i]
            if entry < new_subdata[-1]:
                new_subdata.append(entry)
            else:
                new_subdata.append(new_subdata[-1])
        new_data.append(new_subdata)
    return new_data


def gap_plotter(y_data: list,
                labels: list,
                iterations: int,
                directory: str = DIRECTORY,
                file_name: str = None,
                title: str = None,
                x_label: str = 'Number of iterations',
                y_label: str = 'Primal gap',
                x_scale: str = "log",
                y_scale: str = "log",
                x_lim: tuple = None,
                y_lim: tuple = None,
                linewidth: float = LINEWIDTH,
                styles: list = STYLES,
                colors: list = COLORS,
                markers: list = MARKERS,
                marker_size: int = MARKER_SIZE,
                n_markers: float = N_MARKERS,
                fontsize: int = FONT_SIZE,
                fontsize_legend: int = FONT_SIZE_LEGEND,
                legend: bool = True,
                legend_location: str = "upper right",
                vertical_lines = None):
    """Plots the data that is passed and saves it under "/DIRECTORY/" + str(file_name) + ".png".

    Args:
        y_data: list
            A list of lists. The sublists contain the data.
        labels: list
            The labels corresponding to the list of data. (Default is None.)
        iterations: int
        directory: string, Optional
            (Default is DIRECTORY.)
        file_name: string, None
            (Default is None.)
        title: string, Optional
            (Default is None.)
        x_label: string, Optional
            (Default is 'Number of iterations'.)
        y_label: string, Optional
            (Default is 'Primal gap'.)
        x_scale: string, Optional
            Defines the x_scale, options are "linear" and "log". (Default is "log".)
        y_scale: string, Optional
            Defines the y_scale, options are "linear" and "log". (Default is "log".)
        x_lim: tuple, Optional
            Plot cannot exceed these values in the x-axis. (Default is None.)
        y_lim: tuple, Optional
            Plot cannot exceed these values in the y-axis. (Default is (1, 10e-10).)
        linewidth: float
            (Default is LINEWIDTH.)
        styles: list, Optional
            (Default is STYLES.)
        colors: list, Optional
            (Default is COLORS.)
        markers: list, Optional
            (Default is MARKERS.)
        marker_size: int, Optional
            (Default is MARKER_SIZE.)
        n_markers: float, Optional
            (Default is N_MARKERS.)
        fontsize: int, Optional
            (Default is FONT_SIZE.)
        fontsize_legend: int, Optional
            (Default is FONT_SIZE_LEGEND.)
        legend: bool, Optional
            (Default is TRUE.)
        legend_location: str, Optional
            (Default is "upper right".)
        vertical_lines: list, Optional
            (Default is None.)

    """
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True

    legend_handles = []  # Create a list to store the legend handles

    for i in range(0, len(y_data)):
        # Plot the line with marker
        line = plt.plot(y_data[i], linestyle=styles[i], color=colors[i], label=labels[i], linewidth=linewidth, zorder=1)

        marker_indices = list(np.logspace(0, np.log10(len(y_data[i]) - 1), num=n_markers, base=10, dtype=int))
        marker_values = [y_data[i][idx] for idx in marker_indices]
        scatter = plt.scatter(marker_indices, marker_values, marker=markers[i], color=colors[i], s=marker_size ** 2,
                              label=labels[i], edgecolor="black", linewidth=1, zorder=2)

        legend_handles.append((line[0], scatter))  # Append line and scatter plot handles to the list

    if vertical_lines is not None:
        for line in vertical_lines:
            iteration = line[0]
            if iteration > x_lim[1]:
                iteration = 0
            line_label = line[1]
            if iteration < iterations:
                vert_line = plt.axvline(x=iteration, color='black', label=line_label)
                legend_line = mlines.Line2D([], [], color='black')
                legend_handles = [legend_line] + legend_handles
                labels = [line_label] + labels

    if legend:
        # Create the legend with line and marker handles
        plt.legend(handles=legend_handles, labels=labels, fontsize=fontsize_legend, loc=legend_location)

    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xscale(x_scale)
    plt.yscale(y_scale)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize, width=(linewidth / 1.25), length=(2 * linewidth),
                    direction='in', pad=(2.5 * linewidth))
    plt.grid(linestyle=':', linewidth=max(int(linewidth / 1.5), 1))

    x_num_ticks = iterations + 1
    x_num_ticks = int(np.log10(x_num_ticks))
    x_ticks_space = list(np.logspace(1, x_num_ticks, num=x_num_ticks, base=10))
    plt.xticks(x_ticks_space)

    if file_name is not None:
        directory = "./" + directory + "/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(directory + str(file_name) + ".png", dpi=300, bbox_inches='tight')
    plt.close()


def contour_plotter(x_values: np.ndarray,
                    y_values: np.ndarray,
                    z_values: np.ndarray,
                    y_values_line: np.ndarray = None,
                    directory: str = DIRECTORY,
                    file_name: str = None,
                    cmap: str = "inferno",
                    levels_min: float = 0.8,
                    levels_max: float = 2.2,
                    n_levels: int = 6,
                    ticks_min: float = 0.8,
                    ticks_max: float = 2.2,
                    n_ticks: int = 6,
                    title: str = None,
                    x_label: str = "Dimensions",
                    y_label: str = "Iterations",
                    z_label: str = "Local convergence rate",
                    line_label: str = "Theoretical phase shift iteration",
                    linewidth: int = LINEWIDTH,
                    fontsize: int = FONT_SIZE,
                    fontsize_legend: int = FONT_SIZE_LEGEND,
                    legend: bool = True,
                    ):
    """Makes a contour plot plotting the z_values in colour.

    Args:
        x_values: np.ndarray
        y_values: np.ndarray
        z_values: np.ndarray
        y_values_line: np.ndarray, Optional
            The y_values of the line that is to plot.
        directory: str, Optional
            (Default is DIRECTORY.)
        file_name: str, Optional
            (Default is None.)
        cmap: str, Optional
            (Default is "inferno".)
        levels_min: float, Optional
            (Default is 0.8.)
        levels_max: float, Optional
            (Default is 2.2.)
        n_levels: int, Optional
            (Default is 6.)
        ticks_min: float, Optional
            (Default is 0.8.)
        ticks_max: float, Optional
            (Default is 2.2.)
        n_ticks: int, Optional
            (Default is 6.)
        title: str, Optional
            (Default is None.)
        x_label: str, Optional
            (Default is "Dimensions".)
        y_label: str, Optional
            (Default is "Iterations".)
        z_label: str, Optional
            (Default is "Local convergence rate".)
        line_label: str, Optional
            (Default is "Theoretical phase shift iteration".)
        linewidth: int, Optional
            (Default is LINEWIDTH.)
        fontsize: int, Optional
            (Default is FONT_SIZE.)
        fontsize_legend: int, Optional
            (Default is FONT_SIZE_LEGEND.)
        legend: bool, Optional
            (Default is True.)"""

    X, Y = np.meshgrid(x_values, y_values)

    Z = np.array(z_values).T

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    contour_plot = plt.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(levels_min, levels_max, n_levels))
    if legend:
        cbar = plt.colorbar(contour_plot, ticks=np.linspace(ticks_min, ticks_max, n_ticks))
        cbar.set_label(z_label, fontsize=fontsize)
    ax.autoscale(False)
    if y_values_line is not None:
        ax.plot(x_values, y_values_line, zorder=1, linewidth=linewidth, color="black", label=line_label)
        ax.legend(fontsize=fontsize_legend)

    plt.tick_params(axis='both', which='major', labelsize=fontsize, width=(linewidth / 1.25), length=(2 * linewidth),
                    direction='in', pad=(2.5 * linewidth))

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)

    if file_name is not None:
        directory = "./" + directory + "/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(directory + str(file_name) + ".png", dpi=300, bbox_inches='tight')
    plt.close()
