import matplotlib.pyplot as plt


SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22


class FontPresets:
    def __init__(self):
        self.previous_font_size = None
        self.previous_axes_titlesize = None
        self.previous_axes_labelsize = None
        self.previous_xtick_labelsize = None
        self.previous_ytick_labelsize = None
        self.previous_legend_fontsize = None
        self.previous_figure_titlesize = None

    def __enter__(self):
        self.previous_font_size = plt.rcParams['font.size']
        self.previous_axes_titlesize = plt.rcParams['axes.titlesize']
        self.previous_axes_labelsize = plt.rcParams['axes.labelsize']
        self.previous_xtick_labelsize = plt.rcParams['xtick.labelsize']
        self.previous_ytick_labelsize = plt.rcParams['ytick.labelsize']
        self.previous_legend_fontsize = plt.rcParams['legend.fontsize']
        self.previous_figure_titlesize = plt.rcParams['figure.titlesize']

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

    def __exit__(self, type, value, traceback):
        plt.rc('font', size=self.previous_font_size)
        plt.rc('axes', titlesize=self.previous_axes_titlesize)
        plt.rc('axes', labelsize=self.previous_axes_labelsize)
        plt.rc('xtick', labelsize=self.previous_xtick_labelsize)
        plt.rc('ytick', labelsize=self.previous_ytick_labelsize)
        plt.rc('legend', fontsize=self.previous_legend_fontsize)
        plt.rc('figure', titlesize=self.previous_figure_titlesize)
