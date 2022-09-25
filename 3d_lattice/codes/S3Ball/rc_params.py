from matplotlib import pyplot as plt

FONTSIZE = 14

def init():
    plt.rcParams.update({
        'text.usetex': True, 
        'font.family': 'serif', 
        'font.serif': ['Computer Modern'], 
        'font.size': FONTSIZE, 
    })
