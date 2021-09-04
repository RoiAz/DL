import matplotlib.pyplot as plt

""" Based on examples online, such as:
    # https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
    # https://stackoverflow.com/questions/36008626/how-to-plot-a-graph-in-python-using-txt-file-with-float-value-in-it
"""

def plot_graph(first_file, second_file, graph_title: str, x_title: str, y_title: str):

    x1, y1, x2, y2 = [], [], [], []

    for line in open(first_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'first_file lines: {lines}')
        for idx, num in enumerate(lines):
#             print(num)
            x1.append(float(num))
            y1.append(idx)
#         print(f'X Lines: {x1}')
    
    #plotting first line
    plt.plot(y1, x1, marker = 'o', c = 'g', label = "Discriminator Loss")

    
    for line in open(second_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'second_file lines: {lines}')
        for idx, num in enumerate(lines):
            x2.append(float(num))
            y2.append(idx)
#         print(f'Y Lines: {y2}')
    
    #plotting second line
    plt.plot(y2, x2, marker = 'o', c = 'r', label = "Generator Loss")


    plt.title(graph_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
#     plt.yticks(y2)
    
    #show legend on graph
    plt.legend()
    
    #func to show graph
    plt.show()
    
    
def plot_inception_graph(first_file, graph_title: str, x_title: str, y_title: str):

    x1, y1 = [], []

    for line in open(first_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'first_file lines: {lines}')
        for idx, num in enumerate(lines):
#             print(num)
            x1.append(float(num))
            y1.append(idx)
#         print(f'X Lines: {x1}')
    
    #plotting first line
    plt.plot(y1, x1, marker = 'o', c = 'b', label = "Inception Score")

    
    plt.title(graph_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
#     plt.yticks(y2)
    
    #show legend on graph
    plt.legend()
    
    #func to show graph
    plt.show()
    

def plot_all_inceptions_graph(first_file, second_file, third_file, forth_file, graph_title: str, x_title: str, y_title: str):

    x1, y1, x2, y2, x3, y3, x4, y4 = [], [], [], [], [], [], [], []

    for line in open(first_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'first_file lines: {lines}')
        for idx, num in enumerate(lines):
#             print(num)
            x1.append(float(num))
            y1.append(idx)
#         print(f'X Lines: {x1}')
    
    #plotting first line
    plt.plot(y1, x1, marker = 'o', c = 'g', label = "vanilla_gan")

    
    for line in open(second_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'second_file lines: {lines}')
        for idx, num in enumerate(lines):
            x2.append(float(num))
            y2.append(idx)
#         print(f'Y Lines: {y2}')
    
    #plotting second line
    plt.plot(y2, x2, marker = 'o', c = 'r', label = "sn_gan")

    for line in open(third_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'second_file lines: {lines}')
        for idx, num in enumerate(lines):
            x3.append(float(num))
            y3.append(idx)
#         print(f'Y Lines: {y2}')
    
    #plotting second line
    plt.plot(y3, x3, marker = 'o', c = 'b', label = "w_gan")
    
    for line in open(forth_file, 'r'):
        lines = [float(i) for i in line.split()]
        print(f'second_file lines: {lines}')
        for idx, num in enumerate(lines):
            x4.append(float(num))
            y4.append(idx)
#         print(f'Y Lines: {y2}')
    
    #plotting second line
    plt.plot(y4, x4, marker = 'o', c = 'k', label = "sn_w_gan")
    
    plt.title(graph_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
#     plt.yticks(y2)
    
    #show legend on graph
    plt.legend()
    
    #func to show graph
    plt.show()