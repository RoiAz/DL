import matplotlib.pyplot as plt

# https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
"""
# line 1 points
x1 = [1,2,3]
y1 = [2,4,1]
# plotting the line 1 points
plt.plot(x1, y1, label = "line 1")
 
# line 2 points
x2 = [1,2,3]
y2 = [4,1,3]
# plotting the line 2 points
plt.plot(x2, y2, label = "line 2")
 
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Two lines on same graph!')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()
"""    


# https://stackoverflow.com/questions/36008626/how-to-plot-a-graph-in-python-using-txt-file-with-float-value-in-it
"""
import numpy as np
import matplotlib.pyplot as plt

with open("test-1-14M.txt") as f:
    data = f.read()

data = data.split('\n')

x = [row.split(' ')[0] for row in data]
y = [row.split(' ')[0] for row in data]

fig = plt.figure()

ax1 = fig.add_subplot(70)

ax1.set_title("Plot title...")    
ax1.set_xlabel('your x label..')
ax1.set_ylabel('your y label...')

ax1.plot(x, y, c='r', label='the data')

leg = ax1.legend()

plt.show()

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