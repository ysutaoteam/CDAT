# coding:UTF-8
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import scipy.io as scio
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
import scipy.io as scio
from matplotlib import pyplot as plt
import numpy as np
from networkx import spring_layout


class Graph_Matrix:
    """
    Adjacency Matrix
    """
    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))



####由邻接矩阵生成属性拓扑图
def create_undirected_matrix(my_graph,chuanshu):
    nodes = ['0°-20°', '20°-40°', '40°-60°', '60°-80°', '80°-100°', '100°-120°', '120°-140°', '140°-160°','160°-180°']

    # matrix = [[2, 0, 0, 0, 5, 0, 0, 0, 0],  # a
    #           [0, 1, 0, 0, 0, 0, 8, 0, 0],  # b
    #           [0, 1, 3, 1, 0, 6, 0, 0, 0],  # c
    #           [0, 0, 1, 4, 2, 0, 0, 3, 0],  # d
    #           [0, 0, 0, 2, 4, 2, 0, 0, 0],  # e
    #           [0, 0, 0, 0, 2, 5, 2, 0, 0],  # f
    #           [0, 0, 2, 0, 0, 2, 4, 2, 0],  # g
    #           [0, 3, 0, 0, 0, 0, 1, 2, 1],  # h
    #           [0, 0, 5, 0, 0, 0, 0, 0, 1]]  # i
    #
    matrix = chuanshu

    my_graph = Graph_Matrix(nodes, matrix)

    graph_bian =my_graph.getAllEdges()  # ['0°-20°', '0°-20°', 8.0], ['20°-40°', '0°-20°', 8.0],
    return my_graph,graph_bian



def draw_undircted_graph(my_graph,i,path):
    G = nx.DiGraph()  # 建立一个空的有向图G
    for node in my_graph.vertices:
        G.add_node(str(node))

    # print(my_graph.edges[0][0])

    for edge in my_graph.edges:
        G.add_edge(str(edge[0]), str(edge[1]), weight = str(edge[2]))

    # for node in my_graph.vertices:
    #     G.add_node(str(node))
    # G.add_weighted_edges_from(my_graph.edges_array)

    print("nodes:", G.nodes())  # 输出全部的节点：
    print("edges:", G.edges())  # 输出全部的边：
    print("number of edges:", G.number_of_edges())  # 输出边的数量：


    pos = spring_layout(G)
    nx.draw(G, pos, with_labels=True)# with_labels=True用于显示图中节点上的名称
    # nx.draw(G, pos)

    # 把文件写进本地文件中
    #第一步先创建文件夹
    folder = os.path.exists("C:\\Users\\Administrator\\Desktop\\AT\\yu-1.1\\PD\\"  + path)
    shiwei = i // 3
    gewei = i % 3
    if not folder:
        os.makedirs("C:\\Users\\Administrator\\Desktop\\AT\\yu-1.1\\PD\\"  + path )
        if gewei==1:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/("  +str(shiwei+1) +")" +str(1)+".jpg")
            plt.clf()
        if gewei==2:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/(" +str(shiwei+1) +")"+str(2)+".jpg")
            plt.clf()
        if gewei==0:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/(" +str(shiwei)+")"+str(3)+".jpg")
            plt.clf()

    else:
        if gewei == 1:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/(" + str(shiwei + 1) +")"+str(1)+ ".jpg")
            plt.clf()
        if gewei == 2:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/(" + str(shiwei + 1) +")"+str(2)+ ".jpg")
            plt.clf()
        if gewei == 0:
            plt.savefig("C:/Users/Administrator/Desktop/AT/yu-1.1/PD" + "/" + path + "/(" + str(shiwei)+")"+str(3)+".jpg")
            plt.clf()#添加上这一行，画完第一个图后，重置一下

    # plt.savefig("F:/HOG-Picture/健康语音每个属性拓扑图/" + str(i) + ".jpg")
    # plt.savefig("F:/HOG-Picture/(不包含节点名称)健康语音每个属性拓扑图/" + str(i) + ".jpg")
    # plt.savefig("F:/HOG-Picture/(不包含节点名称)患者语音每个属性拓扑图/" + str(i) + ".jpg")
    # plt.show()


# dataFile = 'F:/HOG-Picture/CPPDD-dataset/SHIYAN/train/0_class/REC001-4-a.jpg.xls' #REC001-4-a.jpg.xls  14(w)-1-a(3).jpg.xls
# # dataFile_1 = 'F:/HOG-Picture/CPPDD-dataset/SHIYAN/train/1_class/14(w)-1-a(3).jpg.xls' #REC001-4-a.jpg.xls  14(w)-1-a(3).jpg.xls
# dataFile = 'F:/HOG-Picture/train/0_class/260602_o_3.jpg.xls' #260602_o_4.jpg.xls
# dataFile_1 = 'F:/HOG-Picture/train/1_class/No2-o(1).jpg.xls' #No2-o(1).jpg.xls  No1-a.jpg.xls
# data = scio.loadmat(dataFile)
# data_1 = scio.loadmat(dataFile_1)

for path in os.listdir( 'C:/Users/Administrator/Desktop/xingshibeijing/yu-1.1/0_class'):  #0_class  1_class
    data =scio.loadmat('C:/Users/Administrator/Desktop/xingshibeijing/yu-1.1/0_class' + "./" + path)
    # data_1 = scio.loadmat('C:/Users/Administrator/Desktop/xingshibeijing/yu-1.1/1_class' + "./" + path)

    # print(type(data))
    # print(data.keys())

    #shuju_1= data_1['1']  #无病人
    shuju= data['0']  #有病人

    # print(np.shape(shuju))

    my_graph = Graph_Matrix()  #实例化一个类
    for i in range(1,int(np.shape(shuju)[1]/9)+1):  #shuju:有病人  shuju_1：健康人
        print(i)
        created_graph,created_weight = create_undirected_matrix(my_graph,shuju[:,9*(i-1):9*i])
        draw_undircted_graph(created_graph,i,path)



#
# # 2，有向图
# G = nx.DiGraph()
# G.add_node(1)
# G.add_node(2)
# G.add_nodes_from([3, 4, 5, 6])
# G.add_cycle([1, 2, 3, 4])
# G.add_edge(1, 3)
# G.add_edges_from([(3, 5), (3, 6), (6, 7)])
# nx.draw(G,with_labels=True)
# # plt.savefig("youxiangtu.png")
# plt.show()
#
# # 4，无向图
# G = nx.Graph()
# G.add_node(1)
# G.add_node(2)
# G.add_nodes_from([3, 4, 5, 6])
# G.add_cycle([1, 2, 3, 4])
# G.add_edge(1, 3)
# G.add_edges_from([(3, 5), (3, 6), (6, 7)])
# nx.draw(G)
# # plt.savefig("wuxiangtu.png")
# plt.show()
#
#
# # 5，颜色节点图
# G = nx.Graph()
# G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (4, 5), (4, 6), (5, 6)])
# pos = nx.spring_layout(G)
#
# colors = [1, 2, 3, 4, 5, 6]
# nx.draw_networkx_nodes(G, pos, node_color=colors)
# nx.draw_networkx_edges(G, pos)
#
# plt.axis('off')
# # plt.savefig("color_nodes.png")
# plt.show()
#
#
