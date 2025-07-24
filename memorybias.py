# Libraries
from abc import ABC, abstractmethod
import numpy as np
import math, statistics
import random
import json
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as color
import pandas as pd
import re


# Principal functions

def forgettingEffect(gammaValues, initValue=0.95, t=1):

    return(initValue*np.power(math.e,(-t/gammaValues)))

def errorRateFunction(data):
    prop = []
    for i in range(data.shape[1]):
        dataUnit = data[:,i]
        total = len(dataUnit) - sum(dataUnit==-1)
        zeros = sum(dataUnit == 0)
        propUnit = zeros/total
        prop = np.append(prop, [propUnit], axis=0)

    return(prop)


# Project: Global
def proj(stud_mem,cond,gammaValues_Mem, gammaValues_Prac, timeForgetting):

    # --------------------------------
    #          T1: Retrieval
    # --------------------------------

    # Get stud and data: retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t1_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: study
    # --------------------------------

    # Get stud and data: study
    stud = np.where(stud_mem.isMem == True)
    studData = stud_mem.answer[stud].copy()

    # mean
    t1_study = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: all
    # --------------------------------

    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t1_all = errorRateFunction(data=studData)


    # --------------------------------
    #     T2: forgetting phase
    # --------------------------------

    # Forgetting
    stud_mem.forgetting_effect_Manual(sameItems=False, gammaValues_Mem=gammaValues_Mem, gammaValues_Prac=gammaValues_Prac, timeForgetting=timeForgetting)

    if (onMasteryBias):
        stud_mem.applyMasteryBias(cond=cond)
    elif (onAttritionBias):
        stud_mem.applyAttritionBias(cond=cond)

    # --------------------------------
    #          T2: Retrieval
    # --------------------------------

    # T2_retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t2_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T2: All
    # --------------------------------

    # T2_all
    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t2_all = errorRateFunction(data=studData)

    # --------------------------------
    #          Plot options
    # --------------------------------

    # Create graphs
    graph_T1 = plt.subplot(2,1,1)
    graph_T2 = plt.subplot(2,1,2)

    # search for the position which the diff is the greatest
    diff_arr = (t2_all-t2_retrieval)

    # xPos plus 5 position to avoid first differences by learning
    #xPos = np.where(diff_arr == max(diff_arr))[0][0] + 5
    xPos = 5

    # Position text
    posText = xPos + 3

    # minimum margin for print
    margin=0.03

    # -- Arrows in T2 --

    # difference of values
    dy = t2_all[xPos] - t2_retrieval[xPos]
    # if it's greater than margin, print arrow!
    if (dy > margin):
        # Vertical line: from retrieval to all
        graph_T2.annotate("", xy=(xPos,t2_retrieval[xPos]),xytext=(xPos,t2_all[xPos]), arrowprops=dict(arrowstyle="<->",color="red"))
        # Horizontal arrow: from the middle
        graph_T2.annotate(str(round(dy,2)), xy=(xPos,t2_retrieval[xPos]+dy/2),xytext=(posText,t2_retrieval[xPos]+dy/2), arrowprops=dict(arrowstyle="<-"))


    # time horizontal values
    t = np.arange(start=0,stop=(len(t1_all)),step=1)

    # Set limits and names
    graph_T1.set_ylim(0,1)
    graph_T2.set_ylim(0,1)

    graph_T1.set_xlabel("Items")
    graph_T2.set_xlabel("Items")

    graph_T1.set_ylabel("Error Rate")
    graph_T2.set_ylabel("Error Rate")

    # graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")
    graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")

    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")


    # Print values: T1 PERIOD
    graph_T1.plot(t,t1_retrieval, color="goldenrod")
    graph_T1.plot(t,t1_study, color="blue")
    graph_T1.plot(t,t1_all, color="black", linestyle="--")

    # Print values: T2 PERIOD
    graph_T2.plot(t,t2_retrieval,color="green")
    graph_T2.plot(t,t2_all,color="m")

    # Legends
    graph_T1.legend(["High performance stud", "Low performance stud", "Global performance stud"])
    graph_T2.legend(["High performance stud", "Global performance stud"])

    # Others props
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)

    plt.show()


def proj_notconsider(stud_mem,stud_mem2,cond,gammaValues_Mem, gammaValues_Prac, timeForgetting):

    # --------------------------------
    #          T1: Retrieval
    # --------------------------------

    # Get stud and data: retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t1_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: study
    # --------------------------------

    # Get stud and data: study
    stud = np.where(stud_mem.isMem == True)
    studData = stud_mem.answer[stud].copy()

    # mean
    t1_study = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: all
    # --------------------------------

    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t1_all = errorRateFunction(data=studData)


    # --------------------------------
    #     T2: forgetting phase
    # --------------------------------

    # Forgetting
    stud_mem.forgetting_effect_Manual(sameItems=False, gammaValues_Mem=gammaValues_Mem, gammaValues_Prac=gammaValues_Prac, timeForgetting=timeForgetting)
    stud_mem2.forgetting_effect_Manual(sameItems=False, gammaValues_Mem=gammaValues_Mem, gammaValues_Prac=gammaValues_Prac, timeForgetting=0)


    if (onMasteryBias):
        stud_mem.applyMasteryBias(cond=cond)
    elif (onAttritionBias):
        stud_mem.applyAttritionBias(cond=cond)

    # --------------------------------
    #          T2: Retrieval
    # --------------------------------

    # T2_retrieval
    studData = stud_mem.answer.copy()

    # error Rate
    t2_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T2: All
    # --------------------------------

    # T2_all
    # Get stud and data: study
    studData = stud_mem2.answer.copy()

    # error Rate
    t2_all = errorRateFunction(data=studData)

    # --------------------------------
    #          Plot options
    # --------------------------------

    # Create graphs
    graph_T1 = plt.subplot(2,1,1)
    graph_T2 = plt.subplot(2,1,2)

    # search for the position which the diff is the greatest
    diff_arr = (t2_all-t2_retrieval)
    print(diff_arr)

    # xPos plus 5 position to avoid first differences by learning
    xPos = 5

    # Position text
    posText = xPos + 3

    # minimum margin for print
    margin=0.03

    # -- Arrows in T2 --

    # difference of values
    dy = abs(t2_all[xPos] - t2_retrieval[xPos])
    # if it's greater than margin, print arrow!
    if (dy > margin):
        # Vertical line: from retrieval to all
        graph_T2.annotate("", xy=(xPos,t2_retrieval[xPos]),xytext=(xPos,t2_all[xPos]), arrowprops=dict(arrowstyle="<->",color="red"))
        # Horizontal arrow: from the middle
        graph_T2.annotate(str(round(dy,2)), xy=(xPos,t2_retrieval[xPos]+dy/2),xytext=(posText,t2_retrieval[xPos]+dy/2), arrowprops=dict(arrowstyle="<-"))


    # time horizontal values
    t = np.arange(start=0,stop=(len(t1_all)),step=1)

    # Set limits and names
    graph_T1.set_ylim(0,1)
    graph_T2.set_ylim(0,1)

    graph_T1.set_xlabel("Items")
    graph_T2.set_xlabel("Items")

    graph_T1.set_ylabel("Error Rate")
    graph_T2.set_ylabel("Error Rate")

    # graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")
    graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")

    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")


    # Print values: T1 PERIOD
    graph_T1.plot(t,t1_retrieval, color="goldenrod")
    graph_T1.plot(t,t1_study, color="blue")
    graph_T1.plot(t,t1_all, color="black", linestyle="--")

    # Print values: T2 PERIOD
    graph_T2.plot(t,t2_retrieval,color="green")
    graph_T2.plot(t,t2_all,color="m")

    # Legends
    graph_T1.legend(["High performance stud", "Low performance stud", "Global performance stud"])
    graph_T2.legend(["High performance stud", "Global performance stud"])

    # Others props
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)

    plt.show()

# Project: Global
def proj_rand(stud_mem,stud_mem2,cond,timeForgetting):

    # --------------------------------
    #          T1: Retrieval
    # --------------------------------

    # Get stud and data: retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t1_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: study
    # --------------------------------

    # Get stud and data: study
    stud = np.where(stud_mem.isMem == True)
    studData = stud_mem.answer[stud].copy()

    # mean
    t1_study = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: all
    # --------------------------------

    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t1_all = errorRateFunction(data=studData)


    # --------------------------------
    #     T2: forgetting phase
    # --------------------------------
    # Forgetting
    stud_mem.forgetting_effect_Manual_rand(sameItems=False, timeForgetting=timeForgetting)
    stud_mem2.forgetting_effect_Manual_rand(sameItems=False, timeForgetting=0)

    if (onMasteryBias):
        stud_mem.applyMasteryBias(cond=cond)
    elif (onAttritionBias):
        stud_mem.applyAttritionBias(cond=cond)

    # --------------------------------
    #          T2: Retrieval
    # --------------------------------

    # T2_retrieval
    #stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer.copy()

    # error Rate
    t2_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T2: All
    # --------------------------------

    # T2_all
    # Get stud and data: study
    studData = stud_mem2.answer.copy()

    # error Rate
    t2_all = errorRateFunction(data=studData)

    print(t2_retrieval)
    print(t2_all)

    # --------------------------------
    #          Plot options
    # --------------------------------

    # Create graphs
    graph_T1 = plt.subplot(2,1,1)
    graph_T2 = plt.subplot(2,1,2)

    # search for the position which the diff is the greatest
    diff_arr = (t2_retrieval-t2_all)
    print(diff_arr)

    # xPos plus 5 position to avoid first differences by learning
    xPos = np.where(diff_arr == max(diff_arr))[0][0] + 5
    print(xPos)

    # Position text
    posText = xPos + 3

    # minimum margin for print
    margin=0.03

    # -- Arrows in T2 --

    # difference of values
    dy = t2_all[xPos] - t2_retrieval[xPos]
    # if it's greater than margin, print arrow!
    if (dy > margin):
        # Vertical line: from retrieval to all
        graph_T2.annotate("", xy=(xPos,t2_retrieval[xPos]),xytext=(xPos,t2_all[xPos]), arrowprops=dict(arrowstyle="<->",color="red"))
        # Horizontal arrow: from the middle
        graph_T2.annotate(str(round(dy,2)), xy=(xPos,t2_retrieval[xPos]+dy/2),xytext=(posText,t2_retrieval[xPos]+dy/2), arrowprops=dict(arrowstyle="<-"))


    # time horizontal values
    t = np.arange(start=0,stop=(len(t1_all)),step=1)

    # Set limits and names
    graph_T1.set_ylim(0,1)
    graph_T2.set_ylim(0,1)

    graph_T1.set_xlabel("Items")
    graph_T2.set_xlabel("Items")

    graph_T1.set_ylabel("Error Rate")
    graph_T2.set_ylabel("Error Rate")

    # graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")
    graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")

    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")


    # Print values: T1 PERIOD
    graph_T1.plot(t,t1_retrieval, color="goldenrod")
    graph_T1.plot(t,t1_study, color="blue")
    graph_T1.plot(t,t1_all, color="black", linestyle="--")

    # Print values: T2 PERIOD
    graph_T2.plot(t,t2_retrieval,color="green")
    graph_T2.plot(t,t2_all,color="m")

    # Legends
    graph_T1.legend(["High performance stud", "Low performance stud", "Global performance stud"])
    graph_T2.legend(["High performance stud", "Global performance stud"])

    # Others props
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)

    plt.show()


# Project: Testing different values of learning!
def proj_learn(stud_mem,gammaValues_Mem, gammaValues_Prac, timeForgetting, learnValues):

    # --------------------------------
    #          T1: Retrieval
    # --------------------------------

    # Get stud and data: retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t1_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: study
    # --------------------------------

    # Get stud and data: study
    stud = np.where(stud_mem.isMem == True)
    studData = stud_mem.answer[stud].copy()

    # mean
    t1_study = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: all
    # --------------------------------

    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t1_all = errorRateFunction(data=studData)


    # --------------------------------
    # T2: Looping with learning values
    # --------------------------------

    valuesT2 = []
    legendValues = []

    for learnUnit in learnValues:

        # Forgetting
        stud_mem.forgetting_effect_Manual_learn(sameItems=False, gammaValues_Mem=gammaValues_Mem, gammaValues_Prac=gammaValues_Prac, timeForgetting=timeForgetting, learnUnit=learnUnit)

        # T2 retrieval
        stud = np.where(stud_mem.isMem == False)
        studData = stud_mem.answer[stud].copy()

        # error Rate
        t2_retrieval = errorRateFunction(data=studData)

        if (learnUnit == learnValues[0]):
            valuesT2 = np.array([t2_retrieval])
            aux = "Learn: " + str(learnUnit)
            legendValues = np.array([aux])
        else:
            valuesT2 = np.append(valuesT2, [t2_retrieval], axis=0)
            aux = "Learn: " + str(learnUnit)
            legendValues = np.append(legendValues,[aux], axis=0)


    # --------------------------------
    #          Plot options
    # --------------------------------

    # Create graphs
    graph_T1 = plt.subplot(2,1,1)
    graph_T2 = plt.subplot(2,1,2)


    # time horizontal values
    t = np.arange(start=0,stop=(len(t1_all)),step=1)

    # Set limits and names
    graph_T1.set_ylim(0,1)
    graph_T2.set_ylim(0,1)

    graph_T1.set_xlabel("Items")
    graph_T2.set_xlabel("Items")

    graph_T1.set_ylabel("Error Rate")
    graph_T2.set_ylabel("Error Rate")

    # graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")
    graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")

    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")


    # Print values: T1 PERIOD
    graph_T1.plot(t,t1_retrieval, color="goldenrod")
    graph_T1.plot(t,t1_study, color="blue")
    graph_T1.plot(t,t1_all, color="black", linestyle="--")

    # Print values: T2 PERIOD
    for dataT2 in valuesT2:
        graph_T2.plot(t,dataT2)

    # Legends
    graph_T1.legend(["High performance stud", "Low performance stud", "Global performance stud"])
    graph_T2.legend(legendValues)

    # Others props
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)

    plt.show()

# Project: Testing different values of memStrength!
def proj_memStrength(stud_mem,gammaValues_Mem, gammaValues_Prac, timeForgetting):

    # --------------------------------
    #          T1: Retrieval
    # --------------------------------

    # Get stud and data: retrieval
    stud = np.where(stud_mem.isMem == False)
    studData = stud_mem.answer[stud].copy()

    # error Rate
    t1_retrieval = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: study
    # --------------------------------

    # Get stud and data: study
    stud = np.where(stud_mem.isMem == True)
    studData = stud_mem.answer[stud].copy()

    # mean
    t1_study = errorRateFunction(data=studData)


    # --------------------------------
    #          T1: all
    # --------------------------------

    # Get stud and data: study
    studData = stud_mem.answer.copy()

    # error Rate
    t1_all = errorRateFunction(data=studData)


    # --------------------------------
    # T2: Looping with mem values
    # --------------------------------

    valuesT2 = []
    legendValues = []

    for memUnit in gammaValues_Prac:

        # Forgetting
        stud_mem.forgetting_effect_Manual(sameItems=False, gammaValues_Mem=gammaValues_Mem, gammaValues_Prac=memUnit, timeForgetting=timeForgetting)

        # T2 retrieval
        stud = np.where(stud_mem.isMem == False)
        studData = stud_mem.answer[stud].copy()

        # error Rate
        t2_retrieval = errorRateFunction(data=studData)

        if (memUnit == gammaValues_Prac[0]):

            # T2 study
            stud = np.where(stud_mem.isMem == True)
            studData = stud_mem.answer.copy()

            # Error rate
            t2_study = errorRateFunction(data=studData)

            valuesT2 = np.array([t2_study])
            aux = "Low performance stud"
            legendValues = np.array([aux])

            # T2 retrieval
            valuesT2 = np.append(valuesT2, [t2_retrieval], axis=0)
            aux = "Memory Strength (\u03B3): " + str(memUnit)
            legendValues = np.append(legendValues,[aux], axis=0)

        else:
            valuesT2 = np.append(valuesT2, [t2_retrieval], axis=0)
            aux = "Memory Strength (\u03B3): " + str(memUnit)
            legendValues = np.append(legendValues,[aux], axis=0)


    # --------------------------------
    #          Plot options
    # --------------------------------

    # Create graphs
    graph_T1 = plt.subplot(2,1,1)
    graph_T2 = plt.subplot(2,1,2)


    # time horizontal values
    t = np.arange(start=0,stop=(len(t1_all)),step=1)

    # Set limits and names
    graph_T1.set_ylim(0,1)
    graph_T2.set_ylim(0,1)

    graph_T1.set_xlabel("Items")
    graph_T2.set_xlabel("Items")

    graph_T1.set_ylabel("Error Rate")
    graph_T2.set_ylabel("Error Rate")

    # graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")
    graph_T1.set_title("T1 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")

    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud) + \
        ". Memory Strength(\u03B3): " + str(gammaValues_Mem) + " -> Mem, " + str(gammaValues_Prac) + " -> Prac. ", loc="left")


    # Print values: T1 PERIOD
    graph_T1.plot(t,t1_retrieval, color="goldenrod")
    graph_T1.plot(t,t1_study, color="blue")
    graph_T1.plot(t,t1_all, color="black", linestyle="--")

    # Print values: T2 PERIOD
    for dataT2 in valuesT2:
        graph_T2.plot(t,dataT2)

    # Legends
    graph_T1.legend(["High performance stud", "Low performance stud", "Global performance stud"])
    graph_T2.legend(legendValues)

    # Others props
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)


    plt.show()

# Project: Testing different values of students!
def proj_students(stud_mem,valuesStudent=np.array([10,100,1000,6000])):

    # Save all data in dataAll
    dataAll = []
    legendValues = []

    # Loop for each student value
    for valueStudent in valuesStudent:

        # int
        valueStudent = int(valueStudent)

        # Generate model
        stud_mem = SimulatedDataBKT("T1_retrieval",valueStudent, 50, perc_stud_stud=0.5)

        # Get stud and data: study
        d_std_all = stud_mem.answer.copy()
        data = errorRateFunction(data=d_std_all)

        # Save data
        if (valueStudent == valuesStudent[0]):
            dataAll = np.array([data])
            aux = "Stud: " + str(valueStudent)
            legendValues = np.array([aux])
        else:
            dataAll = np.append(dataAll,[data],axis=0)
            aux = "Stud: " + str(valueStudent)
            legendValues = np.append(legendValues, [aux], axis=0)

    # Plot information!
    graph_T2 = plt.subplot(2,1,2)
    graph_T2.set_ylim(0,1)
    graph_T2.set_xlabel("Items")
    graph_T2.set_ylabel("Error Rate")
    graph_T2.set_title("T2 period. Prop: " + str(stud_mem.perc_stud_stud), loc="left")


    # time horizontal values
    t = np.arange(start=0,stop=(len(dataAll[0])),step=1)

    # Plot
    for dataT2 in dataAll:
        graph_T2.plot(t,dataT2)

    # Legends
    graph_T2.legend(legendValues)

    # Plot options
    plt.tight_layout()

    # size plot
    figure = plt.gcf()
    figure.set_size_inches(10,10)

    # Plot show!
    plt.show()



# Model BKT
# -- Simulated Data class --
class SimulatedData(ABC): # -> Abstract class

    def __init__(self, type_params, students=1000, nitems=50, perc_stud_stud=0.2):

        params = self.load_params(type_params)                   # Load params

        self.params = [params for i in range(students)]          # Load params for each student
        self.perc_stud_stud=perc_stud_stud                       # percentage students use study strategy
        self.students = students                                 # number of students
        self.items = nitems                                      # number of items
        self.prob = np.empty((students, nitems)) * np.nan        # Matrix of probabilities of each student for each item.
        self.answer = np.empty((students, nitems)) * np.nan      # Matrix of answer of each student for each item
        self.true_mastery = np.ones(students) * nitems           # Item where each student obtained mastery knowledge
        self.isMem = np.ones(students) * False                   # Memorizing effects

        # first functions to start
        self.generate_Mem_stud(p_memStud=perc_stud_stud,type_params="T1_study")  # Generate student for study strategy
        self.generate_probabilities()                                            # Generate first probabilities
        self.generate_answers()                                                  # Generate first answers


    @abstractmethod
    def generate_probabilities(self):
        pass

    # load data function
    def load_params(self,type_params):
        scenario = []

        with open("./scenarios/scenarios.json", "r") as json_file:
            scenarios = json.load(json_file)
            scenario = scenarios[type_params]

        return(scenario["params"])

    def generate_answers(self):
        for s in range(self.students):                 #para cada estudiante...
            for i in range(self.items):
                randVariable =  random.random()              #para cada pregunta:
                if  randVariable < self.prob[s, i]:  #con una probabilidad de "guess" o "1-slip" en la matriz prob... (si respondió correcto:)
                    self.answer[s, i] = 1               #se rellena la matriz answer con 1 si respondió correctamente
                else:
                    self.answer[s, i] = 0               #se rellena la matriz answer con 0 si se equivocó



    def forgetting_effect_Manual(self,timeForgetting, gammaValues_Mem, gammaValues_Prac,sameItems=False):

        # Format attributes
        self.true_mastery[:] = self.items

        if (sameItems):
            params_t2_study = self.load_params("T2_study_learn")
            params_t2_retrieval = self.load_params("T2_retrieval_learn")
        else:
            params_t2_study = self.load_params("T2_study")
            params_t2_study["init"] = forgettingEffect(gammaValues=gammaValues_Mem,t=timeForgetting)
            params_t2_retrieval = self.load_params("T2_retrieval")
            params_t2_retrieval["init"] = forgettingEffect(gammaValues=gammaValues_Prac,t=timeForgetting)

        for s in range(self.students):
            if (self.isMem[s]):
                self.params[s] = params_t2_study
            else:
                self.params[s] = params_t2_retrieval

        self.generate_probabilities()
        self.generate_answers()

    def forgetting_effect_Manual_rand(self,timeForgetting, sameItems=False):

        # Format attributes
        self.true_mastery[:] = self.items
        params_t2_study = self.load_params("T2_study")
        params_t2_retrieval = self.load_params("T2_retrieval")

        for s in range(self.students):
            gamma_rand = random.random()*5
            if (self.isMem[s]):
                self.params[s] = params_t2_study
                params_t2_study["init"] = forgettingEffect(gammaValues=gamma_rand,t=timeForgetting)
            else:
                self.params[s] = params_t2_retrieval
                params_t2_retrieval["init"] = forgettingEffect(gammaValues=gamma_rand,t=timeForgetting)

        self.generate_probabilities()
        self.generate_answers()

    def forgetting_effect_Manual_learn(self,timeForgetting, gammaValues_Mem, gammaValues_Prac, learnUnit,sameItems=False):

        # Format attributes
        self.true_mastery[:] = self.items

        if (sameItems):
            params_t2_study = self.load_params("T2_study_learn")
            params_t2_retrieval = self.load_params("T2_retrieval_learn")
        else:
            params_t2_study = self.load_params("T2_study")
            params_t2_study["init"] = forgettingEffect(gammaValues=gammaValues_Mem,t=timeForgetting)
            params_t2_retrieval = self.load_params("T2_retrieval")
            params_t2_retrieval["init"] = forgettingEffect(gammaValues=gammaValues_Prac,t=timeForgetting)
            params_t2_retrieval["learn"] = learnUnit

        for s in range(self.students):
            if (self.isMem[s]):
                self.params[s] = params_t2_study
            else:
                self.params[s] = params_t2_retrieval

        self.generate_probabilities()
        self.generate_answers()


    def generate_Mem_stud(self, p_memStud=0.5, type_params="T1_study"):
        params_mem = self.load_params(type_params)
        for i in range(int(round(len(self.isMem))*p_memStud)):
            self.isMem[i] = 1.0
            self.params[i] = params_mem


    def applyMasteryBias(self, cond=3):

        for s in range(self.students):
            aux = 0
            for i in range(self.items):

                if (self.answer[s][i] == 1):
                    aux +=1
                else:
                    aux = 0

                if aux==cond:
                    if (random.random() < 0.9): # 10 % do not take mastery to have values to measure the mean
                        self.answer[s][i+1:] = -1
                    break


    def applyAttritionBias(self,cond=5):

        for s in range(self.students):
            aux = 0
            for i in range(self.items):

                if (self.answer[s][i] == 0):
                    aux +=1
                else:
                    aux = 0

                if aux==cond:
                    if (random.random() < 0.9): # 10 % do not take mastery to have values to measure the mean
                        self.answer[s][i+1:] = -1
                    break

# SimulatedDataBKT Class
class SimulatedDataBKT(SimulatedData):

    def generate_probabilities(self):

        # For each student
        for s in range(self.students):

            for i in range(self.items):

                # First probability
                if i==0:
                    self.prob[s][i] = self.params[s]["init"]  # init prob
                    continue

                # Only i > 1

                # Probability of apply the skill correct
                p_correct = self.prob[s][i-1] * (1-self.params[s]["slip"]) + (1-self.prob[s][i-1])*self.params[s]["guess"]

                # Observe correct skill based on p_correct
                obs = int(random.random() < p_correct)

                # Probability based on obs
                p_obs = []
                if (obs == 1):
                    p_obs = ( self.prob[s][i-1] * (1-self.params[s]["slip"]) ) / ( self.prob[s][i-1] * (1-self.params[s]["slip"]) + (1-self.prob[s][i-1])*self.params[s]["guess"] )
                elif (obs==0):
                    p_obs = ( self.prob[s][i-1] * self.params[s]["slip"] ) / ( self.prob[s][i-1] * self.params[s]["slip"] + (1-self.prob[s][i-1]) * (1-self.params[s]["guess"]) )

                # Probabilty of student
                self.prob[s][i] = p_obs + (1-p_obs)*self.params[s]["learn"]


# Global variables
onMasteryBias=False
onAttritionBias=False

# Init
if __name__ == "__main__":

    random.seed(1)

    # Simualtion init
    stud_mem = SimulatedDataBKT("T1_retrieval",6000,50, perc_stud_stud=0.99)


    random.seed(1)

    # Simualtion init
    stud_mem2 = SimulatedDataBKT("T1_retrieval",6000,50, perc_stud_stud=0.1)

    # Quitar el comentario del "proj" que quieran ejecutar!

    proj(stud_mem, cond=5, gammaValues_Mem=np.array([1]), gammaValues_Prac=np.array([5]), timeForgetting=1)
    #proj_notconsider(stud_mem, stud_mem2, cond=3, gammaValues_Mem=np.array([1]), gammaValues_Prac=np.array([1]), timeForgetting=1)
    #proj_rand(stud_mem, stud_mem2, cond=3, timeForgetting=1)
    #proj_learn(stud_mem, gammaValues_Mem=np.array([1]), gammaValues_Prac=np.array([5]), timeForgetting=1,learnValues=np.array([0.1,0.2,0.3,0.4,0.5]))
    #proj_memStrength(stud_mem, gammaValues_Mem=np.array([5]), gammaValues_Prac=np.array([1,2,3,4,5]), timeForgetting=1)
    #proj_students(stud_mem, valuesStudent=np.array([10,100,1000,6000]))
