import sys

import numpy.random


class Task:
    '所有子任务的基类'

    def __init__(self, subId, taskId, dataSize, cload, release_time, source, deadline):
        self. subId=subId
        self. taskId=taskId
        self. dataSize=dataSize
        self. cload=cload
        self. release_time=release_time
        self. source=source
        self. isAllocated = False
        self.preList = []
        # self.resource = resource
        self.deadline = deadline
        self.successFlag = False
        self.path = {} # key of path: time, value of path: [x,y]

    def setSuccessFlag(self, flag):
        self.successFlag = flag

    def setPath(self, path):
        self.path = path

    def setDeadline(self, deadline):
        self.deadline = deadline

    def getPath(self):
        return self.path

    # 设置前置任务list
    def setSucceList(self, taskList):
        self.preList = taskList

    def getSucceList(self):
        return self.preList

    def getResource(self):
        return self.resource

    def setAllocated(self):
        self.isAllocated = True

    def getAllocated(self):
        return self.isAllocated

    def printInfo(self):
        print("subId:"+str(self.subId)+",taskId:"+str(self.taskId)+",type:"+str(self.pre)
              +",dataSize:"+str(self.dataSize)+",cload,"+str(self.cload)
              +",release_time:"+str(self.release_time)+",source:"+str(self.source)
              +",resource:"+str(self.resource)+",deadline:"+str(self.deadline))


