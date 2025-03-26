class EdgeDevice:
    '所有子任务的基类'

    def __init__(self, deviceId, cpuNum, waitTime, resource):
        self.deviceId = deviceId
        self.cpuNum = cpuNum
        self.waitTime = waitTime
        self.resource = resource
        self.failureRate = 0.00099 # DT is 0.0001

    def setfailureRate(self, failureRate):
        self.failureRate = failureRate

    def setResource(self, resource):
        self.resource = resource

    def setWaitTime(self, waitTime):
        self.waitTime = waitTime

    def printInfo(self):
        print("deviceId:"+str(self.deviceId)+",cpuNum:"+
                              str(self.cpuNum)+",waitTime:"+str(self.waitTime))
