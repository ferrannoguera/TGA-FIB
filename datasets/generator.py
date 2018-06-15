
import random
    
    

if __name__ == '__main__':
    res = open('/tmp/testloco','w')
    numP = 20000
    numD = 500
    numK = 2
    numI = 100
    res.write(str(numP)+ ' ')
    res.write(str(numD)+ ' ')
    res.write(str(numK)+ ' ')
    res.write(str(numI)+'\n')
    random.seed()
    for i in range(0, numP):
        for j in range(0, numD):
            res.write(str(random.uniform(0,1000000))+' ')
        res.write(str('\n'))
    res.closed
