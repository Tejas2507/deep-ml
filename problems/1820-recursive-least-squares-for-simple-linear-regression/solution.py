import numpy as np

def online_simple_regression(xs, ys):
    
    meanx=0
    meany = 0
    mxx=0
    mxy=0
    count = 0
    result = []
    for x,y in zip(xs,ys):
        count +=1 
        dx = x-meanx
        dy = y-meany
        meanx+= dx/count 
        meany+=dy/count 
        mxx+= dx*(x-meanx)
        mxy += dx*(y-meany)

        if mxx == 0:
            w = 0.0
        else:
            w = mxy/mxx 
        
        b = meany - w*meanx 
        result.append([b,w])
    
    return np.array(result,dtype=float).reshape(-1,2)


 

