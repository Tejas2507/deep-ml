def matrixmul(a:list[list[int|float]],b:list[list[int|float]])-> list[list[int|float]]:
    m1 = len(a)
    n1 = len(a[0])
    m2 = len(b)
    n2 = len(b[0])

    c = [[] for _ in range(m1)]

    if n1 == m2:
        for i in range(m1):
            for j in range(n2):
                elem = 0 
                for k in range(n1):
                    elem += a[i][k]*b[k][j]
                c[i].append(elem)
        
        return c 
    
    else:
        return -1
                    


    print
