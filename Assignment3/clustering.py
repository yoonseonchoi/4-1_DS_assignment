import sys
import os

# n: number of clusters
# eps: Radius
# minPts: Density threshold
def dbscan(data, n, eps, minPts):
    c = 0       # Undefined label
    noise = -1      # noise
    cluster_cnt = 0
    
    # Iterate over every point
    for i in range(len(data)):
        # Skip processed points
        if data[i][3] != 0:
            continue

        # Fine initial neighbors    
        p = data[i][1:3]
        neighbors = range_query(data, p, eps)
        
        # Non-core points are noise
        if len(neighbors) < minPts:
            data[i][3] = noise
            continue

        # Start a new cluster     
        c += 1
        data[i][3] = c    
        cluster_cnt +=1
        
        # Expand neighborhood
        seed_set = neighbors
        
        for j in seed_set:
            if data[j][3] == noise:
                data[j][3] = c
            if data[j][3] != 0:
                continue
            
            data[j][3] = c
            q = data[j][1:3]
            neighbors = range_query(data, q, eps)
            
            # Core-point check
            if len(neighbors) >= minPts:
                seed_set += neighbors

        # cluster의 개수가 n개에 도달하면 stop
        if cluster_cnt == n:
            break
    return data

# Get neighborhood                        
def range_query(data, pt, eps):
    neighbors = []
    
    for i in range(len(data)):
        q = data[i][1:3]
        if get_distance(pt, q) <= eps:
            neighbors.append(i)
    return neighbors

# Compute the distance beween two points
def get_distance(pt1, pt2):
    x1 = float(pt1[0])
    y1 = float(pt1[1])
    x2 = float(pt2[0])
    y2 = float(pt2[1])
    dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return dist

# 각 cluster의 label과 point index를 clusters dictionary에 저장 
def get_clusters(data):
    clusters = {}

    for i in range(len(data)):
        idx = data[i][0]
        label = data[i][3]
        if label != 0 and label != -1:
            if label in clusters:
                clusters[label].append(idx)
            else:
                clusters[label] = [idx]
    return clusters

# input 파일 읽어오기
def read_file(input):
    data = []
    
    with open(input, 'r') as inFile:
        lines = inFile.readlines()
        for line in lines:
            if line:
                idx, x, y = line.split('\t')
                x, y = float(x), float(y)
                # 기존의 input data 마지막 열에 undefine된 cluster label을 0으로 초기화하여 추가
                data.append([idx, x, y, 0])
    return data

# cluster 결과 작성
def write_objects(input, clusters):
    num = os.path.splitext(input)[0][-1]
    
    for label, objects in clusters.items():
        output = f'input{num}_cluster_{label-1}.txt'
        with open(output, 'w') as file:
            for o in objects:
                file.write(f'{o}\n')

if __name__ == "__main__":
    # 올바르게 compile하지 않으면 다음과 같은 경고가 뜸
    if len(sys.argv) != 5:
        raise Exception("Correct usage: [program] [input] [n] [eps] [minPts]")
    
    input = sys.argv[1]
    n = int(sys.argv[2])
    eps = float(sys.argv[3])
    minPts = int(sys.argv[4])

    data = read_file(input)
    output = dbscan(data, n, eps, minPts)
    clustered_input = get_clusters(output)
    write_objects(input, clustered_input)