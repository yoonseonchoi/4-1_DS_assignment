import sys
from itertools import combinations

if len(sys.argv) != 4:
    raise Exception("Correct usage: [program] [min_support] [input] [output]")

# system argument 입력
min_sup = int(sys.argv[1])
inFile = sys.argv[2]
outFile = sys.argv[3]

# input file open 후 data 읽기
with open(inFile) as f:
    lines = f.readlines()

data = [line.split('\t') for line in lines]
data = [list(map(int, line)) for line in data]

# output file에 결과 작성
f = open(outFile, 'w')

# size가 1인 initial candidate itemset 생성
C_1 = dict()
for trans in data:
    for itemset in trans:
        if itemset in C_1.keys():
            C_1[itemset] += 1
        else:
            C_1.update({itemset: 0})
            C_1[itemset] += 1
# C_1의 support 계산
C_1 = {itemset: sup/len(data)*100 for itemset, sup in C_1.items()}
# minimum support보다 작은 candidate는 삭제
L_1 = {itemset for itemset, sup in C_1.items() if sup >= min_sup}

freq_items = L_1
C_k = C_1
L_k = L_1
k = 1

# frequent itemset의 size가 0이 될 때까지 다음 과정 반복
while len(L_k) != 0:
    if k == 1:
        # L_1에서 C_2생성 (combination 사용)
        nextC_k = list(combinations(L_k, 2))
        nextC_k = {itemset: 0 for itemset in nextC_k}
        for trans in data:
            for itemset in nextC_k.keys():
                if set(itemset) <= set(trans):
                    nextC_k[itemset] += 1
        nextC_k = {itemset: sup/len(data)*100 for itemset, sup in nextC_k.items()}
        # min_support보다 작은 itemset 제거
        nextL_k = {itemset for itemset, sup in nextC_k.items() if sup >= min_sup}

    # item size가 2 이상일 때
    elif k >= 2:
        # combination을 사용하여 self joining
        nextC_k = list(combinations(L_k, 2))
        nextC_k = [tuple(set(itemset[0]) | set(itemset[1])) for itemset in nextC_k]
        
        # self joining 했을 때, item size가 현재 k와 다른 itemset이나 중복된 itemset 제거
        temp = list()
        for itemset in nextC_k:
            if len(itemset) == k+1 and set(itemset) not in temp:
                temp.append(set(itemset))
        nextC_k = temp

        # pruning
        temp = dict()
        for itemset in nextC_k:
            isSubset = 0
            for subset in list(L_k):
                if set(subset) <= itemset:
                    isSubset += 1
                else:
                    continue
            if isSubset == k+1:
                temp[tuple(itemset)] = 0
        nextC_k = temp

        for trans in data:
            for itemset in nextC_k.keys():
                if set(itemset) <= set(trans):
                    nextC_k[itemset] += 1
        # C_k의 support 계산
        nextC_k = {itemset: sup/len(data)*100 for itemset, sup in nextC_k.items()}
        # min_support보다 작은 itemset 제거
        nextL_k = {itemset for itemset, sup in nextC_k.items() if sup >= min_sup}

    freq_items = (freq_items | nextL_k)
    L_k = nextL_k

    # output file에 result writing
    for i in L_k:
        for j in range(len(i)-1):
            for itemset in list(combinations(list(i), j+1)):
                sup_cnt = 0     # support count
                antecedent_cnt = 0     # X -> Y의 confidence를 구할 때, X를 포함한 transaction의 개수
                interaction_cnt = 0     # X -> Y의 confidence를 구할 때, X와 Y를 둘 다 포함한 transaction의 개수
                associative_item_set = set(i) - set(itemset)    # X -> Y의 confidence를 구할 때, Y에 해당하는 itemset
                # sup_cnt, antecedent_cnt, interaction_cnt 구하기
                for trans in data:
                    if set(i) <= set(trans):
                        sup_cnt += 1
                    if set(itemset) <= set(trans):
                        antecedent_cnt += 1
                        if set(associative_item_set) <= set(trans):
                            interaction_cnt += 1
                # support와 confidence percentage로 구하기
                support = (sup_cnt/len(data))*100
                confidence = (interaction_cnt/antecedent_cnt)*100

                f.write('{}\t{}\t{:.2f}\t{:.2f}\n'.format(set(itemset), associative_item_set, support, confidence))
    
    # itemset size 증가시키기
    k += 1

# file close
f.close()