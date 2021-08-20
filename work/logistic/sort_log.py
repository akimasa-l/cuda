with open("./logistic.log") as f:
    l=list(map(str.split,f.read().split("\n")))
l.sort()
for i in l:
    print(*i)
