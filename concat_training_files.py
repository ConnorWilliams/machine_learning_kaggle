fout=open("general_station_train.csv","a")

# first file:
for line in open("Train/station_201_deploy.csv"):
    fout.write(line)
# now the rest:
for num in range(202,276):
    f = open("Train/station_"+str(num)+"_deploy.csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()
