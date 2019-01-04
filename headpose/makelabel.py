import sys
import os

lines=[]

for root,_,filenames in os.walk("/home/hanson/work/FaceAttribute_Pytorch/headpose/300W_LP"):
    for filename in filenames:
        lines.append( os.path.join(root,filename)[:-4] )

with open("labelfile.txt","w") as f:
    lines=list(set(lines))
    for i in lines:
        f.write(i+"\n")

