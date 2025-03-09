import matplotlib.pyplot as plt
import pandas as pd
#allows user to plot intersection point (only one) between a point and a line or between 2 lines

def point_intersection(line, POI):
    for i in range(len(line[0])):
        points = [line[0][i], line[1][i]]
        print(points)
        
        if POI == points:
            new_line = [[], []]
            for k in range(i, len(line[0])):
                new_line[0].append(line[0][k])
                new_line[1].append(line[1][k])
            
    return new_line

def line_intersection(line1, line2):
    if len(line1[0]) != len(line2[0]):
        raise Exception("Both should be same shape lines")
    
    for k in range(len(line1[0])):
        if [line1[0][k], line1[1][k]] == [line2[0][k], line2[1][k]]:
            new_line1 = [[], []]
            new_line2 = [[], []]
            
            for j in range(k, len(line1[0])):
                new_line1[0].append(line1[0][j])
                new_line1[1].append(line1[1][j])
                new_line2[0].append(line2[0][j])
                new_line2[1].append(line2[1][j])
    
    return new_line1, new_line2
         

if __name__ == "__main__":   
    df = pd.read_csv("Flash Floods/data/Rectangle_channel.csv")
    print(df)
    print(df.shape)