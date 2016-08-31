import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="draw input stock data")
    parser.add_argument("file", help="the input csv file to draw")
    args = parser.parse_args()

    try:
        if not args.file[-4:] == ".csv":
            raise "file " + args.file + " isn't a csv file!"
        draw_file = open(args.file)
        reader = csv.reader(draw_file)
    except:
        print "Cannot open file " + args.file + "!"
        sys.exit(0)

    ## show choice ##
    header = reader.next()
    print "choices:",
    for i, name in enumerate(header):
        if not i == 0:
            print "["+str(i)+' '+name+"]",
    print '\n'

    ## access data ##
    data = np.empty((0, len(header)-1), dtype='float')
    for line in reader:
        data = np.append(data, [np.array(line[1:], dtype='float')], axis=0)
    draw_file.close()

    color = ['r-', 'b-']

    plt.ion()
    while True:
        ## get request column to draw ##
        while True:
            column = raw_input(">>>request column: ")
            if column == "q":
                sys.exit(0)
            try:
                column = int(column)
                if column <= 0 or column >= len(header):
                    print "Invalid column:", column, "Request column should be a positive int!"
                    column = None

            except:
                    print "Invalid column:", column, "Request column should be a positive int!"
                    column = None
            if not column == None:
                break
        print "Chosen column:", header[column], '\n'
        plt.plot(data[:,column-1], 'r-')#, data[:,column-1], 'ro')
        #plt.plot(np.arange(100, 200), data[100:200,column-1], 'bo')#, data[:,column-1], 'ro')'''
        plt.title(header[column])
        plt.show()
