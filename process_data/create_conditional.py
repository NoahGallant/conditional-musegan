import SharedArray as sa
import numpy as np
#File of the different functions creating conditional info from the data

#Load in the training data from the shared array
train =  sa.attach('train_x_lpd_5_phr')
# train has shape (102378, 4, 48, 84, 5)

#Use a smaller sample size for teseting
train = train[:56]
#train = np.array(train)


 
def andmask(train):
    #function that computes the notes shared between all the different instrument tracks
    shape = train.shape
    a = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] and train[i][j][k][m][1] and train[i][j][k][m][2] and train[i][j][k][m][3] and train[i][j][k][m][4]:
                        a[i][j][k][m] = 1
    train[..., 2] = a
    return train

#Testing
a = andmask(train)
#print(a[1][1][0][:][:])
print(a.shape)


def ormask(train):
    #function that computes the notes used by any of the different instrument tracks
    shape = train.shape
    o = np.zeros((shape[0], shape[1], shape[2], shape[3], 1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] or train[i][j][k][m][1] or train[i][j][k][m][2] or train[i][j][k][m][3] or train[i][j][k][m][4]:
                        o[i][j][k][m][0] = 1
    return o

#Testing
#o = ormask(train)
#print(o[1][1][0][:][:])
#print(o[1][1][0][:][:].shape)




def xormask(train):
    #function that computes the notes used by only one of the different instrument tracks
    shape = train.shape
    xo = np.zeros((shape[0], shape[1], shape[2], shape[3], 1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] ^ train[i][j][k][m][1] ^ train[i][j][k][m][2] ^ train[i][j][k][m][3] ^ train[i][j][k][m][4]:
                        xo[i][j][k][m][0] = 1
    return xo

#Testing
#xo = ormask(train)
#print(xo[1][1][0][:][:])
#print(xo[1][1][0][:][:].shape)



def height(train):
    #function that computes the difference between the topmost note and the bottommost note
    shape = train.shape
    diff = np.zeros((shape[0], shape[1], shape[2], 1, shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for k in range(shape[2]):
                    top = shape[3] + 1
                    bottom = -1
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        diff[i][j][k][0][l] = bottom - top
                    else: 
                        diff[i][j][k][0][l] = 0
    return diff

#Testing
#print(train.shape)
#h = height(train)
#print(h.shape)
#print(h[1][1][:][:][:].shape)


def maxmin(train):
    #function that computes the index of the topmost note and the bottommost note in each time interval in a bar
    shape = train.shape
    mm = np.zeros((shape[0], shape[1], shape[2], 2, shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for k in range(shape[2]):
                    top = shape[3] + 1
                    bottom = -1
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        mm[i][j][k][0][l] = top
                        mm[i][j][k][1][l] = bottom
                    elif top < 85: 
                        mm[i][j][k][0][l] = top
                        mm[i][j][k][1][l] = top
                    elif bottom > -1: 
                        mm[i][j][k][0][l] = bottom
                        mm[i][j][k][1][l] = bottom
                    else:  
                        mm[i][j][k][0][l] = 0
                        mm[i][j][k][1][l] = 0
    return mm


#Testing
#mm = maxmin(train)
#print(mm[1][1][:][:][:].shape)
#print(mm[10][0][:][:][:])

def density(train):
    #function that computes the number of notes within each interval of a bar
    shape = train.shape
    dens = np.zeros((shape[0], shape[1], shape[2], 1, shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for k in range(shape[2]):
                    density = 0
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            density += 1 
                    dens[i][j][k][0][l] = density
                   
    return dens


#Testing
#d = density(train)
#print(d.shape)
#print(d[1][1][:][:][:])
#print(d[1][1][:][:][:].shape)


def note_repetition(train):
    #function that computes the number of times a note is repeated within a bar
    shape = train.shape
    rep = np.zeros((shape[0], shape[1], 1, shape[3], shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for m in range(shape[3]):
                    repeats = 0
                    for k in range(shape[2]):
                        if train[i][j][k][m][l]:
                            repeats += 1
                    rep[i][j][0][m][l] = repeats
    return rep


#Testing
#n = note_repetition(train)
#print(n.shape)
#print(n[3][0][:][:][:])
#print(n[3][1][:][:][:])
#print(n[2][1][:][:][:].shape)

#notes above a line and below

def halfline(train):
    #function that computes the number of notes within the top half notes and bottom half of notes of an interval of a bar
    shape = train.shape
    hline = np.zeros((shape[0], shape[1], shape[2], 2, shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for k in range(shape[2]):
                    above_line = 0
                    below_line = 0
                    for m in range(shape[3]/2):
                        if train[i][j][k][m][l]:
                            above_line += 1 
                    for m in range(shape[3]/2, shape[3]):
                        if train[i][j][k][m][l]:
                            below_line += 1
                    hline[i][j][k][0][l] = above_line
                    hline[i][j][k][1][l] = below_line
    return hline


#Testing
#h = halfline(train)
#print(h.shape)
#print(h[3][0][:][:][:])
#print(h[3][1][:][:][:])
#print(h[2][1][:][:][:].shape)

def thirdline(train):
    #function that computes the number of notes within the top third, middle third, and bottomm third notes of each interval of a bar
    shape = train.shape
    tline = np.zeros((shape[0], shape[1], shape[2], 3, shape[4]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                for k in range(shape[2]):
                    top = 0
                    middle = 0
                    bottom = 0
                    for m in range(shape[3]/3):
                        if train[i][j][k][m][l]:
                            top += 1 
                    for m in range(shape[3]/3, shape[3]*2/3):
                        if train[i][j][k][m][l]:
                            middle += 1
                    for m in range(shape[3]*2/3, shape[3]):
                        if train[i][j][k][m][l]:
                            bottom += 1
                    tline[i][j][k][0][l] = top
                    tline[i][j][k][1][l] = middle
                    tline[i][j][k][2][l] = bottom
    return tline


#Testomg
#t = thirdline(train)
#print(t.shape)
#print(t[3][0][:][:][:])
#print(t[3][1][:][:][:])
#print(t[2][1][:][:][:].shape)


def barheight(train):
    #function that computes the difference between the topmost note and the bottommost note within each bar
    shape = train.shape
    bdiff = np.zeros((shape[0], shape[1], 1, 1, shape[4]))
    for i in range(shape[0]):
        for l in range(shape[4]):
            for j in range(shape[1]):
                top = shape[3] + 1
                bottom = -1
                for k in range(shape[2]):
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        bdiff[i][j][0][0][l] = bottom - top
                    else: 
                        bdiff[i][j][0][0][l] = 0
    return bdiff

#Testing
#bh = barheight(train)
#print(bh[1][1][:][:][:])
#print(bh[1][1][:][:][:].shape)

def barmaxmin(train):
    #function that computes the index of the topmost note and the bottommost note within each bar
    shape = train.shape
    bmm = np.zeros((shape[0], shape[1], 1, 2, shape[4]))
    for i in range(shape[0]):
        for l in range(shape[4]):
            for j in range(shape[1]):
                top = shape[3] + 1
                bottom = -1
                for k in range(shape[2]):
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        bmm[i][j][0][0][l] = top
                        bmm[i][j][0][1][l] = bottom
                    elif top < 85: 
                        bmm[i][j][0][0][l] = top
                        bmm[i][j][0][1][l] = top
                    elif bottom > -1: 
                        bmm[i][j][0][0][l] = bottom
                        bmm[i][j][0][1][l] = bottom
                    else:  
                        bmm[i][j][0][0][l] = 0
                        bmm[i][j][0][1][l] = 0
    return bmm


#Testing
#bm = barmaxmin(train)
#print(bm[1][1][:][:][:])
#print(bm[1][1][:][:][:].shape)




def alland(train): 
    shape = train.shape
    a = np.zeros((shape[0], shape[1], shape[2], shape[3], 10))
    index = 0
    for i in range(shape[4]-1):
        for j in range(i+1, shape[4]):
            print(a[:][:][:][:][index].shape)
            print(train[:][:][:][:][i].shape)
            a[:][:][:][:][index] = checkand(train[:][:][:][:][i], train[:][:][:][:][j])
            index += 1
    return a


def checkand(train1, train2):
    #function that computes the notes shared between every pair of the different instrument tracks
    shape = train1.shape
    a = np.zeros((shape[0], shape[1], shape[2], shape[3], 1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train1[i][j][k][m] and train2[i][j][k][m]:
                        a[i][j][k][m][0] = 1
    return a

#Testing
#fix these methods (alland, checkand, allor, checkor, allxor, checkxor)
#a = alland(train)
#print(a[1][1][0][:][:])
#pprint(a[1][1][0][:][:].shape)


def allor(train): 
    shape = train.shape
    o = np.zeros((shape[0], shape[1], shape[2], shape[3], 10))
    index = 0
    for i in range(shape[4]-1):
        for j in range(i+1, shape[4]):
            o[:][:][:][:][index] = checkor(train[:][:][:][:][i], train[:][:][:][:][j])
            index += 1
    return o
    

def checkor(train1, train2):
    #function that computes the notes used by any pair of the different instrument tracks
    shape = train1.shape
    o = np.zeros((shape[0], shape[1], shape[2], shape[3], 1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train1[i][j][k][m][0] or train2[i][j][k][m][0]:
                        o[i][j][k][m][0] = 1
    return o
 
#Testing
#o = allor(train)
#print(o[1][1][0][:][:])
#print(o[1][1][0][:][:].shape)


def allxor(train): 
    shape = train.shape
    xo = np.zeros((shape[0], shape[1], shape[2], shape[3], 10))
    index = 0
    for i in range(shape[4]-1):
        for j in range(i+1, shape[4]):
            xo[:][:][:][:][index] = checkxor(train[:][:][:][:][i], train[:][:][:][:][j])
            index += 1
    return xo
    

def checkxor(train1, train2):
    #function that computes the notes used by only one the different instrument tracks
    shape = train1.shape
    xo = np.zeros((shape[0], shape[1], shape[2], shape[3], 0))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train1[i][j][k][m][0] ^ train2[i][j][k][m][0]:
                        xo[i][j][k][m][0] = 1
    return xo


#TODO

# train has shape (102378, 4, 48, 84, 5)
def similarity(train):
    #function that uses some kind of distance metric to compare the different instrument tracks (distance function still TBD)
    shape = train.shape
    similar = np.zeros((shape[0], shape[1], shape[2], 1, 10))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    top = 85
                    bottom = -1
                    for m in range(shape[3]):
                        if train[i][j][k][m][l]:
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        diff[i][j][k][0][l] = bottom - top
                    else: 
                        diff[i][j][k][0][l] = 0
    return similar


