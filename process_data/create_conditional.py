import SharedArray as sa
import numpy as np



train =  sa.attach('train_x_lpd_5_phr')
# train has shape (102378, 4, 48, 84, 5)
shape = train.shape

Height” of MIDI notes
Difference between top and bottom note
Similarity metric of different instrument tracks
Repetition of notes
Measure of note change (piano)
Density of notes


def height(train):
    #function that computes the difference between the topmost note and the bottommost note
    diff = np.zeros(shape[0], shape[1], shape[2], 1, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    top = shape[3] + 1
                    bottom = -1
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        diff(i, j, k, 0, l) = bottom - top
                    else: 
                        diff(i, j, k, 0, l) = 0
    return diff


def maxmin(train):
    #function that computes the index of the topmost note and the bottommost note
    mm = np.zeros(shape[0], shape[1], shape[2], 2, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    top = shape[3] + 1
                    bottom = -1
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        mm(i, j, k, 0, l) = top
                        mm(i, j, k, 1, l) = bottom
                    elif top < 85: 
                        mm(i, j, k, 0, l) = top
                        mm(i, j, k, 1, l) = top
                    elif bottom > -1: 
                        mm(i, j, k, 0, l) = bottom
                        mm(i, j, k, 1, l) = bottom
                    else:  
                        mm(i, j, k, 0, l) = top
                        mm(i, j, k, 1, l) = bottom
    return mm



def density(train):
    #function that computes the number of notes within each interval of a bar
    dens = np.zeros(shape[0], shape[1], shape[2], 1, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    density = 0
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            density += 1 
                    dens(i, j, k, 0, l) = density
                   
    return dens


def note_repetition(train):
    #function that computes the number of times a note is repeated within a bar
    rep = np.zeros(shape[0], shape[1], 1, shape[3], shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for m in range(shape[3]):
                for l in range(shape[4]):
                    repeats = 0
                    for k in range(shape[2]):
                        if train(i, j, k, m, l):
                            repeat += 1
                    rep(i, j, 1, m, l) = repeat
    return diff

#notes above a line and below

def halfline(train):
    #function that computes the number of notes within each interval of a bar
    hline = np.zeros(shape[0], shape[1], shape[2], 2, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    above_line = 0
                    below_line = 0
                    for m in range(shape[3]/2):
                        if train(i, j, k, m, l):
                            above_line += 1 
                    for m in range(shape[3]/2, shape[3]):
                        if train(i, j, k, m, l):
                            below_line += 1
                    hline(i, j, k, 0, l) = above_line
                    hline(i, j, k, 1, l) = below_line
    return hline


def thirdline(train):
    #function that computes the number of notes within each interval of a bar
    tline = np.zeros(shape[0], shape[1], shape[2], 3, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    top = 0
                    middle = 0
                    bottom = 0
                    for m in range(shape[3]/3):
                        if train(i, j, k, m, l):
                            top += 1 
                    for m in range(shape[3]/3, shape[3]*2/3):
                        if train(i, j, k, m, l):
                            middle += 1
                    for m in range(shape[3]*2/3, shape[3]):
                        if train(i, j, j, m, l):
                            bottom += 1
                    tline(i, j, k, 0, l) = top
                    tline(i, j, k, 1, l) = middle
                    tline(i, j, k, 2, l) = bottom
    return tline


def barheight(train):
    #function that computes the difference between the topmost note and the bottommost note
    bdiff = np.zeros(shape[0], shape[1], 1, 1, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                top = shape[3] + 1
                bottom = -1
                for k in range(shape[2])
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        bdiff(i, j, 0, 0, l) = bottom - top
                    else: 
                        bdiff(i, j, 0, 0, l) = 0
    return bdiff


def barmaxmin(train):
    #function that computes the index of the topmost note and the bottommost note
    bmm = np.zeros(shape[0], shape[1], 1, 2, shape[4])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[4]):
                top = shape[3] + 1
                bottom = -1
                for k in range(shape[2]):
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        bmm(i, j, k, 0, l) = top
                        bmm(i, j, k, 1, l) = bottom
                    elif top < 85: 
                        bmm(i, j, k, 0, l) = top
                        bmm(i, j, k, 1, l) = top
                    elif bottom > -1: 
                        bmm(i, j, k, 0, l) = bottom
                        bmm(i, j, k, 1, l) = bottom
                    else:  
                        bmm(i, j, k, 0, l) = top
                        bmm(i, j, k, 1, l) = bottom

    return bmm

#Finish

# train has shape (102378, 4, 48, 84, 5)
def similarity(train):
    #function that computes the difference between the topmost note and the bottommost note
    similar = np.zeros(shape[0], shape[1], shape[2], 1, 7)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[4]):
                    top = 85
                    bottom = -1
                    for m in range(shape[3]):
                        if train(i, j, k, m, l):
                            if m < top: 
                                top = m
                            if m > bottom: 
                                bottom = m 
                    if top < 85 and bottom > -1: 
                        diff(i, j, k, 0, l) = bottom - top
                    else: 
                        diff(i, j, k, 0, l) = 0


#xor?
#binary mask
#some kind of correspondence with other tracks
#corrsepondence with local surroundings

 
    







