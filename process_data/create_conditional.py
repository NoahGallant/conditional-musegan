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
                    else: 
                        mm(i, j, k, 1, l) = bottom

    return mm

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



def density(train):
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
 
    

def note_repetition(train):
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



#above a line and below

def line(train):
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



def barheight(train):
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


def barmaxmin(train):
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
                    else: 
                        mm(i, j, k, 1, l) = bottom

    return mm
