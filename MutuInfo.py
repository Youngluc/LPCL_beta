import numpy as np
import sklearn.metrics


def Mutualinfo(x, y):
    return sklearn.metrics.mutual_info_score(x, y) / np.log(2)


def Entropy(x):
    count = np.histogram(x, 256, (0, 255))[0]
    px = count / np.sum(count)
    hx = - np.sum(px * np.log2(px + 1e-8))
    return hx


def JointEntropy(x, y):
    tx, ty = x.reshape(-1), y.reshape(-1)
    count = np.histogram2d(tx, ty, 256, [[0, 255], [0, 255]])[0]
    pxy = count / np.sum(count)
    hxy = - np.sum(pxy * np.log2(pxy + 1e-8))
    return hxy


def RGBEntropy(x):
    B = x[:,:,0]
    G = x[:,:,1]
    R = x[:,:,2]
    InfoEntropy = np.array([Entropy(B), Entropy(G), Entropy(R)])
    return InfoEntropy


def JointRGBEntropy(x, y):
    B1 = x[:,:,0]
    G1 = x[:,:,1]
    R1 = x[:,:,2]
    B2 = y[:,:,0]
    G2 = y[:,:,1]
    R2 = y[:,:,2]
    JointInfoEntropy = np.array([JointEntropy(B1, B2), JointEntropy(G1, G2), JointEntropy(R1, R2)])
    return JointInfoEntropy


def matchPatch(xset, yset):
    dict_res = {}
    for item in xset:
        name = item[3]
        patch = item[2]
        dict_res[name] = yset[0][3] 
        max_info = 0
        for o in yset:
            num = o[3]
            obj = o[2]
            mutual_info = RGBEntropy(patch).mean() + RGBEntropy(obj).mean() - JointRGBEntropy(patch, obj).mean()
            if mutual_info > max_info:
                dict_res[name] = num
                max_info = mutual_info
    return dict_res


def matchPatch_list(xset, yset):
    x_res = []
    y_res = []
    posx, posy = np.zeros((512, len(xset))), np.zeros((512, len(yset)))
    count = 0
    for item in xset:
        name = item[3]
        patch = item[2]
        x_res.append(name) 
        max_info = 0
        tag = yset[0][3]
        patch1_xpos, patch1_ypos = name // 7, name % 7
        posx[patch1_xpos * 32][count] = 1
        posx[patch1_ypos * 32 + 256][count] = 1
        for o in yset:
            num = o[3]
            obj = o[2]
            #mutual_info = item[0] + o[0] - JointRGBEntropy(patch, obj).mean()            
            #mutual_info = item[0] + o[0] - JointEntropy(patch, obj)
            mutual_info = Mutualinfo(patch[:,:,0].reshape(-1), obj[:,:,0].reshape(-1)) + Mutualinfo(patch[:,:,1].reshape(-1), obj[:,:,1].reshape(-1)) + Mutualinfo(patch[:,:,2].reshape(-1), obj[:,:,2].reshape(-1)) 
            #print(1/3 * mutual_info, item[0] + o[0] - JointRGBEntropy(patch, obj).mean())
            #mutual_info = Mutualinfo(patch.reshape(-1), obj.reshape(-1))
            #print(Mutualinfo(patch.reshape(-1), obj.reshape(-1)), item[0] + o[0] - JointEntropy(patch, obj))
            if mutual_info > max_info:
                tag = num
                max_info = mutual_info
        y_res.append(tag)
        patch2_xpos, patch2_ypos = tag // 7, tag % 7
        posy[patch2_xpos * 32][count] = 1
        posy[patch2_ypos * 32 + 256][count] = 1
    return x_res, y_res, posx, posy


def matchPatch_mutual(xset, yset):
    x_res = np.array([*xset[:, 3]])
    #print("x_res: ", x_res)
    #print("y_res：", np.array([*yset[:, 3]]))
    xpatch = np.array([*xset[:, 2]])
    ypatch = np.array([*yset[:, 2]])
    y_res = np.array([*yset[:, 3]])
    y = []
    posx, posy = np.zeros((512, len(xset))), np.zeros((512, len(yset)))
    for i in range(len(x_res)):
        tag = 0
        num = y_res[0]
        for j in range(len(y_res)):
            d = Mutualinfo(xpatch[i].reshape(-1), ypatch[j].reshape(-1))
            #d = xset[i][0] + yset[j][0] - JointEntropy(xpatch[i].reshape(-1), ypatch[j].reshape(-1))
            #d = Mutualinfo(xpatch[i][:,:,0].reshape(-1), ypatch[j][:,:,0].reshape(-1)) + Mutualinfo(xpatch[i][:,:,1].reshape(-1), ypatch[j][:,:,1].reshape(-1)) + Mutualinfo(xpatch[i][:,:,2].reshape(-1), ypatch[j][:,:,2].reshape(-1))
            if d > tag:
                tag = d
                num = y_res[j]
        y.append(num)
        patch1_xpos, patch1_ypos = x_res[i] // 7, x_res[i] % 7
        posx[patch1_xpos * 32][i] = 1
        posx[patch1_ypos * 32 + 256][i] = 1
        patch2_xpos, patch2_ypos = num // 7, num % 7
        posy[patch2_xpos * 32][i] = 1
        posy[patch2_ypos * 32 + 256][i] = 1
    y_res = np.array(y)

    return x_res, np.array(y_res), posx, posy


def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis


def matchPatch(xset, yset):
    x_res = np.array([*xset[:, 3]])
    #print("x_res: ", x_res)
    #print("y_res：", np.array([*yset[:, 3]]))
    xpatch = np.array([*xset[:, 2]]).reshape(len(xset), -1)
    ypatch = np.array([*yset[:, 2]]).reshape(len(yset), -1)
    print(EuclideanDistance(xpatch, ypatch))
    y_res = EuclideanDistance(xpatch, ypatch).argmin(axis = 1)
    #print("xaxis:", x_res[EuclideanDistance(xpatch, ypatch).argmax(axis = 0)])
    y_res = np.array([*yset[:, 3]])[y_res] # map argmax to the correct order
    #print(y_res)
    posx, posy = np.zeros((512, len(xset))), np.zeros((512, len(yset)))
    for i in range(len(xset)):
        patch1_xpos, patch1_ypos = x_res[i] // 7, x_res[i] % 7
        posx[patch1_xpos * 32][i] = 1
        posx[patch1_ypos * 32 + 256][i] = 1
        patch2_xpos, patch2_ypos = y_res[i] // 7, y_res[i] % 7
        posy[patch2_xpos * 32][i] = 1
        posy[patch2_ypos * 32 + 256][i] = 1
    return x_res, y_res, posx, posy


def matchPatch_dist(xset, yset):
    x_res = np.array([*xset[:, 3]])
    #print("x_res: ", x_res)
    #print("y_res：", np.array([*yset[:, 3]]))
    xpatch = np.array([*xset[:, 2]]).reshape(len(xset), -1)
    ypatch = np.array([*yset[:, 2]]).reshape(len(yset), -1)
    y_res = np.array([*yset[:, 3]])
    y = []
    posx, posy = np.zeros((512, len(xset))), np.zeros((512, len(yset)))
    for i in range(len(x_res)):
        tag = np.sqrt(np.sum((xpatch[i] - ypatch[0])**2))
        num = y_res[0]
        for j in range(len(y_res)):
            d = np.sqrt(np.sum((xpatch[i] - ypatch[j])**2))
            if d < tag:
                tag = d
                num = y_res[j]
        y.append(num)
        patch1_xpos, patch1_ypos = x_res[i] // 7, x_res[i] % 7
        posx[patch1_xpos * 32][i] = 1
        posx[patch1_ypos * 32 + 256][i] = 1
        patch2_xpos, patch2_ypos = num // 7, num % 7
        posy[patch2_xpos * 32][i] = 1
        posy[patch2_ypos * 32 + 256][i] = 1
    y_res = np.array(y)

    return x_res, np.array(y_res), posx, posy


from scipy.spatial.distance import pdist

def matchPatch_cos(xset, yset):
    x_res = np.array([*xset[:, 3]])
    #print("x_res: ", x_res)
    #print("y_res：", np.array([*yset[:, 3]]))
    xpatch = np.array([*xset[:, 2]]).reshape(len(xset), -1)
    ypatch = np.array([*yset[:, 2]]).reshape(len(yset), -1)
    y_res = np.array([*yset[:, 3]])
    y = []
    posx, posy = np.zeros((512, len(xset))), np.zeros((512, len(yset)))
    for i in range(len(x_res)):
        tag = 1 - pdist(np.vstack([xpatch[i],ypatch[0]]),'cosine')
        num = y_res[0]
        for j in range(len(y_res)):
            d = 1 - pdist(np.vstack([xpatch[i],ypatch[j]]),'cosine')
            if d > tag:
                tag = d
                num = y_res[j]
        y.append(num)
        patch1_xpos, patch1_ypos = x_res[i] // 7, x_res[i] % 7
        posx[patch1_xpos * 32][i] = 1
        posx[patch1_ypos * 32 + 256][i] = 1
        patch2_xpos, patch2_ypos = num // 7, num % 7
        posy[patch2_xpos * 32][i] = 1
        posy[patch2_ypos * 32 + 256][i] = 1
    y_res = np.array(y)

    return x_res, np.array(y_res), posx, posy


if __name__ == '__main__':
    print(pdist(np.vstack([np.array([0,1]),np.array([1,0])]),'cosine'))