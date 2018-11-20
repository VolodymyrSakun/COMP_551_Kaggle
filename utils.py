import pickle 

def saveObject(fileName, obj):
    f = open(fileName, "wb")
    pickle.dump(obj, f)
    f.close()
    return

def loadObject(fileName):
    f = open(fileName, "rb")
    obj = pickle.load(f)
    f.close()
    return obj
