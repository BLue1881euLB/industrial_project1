from jutaijiancai import jutaijiancai

def newcalss():
    for i in range(1,6):
        LSTM = jutaijiancai("20180417",i)
        LSTM.process()
    pass

newcalss()
