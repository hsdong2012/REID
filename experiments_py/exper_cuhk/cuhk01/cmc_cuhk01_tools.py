def readList(name_list,name_dict,sequence_flag):
    import random
    # import os
    probes = {}
    gallerys = {}
    if sequence_flag == False:
        
        for key in name_list:
            length = len(name_dict[key][:])
            probe_index = random.randint(0,length-1)
            if not probes.has_key(key):
                probes[key] = []
            probes[key].append(name_dict[key][probe_index])
            gallery_index = random.randint(0,length-1)
            while probe_index == gallery_index:
                gallery_index = random.randint(0,length-1)
            if not gallerys.has_key(key):
                gallerys[key] = []
            gallerys[key].append(name_dict[key][gallery_index])
    else:
        for key in name_list:
            length = len(name_dict[key][:])
            probe_index = random.randint(0,length-1)
            if not probes.has_key(key):
                probes[key] = []
            probes[key].append(name_dict[key][probe_index])
            buffer = []
            buffer = name_dict[key][:]
            del buffer[probe_index]
            if not gallerys.has_key(key):
                gallerys[key] = []
            gallerys[key].append(buffer) 
        
    if len(probes)!=len(gallerys):
        print('something wrong! list length does not match!/n')
        return 0
    else:
        return probes,gallerys
def plotCMC(cmcDict,pathname):
    import matplotlib.pyplot as plt
    get_ipython().magic(u'matplotlib inline')   
    from matplotlib.legend_handler import HandlerLine2D
    import numpy as np

    #plot the cmc curve, record CVPR from the pdf paper.cmc[0,4,8,12,16,21,25,29,33,37,41,45,49]
    rank2show=25
    rankStep=1
    cmcIndex=np.arange(0,rank2show,rankStep)   #0,5,10,15,20,25

    colorList=['rv-','g^-','bs-','yp-','c*-','mv-','kd-','gs-','b^-']
    #start to plot
    plt.ioff()
    fig = plt.figure(figsize=(6,5),dpi=180)
    sortedCmcDict = sorted(cmcDict.items(), key=lambda (k, v): v[1])[::-1]
    for idx in range(len(sortedCmcDict)):
        cmc_dictList=sortedCmcDict[idx]
        cmc_name=cmc_dictList[0]
        cmc_list=cmc_dictList[1]
        #print cmc_name,": ",cmc_list
        #x for plot
        x_point=[item+1 for item in cmcIndex]
        x_line=range(rank2show)
        x_plot=[temp+1 for temp in x_line]
        #start plot
        plt.plot(x_plot, cmc_list[x_line],colorList[idx],label="%02.02f%% %s"%(100*cmc_list[0],cmc_name))
        plt.plot(x_point,cmc_list[cmcIndex],colorList[idx]+'.')
        #plt.legend(loc=4,handler_map={line: HandlerLine2D(numpoints=1)})
        #idx of color +1
        idx+=1
    #something to render

    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.xticks(np.arange(0,rank2show+1,5))
    plt.yticks(np.arange(0,1.01,0.1))
    plt.grid()
    plt.legend(loc=4)
    plt.savefig(pathname)
    plt.show() 

def parse_dataset(DATA_DIR, filename_test):
    import os
    file_list_a=os.listdir(DATA_DIR)
    name_dict={}

    for name in file_list_a:
        if name[-3:]=='png':
            id = name[:4]
            if not name_dict.has_key(id):
                name_dict[id]=[]
            name_dict[id].append(DATA_DIR+name)




    # choose test ids:


    file_object = open(filename_test)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    test_dict = {}

    lines = all_the_text.split('\n')
    for filename in lines:
        if filename!='':
            if name_dict.has_key(filename):
                test_dict[filename] = name_dict[filename]
    print len(test_dict)
    key_list = []
    for key in test_dict.keys():
        key_list.append(key)
    
    return [name_dict, key_list]