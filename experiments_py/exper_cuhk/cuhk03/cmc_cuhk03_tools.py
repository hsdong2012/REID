def readList(list_name, num_list, DATA_DIR, frames_l, balance): 
    import random
    import os
    import glob
    file_object = open(list_name)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    lines = all_the_text.split('\n')
    #print all_the_text
    
    f = open(num_list, 'r')
    try:
        f_list = f.readlines()
    # print f_lines
    finally:
        f.close()
    
    num_dict = {}
    for line in f_list:
        num_dict[line.split(' ')[0]] = int(line.split(' ')[1])
    
#     print lines
    random.shuffle(lines[:])
    
    probes={}
    gallerys={}
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            idkey =  '%02d_%04d'%(campair_no, person_id)
            
            file_num = num_dict[idkey]
#             print file_num
            img_list = []
            count = 0 
            index = 0 
            while count<file_num:
                onefilename = DATA_DIR +  'campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,index+1)
                #print onefilename
                if os.path.isfile(onefilename):
                    img_list.append(onefilename)
                    count += 1
#                     print count
                index += 1
            
            #print int(campair_no)
            probes[idkey] = []
            gallerys[idkey] = []
            
            probe_start = random.randint(0,file_num-frames_l)
            gallery_start = random.randint(0,file_num-frames_l)
            while abs((gallery_start-probe_start))<frames_l/2:
                gallery_start = random.randint(0,file_num-frames_l)
            for numid in range(frames_l):
                if balance:
                    probe_filename = img_list[probe_start+numid]
                else:
                    probe_filename = img_list[probe_start]
                gallery_filename = img_list[gallery_start+numid]
                probes[idkey].append(probe_filename)
                gallerys[idkey].append(gallery_filename)
                
    if len(probes.keys())!=len(gallerys.keys()):
        print('something wrong! list length does not match!/n')
        return 0
    else:
        return probes,gallerys
def plotCMC(cmcDict,pathname):
    import matplotlib.pyplot as plt
    # get_ipython().magic(u'matplotlib inline')   
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
    #plt.show() 

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