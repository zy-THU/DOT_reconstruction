import csv,re
import numpy as np

src = []
src_sphere = []
src0 =[]
src1=[]
real_im_list = []
real_im_list_sphere = []
real_im0_list = []
real_im1_list = []
miu_t = []
miu_t_sphere = []
test_src = []
test_src_sphere = []
test_real_im_list =[]
test_real_im_list_sphere =[]
test_miu_t_sphere = []
test_miu_t = []


        
    



def dataprocess():
    ## ref
    depth_a = []
    with open('./Tdata/ref.csv')as f0:
    #with open('background_fixbackground_with_scan_tradius&depth.csv')as f0:
        f0_csv = csv.reader(f0)
        for row in f0_csv:
            src0.append(row)  #row[0] t_radius row[1] depth row[2] source_n

    pattern = re.compile('.{5}\d*E.\d{1,2}')
    
    for r in src0[5:]:
        comp0=[]
        for signal0 in r[7:]:
      
            c = signal0.replace('i', 'j')
            comp0.append(complex(c))
        resort0 = [comp0[2],comp0[6],comp0[11],comp0[13],comp0[9],comp0[7]
                ,comp0[5],comp0[0],comp0[8],comp0[10],comp0[12],comp0[4],comp0[3],comp0[1]]
        real_im0_list.append(resort0)   
    real_im0=np.array(real_im0_list)
    ref_u = real_im0.reshape((int(len(real_im0_list)/9)),9*14,1)
    ref_u_real = ref_u.real
    ref_u_im = ref_u.imag
    ref_u_up = np.concatenate((ref_u_real,ref_u_im),axis = 1).reshape((int(len(real_im0_list)/9)),9*14*2)
    ''' #other geometry target
    with open('fixbackground_with_scan_tradius&depth.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            src.append(row)  #row[0] t_radius row[1] depth row[2] source_n
    
    for r in src[5:]:
        comp=[]
        for signal in r[4:]:
            c = signal.replace('i', 'j')
            comp.append(complex(c))
        real_im_list.append(comp)
    real_im0 = np.array(real_im0_list*int(len(real_im_list)/9)).reshape((int(len(real_im_list)/9),9*14))
    real_im = (np.array(real_im_list).reshape((int(len(real_im_list)/9),9*14)) / real_im0 -1)#/real_im0
    real_im_real = real_im.real
    real_im_im = real_im.imag
    real_im = np.concatenate((real_im_real,real_im_im),axis = 1)
    #mu0 = np.array([src[6][2],src[6][3]]*(int(len(real_im_list)/9))).reshape((int(len(real_im_list)/9),2))
    #real_im = np.concatenate([real_im, mu0], axis = 1)
    
    for i in range(int((len(src)-5)/9)):    
        miua = np.zeros((16,16, 5))
        radius = float(src[5+i*9][0])
        depth = float(src[5+i*9][1])
        t_miua = float(src[5+i*9][2])
        for numb in range(int(radius / 0.5)):
            if numb > 2:
                numb = 2
            edge_n_l = int(radius/0.25)
            edge_n_r = round(radius/0.25)
            miua[8-edge_n_l:8+edge_n_r,8-edge_n_l:8+edge_n_r,2+numb] = t_miua
            miua[8-edge_n_l:8+edge_n_r,8-edge_n_l:8+edge_n_r,2-numb] = t_miua
        miu_t.append(miua)
        miu_T = np.array(miu_t).reshape((-1,16,16,5))
    '''
    with open('./Tdata/lesion.csv')as f1:
    #with open('.\Tdata\sphere_fixbackground_with_scan_tradius&depth.csv')as f1:
        f1_csv = csv.reader(f1)
        for row in f1_csv:
            src_sphere.append(row)  #row[0] t_radius row[1] depth row[2] source_n
    
    for r in src_sphere[5:]:
        comp_sphere=[]
        for signal in r[7:]:
            c = signal.replace('i', 'j')
            comp_sphere.append(complex(c))
        resort = [comp_sphere[2],comp_sphere[6],comp_sphere[11],comp_sphere[13],comp_sphere[9],comp_sphere[7]
            ,comp_sphere[5],comp_sphere[0],comp_sphere[8],comp_sphere[10],comp_sphere[12],comp_sphere[4],comp_sphere[3],comp_sphere[1]]
        real_im_list_sphere.append(resort)
    # real_im0 = np.array(real_im0_list*int(len(real_im_list_sphere)/9)).reshape((int(len(real_im_list_sphere)/9),9*14))    
    real_im_sphere = ((np.array(real_im_list_sphere).reshape((int(len(real_im_list_sphere)/9)),126) ) )#/ref_u
    real_im_sphere_real = real_im_sphere.real
    real_im_sphere_im = real_im_sphere.imag
    real_im_sphere = np.concatenate((real_im_sphere_real,real_im_sphere_im),axis = 1)

    pert = real_im_sphere
    
    for i in range(int((len(src_sphere)-5)/9)):    
        
        radius = float(src_sphere[5+i*9][1])
        depth = float(src_sphere[5+i*9][4])
        depth_a.append(depth)
        t_miua = float(src_sphere[5+i*9][3])
        miua0 = float(src_sphere[5+i*9][2])
        miua_sphere = np.zeros((16,16, 3))
        for nx in range(16):
            for ny in range(16):
                for nz in range(3):
                    '''
                    if radius < 0.5:
                        if ((nx*0.25-2)**2+(ny*0.25-2)**2)**0.5 <= radius:
                            miua_sphere[nx,ny,1] = t_miua
                    elif radius < 0.75:
                        if ((nx*0.25-2)**2+(ny*0.25-2)**2+0.25**2)**0.5 <= radius:
                            miua_sphere[nx,ny,0] = t_miua
                            miua_sphere[nx,ny,1] = t_miua 
                    else:
                    '''
                    if ((nx*0.25-2)**2+(ny*0.25-2)**2+((nz-1)*0.5)**2)**0.5 <= radius:
                        miua_sphere[nx,ny,nz] = t_miua  
                    else:
                        miua_sphere[nx,ny,nz] = miua0
        miu_t_sphere.append(miua_sphere)
        miu_T_sphere = np.array(miu_t_sphere).reshape((-1,16,16,3))
    miu_T = miu_T_sphere
        
    return ref_u_up,pert, miu_T,np.array(depth_a)
        
        
def test_data():
    with open('./Tdata/test_ref.csv')as f2:
    #with open('./Tdata/test_rref.csv')as f2:
        f2_csv = csv.reader(f2)
        for row in f2_csv:
            src1.append(row)  #row[0] t_radius row[1] depth row[2] source_n

#    pattern = re.compile('.{5}\d*E.\d{1,2}')
    
    for r in src1[5:]:
        comp0=[]
        for signal1 in r[7:]:
      
            c = signal1.replace('i', 'j')
            comp0.append(complex(c))
        resort0 = [comp0[2],comp0[6],comp0[11],comp0[13],comp0[9],comp0[7]
                ,comp0[5],comp0[0],comp0[8],comp0[10],comp0[12],comp0[4],comp0[3],comp0[1]]
        real_im1_list.append(resort0)   
    real_im1=np.array(real_im1_list)
    ref1_u = real_im1.reshape((int(len(real_im1_list)/9)),126,1)
    ref1_u_real = ref1_u.real
    ref1_u_im = ref1_u.imag
    ref_u_up1 = np.concatenate((ref1_u_real,ref1_u_im),axis = 1).reshape((int(len(real_im1_list)/9)),9*14*2)
    '''# other geometry target
    with open('test_fixbackground_with_scan_tradius&depth.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            test_src.append(row)  #row[0] t_radius row[1] depth row[2] source_n
    
    pattern = re.compile('.{5}\d*E.\d{1,2}')
    
    for r in test_src[5:]:
        comp=[]
        for signal in r[4:]:
            c = signal.replace('i', 'j')
            comp.append(complex(c))
        test_real_im_list.append(comp)
    real_im0 = np.array(real_im0_list*int(len(test_real_im_list)/9)).reshape((int(len(test_real_im_list)/9),9*14))
    test_real_im = (np.array(test_real_im_list).reshape((int(len(test_real_im_list)/9),9*14)) / real_im0 -1)#/real_im0
    test_real_im_real = test_real_im.real
    test_real_im_im = test_real_im.imag
    test_real_im = np.concatenate((test_real_im_real,test_real_im_im),axis = 1)
    
    for i in range(int((len(test_src)-5)/9)):    
        miua = np.zeros((16,16, 5))
        radius = float(test_src[5+i*9][0])
        depth = float(test_src[5+i*9][1])
        t_miua = float(test_src[5+i*9][2])
        for numb in range(int(radius / 0.5)):
            if numb > 2:
                numb = 2
            edge_n_l = int(radius/0.25)
            edge_n_r = round(radius/0.25)
            miua[8-edge_n_l:8+edge_n_r,8-edge_n_l:8+edge_n_r,2+numb] = t_miua
            miua[8-edge_n_l:8+edge_n_r,8-edge_n_l:8+edge_n_r,2-numb] = t_miua
        test_miu_t.append(miua)
        test_miu_T = np.array(test_miu_t).reshape((-1,16,16,5))
    '''
    
    with open('./Tdata/test_lesion.csv')as fts:
    #with open('.\Tdata\sphere_test_fixbackground_with_scan_tradius&depth.csv')as fts:
        fts_csv = csv.reader(fts)
        for row in fts_csv:
            test_src_sphere.append(row)  #row[0] t_radius row[1] depth row[2] source_n
    
#    pattern = re.compile('.{5}\d*E.\d{1,2}')
    
    for r in test_src_sphere[5:]:
        comp_sphere=[]
        for signal in r[7:]:
            c = signal.replace('i', 'j')
            comp_sphere.append(complex(c))
        resort = [comp_sphere[2],comp_sphere[6],comp_sphere[11],comp_sphere[13],comp_sphere[9],comp_sphere[7]
            ,comp_sphere[5],comp_sphere[0],comp_sphere[8],comp_sphere[10],comp_sphere[12],comp_sphere[4],comp_sphere[3],comp_sphere[1]]
        test_real_im_list_sphere.append(resort)
    
    test_real_im_sphere = ((np.array(test_real_im_list_sphere).reshape((int(len(test_real_im_list_sphere)/9)),126) ) )#/ref1_u
    test_real_im_sphere_real = test_real_im_sphere.real
    test_real_im_sphere_im = test_real_im_sphere.imag
    test_real_im_sphere = np.concatenate((test_real_im_sphere_real,test_real_im_sphere_im),axis = 1)

    pert_test = test_real_im_sphere 
    
    for i in range(int((len(test_src_sphere)-5)/9)):    
        miua_sphere = np.zeros((16,16, 3))
        radius = float(test_src_sphere[5+i*9][1])
        depth = float(test_src_sphere[5+i*9][4])
        t_miua = float(test_src_sphere[5+i*9][3])
        miua0 = float(test_src_sphere[5+i*9][2])
        for nx in range(16):
            for ny in range(16):
                for nz in range(3):
                    '''
                    if radius < 0.5:
                        if ((nx*0.25-2)**2+(ny*0.25-2)**2)**0.5 <= radius:
                            miua_sphere[nx,ny,1] = t_miua
                    elif radius < 0.75:
                        if ((nx*0.25-2)**2+(ny*0.25-2)**2+0.25**2)**0.5 <= radius:
                            miua_sphere[nx,ny,0] = t_miua
                            miua_sphere[nx,ny,1] = t_miua
                    else:
                    '''
                    if ((nx*0.25-2)**2+(ny*0.25-2)**2+((nz-1)*0.5)**2)**0.5 <= radius:
                        miua_sphere[nx,ny,nz] = t_miua
                    else:
                        miua_sphere[nx,ny,nz] = miua0
        test_miu_t_sphere.append(miua_sphere)
        test_miu_T_sphere = np.array(test_miu_t_sphere).reshape((-1,16,16,3))    
    #test_miu_T = np.concatenate((test_miu_T,test_miu_T_sphere),axis=0) 
    test_miu_T = test_miu_T_sphere
    return ref_u_up1,pert_test,test_miu_T

    '''
    with open('phaton.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            test_src.append(row)
    test_real_im = np.array(test_src).reshape(-1,252)
#    test_real_im_real = test_real_im[:,:126].reshape(-1,9,14)
#    test_real_im_im = test_real_im[:,126:].reshape(-1,9,14)
#    test_real_im = np.concatenate((test_real_im_real,test_real_im_im),axis=0).reshape(-1,252)
    test_real_im = test_real_im.astype('float64')
    return test_real_im
    '''