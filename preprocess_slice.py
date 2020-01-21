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
meas_t=[]
test_meas_t=[]

        
    



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
        meas = np.zeros((7,4))
        for nx in range(7):
            if ((nx-3)*0.5)<radius:
                meas[nx,0] = (radius**2-((nx-3)*0.5)**2)**0.5 #radius
                meas[nx,1] = t_miua #mua
                meas[nx,2] = 0 # tar x position 
                meas[nx,3] = 0 # tar y position 
            else:
                meas[nx,0] = 0 #radius
                meas[nx,1] = 0 #mua
                meas[nx,2] = 0 # tar x position 
                meas[nx,3] = 0 # tar y position                 
        meas_t.append(meas)
        
    return ref_u_up,pert, meas_t,np.array(depth_a)
        
        
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
        radius = float(test_src_sphere[5+i*9][1])
        depth = float(test_src_sphere[5+i*9][4])
        t_miua = float(test_src_sphere[5+i*9][3])
        miua0 = float(test_src_sphere[5+i*9][2])
        meas1 = np.zeros((7,4))
        for nx in range(7):
            if ((nx-3)*0.5)<radius:
                meas1[nx,0] = (radius**2-((nx-3)*0.5)**2)**0.5 #radius
                meas1[nx,1] = t_miua #mua
                meas1[nx,2] = 0 # tar x position 
                meas1[nx,3] = 0 # tar y position 
            else:
                meas1[nx,0] = 0 #radius
                meas1[nx,1] = 0 #mua
                meas1[nx,2] = 0 # tar x position 
                meas1[nx,3] = 0 # tar y position                 
        test_meas_t.append(meas1)
        
    return ref_u_up1,pert_test,test_meas_t
